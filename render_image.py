import os
import json
import gc
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import Process
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import shutil
import math
from pipeline_config import (
    LINE_Y_THRESHOLD,
    LINE_X_GAP_THRESHOLD,
    TEXT_STROKE_WIDTH,
    INPAINT_EXPAND,
    LONG_TEXT_MIN_CHARS,
    LONG_TEXT_EXPAND_PX,
)

# ================= CẤU HÌNH =================
RAW_DIR = "./frames_raw"         # Frames gốc
JSON_DIR = "./json_cache"        # JSON đã dịch
OUTPUT_DIR = "./frames_done"     # Frames output
FONT_PATH = "arial.ttf"
RENDER_THREADS = 8               # Threads per GPU
NUM_GPUS = 2                     # Số GPU

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================= HÀM VẼ =================
def normalize_box(box, width, height):
    """Chuẩn hóa box về int, đúng thứ tự và nằm trong biên ảnh."""
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return None

    x1, y1, x2, y2 = [int(v) for v in box]
    x_min, x_max = sorted((x1, x2))
    y_min, y_max = sorted((y1, y2))

    x_min = max(0, min(x_min, width - 1))
    y_min = max(0, min(y_min, height - 1))
    x_max = max(0, min(x_max, width - 1))
    y_max = max(0, min(y_max, height - 1))

    if x_max <= x_min or y_max <= y_min:
        return None
    return x_min, y_min, x_max, y_max


def build_inpaint_mask(image_shape, boxes, expand=2):
    """Tạo mask tổng cho toàn bộ vùng chữ để inpaint 1 lần."""
    h, w = image_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    for x1, y1, x2, y2 in boxes:
        x1e = max(0, x1 - expand)
        y1e = max(0, y1 - expand)
        x2e = min(w - 1, x2 + expand)
        y2e = min(h - 1, y2 + expand)
        mask[y1e:y2e + 1, x1e:x2e + 1] = 255

    return mask


def expand_box_horizontally(box, width, expand_px):
    x1, y1, x2, y2 = box
    x1 = max(0, x1 - expand_px)
    x2 = min(width - 1, x2 + expand_px)
    if x2 <= x1:
        return box
    return (x1, y1, x2, y2)


def merge_boxes_on_line(items, y_threshold=10, x_gap_threshold=40):
    """Gộp các box OCR nằm cùng dòng để inpaint/render mượt hơn."""
    if not items:
        return []

    sorted_items = sorted(items, key=lambda it: (it["box"][1], it["box"][0]))
    merged = []
    current = dict(sorted_items[0])

    for nxt in sorted_items[1:]:
        cx1, cy1, cx2, cy2 = current["box"]
        nx1, ny1, nx2, ny2 = nxt["box"]

        same_line = abs(ny1 - cy1) <= y_threshold
        gap = nx1 - cx2
        is_near = gap <= x_gap_threshold

        if same_line and is_near:
            current["box"] = [min(cx1, nx1), min(cy1, ny1), max(cx2, nx2), max(cy2, ny2)]
            current["translated"] = (current.get("translated", "") + " " + nxt.get("translated", "")).strip()
        else:
            merged.append(current)
            current = dict(nxt)

    merged.append(current)
    return merged


def get_border_background_color(cv2_img, box, border_thickness=2):
    """Lấy màu nền đại diện từ viền trong của box (BGR)."""
    x1, y1, x2, y2 = box
    roi = cv2_img[y1:y2 + 1, x1:x2 + 1]
    if roi.size == 0:
        return np.array([255, 255, 255], dtype=np.float32)

    h, w = roi.shape[:2]
    b = max(1, min(border_thickness, h // 3, w // 3))

    top = roi[:b, :]
    bottom = roi[-b:, :]
    left = roi[:, :b]
    right = roi[:, -b:]
    border_pixels = np.concatenate(
        [top.reshape(-1, 3), bottom.reshape(-1, 3), left.reshape(-1, 3), right.reshape(-1, 3)],
        axis=0,
    )
    return np.median(border_pixels, axis=0)


def get_original_text_color(cv2_img, box):
    """Ước lượng màu chữ bằng tương phản cao so với màu nền biên box (trả về RGB)."""
    x1, y1, x2, y2 = box
    roi = cv2_img[y1:y2 + 1, x1:x2 + 1]
    if roi.size == 0:
        return (0, 0, 0)

    bg_color = get_border_background_color(cv2_img, box)
    pixels = roi.reshape(-1, 3).astype(np.float32)
    dist = np.linalg.norm(pixels - bg_color, axis=1)

    if dist.size == 0:
        center_bgr = cv2_img[(y1 + y2) // 2, (x1 + x2) // 2].astype(int).tolist()
        return tuple(center_bgr[::-1])

    threshold = float(np.percentile(dist, 80))
    text_pixels = pixels[dist >= threshold]
    if text_pixels.shape[0] == 0:
        text_pixels = pixels

    bgr = np.median(text_pixels, axis=0).astype(int).tolist()
    rgb = tuple(bgr[::-1])

    if isinstance(bg_color, np.ndarray):
        bg_rgb = tuple(bg_color[::-1].astype(int).tolist())
    else:
        bg_rgb = tuple(bg_color)

    # Nếu màu chữ và nền quá gần nhau, fallback sang màu tương phản cao.
    contrast = float(np.linalg.norm(np.array(rgb, dtype=np.float32) - np.array(bg_rgb, dtype=np.float32)))
    if contrast < 30:
        bg_luma = 0.299 * bg_rgb[0] + 0.587 * bg_rgb[1] + 0.114 * bg_rgb[2]
        rgb = (0, 0, 0) if bg_luma > 140 else (255, 255, 255)

    return rgb


def wrap_text_by_width(draw, text, font, max_width):
    words = text.split()
    lines = []
    line = ""
    for word in words:
        test_line = (line + " " + word).strip()
        bbox = draw.textbbox((0, 0), test_line, font=font)
        w = bbox[2] - bbox[0]
        if w <= max_width:
            line = test_line
        else:
            if line: 
                lines.append(line)
            line = word
    if line: 
        lines.append(line)
    return lines


def get_optimal_font_and_lines(draw, text, font_path, box_width, box_height, padding=4):
    max_size = min(int(box_height), 120)
    min_size = 10
    
    if not os.path.exists(font_path):
        font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    
    safe_width = box_width - (padding * 2)
    safe_height = box_height - (padding * 2)
    spacing = 4
    default_font = ImageFont.load_default()

    for size in range(max_size, min_size, -2):
        if size <= 0: 
            break
        try:
            font = ImageFont.truetype(font_path, size)
        except:
            font = default_font
            break
        
        lines = wrap_text_by_width(draw, text, font, safe_width)
        bbox_sample = draw.textbbox((0, 0), "Ay", font=font)
        line_height = bbox_sample[3] - bbox_sample[1]
        total_text_height = (len(lines) * line_height) + ((len(lines) - 1) * spacing)
        
        if total_text_height <= safe_height:
            return font, lines, total_text_height, line_height

    try: 
        font = ImageFont.truetype(font_path, min_size)
    except: 
        font = default_font
    
    lines = wrap_text_by_width(draw, text, font, safe_width)
    return font, lines, safe_height, 12


def render_text_in_box(draw, translated, font_path, x_min, y_min, x_max, y_max, text_color):
    box_width = x_max - x_min
    box_height = y_max - y_min
    
    if box_width < 10 or box_height < 10: 
        return
    
    font, lines, text_block_height, line_height = get_optimal_font_and_lines(
        draw, translated, font_path, box_width, box_height
    )
    
    # Tính vị trí bắt đầu (center vertical)
    start_y = y_min + (box_height - text_block_height) // 2
    if start_y < y_min: 
        start_y = y_min + 2
    
    current_y = start_y
    spacing = 4
    
    # Vẽ từng dòng text (center horizontal)
    shadow_color = (0, 0, 0) if np.mean(text_color) > 128 else (255, 255, 255)
    stroke_color = shadow_color

    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        line_w = bbox[2] - bbox[0]
        start_x = x_min + (box_width - line_w) // 2
        draw.text((start_x + 1, current_y + 1), line, fill=shadow_color, font=font)
        draw.text(
            (start_x, current_y),
            line,
            fill=text_color,
            font=font,
            stroke_width=TEXT_STROKE_WIDTH,
            stroke_fill=stroke_color,
        )
        current_y += line_height + spacing


# ================= RENDER WORKER =================
def render_image_worker(task):
    """Render 1 ảnh từ JSON"""
    img_path, json_path, out_path = task
    
    try:
        # Kiểm tra JSON có tồn tại không
        if not os.path.exists(json_path):
            # Không có JSON → copy ảnh gốc
            shutil.copy(img_path, out_path)
            return True
        
        # Load JSON
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        ocr_data = data.get('texts', [])
        
        # Không có text → copy ảnh gốc
        if len(ocr_data) == 0:
            shutil.copy(img_path, out_path)
            return True
        
        # Load ảnh bằng OpenCV để xử lý inpainting nền
        cv2_img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if cv2_img is None:
            print(f"\n❌ Cannot read image: {img_path}")
            return False

        h, w = cv2_img.shape[:2]
        raw_items = []

        # Chuẩn bị dữ liệu render từ OCR gốc
        for item in ocr_data:
            # Ưu tiên dùng 'translated', fallback sang 'text'
            translated = item.get('translated', '').strip()
            if not translated:
                translated = item.get('text', '').strip()
            
            if not translated:
                continue
            
            normalized_box = normalize_box(item.get('box', []), w, h)
            if not normalized_box:
                continue

            raw_items.append({
                "translated": translated,
                "box": list(normalized_box),
            })

        # Gộp word-level boxes thành line-level boxes
        merged_items = merge_boxes_on_line(
            raw_items,
            y_threshold=LINE_Y_THRESHOLD,
            x_gap_threshold=LINE_X_GAP_THRESHOLD,
        )

        render_items = []
        boxes = []
        for item in merged_items:
            box = tuple(item["box"])
            translated_text = item["translated"]
            if len(translated_text) >= LONG_TEXT_MIN_CHARS and LONG_TEXT_EXPAND_PX > 0:
                box = expand_box_horizontally(box, w, LONG_TEXT_EXPAND_PX)

            color = get_original_text_color(cv2_img, box)
            boxes.append(box)
            render_items.append((translated_text, box, color))

        if len(render_items) == 0:
            shutil.copy(img_path, out_path)
            return True

        # Xóa chữ cũ bằng inpainting để giữ kết cấu nền tự nhiên
        inpaint_mask = build_inpaint_mask(cv2_img.shape, boxes, expand=INPAINT_EXPAND)
        cleaned_cv2 = cv2.inpaint(cv2_img, inpaint_mask, 3, cv2.INPAINT_TELEA)

        # Chuyển sang PIL để vẽ text đa ngôn ngữ đẹp hơn
        img_pil = Image.fromarray(cv2.cvtColor(cleaned_cv2, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)

        # Render chữ mới không vẽ box nền
        for translated, (x1, y1, x2, y2), color in render_items:
            render_text_in_box(draw, translated, FONT_PATH, x1, y1, x2, y2, color)

        # Save
        img_pil.save(out_path, quality=95)
        return True
        
    except Exception as e:
        print(f"\n❌ Error rendering {os.path.basename(img_path)}: {e}")
        return False


# ================= WORKER PROCESS PER GPU =================
def gpu_worker(gpu_id, tasks):
    """Worker chạy trên 1 GPU với ThreadPoolExecutor"""
    # Set GPU (không thực sự cần vì render chỉ dùng CPU, nhưng để tránh conflict)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    print(f"🚀 GPU Worker {gpu_id} started with {len(tasks)} tasks")
    
    success_count = 0
    
    # Render với threads
    with ThreadPoolExecutor(max_workers=RENDER_THREADS) as executor:
        futures = {executor.submit(render_image_worker, task): task for task in tasks}
        
        for future in as_completed(futures):
            if future.result():
                success_count += 1
    
    print(f"✅ GPU Worker {gpu_id} completed: {success_count}/{len(tasks)}")
    return success_count


# ================= MAIN =================
def main():
    print("=" * 70)
    print("🎨 MULTI-GPU RENDER IMAGES FROM TRANSLATED JSON")
    print("=" * 70)
    print(f"Raw frames: {RAW_DIR}")
    print(f"JSON cache: {JSON_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"GPUs: {NUM_GPUS}")
    print(f"Threads per GPU: {RENDER_THREADS}")
    print(f"Total workers: {NUM_GPUS * RENDER_THREADS}")
    print("=" * 70)
    
    # Collect tasks
    tasks = []
    
    for root, dirs, files in os.walk(RAW_DIR):
        for f in files:
            if f.lower().endswith(('.jpg', '.png', '.jpeg')):
                img_path = os.path.join(root, f)
                
                # JSON path
                rel_path = os.path.relpath(img_path, RAW_DIR)
                rel_dir = os.path.dirname(rel_path)
                json_name = f.replace('.jpg', '.json').replace('.png', '.json').replace('.jpeg', '.json')
                json_path = os.path.join(JSON_DIR, rel_dir, json_name)
                
                # Output path
                out_path = os.path.join(OUTPUT_DIR, rel_path)
                out_dir = os.path.dirname(out_path)
                os.makedirs(out_dir, exist_ok=True)
                
                # Chỉ render nếu chưa có output
                if not os.path.exists(out_path):
                    tasks.append((img_path, json_path, out_path))
    
    total = len(tasks)
    
    if total == 0:
        print("✅ All images already rendered!")
        return
    
    print(f"📦 Found {total} images to render\n")
    
    # Chia tasks cho mỗi GPU
    import random
    random.shuffle(tasks)  # Shuffle để load balance tốt hơn
    
    chunk_size = math.ceil(total / NUM_GPUS)
    task_chunks = [tasks[i:i + chunk_size] for i in range(0, total, chunk_size)]
    
    print(f"📊 Task distribution:")
    for i, chunk in enumerate(task_chunks):
        print(f"   GPU {i}: {len(chunk)} images")
    print()
    
    # Spawn processes cho mỗi GPU
    processes = []
    for gpu_id in range(min(NUM_GPUS, len(task_chunks))):
        p = Process(target=gpu_worker, args=(gpu_id, task_chunks[gpu_id]))
        p.start()
        processes.append(p)
    
    # Wait for all processes
    for p in processes:
        p.join()
    
    print("\n" + "=" * 70)
    print(f"✅ ALL GPU WORKERS COMPLETED")
    print(f"📁 Output: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    # Để multiprocessing hoạt động đúng
    try:
        from multiprocessing import set_start_method
        set_start_method('spawn')
    except RuntimeError:
        pass
    
    main()