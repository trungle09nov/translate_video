import os
import json
import gc
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import shutil

# ================= CẤU HÌNH =================
RAW_DIR = "./frames_raw"         # Frames gốc
JSON_DIR = "./json_cache"        # JSON đã dịch
OUTPUT_DIR = "./frames_done"     # Frames output
FONT_PATH = "arial.ttf"
RENDER_THREADS = 8               # Số thread render song song

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================= HÀM VẼ =================
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


def render_text_in_box(draw, translated, font_path, x_min, y_min, x_max, y_max):
    box_width = x_max - x_min
    box_height = y_max - y_min
    
    if box_width < 10 or box_height < 10: 
        return
    
    font, lines, text_block_height, line_height = get_optimal_font_and_lines(
        draw, translated, font_path, box_width, box_height
    )
    
    # Vẽ background trắng
    draw.rectangle([(x_min, y_min), (x_max, y_max)], fill="white")
    
    # Tính vị trí bắt đầu (center vertical)
    start_y = y_min + (box_height - text_block_height) // 2
    if start_y < y_min: 
        start_y = y_min + 2
    
    current_y = start_y
    spacing = 4
    
    # Vẽ từng dòng text (center horizontal)
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        line_w = bbox[2] - bbox[0]
        start_x = x_min + (box_width - line_w) // 2
        draw.text((start_x, current_y), line, fill="black", font=font)
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
        
        # Load ảnh
        img_pil = Image.open(img_path).convert("RGB")
        draw = ImageDraw.Draw(img_pil)

        # Render từng text box
        for item in ocr_data:
            # Ưu tiên dùng 'translated', fallback sang 'text'
            translated = item.get('translated', '').strip()
            if not translated:
                translated = item.get('text', '').strip()
            
            if not translated:
                continue
            
            box = item.get('box', [])
            if len(box) == 4:
                x1, y1, x2, y2 = box
                render_text_in_box(draw, translated, FONT_PATH, x1, y1, x2, y2)

        # Save
        img_pil.save(out_path, quality=95)
        return True
        
    except Exception as e:
        print(f"\n❌ Error rendering {os.path.basename(img_path)}: {e}")
        return False


# ================= MAIN =================
def main():
    print("=" * 70)
    print("🎨 RENDER IMAGES FROM TRANSLATED JSON")
    print("=" * 70)
    print(f"Raw frames: {RAW_DIR}")
    print(f"JSON cache: {JSON_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Threads: {RENDER_THREADS}")
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
    
    # Render with progress bar
    success_count = 0
    
    with ThreadPoolExecutor(max_workers=RENDER_THREADS) as executor:
        # Submit all tasks
        futures = {executor.submit(render_image_worker, task): task for task in tasks}
        
        # Progress bar
        with tqdm(total=total, desc="Rendering", unit="img") as pbar:
            for future in as_completed(futures):
                if future.result():
                    success_count += 1
                pbar.update(1)
    
    print("\n" + "=" * 70)
    print(f"✅ COMPLETED: {success_count}/{total} images rendered")
    print(f"📁 Output: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()