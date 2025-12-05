import os
import glob
import json
import cv2
import numpy as np
import math
import time
from multiprocessing import Process
import paddle
from PIL import Image, ImageDraw, ImageFont
from paddleocr import PaddleOCR
from deep_translator import GoogleTranslator

# ================= CẤU HÌNH PHẦN CỨNG =================
NUM_GPUS = 4           # Bạn có 4 GPU
WORKERS_PER_GPU = 1    # 1 process cho mỗi GPU (Nếu VRAM 4060 8GB dư thì tăng lên 2)
BATCH_SIZE_OCR = 16    # Số ảnh tống vào VRAM cùng lúc trên mỗi GPU

# ================= CẤU HÌNH =================
RAW_DIR = "./frames_raw"         # Ảnh gốc
JSON_DIR = "./json_cache"        # Nơi lưu JSON
TRANSLATED_DIR = "./frames_done" # Ảnh kết quả
FONT_PATH = "arial.ttf"          # Đổi thành đường dẫn font Linux nếu cần

LANG_SOURCE = 'de' 
LANG_TARGET = 'en'
BATCH_SIZE_OCR = 8  # Số ảnh OCR cùng lúc (Tăng lên nếu có GPU)
BATCH_SIZE_TRANS = 50 # Số từ dịch cùng lúc

# ================= KHỞI TẠO =================
# PaddleOCR
try:
    ocr_engine = PaddleOCR(lang='german')
except:
    print("⚠️ Cảnh báo: Không load được PaddleOCR.")
    exit()

translator = GoogleTranslator(source=LANG_SOURCE, target=LANG_TARGET)


# ==========================================================
#  CÁC HÀM VẼ TỐI ƯU (Của bạn cung cấp)
# ==========================================================

def wrap_text_by_width(draw, text, font, max_width):
    """Trả về list các dòng text đã wrap theo max_width."""
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
            if line: lines.append(line)
            line = word
    if line: lines.append(line)
    return lines

def get_optimal_font_and_lines(draw, text, font_path, box_width, box_height, padding=4):
    """
    Thử font từ size lớn xuống nhỏ.
    Với mỗi size: wrap text -> đo tổng chiều cao -> nếu vừa thì chốt.
    """
    max_size = min(int(box_height), 120) # Giới hạn size max
    min_size = 10
    
    # Check font path, fallback nếu lỗi
    if not os.path.exists(font_path):
        # Đường dẫn font dự phòng cho Linux
        font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    
    safe_width = box_width - (padding * 2)
    safe_height = box_height - (padding * 2)
    spacing = 4 

    # Loop giảm dần size
    best_font = None
    best_lines = [text]
    best_total_h = 0
    
    # Load default font để fallback cuối cùng
    default_font = ImageFont.load_default()

    for size in range(max_size, min_size, -2):
        if size <= 0: break
        try:
            font = ImageFont.truetype(font_path, size)
        except:
            font = default_font
            break

        lines = wrap_text_by_width(draw, text, font, safe_width)
        
        # Đo chiều cao 1 dòng mẫu
        bbox_sample = draw.textbbox((0, 0), "Ay", font=font)
        line_height = bbox_sample[3] - bbox_sample[1]
        
        total_text_height = (len(lines) * line_height) + ((len(lines) - 1) * spacing)

        if total_text_height <= safe_height:
            return font, lines, total_text_height, line_height

    # Fallback về size nhỏ nhất
    try:
        font = ImageFont.truetype(font_path, min_size)
    except:
        font = default_font
        
    lines = wrap_text_by_width(draw, text, font, safe_width)
    return font, lines, safe_height, 12


def render_text_in_box(draw, translated, font_path, x_min, y_min, x_max, y_max):
    """Hàm vẽ chính: Vẽ nền trắng và Text căn giữa"""
    box_width = x_max - x_min
    box_height = y_max - y_min
    
    if box_width < 10 or box_height < 10: return

    # Lấy font và lines tối ưu
    font, lines, text_block_height, line_height = get_optimal_font_and_lines(
        draw, translated, font_path, box_width, box_height
    )

    # 1. Vẽ nền trắng che chữ cũ
    draw.rectangle([(x_min, y_min), (x_max, y_max)], fill="white")

    # 2. Tính toán căn giữa dọc (Vertical Center)
    start_y = y_min + (box_height - text_block_height) // 2
    if start_y < y_min: start_y = y_min + 2

    # 3. Vẽ từng dòng
    current_y = start_y
    spacing = 4
    
    for line in lines:
        # Căn giữa ngang (Horizontal Center)
        bbox = draw.textbbox((0, 0), line, font=font)
        line_w = bbox[2] - bbox[0]
        start_x = x_min + (box_width - line_w) // 2

        draw.text((start_x, current_y), line, fill="black", font=font)
        current_y += line_height + spacing


# ================= HÀM XỬ LÝ CỦA TỪNG WORKER (GPU) =================
def worker_ocr_process(gpu_id, image_files):
    """
    Hàm này sẽ chạy trên một Process riêng biệt.
    Nó sẽ chiếm dụng riêng 1 GPU được chỉ định.
    """
    # 1. Cấu hình để Process này chỉ nhìn thấy 1 GPU duy nhất
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    print(f"🚀 Worker khởi động trên GPU {gpu_id} | Xử lý {len(image_files)} ảnh...")

    # 2. Khởi tạo PaddleOCR (Phải khởi tạo bên trong process)
    # use_gpu=True là bắt buộc
    try:
        ocr_engine = PaddleOCR(lang='german', use_gpu=True)
    except Exception as e:
        print(f"❌ Lỗi khởi tạo GPU {gpu_id}: {e}")
        return

    # 3. Chạy vòng lặp xử lý Batch
    total_files = len(image_files)
    
    # Chia nhỏ danh sách file thành các batch nhỏ hơn để tống vào GPU
    for i in range(0, total_files, BATCH_SIZE_OCR):
        batch_items = image_files[i : i + BATCH_SIZE_OCR]
        batch_imgs = []
        valid_items = []

        # Load ảnh vào RAM
        for img_path, json_path, filename in batch_items:
            img = cv2.imread(img_path)
            if img is not None:
                batch_imgs.append(img)
                valid_items.append((img_path, json_path, filename))
        
        if not batch_imgs: continue

        try:
            # Gửi batch vào GPU
            results = ocr_engine.ocr(batch_imgs)
            
            # Xử lý kết quả trả về
            for idx, res in enumerate(results):
                _, json_out_path, fname = valid_items[idx]
                ocr_data = []

                if res:
                    # Xử lý output format (Dict hoặc List)
                    if isinstance(res, dict) and 'rec_texts' in res: # New version
                        texts = res.get('rec_texts', [])
                        boxes = res.get('dt_polys', [])
                        scores = res.get('rec_scores', [])
                        for b, t, c in zip(boxes, texts, scores):
                            if c > 0.5:
                                xs, ys = [p[0] for p in b], [p[1] for p in b]
                                ocr_data.append({
                                    "box": [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))],
                                    "text": t, "confidence": float(c), "translated": ""
                                })
                    elif isinstance(res, list): # Old version
                        for line in res:
                            content = line[1]
                            txt = content if isinstance(content, str) else content[0]
                            cnf = 1.0 if isinstance(content, str) else content[1]
                            if cnf > 0.5:
                                pts = line[0]
                                xs, ys = [p[0] for p in pts], [p[1] for p in pts]
                                ocr_data.append({
                                    "box": [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))],
                                    "text": txt, "confidence": float(cnf), "translated": ""
                                })

                # Lưu JSON
                with open(json_out_path, 'w', encoding='utf-8') as f:
                    json.dump({"frame": fname, "texts": ocr_data}, f, ensure_ascii=False, indent=2)

        except Exception as e:
            print(f"⚠️ Lỗi tại GPU {gpu_id}: {e}")

        # Log tiến độ đơn giản
        if i % (BATCH_SIZE_OCR * 5) == 0:
            print(f"   [GPU {gpu_id}] Đã xong {i}/{total_files}...", end="\r")

    print(f"✅ [GPU {gpu_id}] HOÀN TẤT.")

# ================= BƯỚC 1: QUẢN LÝ ĐA GPU =================
def step1_multi_gpu_ocr():
    print(f"\n🔹 BƯỚC 1: SCAN OCR VỚI {NUM_GPUS} GPU...")
    
    # 1. Quét toàn bộ file
    all_tasks = []
    for root, dirs, files in os.walk(RAW_DIR):
        rel_subdir = os.path.relpath(root, RAW_DIR)
        if rel_subdir == ".": rel_subdir = ""
        os.makedirs(os.path.join(JSON_DIR, rel_subdir), exist_ok=True)

        for f in files:
            if f.lower().endswith((".jpg", ".png", ".jpeg")):
                json_path = os.path.join(JSON_DIR, rel_subdir, f.replace(".jpg", ".json").replace(".png", ".json"))
                # Chỉ thêm ảnh chưa có JSON
                if not os.path.exists(json_path):
                    all_tasks.append((os.path.join(root, f), json_path, f))

    total_images = len(all_tasks)
    if total_images == 0:
        print("✅ Tất cả ảnh đã được OCR trước đó.")
        return

    print(f"📦 Tổng số ảnh cần xử lý: {total_images}")

    # 2. Chia đều công việc cho các GPU
    # Ví dụ: 1000 ảnh / 4 GPU = 250 ảnh/GPU
    chunk_size = math.ceil(total_images / NUM_GPUS)
    chunks = [all_tasks[i:i + chunk_size] for i in range(0, total_images, chunk_size)]

    processes = []

    # 3. Khởi chạy các Process
    start_time = time.time()
    
    for i in range(len(chunks)):
        # Nếu worker ít hơn GPU (trường hợp chia dư), chỉ chạy số lượng worker cần thiết
        if not chunks[i]: continue
        
        gpu_id = i % NUM_GPUS # 0, 1, 2, 3
        
        p = Process(target=worker_ocr_process, args=(gpu_id, chunks[i]))
        p.start()
        processes.append(p)

    # 4. Chờ tất cả hoàn thành
    for p in processes:
        p.join()

    end_time = time.time()
    print(f"\n✅ Hoàn tất toàn bộ OCR trong {end_time - start_time:.2f} giây.")

# ==========================================================
# BƯỚC 2: DỊCH (Batch)
# ==========================================================
def step2_translate_batch():
    print("\n🔹 BƯỚC 2: DỊCH THUẬT...")
    all_jsons = glob.glob(f"{JSON_DIR}/**/*.json", recursive=True)
    
    # Gom text cần dịch
    need_trans = set()
    for js in all_jsons:
        with open(js, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for item in data.get('texts', []):
                if not item.get('translated'):
                    txt = item['text'].strip()
                    if len(txt) > 1 and not txt.isdigit():
                        need_trans.add(txt)
    
    text_list = list(need_trans)
    if not text_list:
        print("✅ Tất cả đã được dịch.")
        return

    print(f"   ☁️  Dịch {len(text_list)} cụm từ...")
    trans_map = {}
    
    # Dịch batch
    for i in range(0, len(text_list), BATCH_SIZE_TRANS):
        batch = text_list[i:i+BATCH_SIZE_TRANS]
        try:
            res = translator.translate_batch(batch)
            for s, d in zip(batch, res): trans_map[s] = d
        except:
            for s in batch:
                try: trans_map[s] = translator.translate(s)
                except: pass

    # Update JSON
    cnt = 0
    for js in all_jsons:
        with open(js, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        dirty = False
        for item in data.get('texts', []):
            orig = item['text'].strip()
            if not item.get('translated') and orig in trans_map:
                item['translated'] = trans_map[orig]
                dirty = True
        
        if dirty:
            with open(js, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            cnt += 1
            
    print(f"✅ Đã cập nhật {cnt} file JSON.")


# ==========================================================
# BƯỚC 3: RENDER (Áp dụng Code tối ưu của bạn)
# ==========================================================
def step3_render_images():
    print("\n🔹 BƯỚC 3: VẼ ẢNH KẾT QUẢ (OPTIMIZED)...")

    for root, dirs, files in os.walk(RAW_DIR):
        rel_subdir = os.path.relpath(root, RAW_DIR)
        if rel_subdir == ".": rel_subdir = ""

        out_subdir = os.path.join(TRANSLATED_DIR, rel_subdir)
        json_subdir = os.path.join(JSON_DIR, rel_subdir)
        os.makedirs(out_subdir, exist_ok=True)

        for file in files:
            if not file.lower().endswith((".jpg", ".png")): continue
            
            img_path = os.path.join(root, file)
            json_path = os.path.join(json_subdir, file.replace(".jpg", ".json").replace(".png", ".json"))
            out_path = os.path.join(out_subdir, file)

            if not os.path.exists(json_path): continue
            
            print(f"   Render: {file}", end="\r")

            # Load JSON
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Load Ảnh
            img_pil = Image.open(img_path).convert("RGB")
            draw = ImageDraw.Draw(img_pil)

            # Vẽ từng box
            for item in data.get('texts', []):
                # Ưu tiên text dịch, không thì dùng gốc
                text_content = item.get('translated')
                if not text_content: 
                    text_content = item['text']

                box = item['box']
                x1, y1, x2, y2 = box

                # --- GỌI HÀM VẼ TỐI ƯU CỦA BẠN ---
                render_text_in_box(draw, text_content, FONT_PATH, x1, y1, x2, y2)

            img_pil.save(out_path)

    print("\n✅ Hoàn tất toàn bộ quy trình!")


# ================= MAIN =================
def main():
    step1_multi_gpu_ocr()
    step2_translate_batch()
    step3_render_images()

if __name__ == "__main__":
    main()