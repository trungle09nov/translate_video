import os
import glob
import json
import cv2
import numpy as np
import math
import time
from multiprocessing import Process, set_start_method
# Lưu ý: Không import paddle hoặc khởi tạo OCR ở global scope để tránh xung đột CUDA

from PIL import Image, ImageDraw, ImageFont
from deep_translator import GoogleTranslator

# ================= CẤU HÌNH PHẦN CỨNG =================
NUM_GPUS = 4           # Số lượng GPU
BATCH_SIZE_OCR = 16    # Số ảnh xử lý trong 1 lần load vào RAM (Batch logic của code)

# ================= CẤU HÌNH THƯ MỤC =================
RAW_DIR = "./frames_raw"         
JSON_DIR = "./json_cache"        
TRANSLATED_DIR = "./frames_done" 
FONT_PATH = "arial.ttf"          

LANG_SOURCE = 'de' 
LANG_TARGET = 'en'
BATCH_SIZE_TRANS = 50 

# Khởi tạo Translator (Global OK vì nó dùng CPU/API request)
try:
    translator = GoogleTranslator(source=LANG_SOURCE, target=LANG_TARGET)
except:
    pass # Xử lý sau trong hàm dịch

# ==========================================================
#  CÁC HÀM VẼ (Giữ nguyên logic của bạn)
# ==========================================================
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
            if line: lines.append(line)
            line = word
    if line: lines.append(line)
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
        if size <= 0: break
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

    try: font = ImageFont.truetype(font_path, min_size)
    except: font = default_font
    lines = wrap_text_by_width(draw, text, font, safe_width)
    return font, lines, safe_height, 12

def render_text_in_box(draw, translated, font_path, x_min, y_min, x_max, y_max):
    box_width = x_max - x_min
    box_height = y_max - y_min
    if box_width < 10 or box_height < 10: return
    font, lines, text_block_height, line_height = get_optimal_font_and_lines(
        draw, translated, font_path, box_width, box_height
    )
    draw.rectangle([(x_min, y_min), (x_max, y_max)], fill="white")
    start_y = y_min + (box_height - text_block_height) // 2
    if start_y < y_min: start_y = y_min + 2
    current_y = start_y
    spacing = 4
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        line_w = bbox[2] - bbox[0]
        start_x = x_min + (box_width - line_w) // 2
        draw.text((start_x, current_y), line, fill="black", font=font)
        current_y += line_height + spacing

# ================= HÀM XỬ LÝ CỦA TỪNG WORKER (GPU) =================
def worker_ocr_process(gpu_id, image_files):
    """
    Worker xử lý OCR trên GPU với cơ chế bắt lỗi an toàn (Safe Parsing)
    """
    # 1. Gán cứng GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    # 2. Import Paddle
    import paddle
    from paddleocr import PaddleOCR

    # 3. Khởi tạo Engine
    print(f"🚀 Worker GPU {gpu_id} khởi động...")
    try:
        # Tắt log để đỡ rối terminal
        ocr_engine = PaddleOCR(lang='german', use_angle_cls=False)
    except Exception as e:
        print(f"❌ GPU {gpu_id} lỗi Init: {e}")
        return

    total_files = len(image_files)
    
    # Loop xử lý batch
    for i in range(0, total_files, BATCH_SIZE_OCR):
        batch_items = image_files[i : i + BATCH_SIZE_OCR]
        loaded_images = [] 
        
        # Load ảnh
        for img_path, json_path, filename in batch_items:
            try:
                img = cv2.imread(img_path)
                if img is not None:
                    loaded_images.append((img, json_path, filename))
                else:
                    print(f"⚠️ Không đọc được ảnh: {filename}")
            except:
                pass
        
        if not loaded_images: continue

        # OCR từng ảnh trong batch
        for img, json_out_path, fname in loaded_images:
            try:
                result = ocr_engine.predict(img)
                ocr_data = []

                # --- ĐOẠN CODE SỬA LỖI (SAFE PARSING) ---
                if result and isinstance(result, list) and result[0]:
                    for line in result[0]:
                        try:
                            # line chuẩn: [ [x,y...], ('text', 0.99) ]
                            box = line[0]
                            content = line[1]

                            # Kiểm tra kỹ cấu trúc content
                            if isinstance(content, (list, tuple)) and len(content) >= 2:
                                text = content[0]
                                score = content[1]
                            elif isinstance(content, str):
                                # Trường hợp hiếm: content chỉ là string
                                text = content
                                score = 1.0
                            else:
                                # Dữ liệu rác -> Bỏ qua
                                continue

                            # Chỉ lấy tin cậy > 0.5
                            if isinstance(score, (int, float)) and score > 0.5:
                                xs = [pt[0] for pt in box]
                                ys = [pt[1] for pt in box]
                                
                                ocr_data.append({
                                    "box": [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))],
                                    "text": str(text), # Ép kiểu string cho chắc
                                    "confidence": float(score),
                                    "translated": ""
                                })
                        except Exception as parse_err:
                            # Nếu 1 dòng lỗi, bỏ qua dòng đó, không crash cả chương trình
                            # print(f"⚠️ Lỗi parse dòng trong {fname}: {parse_err}") 
                            continue
                # ------------------------------------------

                # Lưu JSON
                with open(json_out_path, 'w', encoding='utf-8') as f:
                    json.dump({"frame": fname, "texts": ocr_data}, f, ensure_ascii=False, indent=2)

            except Exception as e:
                # Nếu ảnh lỗi nặng, in ra để biết nhưng KHÔNG DỪNG worker
                print(f"\n❌ Lỗi file {fname} trên GPU {gpu_id}: {e}")

        # Log tiến độ (in cùng dòng)
        if i % BATCH_SIZE_OCR == 0:
            print(f"   [GPU {gpu_id}] Xử lý: {i}/{total_files} ảnh...", end="\r")

    print(f"✅ [GPU {gpu_id}] HOÀN TẤT.")

# ================= BƯỚC 1: QUẢN LÝ ĐA GPU =================
def step1_multi_gpu_ocr():
    print(f"\n🔹 BƯỚC 1: SCAN OCR VỚI {NUM_GPUS} GPU...")
    
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

    # Chia đều công việc
    chunk_size = math.ceil(total_images / NUM_GPUS)
    chunks = [all_tasks[i:i + chunk_size] for i in range(0, total_images, chunk_size)]

    processes = []
    start_time = time.time()
    
    # Khởi chạy Process
    # Lưu ý: Mỗi Process sẽ nhận 1 gpu_id từ 0 đến 3 (tương ứng biến môi trường thực tế)
    for i in range(len(chunks)):
        if not chunks[i]: continue
        # Nếu máy có 4 GPU vật lý: 0, 1, 2, 3. 
        # Worker sẽ thấy mình đang chạy trên "GPU 0" của context riêng nó nhờ biến môi trường.
        real_gpu_id = i % NUM_GPUS 
        
        p = Process(target=worker_ocr_process, args=(real_gpu_id, chunks[i]))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    end_time = time.time()
    print(f"\n✅ Hoàn tất OCR trong {end_time - start_time:.2f} giây.")

# ================= CÁC BƯỚC CÒN LẠI (GIỮ NGUYÊN) =================
def step2_translate_batch():
    print("\n🔹 BƯỚC 2: DỊCH THUẬT...")
    all_jsons = glob.glob(f"{JSON_DIR}/**/*.json", recursive=True)
    
    need_trans = set()
    for js in all_jsons:
        with open(js, 'r', encoding='utf-8') as f:
            try: data = json.load(f)
            except: continue
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
    
    for i in range(0, len(text_list), BATCH_SIZE_TRANS):
        batch = text_list[i:i+BATCH_SIZE_TRANS]
        try:
            res = translator.translate_batch(batch)
            for s, d in zip(batch, res): trans_map[s] = d
        except:
            pass

    cnt = 0
    for js in all_jsons:
        with open(js, 'r', encoding='utf-8') as f:
            try: data = json.load(f)
            except: continue
        
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

def step3_render_images():
    print("\n🔹 BƯỚC 3: VẼ ẢNH KẾT QUẢ...")
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
            
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            img_pil = Image.open(img_path).convert("RGB")
            draw = ImageDraw.Draw(img_pil)

            for item in data.get('texts', []):
                text_content = item.get('translated') if item.get('translated') else item['text']
                x1, y1, x2, y2 = item['box']
                render_text_in_box(draw, text_content, FONT_PATH, x1, y1, x2, y2)

            img_pil.save(out_path)
            print(f"Rendered: {file}", end='\r')

    print("\n✅ Hoàn tất toàn bộ quy trình!")

def main():
    # Set start method thành spawn để an toàn với CUDA
    try:
        set_start_method('spawn')
    except RuntimeError:
        pass
        
    step1_multi_gpu_ocr()
    step2_translate_batch()
    step3_render_images()

if __name__ == "__main__":
    main()