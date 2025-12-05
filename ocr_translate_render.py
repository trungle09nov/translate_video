import os
import glob
import json
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from paddleocr import PaddleOCR
from deep_translator import GoogleTranslator

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
    ocr_engine = PaddleOCR(lang='german', use_angle_cls=True, show_log=False)
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


# ==========================================================
# BƯỚC 1: OCR (Batch)
# ==========================================================
def step1_ocr_scan():
    print(f"\n🔹 BƯỚC 1: QUÉT ẢNH VÀ TẠO FILE JSON (Batch: {BATCH_SIZE_OCR})...")
    
    # Tìm ảnh chưa có JSON
    all_tasks = []
    for root, dirs, files in os.walk(RAW_DIR):
        rel_subdir = os.path.relpath(root, RAW_DIR)
        if rel_subdir == ".": rel_subdir = ""
        os.makedirs(os.path.join(JSON_DIR, rel_subdir), exist_ok=True)
        
        for f in files:
            if f.lower().endswith((".jpg", ".png")):
                json_path = os.path.join(JSON_DIR, rel_subdir, f.replace(".jpg", ".json").replace(".png", ".json"))
                if not os.path.exists(json_path):
                    all_tasks.append((os.path.join(root, f), json_path, f))

    if not all_tasks:
        print("✅ Đã có đủ JSON cache.")
        return

    # Chạy Batch
    for i in range(0, len(all_tasks), BATCH_SIZE_OCR):
        batch = all_tasks[i:i+BATCH_SIZE_OCR]
        imgs = []
        valid_batch = []
        
        for img_path, js_path, fname in batch:
            im = cv2.imread(img_path)
            if im is not None:
                imgs.append(im)
                valid_batch.append((js_path, fname))
        
        if not imgs: continue
        print(f"   🚀 OCR {i}/{len(all_tasks)}...", end="\r")

        try:
            results = ocr_engine.ocr(imgs, cls=True)
        except:
            continue

        for idx, res in enumerate(results):
            js_path, fname = valid_batch[idx]
            ocr_data = []
            
            # Xử lý format output Paddle
            if res:
                # Format Dict (Paddle mới)
                if isinstance(res, dict) and 'rec_texts' in res:
                    for box, text, conf in zip(res['dt_polys'], res['rec_texts'], res['rec_scores']):
                        if conf > 0.5:
                            xs = [p[0] for p in box]; ys = [p[1] for p in box]
                            ocr_data.append({
                                "box": [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))],
                                "text": text, "confidence": float(conf), "translated": ""
                            })
                # Format List (Paddle cũ)
                elif isinstance(res, list):
                    for line in res:
                        # Fix lỗi index string
                        content = line[1]
                        text = content if isinstance(content, str) else content[0]
                        conf = 1.0 if isinstance(content, str) else content[1]
                        
                        if conf > 0.5:
                            pts = line[0]
                            xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
                            ocr_data.append({
                                "box": [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))],
                                "text": text, "confidence": float(conf), "translated": ""
                            })

            with open(js_path, 'w', encoding='utf-8') as f:
                json.dump({"frame": fname, "texts": ocr_data}, f, ensure_ascii=False, indent=2)

    print("\n✅ Hoàn tất Bước 1.")


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
    step1_ocr_scan()
    step2_translate_batch()
    step3_render_images()

if __name__ == "__main__":
    main()