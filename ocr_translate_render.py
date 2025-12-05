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
FONT_PATH = "arial.ttf"          # Font chữ

LANG_SOURCE = 'de' 
LANG_TARGET = 'en'
BATCH_SIZE = 50 # Số lượng từ dịch một lần (để tránh bị Google chặn)

# ================= KHỞI TẠO =================
# Khởi tạo PaddleOCR
try:
    ocr_engine = PaddleOCR(lang='german')
except:
    print("⚠️ Cảnh báo: Không load được PaddleOCR, hãy kiểm tra cài đặt.")
    exit()

translator = GoogleTranslator(source=LANG_SOURCE, target=LANG_TARGET)

# ================= HÀM HỖ TRỢ FONT (Giữ nguyên) =================
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

def get_optimal_font(draw, text, box_w, box_h, font_path):
    max_size = 120
    min_size = 10
    safe_w = box_w - 8
    safe_h = box_h - 8

    # Fallback font hệ thống nếu không tìm thấy font chỉ định
    if not os.path.exists(font_path):
        # Font mặc định của Linux thường ở đây, hoặc dùng "DejaVuSans.ttf"
        font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf" 

    for size in range(max_size, min_size, -2):
        if size > box_h: continue
        try:
            font = ImageFont.truetype(font_path, size)
        except:
            font = ImageFont.load_default()
            break

        lines = wrap_text_by_width(draw, text, font, safe_w)
        bbox_sample = draw.textbbox((0, 0), "Ay", font=font)
        line_h = bbox_sample[3] - bbox_sample[1]
        total_h = (len(lines) * line_h) + ((len(lines) - 1) * 4)

        if total_h <= safe_h:
            return font, lines, total_h, line_h

    font = ImageFont.load_default()
    return font, [text], safe_h, 12

# ================= BƯỚC 1: QUÉT OCR & TẠO JSON =================
def step1_ocr_scan():
    print("\n🔹 BƯỚC 1: QUÉT ẢNH VÀ TẠO FILE JSON (OCR GỐC)...")
    
    count = 0
    for root, dirs, files in os.walk(RAW_DIR):
        rel_subdir = os.path.relpath(root, RAW_DIR)
        if rel_subdir == ".": rel_subdir = ""

        # Tạo thư mục lưu json
        current_json_dir = os.path.join(JSON_DIR, rel_subdir)
        os.makedirs(current_json_dir, exist_ok=True)

        jpg_files = sorted([f for f in files if f.lower().endswith((".jpg", ".png", ".jpeg"))])
        
        for file in jpg_files:
            img_path = os.path.join(root, file)
            json_filename = file.replace(".jpg", ".json").replace(".png", ".json")
            json_path = os.path.join(current_json_dir, json_filename)

            # 1. Nếu đã có JSON rồi thì bỏ qua (Resume)
            if os.path.exists(json_path):
                continue

            print(f"   OCR: {file}", end="\r")
            
            img = cv2.imread(img_path)
            if img is None: continue
            
            # --- CHẠY OCR ---
            try:
                result = ocr_engine.ocr(img)
            except Exception as e:
                print(f"\n   ⚠️ Lỗi khi OCR ảnh {file}: {e}")
                continue

            ocr_data = []
            
            if result:
                # ================= XỬ LÝ FORMAT DỮ LIỆU =================
                # Kiểm tra xem result[0] là kiểu mới (Dict) hay kiểu cũ (List)
                first_res = result[0]
                
                # TRƯỜNG HỢP 1: Format mới (như hình bạn gửi: có rec_texts, dt_polys...)
                if isinstance(first_res, dict) and 'rec_texts' in first_res and 'dt_polys' in first_res:
                    texts = first_res.get('rec_texts', [])
                    boxes = first_res.get('dt_polys', [])
                    scores = first_res.get('rec_scores', [])
                    
                    # Duyệt qua từng phần tử trong các mảng song song
                    for box_points, text, conf in zip(boxes, texts, scores):
                        if conf > 0.5:
                            # Chuyển đổi box polygon thành box chữ nhật [x1, y1, x2, y2]
                            xs = [p[0] for p in box_points]
                            ys = [p[1] for p in box_points]
                            box = [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]
                            
                            ocr_data.append({
                                "box": box,
                                "text": text,           # Text gốc
                                "confidence": float(conf),
                                "translated": ""        # ĐỂ TRỐNG (chờ bước 2)
                            })

                # TRƯỜNG HỢP 2: Format cổ điển (List of Lists)
                elif isinstance(first_res, list):
                    for line in first_res:
                        # line dạng: [ [[x1,y1]...], ("text", 0.9) ]
                        points = line[0]
                        content = line[1]

                        if isinstance(content, str): # Fix lỗi index string
                            text = content
                            conf = 1.0
                        else:
                            text = content[0]
                            conf = content[1]
                        
                        if conf > 0.5:
                            xs = [p[0] for p in points]
                            ys = [p[1] for p in points]
                            box = [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]

                            ocr_data.append({
                                "box": box,
                                "text": text,
                                "confidence": float(conf),
                                "translated": "" 
                            })
                # ========================================================

            # 2. Tạo cấu trúc JSON đúng như bạn mong muốn
            output_json = {
                "frame": file,
                "texts": ocr_data
            }

            # Lưu file
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(output_json, f, ensure_ascii=False, indent=2)
            
            count += 1

    print(f"\n✅ Bước 1 hoàn tất: Đã tạo {count} file JSON.")

# ================= BƯỚC 2: DỊCH BATCH (NHANH HƠN) =================
def step2_translate_batch():
    print("\n🔹 BƯỚC 2: DỊCH THUẬT (BATCH TRANSLATE)...")

    # 1. Quét tất cả file JSON để tìm từ chưa dịch
    all_json_files = []
    for root, _, files in os.walk(JSON_DIR):
        for file in files:
            if file.endswith(".json"):
                all_json_files.append(os.path.join(root, file))

    if not all_json_files:
        print("⚠️ Không tìm thấy file JSON nào.")
        return

    # Gom các từ cần dịch (dùng set để loại bỏ từ trùng lặp)
    texts_to_translate = set()
    print("   -> Đang quét text chưa dịch...")
    
    for js_path in all_json_files:
        with open(js_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for item in data:
                if not item.get('translated'):
                    txt = item['text_original'].strip()
                    if len(txt) > 1 and not txt.isdigit():
                        texts_to_translate.add(txt)

    text_list = list(texts_to_translate)
    if not text_list:
        print("✅ Tất cả đã được dịch từ trước.")
        return

    print(f"   ☁️  Tìm thấy {len(text_list)} từ mới. Đang gửi Google Dịch...")

    # 2. Dịch Batch (chia nhỏ danh sách để gửi)
    translation_map = {}
    
    for i in range(0, len(text_list), BATCH_SIZE):
        batch = text_list[i : i + BATCH_SIZE]
        try:
            results = translator.translate_batch(batch)
            for src, dest in zip(batch, results):
                translation_map[src] = dest
            print(f"      Đã dịch {i + len(batch)}/{len(text_list)} từ...", end="\r")
        except Exception as e:
            print(f"\n      ⚠️ Lỗi batch tại {i}, chuyển sang dịch lẻ: {e}")
            for txt in batch:
                try:
                    translation_map[txt] = translator.translate(txt)
                except:
                    pass

    # 3. Cập nhật lại vào file JSON
    print("\n   💾 Đang cập nhật JSON...")
    updated_files = 0
    for js_path in all_json_files:
        with open(js_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        dirty = False
        for item in data:
            orig = item['text_original'].strip()
            if not item.get('translated') and orig in translation_map:
                item['translated'] = translation_map[orig]
                dirty = True
        
        if dirty:
            with open(js_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            updated_files += 1

    print(f"✅ Đã cập nhật bản dịch vào {updated_files} file JSON.")

# ================= BƯỚC 3: RENDER ẢNH =================
def step3_render_images():
    print("\n🔹 BƯỚC 3: VẼ ẢNH KẾT QUẢ...")

    for root, dirs, files in os.walk(RAW_DIR):
        rel_subdir = os.path.relpath(root, RAW_DIR)
        if rel_subdir == ".": rel_subdir = ""

        # Tạo thư mục output tương ứng
        out_subdir = os.path.join(TRANSLATED_DIR, rel_subdir)
        json_subdir = os.path.join(JSON_DIR, rel_subdir)
        os.makedirs(out_subdir, exist_ok=True)

        jpg_files = sorted([f for f in files if f.lower().endswith(".jpg")])

        for file in jpg_files:
            img_path = os.path.join(root, file)
            json_path = os.path.join(json_subdir, file.replace(".jpg", ".json"))
            out_path = os.path.join(out_subdir, file)

            # Chỉ render nếu có file JSON
            if not os.path.exists(json_path):
                continue
            
            # Kiểm tra nếu ảnh đã render rồi thì bỏ qua (tùy chọn)
            # if os.path.exists(out_path): continue 

            print(f"   Render: {file}", end="\r")

            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            img_pil = Image.open(img_path).convert("RGB")
            draw = ImageDraw.Draw(img_pil)

            for item in data:
                # Ưu tiên lấy text dịch, nếu không có thì lấy gốc
                text = item.get('translated')
                if not text: text = item['text_original']

                box = item['box']
                x1, y1, x2, y2 = box
                w, h = x2 - x1, y2 - y1
                
                if w < 10 or h < 10: continue

                # Vẽ
                draw.rectangle([x1, y1, x2, y2], fill="white")
                font, lines, text_h, line_h = get_optimal_font(draw, text, w, h, FONT_PATH)
                
                start_y = y1 + (h - text_h) // 2
                curr_y = start_y
                for line in lines:
                    bbox = draw.textbbox((0, 0), line, font=font)
                    lw = bbox[2] - bbox[0]
                    start_x = x1 + (w - lw) // 2
                    draw.text((start_x, curr_y), line, font=font, fill="black")
                    curr_y += line_h + 4

            img_pil.save(out_path)

    print("\n✅ Hoàn tất Render!")

# ================= MAIN =================
def main():
    # Bước 1: OCR toàn bộ ảnh -> JSON
    step1_ocr_scan()
    
    # Bước 2: Dịch toàn bộ JSON (Nhanh, tiết kiệm API)
    step2_translate_batch()
    
    # Bước 3: Đọc JSON và vẽ ảnh
    step3_render_images()

    print("\n🎉🎉🎉 XỬ LÝ HOÀN TẤT TOÀN BỘ!")

if __name__ == "__main__":
    main()