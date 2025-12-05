import os
import glob
import json
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from paddleocr import PaddleOCR
from deep_translator import GoogleTranslator

# ================= CẤU HÌNH =================
RAW_DIR = "workspace/frames_raw"         # Ảnh gốc (từ phần 1)
JSON_DIR = "workspace/json_cache"        # Nơi lưu kết quả OCR
TRANSLATED_DIR = "workspace/frames_done" # Ảnh đã dịch và vẽ
FONT_PATH = "arial.ttf"                  # Đường dẫn font (Window: C:/Windows/Fonts/arial.ttf)

# Cấu hình ngôn ngữ
LANG_SOURCE = 'de' # Tiếng Đức
LANG_TARGET = 'en' # Tiếng Anh

# ================= KHỞI TẠO =================
# PaddleOCR (chạy lần đầu sẽ tải model hơi lâu)
ocr_engine = PaddleOCR(use_angle_cls=True, lang='german', show_log=False)
translator = GoogleTranslator(source=LANG_SOURCE, target=LANG_TARGET)

os.makedirs(JSON_DIR, exist_ok=True)
os.makedirs(TRANSLATED_DIR, exist_ok=True)

# Cache dịch để đỡ gọi Google nhiều lần cho cùng 1 từ
memory_trans_cache = {}

# ================= HÀM HỖ TRỢ AUTO-FIT FONT =================
def wrap_text_by_width(draw, text, font, max_width):
    """Cắt dòng văn bản sao cho vừa chiều rộng"""
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
    """
    Thuật toán Binary Search hoặc Loop giảm dần để tìm font size to nhất
    vừa khít với box_h và box_w
    """
    max_size = 150 # Giới hạn size to nhất
    min_size = 10
    padding = 4
    
    safe_w = box_w - (padding * 2)
    safe_h = box_h - (padding * 2)

    # Thử từ size lớn xuống nhỏ
    best_font = None
    best_lines = []
    best_total_h = 0
    best_line_h = 0
    
    # Load default nếu lỗi
    try:
        font_test = ImageFont.truetype(font_path, min_size)
    except:
        font_path = "arial.ttf" # Fallback

    for size in range(max_size, min_size, -2):
        if size > box_h: continue # Bỏ qua nếu size chữ > chiều cao box
        
        try:
            font = ImageFont.truetype(font_path, size)
        except:
            continue

        lines = wrap_text_by_width(draw, text, font, safe_w)
        
        # Tính chiều cao khối text
        bbox_sample = draw.textbbox((0, 0), "Ay", font=font)
        line_h = bbox_sample[3] - bbox_sample[1]
        total_h = (len(lines) * line_h) + ((len(lines) - 1) * 4) # 4 là spacing

        if total_h <= safe_h:
            return font, lines, total_h, line_h

    # Fallback về size nhỏ nhất
    font = ImageFont.truetype(font_path, min_size)
    lines = wrap_text_by_width(draw, text, font, safe_w)
    return font, lines, safe_h, 12

# ================= XỬ LÝ CHÍNH =================
def process_frame(img_path):
    filename = os.path.basename(img_path)
    json_path = os.path.join(JSON_DIR, filename.replace(".jpg", ".json"))
    out_path = os.path.join(TRANSLATED_DIR, filename)

    # 1. KIỂM TRA CACHE OCR (Nếu có rồi thì không OCR lại)
    if os.path.exists(json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    else:
        # Chưa có cache -> Chạy OCR
        img = cv2.imread(img_path)
        result = ocr_engine.ocr(img, cls=True)
        data = []
        
        if result and result[0]:
            for line in result[0]:
                # Paddle trả về: [[[x1,y1],[x2,y2],[x3,y3],[x4,y4]], ("text", conf)]
                points = line[0]
                text = line[1][0]
                conf = line[1][1]
                
                # Chuẩn hóa box thành [xmin, ymin, xmax, ymax]
                xs = [p[0] for p in points]
                ys = [p[1] for p in points]
                box = [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]
                
                if conf > 0.5: # Chỉ lấy text rõ
                    data.append({
                        "box": box,
                        "text_original": text,
                        "translated": "" # Để trống, bước sau dịch
                    })
        
        # Lưu JSON ngay sau khi OCR
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    # 2. DỊCH VĂN BẢN (Nếu chưa dịch)
    dirty = False
    for item in data:
        orig = item['text_original']
        if not item.get('translated'):
            # Check memory cache
            if orig in memory_trans_cache:
                item['translated'] = memory_trans_cache[orig]
                dirty = True
            else:
                try:
                    trans = translator.translate(orig)
                    memory_trans_cache[orig] = trans
                    item['translated'] = trans
                    dirty = True
                    print(f"   Trans: {orig} -> {trans}")
                except Exception as e:
                    print(f"   Err trans: {e}")
                    item['translated'] = orig # Fallback

    # Cập nhật lại JSON nếu có dịch mới
    if dirty:
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    # 3. VẼ ẢNH (RENDER)
    # Mở ảnh gốc bằng Pillow
    img_pil = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img_pil)

    for item in data:
        text = item.get('translated', item['text_original'])
        box = item['box'] # [x1, y1, x2, y2]
        x1, y1, x2, y2 = box
        
        w = x2 - x1
        h = y2 - y1
        if w < 10 or h < 10: continue

        # A. Vẽ nền trắng che chữ cũ
        draw.rectangle([x1, y1, x2, y2], fill="white")

        # B. Tìm font to nhất
        font, lines, text_h, line_h = get_optimal_font(draw, text, w, h, FONT_PATH)

        # C. Căn giữa theo chiều dọc
        start_y = y1 + (h - text_h) // 2
        
        # D. Vẽ từng dòng
        curr_y = start_y
        for line in lines:
            # Căn giữa theo chiều ngang
            bbox = draw.textbbox((0, 0), line, font=font)
            lw = bbox[2] - bbox[0]
            start_x = x1 + (w - lw) // 2
            
            draw.text((start_x, curr_y), line, font=font, fill="black")
            curr_y += line_h + 4

    img_pil.save(out_path)

def main():
    files = sorted(glob.glob(f"{RAW_DIR}/*.jpg"))
    print(f"Tìm thấy {len(files)} frames. Bắt đầu xử lý...")
    
    for i, f in enumerate(files):
        print(f"[{i+1}/{len(files)}] Processing: {os.path.basename(f)}", end='\r')
        process_frame(f)
    print("\n✅ Hoàn tất xử lý ảnh!")

if __name__ == "__main__":
    main()