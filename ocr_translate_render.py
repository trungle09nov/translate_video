import os
import glob
import json
import cv2
import numpy as np
import math
import time
import gc
from multiprocessing import Process, set_start_method

from PIL import Image, ImageDraw, ImageFont
from deep_translator import GoogleTranslator

# ================= CẤU HÌNH PHẦN CỨNG =================
NUM_GPUS = 4
BATCH_SIZE_OCR = 32  # Số ảnh xử lý cùng lúc trên GPU
TRANSLATE_BATCH_SIZE = 50  # Số text dịch cùng lúc

# ================= CẤU HÌNH THƯ MỤC =================
RAW_DIR = "./frames_raw"         
JSON_DIR = "./json_cache"        
TRANSLATED_DIR = "./frames_done" 
FONT_PATH = "arial.ttf"          

LANG_SOURCE = 'de' 
LANG_TARGET = 'en'

# ==========================================================
#  CÁC HÀM VẼ & HỖ TRỢ
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

# ================= WORKER OCR + TRANSLATE + RENDER =================
def worker_ocr_translate_render(gpu_id, image_files):
    """
    Worker xử lý OCR + Translate + Render ngay trong batch
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    import paddle
    from paddleocr import PaddleOCR

    paddle.device.set_device('gpu:0')

    print(f"🚀 Worker GPU {gpu_id} (PID {os.getpid()}) khởi động...")
    
    # Init OCR
    try:
        ocr_engine = PaddleOCR(
            lang='german', 
            use_angle_cls=False,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
        )
    except Exception as e:
        print(f"❌ GPU {gpu_id} lỗi Init OCR: {e}")
        return
    
    # Init Translator
    try:
        translator = GoogleTranslator(source=LANG_SOURCE, target=LANG_TARGET)
        print(f"   [GPU {gpu_id}] Translator initialized")
    except Exception as e:
        print(f"   ⚠️ [GPU {gpu_id}] Translator warning: {e}")
        translator = None

    total_files = len(image_files)
    processed_count = 0
    success_count = 0
    error_count = 0
    
    debug_dir = os.path.join(JSON_DIR, f"debug_gpu_{gpu_id}")
    os.makedirs(debug_dir, exist_ok=True)
    
    for i in range(0, total_files, BATCH_SIZE_OCR):
        batch_items = image_files[i : i + BATCH_SIZE_OCR]
        loaded_images = [] 
        
        for img_path, json_path, filename in batch_items:
            try:
                if os.path.exists(img_path):
                    loaded_images.append((img_path, json_path, filename))
            except:
                pass
        
        if not loaded_images: 
            continue
        
        current_batch_count = len(loaded_images)
        
        # ==========================================================
        # STEP 1: OCR BATCH
        # ==========================================================
        batch_ocr_results = []  # Store OCR results for all images in batch
        
        for img_path, json_out_path, fname in loaded_images:
            ocr_data = []
            
            try:
                # ✅ FIX: Predict từng ảnh, KHÔNG batch predict
                result = ocr_engine.predict(img_path)  # Single predict
                
                if not result or not isinstance(result, list) or len(result) == 0:
                    batch_ocr_results.append((json_out_path, fname, img_path, []))
                    continue

                # Parse OCR result
                try:
                    # Method 1: save_to_json
                    result[0].save_to_json(debug_dir)
                    
                    json_filename = fname.replace('.jpg', '_res.json').replace('.png', '_res.json').replace('.jpeg', '_res.json')
                    result_json_path = os.path.join(debug_dir, json_filename)
                    
                    if os.path.exists(result_json_path):
                        with open(result_json_path, 'r', encoding='utf-8') as f:
                            parsed_data = json.load(f)
                        
                        rec_texts = parsed_data.get('rec_texts', [])
                        rec_scores = parsed_data.get('rec_scores', [])
                        rec_boxes = parsed_data.get('rec_boxes', [])
                        
                        for j in range(len(rec_texts)):
                            text = rec_texts[j]
                            score = float(rec_scores[j])
                            box = rec_boxes[j]
                            
                            if not text or not str(text).strip() or score < 0.25:
                                continue
                            
                            ocr_data.append({
                                "box": [int(box[0]), int(box[1]), int(box[2]), int(box[3])],
                                "text": str(text),
                                "confidence": float(score),
                                "translated": ""
                            })
                        
                        # Clean up debug file
                        try:
                            os.remove(result_json_path)
                        except:
                            pass
                    else:
                        raise Exception("Debug JSON not found")
                        
                except Exception as e:
                    # Method 2: Fallback - parse trực tiếp
                    if result[0] is not None:
                        for line in result[0]:
                            try:
                                box_coords = line[0]
                                content = line[1]
                                
                                text = content[0]
                                score = content[1]
                                
                                if score < 0.25 or not str(text).strip():
                                    continue
                                
                                xs = [pt[0] for pt in box_coords]
                                ys = [pt[1] for pt in box_coords]
                                
                                ocr_data.append({
                                    "box": [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))],
                                    "text": str(text),
                                    "confidence": float(score),
                                    "translated": ""
                                })
                            except:
                                continue
                
                # Store result
                batch_ocr_results.append((json_out_path, fname, img_path, ocr_data))
                success_count += 1

            except Exception as e:
                error_count += 1
                batch_ocr_results.append((json_out_path, fname, img_path, []))
        
        # ==========================================================
        # STEP 2: TRANSLATE BATCH (tất cả texts trong batch cùng lúc)
        # ==========================================================
        if translator:
            # Collect tất cả unique texts từ batch
            all_texts_to_translate = set()
            for _, _, _, ocr_data in batch_ocr_results:
                for item in ocr_data:
                    txt = item['text'].strip()
                    if len(txt) > 1 and not txt.isdigit():
                        all_texts_to_translate.add(txt)
            
            # Translate batch
            trans_map = {}
            if all_texts_to_translate:
                text_list = list(all_texts_to_translate)
                try:
                    # Translate theo chunks
                    for chunk_start in range(0, len(text_list), TRANSLATE_BATCH_SIZE):
                        chunk = text_list[chunk_start:chunk_start + TRANSLATE_BATCH_SIZE]
                        try:
                            translated_chunk = translator.translate_batch(chunk)
                            for orig, trans in zip(chunk, translated_chunk):
                                trans_map[orig] = trans
                        except Exception as e2:
                            # Fallback: translate one by one
                            for orig in chunk[:10]:  # Limit to 10 to avoid timeout
                                try:
                                    trans_map[orig] = translator.translate(orig)
                                except:
                                    trans_map[orig] = orig
                except Exception as e:
                    print(f"   ⚠️ [GPU {gpu_id}] Translation error: {e}")
            
            # Update translations vào ocr_data
            for _, _, _, ocr_data in batch_ocr_results:
                for item in ocr_data:
                    txt = item['text'].strip()
                    if txt in trans_map:
                        item['translated'] = trans_map[txt]
        
        # ==========================================================
        # STEP 3: SAVE JSON + RENDER (cho từng ảnh)
        # ==========================================================
        for json_out_path, fname, img_path, ocr_data in batch_ocr_results:
            try:
                # Save JSON
                with open(json_out_path, 'w', encoding='utf-8') as f:
                    json.dump({
                        "frame": fname, 
                        "texts": ocr_data
                    }, f, ensure_ascii=False, indent=2)
                
                # Render image
                if os.path.exists(img_path):
                    # Output path
                    rel_path = os.path.relpath(img_path, RAW_DIR)
                    rel_dir = os.path.dirname(rel_path)
                    out_subdir = os.path.join(TRANSLATED_DIR, rel_dir)
                    os.makedirs(out_subdir, exist_ok=True)
                    out_path = os.path.join(out_subdir, os.path.basename(img_path))
                    
                    if len(ocr_data) == 0:
                        # No text, copy original
                        import shutil
                        shutil.copy(img_path, out_path)
                    else:
                        # Render with translations
                        img_pil = Image.open(img_path).convert("RGB")
                        draw = ImageDraw.Draw(img_pil)

                        for item in ocr_data:
                            text_content = item.get('translated') if item.get('translated') else item['text']
                            x1, y1, x2, y2 = item['box']
                            render_text_in_box(draw, text_content, FONT_PATH, x1, y1, x2, y2)

                        img_pil.save(out_path)
            
            except Exception as e:
                print(f"   ⚠️ [GPU {gpu_id}] Error saving/rendering {fname}: {e}")
        
        # Clean up memory
        del loaded_images
        del batch_items
        del batch_ocr_results
        gc.collect()
        
        try:
            paddle.device.cuda.empty_cache()
        except:
            pass 
        
        processed_count += current_batch_count
        
        # Progress update
        if i % (BATCH_SIZE_OCR * 2) == 0:
            print(f"   [GPU {gpu_id}] Progress: {processed_count}/{total_files} ({success_count} OK, {error_count} errors)")

    print(f"✅ [GPU {gpu_id}] COMPLETED: {success_count} success, {error_count} errors")


# ================= MAIN PIPELINE =================
def step1_multi_gpu_ocr_translate_render():
    """
    Một bước duy nhất: OCR + Translate + Render
    """
    print("🔹 Pre-check: Warm-up model PaddleOCR...")
    try:
        from paddleocr import PaddleOCR
        PaddleOCR(lang='german', use_angle_cls=False, show_log=False)
        print("✅ Model check OK.")
    except Exception as e:
        print(f"⚠️ Warning: {e}")

    print(f"\n🔹 ALL-IN-ONE: OCR + TRANSLATE + RENDER WITH {NUM_GPUS} GPUs...")
    
    all_tasks = []
    for root, dirs, files in os.walk(RAW_DIR):
        rel_subdir = os.path.relpath(root, RAW_DIR)
        if rel_subdir == ".": rel_subdir = ""
        os.makedirs(os.path.join(JSON_DIR, rel_subdir), exist_ok=True)

        for f in files:
            if f.lower().endswith((".jpg", ".png", ".jpeg")):
                json_path = os.path.join(JSON_DIR, rel_subdir, f.replace(".jpg", ".json").replace(".png", ".json").replace(".jpeg", ".json"))
                out_rel_path = os.path.relpath(os.path.join(root, f), RAW_DIR)
                out_path = os.path.join(TRANSLATED_DIR, out_rel_path)
                
                # Check if BOTH JSON and rendered image need processing
                if not os.path.exists(json_path) or not os.path.exists(out_path):
                    all_tasks.append((os.path.join(root, f), json_path, f))

    total_images = len(all_tasks)
    if total_images == 0:
        print("✅ All images already processed.")
        return

    print(f"📦 Total images to process: {total_images}")
    
    import random
    random.shuffle(all_tasks)

    chunk_size = math.ceil(total_images / NUM_GPUS)
    chunks = [all_tasks[i:i + chunk_size] for i in range(0, total_images, chunk_size)]

    processes = []
    start_time = time.time()
    
    for i in range(len(chunks)):
        if not chunks[i]: continue
        real_gpu_id = i % NUM_GPUS 
        p = Process(target=worker_ocr_translate_render, args=(real_gpu_id, chunks[i]))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    elapsed = time.time() - start_time
    print(f"\n✅ ALL STEPS completed in {elapsed:.2f}s ({total_images/elapsed:.2f} images/sec)")


def main():
    try:
        set_start_method('spawn')
    except RuntimeError:
        pass
    
    print("=" * 70)
    print("🎬 VIDEO TRANSLATION PIPELINE - ALL-IN-ONE MODE")
    print("=" * 70)
    print(f"Configuration:")
    print(f"  GPUs: {NUM_GPUS}")
    print(f"  OCR Batch: {BATCH_SIZE_OCR}")
    print(f"  Translate Batch: {TRANSLATE_BATCH_SIZE}")
    print(f"  Source: {RAW_DIR}")
    print(f"  Output: {TRANSLATED_DIR}")
    print(f"  Process: OCR → Translate → Render (in same batch)")
    print("=" * 70)
    
    start_time = time.time()
    
    step1_multi_gpu_ocr_translate_render()  # All in one!
    
    total_time = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"🎉 PIPELINE COMPLETED in {total_time:.2f}s ({total_time/60:.2f} min)")
    print("=" * 70)

if __name__ == "__main__":
    main()