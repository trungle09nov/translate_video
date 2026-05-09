import os
import json
import math
import time
import gc
from multiprocessing import Process, set_start_method

# ================= CẤU HÌNH PHẦN CỨNG =================
NUM_GPUS = 2
BATCH_SIZE_OCR = 32  # Giảm lại để ổn định

# ================= CẤU HÌNH THƯ MỤC =================
RAW_DIR = "./frames_raw"         
JSON_DIR = "./json_cache"        

# ================= WORKER OCR =================
def worker_ocr_only(gpu_id, image_files):
    """
    Worker xử lý OCR theo batch và ghi JSON thô
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
    
    total_files = len(image_files)
    processed_count = 0
    success_count = 0
    error_count = 0
    empty_ocr_count = 0
    
    debug_dir = os.path.join(JSON_DIR, f"debug_gpu_{gpu_id}")
    os.makedirs(debug_dir, exist_ok=True)
    
    for i in range(0, total_files, BATCH_SIZE_OCR):
        batch_items = image_files[i : i + BATCH_SIZE_OCR]
        loaded_images = [] 
        
        # Load batch
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
        # STEP 1: OCR BATCH - Dùng predict() như code cũ
        # ==========================================================
        batch_ocr_results = []
        
        for img_path, json_out_path, fname in loaded_images:
            ocr_data = []
            
            try:
                # ✅ Dùng predict() từng ảnh
                result = ocr_engine.predict(img_path)
                
                if not result or not isinstance(result, list) or len(result) == 0:
                    batch_ocr_results.append((json_out_path, fname, img_path, []))
                    empty_ocr_count += 1
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
                if len(ocr_data) == 0:
                    empty_ocr_count += 1
                else:
                    success_count += 1
                    
                batch_ocr_results.append((json_out_path, fname, img_path, ocr_data))

            except Exception as e:
                error_count += 1
                batch_ocr_results.append((json_out_path, fname, img_path, []))
        
        # ==========================================================
        # STEP 2: SAVE JSON
        # ==========================================================
        for json_out_path, fname, img_path, ocr_data in batch_ocr_results:
            try:
                # Save JSON
                with open(json_out_path, 'w', encoding='utf-8') as f:
                    json.dump({
                        "frame": fname, 
                        "texts": ocr_data
                    }, f, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"   ⚠️ [GPU {gpu_id}] Error saving JSON {fname}: {e}")
        
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
            print(f"   [GPU {gpu_id}] Progress: {processed_count}/{total_files} ({success_count} OK, {empty_ocr_count} empty, {error_count} errors)")

    print(f"✅ [GPU {gpu_id}] COMPLETED: {success_count} success, {empty_ocr_count} empty OCR, {error_count} errors")


# ================= MAIN PIPELINE =================
def step1_multi_gpu_ocr_only():
    """
    Một bước duy nhất: OCR và xuất JSON thô
    """
    print("🔹 Pre-check: Warm-up model PaddleOCR...")
    try:
        from paddleocr import PaddleOCR
        PaddleOCR(lang='german', use_angle_cls=False)
        print("✅ Model check OK.")
    except Exception as e:
        print(f"⚠️ Warning: {e}")

    print(f"\n🔹 OCR-ONLY PIPELINE WITH {NUM_GPUS} GPUs...")
    
    all_tasks = []
    for root, dirs, files in os.walk(RAW_DIR):
        rel_subdir = os.path.relpath(root, RAW_DIR)
        if rel_subdir == ".": rel_subdir = ""
        os.makedirs(os.path.join(JSON_DIR, rel_subdir), exist_ok=True)

        for f in files:
            if f.lower().endswith((".jpg", ".png", ".jpeg")):
                json_path = os.path.join(JSON_DIR, rel_subdir, f.replace(".jpg", ".json").replace(".png", ".json").replace(".jpeg", ".json"))

                # OCR-only: chỉ xử lý nếu JSON chưa có
                if not os.path.exists(json_path):
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
        p = Process(target=worker_ocr_only, args=(real_gpu_id, chunks[i]))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    elapsed = time.time() - start_time
    print(f"\n✅ OCR step completed in {elapsed:.2f}s ({total_images/elapsed:.2f} images/sec)")


def main():
    try:
        set_start_method('spawn')
    except RuntimeError:
        pass
    
    print("=" * 70)
    print("🎬 VIDEO OCR PIPELINE - JSON ONLY")
    print("=" * 70)
    print(f"Configuration:")
    print(f"  GPUs: {NUM_GPUS}")
    print(f"  OCR Batch: {BATCH_SIZE_OCR}")
    print(f"  Source: {RAW_DIR}")
    print(f"  Output JSON: {JSON_DIR}")
    print(f"  Process: OCR (predict) → Save JSON")
    print("=" * 70)
    
    start_time = time.time()
    
    step1_multi_gpu_ocr_only()
    
    total_time = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"🎉 PIPELINE COMPLETED in {total_time:.2f}s ({total_time/60:.2f} min)")
    print("=" * 70)

if __name__ == "__main__":
    main()