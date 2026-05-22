import os
import json
import math
import time
import gc
import traceback
from multiprocessing import Process, set_start_method

# ================= CẤU HÌNH PHẦN CỨNG =================
NUM_GPUS = 2
BATCH_SIZE_OCR = 32  # Giảm lại để ổn định
OCR_SCORE_THRESHOLD = 0.35

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

    def parse_ocr_result(result):
        """Normalize PaddleOCR output into a list of OCR items."""
        ocr_data = []
        if not result:
            return ocr_data

        lines = result[0] if isinstance(result, list) and len(result) > 0 else result
        if not lines:
            return ocr_data

        for line in lines:
            try:
                box_coords = line[0]
                content = line[1]
                text = content[0]
                score = float(content[1])

                if score < OCR_SCORE_THRESHOLD or not str(text).strip():
                    continue

                xs = [pt[0] for pt in box_coords]
                ys = [pt[1] for pt in box_coords]

                ocr_data.append({
                    "box": [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))],
                    "text": str(text),
                    "confidence": score,
                    "translated": "",
                })
            except Exception:
                continue

        return ocr_data

    def run_ocr(img_path):
        """Try predict() first, then fallback to ocr()."""
        last_error = None
        if hasattr(ocr_engine, "predict"):
            try:
                return ocr_engine.predict(img_path)
            except Exception as exc:
                last_error = exc

        if hasattr(ocr_engine, "ocr"):
            try:
                return ocr_engine.ocr(img_path, cls=False)
            except Exception as exc:
                last_error = exc

        if last_error is not None:
            raise last_error
        raise RuntimeError("PaddleOCR engine has no predict() or ocr() method")
    
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
                # Dùng predict() / ocr() trực tiếp, rồi parse output theo format chuẩn.
                result = run_ocr(img_path)
                ocr_data = parse_ocr_result(result)

                if len(ocr_data) == 0:
                    empty_ocr_count += 1
                    batch_ocr_results.append((json_out_path, fname, img_path, []))
                    continue
                
                # Store result
                success_count += 1
                    
                batch_ocr_results.append((json_out_path, fname, img_path, ocr_data))

            except Exception as e:
                error_count += 1
                print(f"   ❌ [GPU {gpu_id}] OCR failed: {os.path.basename(img_path)}")
                print(f"      {type(e).__name__}: {e}")
                tb = traceback.format_exc(limit=2)
                print("      " + "\n      ".join(tb.strip().splitlines()[-2:]))
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