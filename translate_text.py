import os
import json
import torch
from transformers import MarianMTModel, MarianTokenizer
from tqdm import tqdm
import time
import re
from pipeline_config import LINE_Y_THRESHOLD, LINE_X_GAP_THRESHOLD

# ================= CẤU HÌNH =================
JSON_DIR = "./json_cache"
JSON_OUTPUT_DIR = "./json_cache"
LANG_SOURCE = 'de'
LANG_TARGET = 'en'
TRANSLATE_BATCH_SIZE = 16
NUM_BEAMS = 4

# ✅ Custom cache directory (có quyền write)
CACHE_DIR = "./model_cache"  # Hoặc bất kỳ folder nào bạn có quyền
os.makedirs(CACHE_DIR, exist_ok=True)

# ================= INIT TRANSLATOR (không cần HF login) =================
print(f"\n🔄 Loading MarianMT model: opus-mt-{LANG_SOURCE}-{LANG_TARGET}...")
print(f"📁 Cache directory: {CACHE_DIR}")
model_name = f'Helsinki-NLP/opus-mt-{LANG_SOURCE}-{LANG_TARGET}'

try:
    tokenizer = MarianTokenizer.from_pretrained(
        model_name,
        cache_dir=CACHE_DIR  # ✅ Dùng cache folder riêng
    )
    model = MarianMTModel.from_pretrained(
        model_name,
        cache_dir=CACHE_DIR  # ✅ Dùng cache folder riêng
    )
    
    if torch.cuda.is_available():
        model = model.cuda()
        print(f"✅ Model loaded on GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("✅ Model loaded on CPU")
    
    model.eval()
    
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    print("\nTroubleshooting:")
    print("1. Check internet connection")
    print("2. Try: pip install --upgrade transformers tokenizers")
    print(f"3. Check if you have write permission to: {CACHE_DIR}")
    exit(1)

# ================= TRANSLATION FUNCTION =================
def translate_batch_marian(texts, max_length=512, num_beams=4):
    """Dịch batch với MarianMT"""
    if not texts:
        return []
    
    inputs = tokenizer(
        texts, 
        return_tensors="pt", 
        padding=True, 
        truncation=True, 
        max_length=max_length
    )
    
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
    
    with torch.no_grad():
        translated = model.generate(
            **inputs,
            max_length=max_length,
            num_beams=num_beams,
            early_stopping=True,
        )
    
    return [tokenizer.decode(t, skip_special_tokens=True) for t in translated]


def clean_text(text):
    """Làm sạch nhiễu OCR cơ bản trước khi đưa vào model."""
    if text is None:
        return ""

    text = str(text)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return ""

    # Loại chuỗi chỉ gồm ký tự đặc biệt hoặc quá ngắn
    if len(text) <= 1:
        return ""
    if not re.search(r"\w", text, flags=re.UNICODE):
        return ""
    if re.fullmatch(r"[\W_]+", text, flags=re.UNICODE):
        return ""

    return text


def normalize_box(box):
    """Chuẩn hóa box OCR về [x1, y1, x2, y2] dạng int."""
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return None
    try:
        x1, y1, x2, y2 = [int(v) for v in box]
    except Exception:
        return None

    x_min, x_max = sorted((x1, x2))
    y_min, y_max = sorted((y1, y2))
    if x_max <= x_min or y_max <= y_min:
        return None
    return [x_min, y_min, x_max, y_max]


def merge_boxes_on_line(texts_data, y_threshold=10, x_gap_threshold=40):
    """Gộp các box gần nhau trên cùng dòng thành 1 cụm để dịch theo câu."""
    candidates = []
    for idx, item in enumerate(texts_data):
        cleaned = clean_text(item.get('text', ''))
        if not cleaned:
            continue

        box = normalize_box(item.get('box', []))
        if not box:
            continue

        candidates.append({
            'index': idx,
            'box': box,
            'text': cleaned,
        })

    if not candidates:
        return []

    candidates.sort(key=lambda x: (x['box'][1], x['box'][0]))
    groups = []

    current = {
        'indices': [candidates[0]['index']],
        'box': list(candidates[0]['box']),
        'text_parts': [candidates[0]['text']],
    }

    for nxt in candidates[1:]:
        cx1, cy1, cx2, cy2 = current['box']
        nx1, ny1, nx2, ny2 = nxt['box']

        same_line = abs(ny1 - cy1) <= y_threshold
        gap = nx1 - cx2
        is_near = gap <= x_gap_threshold

        if same_line and is_near:
            current['indices'].append(nxt['index'])
            current['box'] = [min(cx1, nx1), min(cy1, ny1), max(cx2, nx2), max(cy2, ny2)]
            current['text_parts'].append(nxt['text'])
        else:
            groups.append({
                'indices': current['indices'],
                'box': current['box'],
                'source_text': ' '.join(current['text_parts']).strip(),
            })
            current = {
                'indices': [nxt['index']],
                'box': list(nxt['box']),
                'text_parts': [nxt['text']],
            }

    groups.append({
        'indices': current['indices'],
        'box': current['box'],
        'source_text': ' '.join(current['text_parts']).strip(),
    })
    return groups


def collect_json_files(json_dir):
    files = []
    for root, dirs, filenames in os.walk(json_dir):
        for name in filenames:
            if name.endswith('.json') and not name.endswith('_res.json'):
                files.append(os.path.join(root, name))
    return files


def translate_logic():
    """Pipeline dịch tối ưu: clean OCR -> merge line -> cross-file batching -> write back."""
    all_json_files = collect_json_files(JSON_DIR)
    if not all_json_files:
        print("⚠️  No JSON files found!")
        return 0, 0

    print(f"📦 Collecting from {len(all_json_files)} JSON files...")

    payloads = []
    unique_texts = set()

    for json_path in tqdm(all_json_files, desc="Scanning", unit="file"):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            texts_data = data.get('texts', [])
            groups = merge_boxes_on_line(
                texts_data,
                y_threshold=LINE_Y_THRESHOLD,
                x_gap_threshold=LINE_X_GAP_THRESHOLD,
            )

            for g in groups:
                src = g.get('source_text', '')
                if src:
                    unique_texts.add(src)

            payloads.append({
                'json_path': json_path,
                'data': data,
                'groups': groups,
            })
        except Exception as e:
            print(f"❌ Scan error: {json_path}: {e}")

    text_list = list(unique_texts)
    print(f"📝 Total unique merged phrases: {len(text_list)}")

    trans_map = {}
    for i in tqdm(range(0, len(text_list), TRANSLATE_BATCH_SIZE), desc="Translating", unit="batch"):
        chunk = text_list[i:i + TRANSLATE_BATCH_SIZE]
        translated_chunk = translate_batch_marian(chunk, num_beams=NUM_BEAMS)
        for orig, trans in zip(chunk, translated_chunk):
            trans_map[orig] = trans

    print("💾 Writing translations back to JSON...")
    success_count = 0

    for payload in tqdm(payloads, desc="Saving", unit="file"):
        json_path = payload['json_path']
        data = payload['data']
        groups = payload['groups']

        try:
            texts_data = data.get('texts', [])

            # Mặc định clear translated cho các box nhiễu để render không vẽ rác OCR
            for item in texts_data:
                cleaned = clean_text(item.get('text', ''))
                item['translated'] = cleaned if cleaned else ''

            # Gán bản dịch theo cách B: chỉ box đầu có nội dung, box sau để rỗng
            for group in groups:
                idxs = group.get('indices', [])
                if not idxs:
                    continue

                source_text = group.get('source_text', '')
                translated_text = trans_map.get(source_text, source_text)

                head_idx = idxs[0]
                if 0 <= head_idx < len(texts_data):
                    texts_data[head_idx]['translated'] = translated_text

                for idx in idxs[1:]:
                    if 0 <= idx < len(texts_data):
                        texts_data[idx]['translated'] = ''

            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            success_count += 1
        except Exception as e:
            print(f"❌ Save error: {json_path}: {e}")

    return success_count, len(all_json_files)


# ================= MAIN =================
def main():
    print("=" * 70)
    print("🔄 RE-TRANSLATE JSON FILES WITH MarianMT")
    print("=" * 70)
    print(f"Directory: {JSON_DIR}")
    print(f"Model: {model_name}")
    print(f"Cache: {CACHE_DIR}")
    print(f"Batch size: {TRANSLATE_BATCH_SIZE}")
    print(f"Beam size: {NUM_BEAMS}")
    print(f"Line merge y-threshold: {LINE_Y_THRESHOLD}")
    print(f"Line merge x-gap threshold: {LINE_X_GAP_THRESHOLD}")
    print(f"Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
    print("=" * 70)
    
    start_time = time.time()
    success_count, total_files = translate_logic()
    
    elapsed = time.time() - start_time

    if total_files == 0:
        return
    
    print("\n" + "=" * 70)
    print(f"✅ COMPLETED in {elapsed:.2f}s ({elapsed/60:.2f} min)")
    print(f"   Success: {success_count}/{total_files}")
    print(f"   Speed: {total_files/elapsed:.2f} files/sec")
    print("=" * 70)
    print(f"\n📝 JSON files updated in place: {JSON_DIR}")
    print("   → Now run your render script to generate images")


if __name__ == "__main__":
    main()