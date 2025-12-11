import os
import json
import torch
from transformers import MarianMTModel, MarianTokenizer
from tqdm import tqdm
import time

# ================= CẤU HÌNH =================
JSON_DIR = "./json_cache"
JSON_OUTPUT_DIR = "./json_cache"
LANG_SOURCE = 'de'
LANG_TARGET = 'en'
TRANSLATE_BATCH_SIZE = 16

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
def translate_batch_marian(texts, max_length=512):
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
        translated = model.generate(**inputs, max_length=max_length)
    
    return [tokenizer.decode(t, skip_special_tokens=True) for t in translated]


# ================= PROCESS JSON =================
def retranslate_json_file(json_path):
    """Đọc JSON, dịch lại field 'translated', ghi đè"""
    try:
        # Đọc JSON
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        texts_data = data.get('texts', [])
        
        if len(texts_data) == 0:
            return True
        
        # Collect unique texts cần dịch
        unique_texts = set()
        for item in texts_data:
            txt = item.get('text', '').strip()
            if len(txt) > 1 and not txt.isdigit():
                unique_texts.add(txt)
        
        if not unique_texts:
            return True
        
        # Translate batch
        trans_map = {}
        text_list = list(unique_texts)
        
        for i in range(0, len(text_list), TRANSLATE_BATCH_SIZE):
            chunk = text_list[i:i + TRANSLATE_BATCH_SIZE]
            translated_chunk = translate_batch_marian(chunk)
            
            for orig, trans in zip(chunk, translated_chunk):
                trans_map[orig] = trans
        
        # Update 'translated' field
        for item in texts_data:
            txt = item.get('text', '').strip()
            if txt in trans_map:
                item['translated'] = trans_map[txt]
            else:
                item['translated'] = txt
        
        # Ghi đè JSON
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {json_path}: {e}")
        return False


# ================= MAIN =================
def main():
    print("=" * 70)
    print("🔄 RE-TRANSLATE JSON FILES WITH MarianMT")
    print("=" * 70)
    print(f"Directory: {JSON_DIR}")
    print(f"Model: {model_name}")
    print(f"Cache: {CACHE_DIR}")
    print(f"Batch size: {TRANSLATE_BATCH_SIZE}")
    print(f"Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
    print("=" * 70)
    
    # Collect all JSON files
    json_files = []
    for root, dirs, files in os.walk(JSON_DIR):
        for f in files:
            if f.endswith('.json') and not f.endswith('_res.json'):
                json_files.append(os.path.join(root, f))
    
    total_files = len(json_files)
    
    if total_files == 0:
        print("⚠️  No JSON files found!")
        return
    
    print(f"📦 Found {total_files} JSON files\n")
    
    start_time = time.time()
    success_count = 0
    
    # Process with progress bar
    for json_path in tqdm(json_files, desc="Translating", unit="file"):
        if retranslate_json_file(json_path):
            success_count += 1
    
    elapsed = time.time() - start_time
    
    print("\n" + "=" * 70)
    print(f"✅ COMPLETED in {elapsed:.2f}s ({elapsed/60:.2f} min)")
    print(f"   Success: {success_count}/{total_files}")
    print(f"   Speed: {total_files/elapsed:.2f} files/sec")
    print("=" * 70)
    print(f"\n📝 JSON files updated in place: {JSON_DIR}")
    print("   → Now run your render script to generate images")


if __name__ == "__main__":
    main()