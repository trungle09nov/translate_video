#!/usr/bin/env python3
"""
translate_srt.py - Dịch SRT tiếng Anh → tiếng Việt qua Google Translate

Input:  subtitle_en.srt (timing chuẩn từ audio EN)
Output: subtitle_vi.srt (cùng timing, text VI)

Logic:
  - Gom entries thành batch (--batch-size) để giảm số lần gọi API
  - Dịch bằng deep-translator (GoogleTranslator)
  - Giữ nguyên index + timing, chỉ thay text

Usage:
    python translate_srt.py
    python translate_srt.py --srt audio_en_full_eng.srt --output subtitle_vi.srt
    python translate_srt.py --batch-size 30
"""

import sys
import re
import time
import argparse
from pathlib import Path

try:
    from deep_translator import GoogleTranslator
except ImportError:
    print("ERROR: deep-translator not installed. Run: pip install deep-translator")
    sys.exit(1)

# ── Config ────────────────────────────────────────────────────────────────────
SRT_INPUT   = "data_translate/transcript/audio_en_full_eng.srt"
SRT_OUTPUT  = "data_translate/transcript/subtitle_vi.srt"
BATCH_SIZE  = 20    # entries per API call
SOURCE_LANG = "en"
TARGET_LANG = "vi"
RETRY_DELAY = 2.0   # giây chờ khi gặp lỗi trước khi retry


# ── SRT parse / write ─────────────────────────────────────────────────────────

def srt_to_sec(t: str) -> float:
    h, m, rest = t.split(":")
    s, ms = rest.split(",")
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000


def parse_srt(content: str) -> list[dict]:
    pattern = re.compile(
        r"(\d+)\n(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})\n(.+?)(?=\n\n|\Z)",
        re.DOTALL,
    )
    entries = []
    for idx, start, end, text in pattern.findall(content):
        entries.append({
            "index": int(idx),
            "start": start,
            "end":   end,
            "text":  text.replace("\n", " ").strip(),
        })
    return entries


def write_srt(entries: list[dict], path: str):
    lines = []
    for i, e in enumerate(entries, 1):
        lines.append(str(i))
        lines.append(f"{e['start']} --> {e['end']}")
        lines.append(e["text_vi"])
        lines.append("")
    Path(path).write_text("\n".join(lines), encoding="utf-8")


# ── Translation ───────────────────────────────────────────────────────────────

def translate_batch(texts: list[str], translator: GoogleTranslator) -> list[str]:
    """
    Dịch một batch texts. Retry 1 lần nếu lỗi.
    Nếu batch fail → dịch từng entry riêng lẻ để không mất data.
    """
    try:
        return translator.translate_batch(texts)
    except Exception as e:
        print(f"\n    [warn] Batch translate failed: {e}")
        print(f"    Retrying individually ({len(texts)} entries)...")
        time.sleep(RETRY_DELAY)
        results = []
        for text in texts:
            try:
                results.append(translator.translate(text))
                time.sleep(0.3)
            except Exception as e2:
                print(f"    [warn] Entry failed, keeping original: {e2}")
                results.append(text)   # fallback: giữ nguyên EN
        return results


def translate_entries(entries: list[dict], batch_size: int) -> list[dict]:
    translator = GoogleTranslator(source=SOURCE_LANG, target=TARGET_LANG)
    total = len(entries)
    result = list(entries)   # copy để giữ nguyên cấu trúc

    for start in range(0, total, batch_size):
        batch = entries[start : start + batch_size]
        texts = [e["text"] for e in batch]

        translated = translate_batch(texts, translator)

        for i, (entry, vi_text) in enumerate(zip(batch, translated)):
            result[start + i]["text_vi"] = vi_text or entry["text"]  # fallback EN nếu None

        done = min(start + batch_size, total)
        print(f"  [{done:>4}/{total}] {batch[-1]['start']} → {batch[-1]['end']}", end="\r")

    print()  # newline sau progress
    return result


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Dịch SRT EN → VI qua Google Translate")
    parser.add_argument("--srt",        default=SRT_INPUT,
                        help=f"Input EN SRT file (default: {SRT_INPUT})")
    parser.add_argument("--output",     default=SRT_OUTPUT,
                        help=f"Output VI SRT file (default: {SRT_OUTPUT})")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help=f"Entries per API call (default: {BATCH_SIZE})")
    args = parser.parse_args()

    print(f"[1] Loading: {args.srt}")
    content = Path(args.srt).read_text(encoding="utf-8")
    entries = parse_srt(content)
    print(f"    {len(entries)} entries")

    print(f"\n[2] Translating EN → VI (batch_size={args.batch_size})...")
    entries = translate_entries(entries, args.batch_size)

    write_srt(entries, args.output)
    print(f"\n[3] Written: {args.output}")
    print(f"    {len(entries)} entries")


if __name__ == "__main__":
    main()
