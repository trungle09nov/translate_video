#!/usr/bin/env python3
"""
translate_srt.py - Dịch SRT tiếng Anh → tiếng Việt qua vLLM (Qwen)

Input:  subtitle_en.srt (timing chuẩn từ audio EN)
Output: subtitle_vi.srt (cùng timing, text VI)

Logic:
  - Gom entries thành batch (--batch-size) để giảm số lần gọi API
  - Dịch bằng vLLM (Qwen3-14B-AWQ) qua llm_translate.py
  - Giữ nguyên index + timing, chỉ thay text

Usage:
    python translate_srt.py
    python translate_srt.py --srt audio_en_full_eng.srt --output subtitle_vi.srt
    python translate_srt.py --batch-size 20
"""

import sys
import re
import argparse
from pathlib import Path

from llm_translate import translate_entries

# ── Config ────────────────────────────────────────────────────────────────────
SRT_INPUT  = "data_translate/transcript/sageactive_2.mp3_eng.srt"
SRT_OUTPUT = "data_translate/transcript/sageactive_2.mp3_vi.srt"
BATCH_SIZE = 20    # entries per API call


# ── SRT parse / write ─────────────────────────────────────────────────────────

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


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Dịch SRT EN → VI qua vLLM (Qwen)")
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

    print(f"\n[2] Translating EN → VI via LLM (batch_size={args.batch_size})...")
    entries = translate_entries(entries, args.batch_size)

    write_srt(entries, args.output)
    print(f"\n[3] Written: {args.output}")
    print(f"    {len(entries)} entries")


if __name__ == "__main__":
    main()
