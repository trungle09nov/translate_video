#!/usr/bin/env python3
"""
split_srt.py - Chia entries SRT dài thành ngắn hơn, bỏ tên người nói

Input:  aligned_vi.srt
Output: aligned_vi_split.srt

Logic:
  1. Bỏ "Tên: " ở đầu mỗi entry
  2. Tách câu theo dấu câu (. ! ?)
  3. Tách tiếp theo dấu phẩy nếu vẫn dài
  4. Tách theo max_words nếu vẫn còn dài
  5. Phân bổ timing tỉ lệ theo số ký tự
  6. Cắt display time theo max_duration

Usage:
    python split_srt.py --srt aligned_vi.srt --max-words 8 --max-duration 4 --output aligned_vi_split.srt
"""

import re
import argparse
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
SRT_INPUT   = "data_translate/transcript/translate-to-VN.srt"
SRT_OUTPUT  = "data_translate/transcript/subtitle_vi_split.srt"
MAX_WORDS   = 8    # Số từ tối đa mỗi entry
MAX_DURATION = 2 # Giây hiển thị tối đa mỗi entry
MIN_DURATION = 0.5 # Giây tối thiểu (bỏ chunk quá ngắn)


# ── Time helpers ──────────────────────────────────────────────────────────────

def srt_to_sec(t: str) -> float:
    h, m, rest = t.split(":")
    s, ms = rest.split(",")
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000


def sec_to_srt(sec: float) -> str:
    sec = max(0.0, sec)
    h   = int(sec // 3600)
    m   = int((sec % 3600) // 60)
    s   = int(sec % 60)
    ms  = min(999, int(round((sec - int(sec)) * 1000)))
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


# ── SRT parse / write ─────────────────────────────────────────────────────────

def parse_srt(content: str) -> list[dict]:
    pattern = re.compile(
        r"\d+\n(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})\n(.+?)(?=\n\n|\Z)",
        re.DOTALL,
    )
    entries = []
    for start, end, text in pattern.findall(content):
        entries.append({
            "start_sec": srt_to_sec(start),
            "end_sec":   srt_to_sec(end),
            "text":      text.replace("\n", " ").strip(),
        })
    return entries


def write_srt(entries: list[dict], path: str):
    lines = []
    for i, e in enumerate(entries, 1):
        lines.append(str(i))
        lines.append(f"{sec_to_srt(e['start_sec'])} --> {sec_to_srt(e['end_sec'])}")
        lines.append(e["text"])
        lines.append("")
    Path(path).write_text("\n".join(lines), encoding="utf-8")


# ── Text processing ───────────────────────────────────────────────────────────

SPEAKER_RE = re.compile(r'^[A-Z][^\n:]{1,40}:\s*')

def remove_speaker(text: str) -> str:
    """Xóa 'Tên người nói: ' ở đầu."""
    return SPEAKER_RE.sub('', text).strip()


def split_text(text: str, max_words: int) -> list[str]:
    """
    Tách text thành chunks ngắn:
      1. Tách theo dấu câu kết thúc (. ! ?)
      2. Tách theo dấu phẩy nếu vẫn dài
      3. Tách theo max_words
    """
    # Bước 1: tách theo câu
    sentences = re.split(r'(?<=[.!?])\s+', text)

    chunks = []
    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue

        if len(sent.split()) <= max_words:
            chunks.append(sent)
        else:
            # Bước 2: tách theo dấu phẩy
            comma_parts = re.split(r',\s*', sent)
            buffer = ""
            for part in comma_parts:
                part = part.strip()
                if not part:
                    continue
                candidate = (buffer + ", " + part).strip(", ") if buffer else part
                if len(candidate.split()) <= max_words:
                    buffer = candidate
                else:
                    if buffer:
                        chunks.append(buffer)
                    buffer = part

            if buffer:
                # Bước 3: nếu buffer vẫn quá dài → tách theo từ
                words = buffer.split()
                if len(words) > max_words:
                    for i in range(0, len(words), max_words):
                        chunks.append(" ".join(words[i:i + max_words]))
                else:
                    chunks.append(buffer)

    # Fallback: nếu không có gì (không có dấu câu)
    if not chunks:
        words = text.split()
        for i in range(0, len(words), max_words):
            chunks.append(" ".join(words[i:i + max_words]))

    return [c for c in chunks if c.strip()]


def assign_timing(
    chunks: list[str],
    start_sec: float,
    end_sec: float,
    max_duration: float,
    min_duration: float,
) -> list[dict]:
    """
    Phân bổ timing tỉ lệ theo số ký tự.
    - Start time: tỉ lệ với vị trí ký tự
    - End time: start + min(tỉ lệ, max_duration)
    - Bỏ chunk quá ngắn (< min_duration)
    """
    total_dur = end_sec - start_sec
    weights = [max(1, len(c)) for c in chunks]
    total_weight = sum(weights)

    entries = []
    cumulative = 0
    for chunk, weight in zip(chunks, weights):
        prop_start = start_sec + total_dur * cumulative / total_weight
        prop_end   = start_sec + total_dur * (cumulative + weight) / total_weight
        actual_end = min(prop_end, prop_start + max_duration)

        if actual_end - prop_start >= min_duration:
            entries.append({
                "start_sec": prop_start,
                "end_sec":   actual_end,
                "text":      chunk,
            })
        cumulative += weight

    return entries


# ── Main processing ───────────────────────────────────────────────────────────

def process(
    entries: list[dict],
    max_words: int,
    max_duration: float,
    min_duration: float,
) -> list[dict]:
    result = []
    skipped = 0

    for entry in entries:
        text = remove_speaker(entry["text"])
        if not text:
            skipped += 1
            continue

        duration = entry["end_sec"] - entry["start_sec"]

        # Entry ngắn → giữ nguyên (sau khi bỏ tên)
        word_count = len(text.split())
        if word_count <= max_words and duration <= max_duration:
            result.append({
                "start_sec": entry["start_sec"],
                "end_sec":   min(entry["end_sec"], entry["start_sec"] + max_duration),
                "text":      text,
            })
            continue

        # Entry dài → chia nhỏ
        chunks = split_text(text, max_words)
        if len(chunks) == 1:
            result.append({
                "start_sec": entry["start_sec"],
                "end_sec":   min(entry["end_sec"], entry["start_sec"] + max_duration),
                "text":      chunks[0],
            })
        else:
            sub_entries = assign_timing(chunks, entry["start_sec"], entry["end_sec"],
                                        max_duration, min_duration)
            result.extend(sub_entries)

    return result, skipped


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Split long SRT entries into short subtitles")
    parser.add_argument("--max-words",    type=int,   default=MAX_WORDS,
                        help=f"Max words per subtitle entry (default: {MAX_WORDS})")
    parser.add_argument("--max-duration", type=float, default=MAX_DURATION,
                        help=f"Max display seconds per entry (default: {MAX_DURATION})")
    parser.add_argument("--min-duration", type=float, default=MIN_DURATION,
                        help=f"Min display seconds (drop shorter, default: {MIN_DURATION})")
    parser.add_argument("--srt",    default=SRT_INPUT,  help="Input aligned VI SRT file")
    parser.add_argument("--output", default=SRT_OUTPUT, help="Output split SRT file")
    args = parser.parse_args()

    print(f"[1] Loading: {args.srt}")
    entries = parse_srt(Path(args.srt).read_text(encoding="utf-8"))
    print(f"    {len(entries)} entries")

    print(f"\n[2] Splitting (max_words={args.max_words}, max_duration={args.max_duration}s)...")
    result, skipped = process(entries, args.max_words, args.max_duration, args.min_duration)

    print(f"    {len(entries)} → {len(result)} entries  (skipped {skipped} empty)")

    write_srt(result, args.output)
    print(f"\n[3] Written: {args.output}")
    if result:
        print(f"    Duration: {result[-1]['end_sec']:.1f}s")


if __name__ == "__main__":
    main()
