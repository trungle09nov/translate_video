#!/usr/bin/env python3
"""
merge_audio_srt.py - Ghép 2 audio + 2 SRT thành 1 file

Bước 1: Lấy duration của audio 1 bằng ffprobe
Bước 2: Offset timing SRT 2 thêm duration đó
Bước 3: Nối SRT 1 + SRT 2 (offset) → file SRT gộp
Bước 4: Ghép audio 1 + audio 2 bằng FFmpeg → file audio gộp

Usage:
    python merge_audio_srt.py
    python merge_audio_srt.py --audio1 a1.mp3 --audio2 a2.mp3 --srt1 s1.srt --srt2 s2.srt
"""

import re
import os
import sys
import json
import shutil
import argparse
import subprocess
from pathlib import Path

TRANSCRIPT_DIR = Path("data_translate/transcript")
AUDIO_DIR      = Path("data_translate/audio")

AUDIO1 = str(AUDIO_DIR / "sageactive_1.mp3.mp3")
AUDIO2 = str(AUDIO_DIR / "sageactive_2.mp3.mp3")
SRT1   = str(TRANSCRIPT_DIR / "sageactive_1.mp3_vi.srt")
SRT2   = str(TRANSCRIPT_DIR / "sageactive_2.mp3_vi.srt")
OUT_AUDIO = str(AUDIO_DIR / "sageactive_merged.mp3")
OUT_SRT   = str(TRANSCRIPT_DIR / "sageactive_merged_vi.srt")
FFMPEG_BIN_DIR = Path(__file__).resolve().parent / "FFmpeg" / "bin"


# ── Helpers ───────────────────────────────────────────────────────────────────

def resolve_ff_binary(tool_name: str) -> str:
    """Ưu tiên dùng binary trong FFmpeg/bin của project, fallback sang PATH."""
    exe_name = f"{tool_name}.exe" if os.name == "nt" else tool_name
    local_bin = FFMPEG_BIN_DIR / exe_name
    if local_bin.exists():
        return str(local_bin)

    from_path = shutil.which(tool_name) or shutil.which(exe_name)
    if from_path:
        return from_path

    print(
        "ERROR: Không tìm thấy binary cho "
        f"'{tool_name}'.\n"
        f"Đã tìm tại: {local_bin}\n"
        "Hãy kiểm tra thư mục FFmpeg/bin hoặc thêm FFmpeg vào PATH."
    )
    sys.exit(1)

def get_duration(audio_path: str) -> float:
    """Lấy duration (giây) của file audio bằng ffprobe."""
    cmd = [
        resolve_ff_binary("ffprobe"), "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        audio_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ERROR: ffprobe thất bại:\n{result.stderr}")
        sys.exit(1)
    data = json.loads(result.stdout)
    return float(data["format"]["duration"])


def tc_to_ms(tc: str) -> int:
    """'HH:MM:SS,mmm' → milliseconds"""
    h, m, rest = tc.split(":")
    s, ms = rest.split(",")
    return int(h) * 3600_000 + int(m) * 60_000 + int(s) * 1000 + int(ms)


def ms_to_tc(ms: int) -> str:
    """milliseconds → 'HH:MM:SS,mmm'"""
    h   = ms // 3600_000;  ms %= 3600_000
    m   = ms // 60_000;    ms %= 60_000
    s   = ms // 1000;      ms %= 1000
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


def parse_srt(path: str) -> list[dict]:
    content = Path(path).read_text(encoding="utf-8")
    pattern = re.compile(
        r"(\d+)\n(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})\n(.+?)(?=\n\n|\Z)",
        re.DOTALL,
    )
    entries = []
    for idx, start, end, text in pattern.findall(content):
        entries.append({
            "start_ms": tc_to_ms(start),
            "end_ms":   tc_to_ms(end),
            "text":     text.replace("\n", " ").strip(),
        })
    return entries


def offset_entries(entries: list[dict], offset_ms: int) -> list[dict]:
    return [
        {**e, "start_ms": e["start_ms"] + offset_ms, "end_ms": e["end_ms"] + offset_ms}
        for e in entries
    ]


def write_srt(entries: list[dict], path: str):
    lines = []
    for i, e in enumerate(entries, 1):
        lines.append(str(i))
        lines.append(f"{ms_to_tc(e['start_ms'])} --> {ms_to_tc(e['end_ms'])}")
        lines.append(e["text"])
        lines.append("")
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def concat_audio(audio1: str, audio2: str, output: str):
    """Ghép 2 audio bằng FFmpeg (concat demuxer)."""
    list_file = Path("_concat_list.txt")
    list_file.write_text(
        f"file '{Path(audio1).resolve()}'\nfile '{Path(audio2).resolve()}'\n",
        encoding="utf-8",
    )
    cmd = [
        resolve_ff_binary("ffmpeg"), "-y",
        "-f", "concat", "-safe", "0",
        "-i", str(list_file),
        "-c", "copy",
        output,
    ]
    print(f"  $ {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    list_file.unlink(missing_ok=True)
    if result.returncode != 0:
        print(f"ERROR: FFmpeg thất bại:\n{result.stderr}")
        sys.exit(1)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Ghép 2 audio + 2 SRT thành 1")
    parser.add_argument("--audio1",     default=AUDIO1)
    parser.add_argument("--audio2",     default=AUDIO2)
    parser.add_argument("--srt1",       default=SRT1)
    parser.add_argument("--srt2",       default=SRT2)
    parser.add_argument("--out-audio",  default=OUT_AUDIO)
    parser.add_argument("--out-srt",    default=OUT_SRT)
    parser.add_argument("--no-audio",   action="store_true",
                        help="Chỉ ghép SRT, bỏ qua bước ghép audio")
    args = parser.parse_args()

    # Bước 1: duration audio 1
    print(f"[1] Lấy duration: {args.audio1}")
    dur1_sec = get_duration(args.audio1)
    dur1_ms  = int(dur1_sec * 1000)
    print(f"    Duration audio 1: {dur1_sec:.3f}s ({dur1_ms} ms)")

    # Bước 2+3: ghép SRT
    print(f"\n[2] Parse SRT...")
    entries1 = parse_srt(args.srt1)
    entries2 = parse_srt(args.srt2)
    print(f"    SRT 1: {len(entries1)} entries")
    print(f"    SRT 2: {len(entries2)} entries")

    entries2_offset = offset_entries(entries2, dur1_ms)
    merged = entries1 + entries2_offset

    write_srt(merged, args.out_srt)
    print(f"\n[3] SRT gộp: {args.out_srt}")
    print(f"    {len(merged)} entries tổng")

    # Bước 4: ghép audio
    if args.no_audio:
        print("\n[4] Bỏ qua ghép audio (--no-audio)")
        return

    print(f"\n[4] Ghép audio...")
    concat_audio(args.audio1, args.audio2, args.out_audio)
    print(f"    Audio gộp: {args.out_audio}")
    print("\nDone!")


if __name__ == "__main__":
    main()
