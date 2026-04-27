#!/usr/bin/env python3
"""
final_assemble.py - Combine EN slides video + EN audio + VI subtitles

Steps:
  1. Merge audio Part 1 + Part 2 → audio_en_full.mp3  (if not already merged)
  2. Burn VI subtitles into video + mux EN audio → output_final.mp4

Usage:
    python final_assemble.py --video video_output/video_translated.mp4 --audio audio_en_full.mp3 --subtitle aligned_vi_split.srt --output output_final.mp4
"""

import subprocess
import sys
import os
import shutil
import argparse
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
FFMPEG  = r"D:\repo\translate_video\FFmpeg\bin\ffmpeg.exe"
FFPROBE = r"D:\repo\translate_video\FFmpeg\bin\ffprobe.exe"


# ── Helpers ───────────────────────────────────────────────────────────────────

def run(cmd: list, description: str):
    """Run subprocess, print output on error."""
    print(f"  $ {' '.join(cmd[:6])}{'...' if len(cmd) > 6 else ''}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ERROR: {description} failed")
        print(result.stderr[-1000:])
        sys.exit(1)
    return result


def subtitle_path_ffmpeg(path: str) -> str:
    """
    Convert a path to the format ffmpeg subtitles filter expects on Windows.
    Drive letter colon must be escaped: C:/path → C\:/path
    """
    abs_path = str(Path(path).resolve())
    abs_path = abs_path.replace("\\", "/")
    if len(abs_path) >= 2 and abs_path[1] == ":":
        abs_path = abs_path[0] + "\\:" + abs_path[2:]
    return abs_path


def get_video_size(video_path: str) -> tuple[int, int]:
    """Return (width, height) of video using ffprobe."""
    result = subprocess.run([
        FFPROBE, "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "csv=p=0",
        video_path,
    ], capture_output=True, text=True)
    if result.returncode == 0 and result.stdout.strip():
        parts = result.stdout.strip().split(",")
        return int(parts[0]), int(parts[1])
    return 1920, 1080  # fallback


def file_size_mb(path: str) -> float:
    return Path(path).stat().st_size / (1024 * 1024)


def get_duration(path: str) -> float:
    """Get media file duration in seconds via ffprobe."""
    result = subprocess.run([
        FFPROBE, "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        path,
    ], capture_output=True, text=True)
    try:
        return float(result.stdout.strip())
    except ValueError:
        return 0.0


def get_srt_duration(path: str) -> float:
    """Get last end timestamp from SRT file."""
    import re
    content = Path(path).read_text(encoding="utf-8")
    times = re.findall(r'-->\s*(\d{2}:\d{2}:\d{2},\d{3})', content)
    if not times:
        return 0.0
    last = times[-1]
    h, m, rest = last.split(":")
    s, ms = rest.split(",")
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000


def check_durations(video_path: str, audio_path: str, srt_path: str):
    """Warn if durations differ by more than 5%."""
    vid_dur = get_duration(video_path)
    aud_dur = get_duration(audio_path)
    srt_dur = get_srt_duration(srt_path)

    print(f"  Video : {vid_dur:.1f}s")
    print(f"  Audio : {aud_dur:.1f}s")
    print(f"  SRT   : {srt_dur:.1f}s")

    ref = vid_dur if vid_dur > 0 else aud_dur
    if ref > 0:
        if abs(aud_dur - ref) / ref > 0.05:
            print(f"  WARNING: audio/video duration mismatch > 5%")
        if srt_dur > 0 and abs(srt_dur - ref) / ref > 0.05:
            print(f"  WARNING: SRT duration mismatch > 5% (normal if SRT ends before video)")


# ── Assembly ──────────────────────────────────────────────────────────────────

def assemble(video_path: str, audio_path: str, subtitle_path: str, output_path: str, use_ass: bool):
    for path, label in [(video_path, "video"), (audio_path, "audio"), (subtitle_path, "subtitle")]:
        if not Path(path).exists():
            print(f"  ERROR: {label} file not found: {path}")
            sys.exit(1)

    print(f"  Video   : {video_path}")
    print(f"  Audio   : {audio_path}")
    print(f"  Subtitle: {subtitle_path}")

    # Copy subtitle to CWD — avoids Windows drive-letter path escaping in ffmpeg filter
    sub_ext = ".ass" if use_ass else ".srt"
    tmp_sub = f"_tmp_subtitle{sub_ext}"
    shutil.copy(subtitle_path, tmp_sub)

    if use_ass:
        vf = f"ass={tmp_sub}"
    else:
        style = "force_style='FontSize=24,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,Outline=2,Shadow=1'"
        vf = f"subtitles={tmp_sub}:{style}"

    # Determine audio/video length relationship
    vid_dur = get_duration(video_path)
    aud_dur = get_duration(audio_path)
    if aud_dur <= vid_dur:
        length_flags = ["-shortest"]          # audio shorter → stop at audio end
    else:
        length_flags = ["-t", str(vid_dur)]   # audio longer  → cut to video duration

    try:
        run([
            FFMPEG, "-y",
            "-i", video_path,
            "-i", audio_path,
            "-vf", vf,
            "-c:v", "libx264", "-preset", "medium", "-crf", "23",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k",
            "-map", "0:v:0", "-map", "1:a:0",
            *length_flags,
            output_path,
        ], "ffmpeg final assembly")
    finally:
        Path(tmp_sub).unlink(missing_ok=True)

    print(f"  Written: {output_path}  ({file_size_mb(output_path):.1f} MB)")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Assemble final EN video with VI subtitles")
    parser.add_argument("--video",    required=True,              help="EN slides video file")
    parser.add_argument("--audio",    required=True,              help="EN audio file (merged)")
    parser.add_argument("--subtitle", required=True,              help="VI subtitle SRT/ASS file")
    parser.add_argument("--output",   default="output_final.mp4", help="Output video file")
    parser.add_argument("--use-ass",  action="store_true",        help="Treat subtitle as ASS format")
    args = parser.parse_args()

    print("=" * 60)
    print("FINAL ASSEMBLE")
    print("=" * 60)

    # Step 1: Duration check
    print("\n[1] Duration check...")
    check_durations(args.video, args.audio, args.subtitle)

    # Step 2: Assemble
    print("\n[2] Assembling final video...")
    assemble(args.video, args.audio, args.subtitle, args.output, use_ass=args.use_ass)

    print("\nDone: " + args.output)
    print("=" * 60)



if __name__ == "__main__":
    main()
