import subprocess
import os
import json
from pipeline_config import ASSEMBLE_OUTPUT_FPS_MODE

FRAMES_ROOT = "./frames_done"      # Frames đã dịch
RAW_FRAMES_ROOT = "./frames_raw"   # Frames gốc (chứa _extract_meta.json từ bước extract)
VIDEO_ROOT = "./data"              # Video gốc (có audio)
OUTPUT_ROOT = "./video_output"     # Video output

os.makedirs(OUTPUT_ROOT, exist_ok=True)


def get_video_fps_fraction(video_path):
    """Lấy FPS gốc dưới dạng phân số (ví dụ: 30000/1001)"""
    try:
        cmd = [
            "ffprobe", "-v", "0",
            "-select_streams", "v:0",
            "-show_entries", "stream=r_frame_rate",
            "-of", "default=noprint_wrappers=1:nokey=1",
            video_path
        ]
        return subprocess.check_output(cmd).decode().strip()
    except:
        return "30/1"


def get_video_duration(video_path):
    """Lấy thời lượng video (seconds)"""
    try:
        cmd = [
            "ffprobe", "-v", "0",
            "-select_streams", "v:0",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            video_path
        ]
        return float(subprocess.check_output(cmd).decode().strip())
    except:
        return 0


def count_frames_in_dir(frames_dir):
    """Đếm số frames trong thư mục"""
    return len([f for f in os.listdir(frames_dir) if f.endswith('.jpg')])


def parse_fps_fraction(fps_fraction):
    try:
        if "/" in fps_fraction:
            n, d = fps_fraction.split("/", 1)
            n = float(n)
            d = float(d)
            if d == 0:
                return 0.0
            return n / d
        return float(fps_fraction)
    except Exception:
        return 0.0


def get_extract_meta(frames_dir):
    meta_path = os.path.join(frames_dir, "_extract_meta.json")
    if not os.path.exists(meta_path):
        return None

    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def infer_extract_fps_fraction(num_frames, duration_sec):
    """Suy ra FPS tách frame từ số frame và duration video gốc (khi thiếu meta)."""
    if duration_sec <= 0 or num_frames <= 0:
        return None
    inferred = num_frames / duration_sec
    if inferred <= 0:
        return None
    return f"{inferred:.6f}"


def choose_output_fps(source_fps_fraction, extract_fps_fraction):
    mode = ASSEMBLE_OUTPUT_FPS_MODE
    if mode == "source":
        return source_fps_fraction
    if mode == "extracted":
        return extract_fps_fraction

    # auto mode
    src = parse_fps_fraction(source_fps_fraction)
    ext = parse_fps_fraction(extract_fps_fraction)
    if src <= 0 or ext <= 0:
        return source_fps_fraction

    if abs(src - ext) <= 0.01:
        return source_fps_fraction
    return extract_fps_fraction


def assemble_video_for_folder(subdir):
    frames_dir = os.path.join(FRAMES_ROOT, subdir)
    raw_frames_dir = os.path.join(RAW_FRAMES_ROOT, subdir)
    video_source = os.path.join(VIDEO_ROOT, f"{subdir}.mp4")
    output_video = os.path.join(OUTPUT_ROOT, f"{subdir}_translated.mp4")

    if not os.path.exists(video_source):
        print(f"⚠️  Không tìm thấy video gốc: {video_source}")
        return

    if not os.path.exists(frames_dir):
        print(f"⚠️  Không có thư mục frames: {frames_dir}")
        return

    # Lấy thông tin video gốc + metadata extract
    fps_fraction = get_video_fps_fraction(video_source)
    duration = get_video_duration(video_source)
    num_frames = count_frames_in_dir(frames_dir)

    meta = get_extract_meta(frames_dir)
    meta_source = "frames_done/_extract_meta.json"
    if not meta:
        # Bước render không copy _extract_meta.json sang frames_done,
        # nên fallback đọc trực tiếp từ frames_raw.
        meta = get_extract_meta(raw_frames_dir)
        if meta:
            meta_source = "frames_raw/_extract_meta.json"

    extract_fps_fraction = fps_fraction
    if meta:
        extract_fps_fraction = str(meta.get("extract_fps", fps_fraction))
    else:
        inferred = infer_extract_fps_fraction(num_frames, duration)
        if inferred:
            extract_fps_fraction = inferred
            meta_source = "inferred_from_frame_count"
        else:
            meta_source = "fallback_source_fps"

    output_fps_fraction = choose_output_fps(fps_fraction, extract_fps_fraction)
    
    print(f"🎬 Ghép video: {subdir}")
    print(f"   FPS gốc: {fps_fraction}")
    print(f"   FPS tách frame: {extract_fps_fraction} ({meta_source})")
    print(f"   FPS output: {output_fps_fraction} (mode: {ASSEMBLE_OUTPUT_FPS_MODE})")
    print(f"   Duration: {duration:.2f}s")
    print(f"   Frames: {num_frames}")

    # ✅ Ghép: frames -> video + audio gốc, luôn giữ đúng duration của video nguồn.
    cmd = [
        "ffmpeg", "-y",
        "-framerate", extract_fps_fraction,
        "-i", f"{frames_dir}/frame_%06d.jpg",   # Input: frames đã dịch
        "-i", video_source,                      # Input: video gốc (lấy audio)
        "-c:v", "libx264",                       # Codec video
        "-preset", "medium",                     # Preset encode
        "-crf", "23",                            # Chất lượng (18-28, thấp = chất lượng cao)
        "-pix_fmt", "yuv420p",                   # Format tương thích
        "-r", output_fps_fraction,
        "-map", "0:v:0",                         # Map video từ frames
        "-map", "1:a:0?",                        # Map audio từ video gốc (? = optional)
        "-c:a", "aac",                           # Encode audio
        "-b:a", "192k",                          # Bitrate audio
        "-af", "apad",                           # Nếu audio ngắn hơn thì pad silence để không cắt video
        "-t", f"{duration:.6f}",                 # Ép output bằng đúng thời lượng video gốc
        output_video
    ]

    # Run ffmpeg
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"   ❌ Lỗi ffmpeg:")
        print(f"   {result.stderr[-500:]}")  # In 500 ký tự cuối của error
    elif os.path.exists(output_video):
        size_mb = os.path.getsize(output_video) / (1024*1024)
        
        # Verify output duration
        output_duration = get_video_duration(output_video)
        print(f"   ✔ Done: {output_video}")
        print(f"   Size: {size_mb:.2f} MB")
        print(f"   Duration: {output_duration:.2f}s (expected: {duration:.2f}s)")
        
        # Warning nếu duration không khớp
        if abs(output_duration - duration) > 1.0:
            print(f"   ⚠️  WARNING: Duration mismatch! Check if frames count is correct.")
    else:
        print(f"   ❌ Không tạo được file output")


def main():
    subdirs = [
        d for d in os.listdir(FRAMES_ROOT)
        if os.path.isdir(os.path.join(FRAMES_ROOT, d))
    ]

    if not subdirs:
        print("⚠️  Không tìm thấy thư mục frames nào trong ./frames_done")
        return

    print("=" * 70)
    print("🎬 VIDEO ASSEMBLY - FRAMES TO VIDEO")
    print("=" * 70)
    print(f"🔍 Found {len(subdirs)} videos to assemble\n")

    success_count = 0
    for i, subdir in enumerate(subdirs, 1):
        print(f"\n[{i}/{len(subdirs)}] ", end="")
        try:
            assemble_video_for_folder(subdir)
            success_count += 1
        except Exception as e:
            print(f"   ❌ Exception: {e}")

    print("\n" + "=" * 70)
    print(f"🎉 Completed: {success_count}/{len(subdirs)} videos")
    print(f"📁 Output directory: {OUTPUT_ROOT}")
    print("=" * 70)


if __name__ == "__main__":
    main()