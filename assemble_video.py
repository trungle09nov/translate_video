import subprocess
import os

FRAMES_ROOT = "./frames_done"      # Frames đã dịch
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


def assemble_video_for_folder(subdir):
    frames_dir = os.path.join(FRAMES_ROOT, subdir)
    video_source = os.path.join(VIDEO_ROOT, f"{subdir}.mp4")
    output_video = os.path.join(OUTPUT_ROOT, f"{subdir}_translated.mp4")

    if not os.path.exists(video_source):
        print(f"⚠️  Không tìm thấy video gốc: {video_source}")
        return

    if not os.path.exists(frames_dir):
        print(f"⚠️  Không có thư mục frames: {frames_dir}")
        return

    # Lấy thông tin video gốc
    fps_fraction = get_video_fps_fraction(video_source)
    duration = get_video_duration(video_source)
    num_frames = count_frames_in_dir(frames_dir)
    
    print(f"🎬 Ghép video: {subdir}")
    print(f"   FPS gốc: {fps_fraction}")
    print(f"   Duration: {duration:.2f}s")
    print(f"   Frames: {num_frames}")

    # ✅ Ghép: frames (1fps) → video (fps gốc) + audio gốc
    cmd = [
        "ffmpeg", "-y",
        "-framerate", "1",                       # ✅ Đọc frames với 1 fps
        "-i", f"{frames_dir}/frame_%06d.jpg",   # Input: frames đã dịch
        "-i", video_source,                      # Input: video gốc (lấy audio)
        "-c:v", "libx264",                       # Codec video
        "-preset", "medium",                     # Preset encode
        "-crf", "23",                            # Chất lượng (18-28, thấp = chất lượng cao)
        "-pix_fmt", "yuv420p",                   # Format tương thích
        "-r", fps_fraction,                      # ✅ Output FPS = FPS gốc
        "-map", "0:v:0",                         # Map video từ frames
        "-map", "1:a:0?",                        # Map audio từ video gốc (? = optional)
        "-c:a", "aac",                           # Encode audio
        "-b:a", "192k",                          # Bitrate audio
        "-shortest",                             # Dừng khi hết frames hoặc audio (tùy cái nào ngắn hơn)
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