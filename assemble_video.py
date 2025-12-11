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

    # Lấy FPS gốc
    fps_fraction = get_video_fps_fraction(video_source)
    print(f"🎬 Ghép video: {subdir}  |  FPS gốc: {fps_fraction}")

    # ✅ Ghép đúng: frames (1fps) → video (fps gốc) + audio gốc
    cmd = [
        "ffmpeg", "-y",
        "-framerate", "1",                       # ✅ Đọc frames với 1 fps (vì tách với fps=1)
        "-i", f"{frames_dir}/frame_%06d.jpg",   # Input: frames đã dịch
        "-i", video_source,                      # Input: video gốc (lấy audio)
        "-c:v", "libx264",                       # Codec video
        "-preset", "medium",                     # Preset encode
        "-crf", "23",                            # Chất lượng
        "-pix_fmt", "yuv420p",                   # Format tương thích
        "-r", fps_fraction,                      # ✅ Output FPS = FPS gốc
        "-map", "0:v:0",                         # Map video từ frames
        "-map", "1:a:0?",                        # Map audio từ video gốc (? = optional nếu không có audio)
        "-c:a", "aac",                           # Encode audio (hoặc 'copy' nếu muốn giữ nguyên)
        "-b:a", "192k",                          # Bitrate audio
        "-shortest",                             # Video dừng khi hết frames hoặc audio
        output_video
    ]

    print(f"   🔧 Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"   ❌ Lỗi ffmpeg:\n{result.stderr}")
    elif os.path.exists(output_video):
        size_mb = os.path.getsize(output_video) / (1024*1024)
        print(f"   ✔ Done: {output_video} ({size_mb:.2f} MB)")
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

    print(f"🔍 Tìm thấy {len(subdirs)} video cần ghép\n")

    for i, subdir in enumerate(subdirs, 1):
        print(f"[{i}/{len(subdirs)}] ", end="")
        assemble_video_for_folder(subdir)

    print("\n🎉 Hoàn tất ghép tất cả video!")


if __name__ == "__main__":
    main()