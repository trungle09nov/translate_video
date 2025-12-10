import subprocess
import os

FRAMES_ROOT = "./frames_done"      # Nhiều thư mục con
VIDEO_ROOT = "./data"              # Chứa video gốc
OUTPUT_ROOT = "./video_output"     # Video sau khi ghép

os.makedirs(OUTPUT_ROOT, exist_ok=True)


def get_fps(video_path):
    """Lấy FPS gốc của video bằng ffprobe"""
    try:
        cmd = [
            "ffprobe", "-v", "0",
            "-select_streams", "v:0",
            "-show_entries", "stream=r_frame_rate",
            "-of", "default=noprint_wrappers=1:nokey=1",
            video_path
        ]
        raw = subprocess.check_output(cmd).decode().strip()

        if "/" in raw:
            num, den = raw.split("/")
            return float(num) / float(den)
        return float(raw)
    except Exception as e:
        print(f"⚠️  Không lấy được FPS từ {video_path}, dùng 30 FPS mặc định")
        return 30.0


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

    # Lấy FPS gốc cho từng video
    fps = get_fps(video_source)
    print(f"🎬 Ghép video: {subdir}  |  FPS gốc: {fps}")

    # Ghép lại: dùng FPS gốc → khớp 100% với video ban đầu
    cmd = [
        "ffmpeg", "-y",
        "-framerate", "1", # str(fps),                  # FPS từ video gốc
        "-i", f"{frames_dir}/frame_%06d.jpg",    # Frames đã xử lý
        "-i", video_source,                      # Lấy audio từ video gốc
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-map", "0:v",                           # Video = frames
        "-map", "1:a",                           # Audio = audio gốc
        "-c:a", "copy",                          # Copy audio gốc
        "-shortest",                             # Video = độ dài frames
        output_video
    ]

    subprocess.run(cmd)

    if os.path.exists(output_video):
        print(f"   ✔ Done: {output_video}")
    else:
        print(f"   ❌ Lỗi khi tạo video {subdir}")


def main():
    subdirs = [
        d for d in os.listdir(FRAMES_ROOT)
        if os.path.isdir(os.path.join(FRAMES_ROOT, d))
    ]

    print(f"🔍 Tìm thấy {len(subdirs)} video cần ghép")

    for subdir in subdirs:
        assemble_video_for_folder(subdir)

    print("\n🎉 Hoàn tất ghép tất cả video!")


if __name__ == "__main__":
    main()
