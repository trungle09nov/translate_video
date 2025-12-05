import subprocess
import os

FRAMES_ROOT = "./frames_done"   # Nhiều thư mục con
VIDEO_ROOT = "./data"                   # Nơi lưu video gốc
OUTPUT_ROOT = "./video_output"           # Video xuất ra
FPS = 30

os.makedirs(OUTPUT_ROOT, exist_ok=True)

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

    print(f"🎬 Ghép video: {subdir}")

    cmd = [
        'ffmpeg', '-y',
        '-framerate', str(FPS),
        '-i', f'{frames_dir}/frame_%06d.jpg',
        '-i', video_source,
        '-c:v', 'libx264',
        '-preset', 'medium',
        '-crf', '23',
        '-pix_fmt', 'yuv420p',
        '-map', '0:v',
        '-map', '1:a',
        '-c:a', 'copy',
        '-shortest',
        output_video
    ]

    subprocess.run(cmd)

    if os.path.exists(output_video):
        print(f"   ✔ Done: {output_video}")
    else:
        print(f"   ❌ Lỗi khi tạo video {subdir}")

def main():
    subdirs = [d for d in os.listdir(FRAMES_ROOT) if os.path.isdir(os.path.join(FRAMES_ROOT, d))]
    
    print(f"🔍 Tìm thấy {len(subdirs)} video cần ghép")

    for subdir in subdirs:
        assemble_video_for_folder(subdir)

    print("\n🎉 Hoàn tất ghép tất cả video!")

if __name__ == "__main__":
    main()
