import os
import subprocess
import shutil
import json

VIDEO_FOLDER = "data"                   # Thư mục chứa các file video
OUTPUT_ROOT = "frames_raw"     # Thư mục gốc để chứa frames

def get_video_fps_fraction(path):
    """Lấy FPS dạng phân số từ video (vd: '30000/1001')"""
    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_streams",
        path
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    data = json.loads(result.stdout)

    for stream in data.get("streams", []):
        if stream.get("codec_type") == "video":
            return stream.get("r_frame_rate")

    raise ValueError("Không tìm thấy stream FPS.")

def extract_frames_for_video(video_path):
    filename = os.path.basename(video_path)
    video_name = os.path.splitext(filename)[0]  # bỏ đuôi .mp4
    
    output_folder = os.path.join(OUTPUT_ROOT, video_name)

    # Xóa folder cũ nếu có
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder, exist_ok=True)

    # Lấy FPS chuẩn
    fps_fraction = get_video_fps_fraction(video_path)
    print(f"🎥 {filename}: FPS = {fps_fraction}")

    print(f"🚀 Đang tách frame -> {output_folder}")

    cmd = [
        "ffmpeg",
        "-i", video_path,
        "-vf", f"fps={fps_fraction}",
        "-q:v", "2",
        f"{output_folder}/frame_%06d.jpg"
    ]

    subprocess.run(cmd)
    print(f"✅ Hoàn tất {filename}\n")

def process_all_videos():
    # Tạo thư mục root nếu chưa có
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    # Lặp qua tất cả file video trong thư mục
    for file in os.listdir(VIDEO_FOLDER):
        if file.lower().endswith((".mp4", ".mov", ".avi", ".mkv")):
            video_path = os.path.join(VIDEO_FOLDER, file)
            extract_frames_for_video(video_path)

if __name__ == "__main__":
    process_all_videos()
