import os
import subprocess
import shutil

# ================= CẤU HÌNH =================
VIDEO_INPUT = "video/Impower.mp4"       # File video gốc
OUTPUT_FOLDER = "workspace/frames_raw"  # Nơi chứa ảnh tách ra
FPS_EXTRACT = 30                        # Số khung hình/giây (Nên khớp với video gốc)

def extract_frames():
    # 1. Dọn dẹp folder cũ
    if os.path.exists(OUTPUT_FOLDER):
        shutil.rmtree(OUTPUT_FOLDER)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    print(f"🚀 Đang tách frame từ {VIDEO_INPUT} bằng FFmpeg...")

    # 2. Lệnh FFmpeg
    # %06d.jpg nghĩa là đặt tên file: 000001.jpg, 000002.jpg...
    cmd = [
        'ffmpeg',
        '-i', VIDEO_INPUT,
        '-vf', f'fps={FPS_EXTRACT}', 
        '-q:v', '2',  # Chất lượng ảnh (1-31, 2 là rất tốt)
        f'{OUTPUT_FOLDER}/frame_%06d.jpg'
    ]
    
    subprocess.run(cmd)
    print(f"✅ Đã tách xong ảnh vào thư mục: {OUTPUT_FOLDER}")

if __name__ == "__main__":
    extract_frames()