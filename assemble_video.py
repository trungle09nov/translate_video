import subprocess
import os

# ================= CẤU HÌNH =================
FRAMES_DIR = "workspace/frames_done"     # Ảnh đầu vào (đã dịch)
ORIGINAL_VIDEO = "video/Impower.mp4"     # Video gốc (để lấy tiếng)
OUTPUT_VIDEO = "video/Final_Translated.mp4"
FPS = 30                                 # Phải khớp với Phần 1

def assemble_video():
    if not os.path.exists(FRAMES_DIR):
        print("❌ Không tìm thấy thư mục ảnh đã dịch!")
        return

    print("🎬 Đang ghép video bằng FFmpeg...")

    # Cấu trúc lệnh FFmpeg:
    # -framerate: Tốc độ đọc ảnh
    # -i frames: Đầu vào ảnh
    # -i video: Đầu vào video gốc (lấy audio)
    # -map 0:v: Lấy hình từ input 0 (ảnh)
    # -map 1:a: Lấy tiếng từ input 1 (video gốc)
    # -c:a copy: Copy âm thanh gốc không cần nén lại (giữ nguyên chất lượng)
    # -pix_fmt yuv420p: Để tương thích mọi trình phát
    
    cmd = [
        'ffmpeg', '-y',                  # Overwrite nếu file tồn tại
        '-framerate', str(FPS),
        '-i', f'{FRAMES_DIR}/frame_%06d.jpg',
        '-i', ORIGINAL_VIDEO,
        '-c:v', 'libx264',
        '-preset', 'medium',
        '-crf', '23',                    # Chất lượng nén (thấp hơn là nét hơn)
        '-pix_fmt', 'yuv420p',
        '-map', '0:v',
        '-map', '1:a',
        '-c:a', 'copy',
        '-shortest',                     # Kết thúc khi luồng ngắn nhất (ảnh) hết
        OUTPUT_VIDEO
    ]
    
    # Chạy lệnh (ẩn bớt log rác)
    subprocess.run(cmd)
    
    if os.path.exists(OUTPUT_VIDEO):
        print(f"\n🎉 XONG! Video của bạn tại: {OUTPUT_VIDEO}")
    else:
        print("\n❌ Có lỗi xảy ra, không thấy file output.")

if __name__ == "__main__":
    assemble_video()