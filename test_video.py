from picamera import PiCamera
import time
import subprocess  # Thêm thư viện subprocess để gọi lệnh terminal
#import cv2

# Khởi tạo Pi Camera
camera = PiCamera()

# Bật tự động cân bằng trắng và tự động điều chỉnh độ sáng
camera.awb_mode = 'auto'         # Tự động cân bằng trắng
camera.exposure_mode = 'auto'    # Tự động điều chỉnh độ sáng

# Bật tự động lấy nét
camera.start_preview()  # Bắt đầu hiển thị hình ảnh trước khi lấy nét
camera.resolution = (640, 480)  # Độ phân giải cho việc lấy nét
#cv2.waitKey(0)
#cv2.destroyAllWindows()

time.sleep(10)  # Chờ 2 giây để camera ổn định
camera.stop_preview()  # Dừng hiển thị hình ảnh

# Đặt cấu hình cho video
camera.resolution = (640, 480)  # Độ phân giải
camera.framerate = 24            # Tốc độ khung hình

# Bắt đầu ghi video
video_path = 'video.h264'  # Đường dẫn của tệp video
camera.start_recording(video_path)  # Ghi video vào tệp video.h264
print("Bắt đầu ghi video...")
camera.wait_recording(10)  # Ghi video trong 10 giây

# Kết thúc ghi video
camera.stop_recording()
print("Kết thúc ghi video")

# Đổi định dạng video từ h264 sang mp4
mp4_video_path = 'video.mp4'  # Đường dẫn mới hoặc tên mới của tệp video MP4
subprocess.call(['ffmpeg', '-i', video_path, mp4_video_path])  # Gọi lệnh ffmpeg

# Xóa tệp video h264 gốc nếu cần
# import os
# os.remove(video_path)

# Giải phóng tài nguyên camera
camera.close()
