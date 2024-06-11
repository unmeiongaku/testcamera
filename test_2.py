import cv2
from picamera import PiCamera
from picamera.array import PiRGBArray
import time

# Khởi tạo Pi Camera
camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(640, 480))

# Cho camera thời gian để khởi động
time.sleep(0.1)

# Chụp một bức ảnh
camera.capture(rawCapture, format="bgr")
image = rawCapture.array

# Lưu ảnh vào file
cv2.imwrite("captured_image.jpg", image)

# Giải phóng camera
camera.close()
