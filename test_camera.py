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
time.sleep(1)

# Chụp một bức ảnh
camera.capture(rawCapture, format="bgr")
image = rawCapture.array

# Hiển thị ảnh sử dụng OpenCV
cv2.imshow("Image", image)
#cv2.waitKey(0)

image_crop = image[200:280, : ]
cv2.imshow("Image_crop", image_crop)
cv2.waitKey(0)
# Lưu ảnh vào file
cv2.imwrite("test_image.jpg", image)

# Giải phóng camera và đóng cửa sổ
camera.close()
#cv2.destroyAllWindows()
