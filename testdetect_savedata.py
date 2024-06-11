import cv2
import numpy as np
import datetime
import os
import pandas as pd
from time import sleep
from threading import Thread
from picamera import PiCamera
from picamera.array import PiRGBArray

# Hàm phát hiện ống nghiệm và trích xuất màu RGB
def detect_test_tubes(image):
    if image is None:
        print("Không thể đọc ảnh. Hãy kiểm tra lại.")
        return None, None

    image = image[110:165, :]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY_INV)
    inverted_thresh = cv2.bitwise_not(thresh)
    contours, _ = cv2.findContours(inverted_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    test_tubes = []
    for contour in contours:
        if cv2.contourArea(contour) < 100:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = h / w
        if 2 <= aspect_ratio <= 5:
            test_tubes.append((x, y, w, h))

    colors = []
    for (x, y, w, h) in test_tubes:
        roi = image[y:y+h, x:x+w]
        avg_color = cv2.mean(roi)[:3]
        colors.append(avg_color)

    return len(test_tubes), colors

# Chức năng chụp ảnh từ camera Pi
def capture_image_from_camera():
    # Mở camera
    cap = PiCamera()
    cap.resolution = (640, 480)
    cap.framerate = 32
    frame = PiRGBArray(cap, size=(640, 480))
    cap.capture(frame, format="bgr")

    captured_image = frame.array
        
    cv2.imshow('Camera', captured_image)
    return captured_image

# Tạo thư mục lưu ảnh nếu chưa tồn tại
image_dir = 'Test_Image'
if not os.path.exists(image_dir):
    os.makedirs(image_dir)

# Tạo DataFrame để lưu giá trị RGB theo thời gian
columns = ['Timestamp'] + [f'Tube_{i}_R' for i in range(1, 11)] + [f'Tube_{i}_G' for i in range(1, 11)] + [f'Tube_{i}_B' for i in range(1, 11)]
df = pd.DataFrame(columns=columns)

# Hàm để thực hiện công việc chụp ảnh và lưu dữ liệu RGB
def capture_and_save(interval=20):
    while True:
        image = capture_image_from_camera()
        if image is None:
            continue

        num_test_tubes, test_tube_colors = detect_test_tubes(image)

        if num_test_tubes is not None:
            timestamp = datetime.datetime.now()
            row = [timestamp]
            for color in test_tube_colors:
                row.extend(color)
            # Thêm giá trị None để duy trì độ dài hàng
            row.extend([None] * (30 - len(row)))
            df.loc[len(df)] = row

            # Lưu DataFrame vào file CSV
            df.to_csv('test_tube_rgb_values.csv', index=False)
        
        sleep(interval)

# Tạo và chạy thread để chụp ảnh và lưu dữ liệu RGB
capture_thread = Thread(target=capture_and_save, args=(20,))  # Interval: 20 seconds
capture_thread.start()
