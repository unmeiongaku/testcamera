
import cv2
import numpy as np
from picamera import PiCamera

from picamera.array import PiRGBArray
def capture_image_from_camera():
	# Mở camera
	cap = PiCamera()
	cap.resolution = (640,480)
	cap.framerate = 32
	frame = PiRGBArray(cap,size=(640,480))
	cap.capture(frame, format="bgr")

	captured_image = frame.array
        # Cân bằng sáng
#        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
#        l_channel, a_channel, b_channel = cv2.split(frame)
#        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
#        l_channel = clahe.apply(l_channel)
#        frame = cv2.merge((l_channel, a_channel, b_channel))
#        frame = cv2.cvtColor(frame, cv2.COLOR_LAB2BGR) 	
#         Hiển thị frame
	cv2.imshow('Camera', captured_image)

#        # Nhấn phím 'c' để chụp ảnh
#        if cv2.waitKey(1) & 0xFF == ord('c'):
#            captured_image = frame
#            break

    # Giải phóng camera và đóng cửa sổ
	#cap.release()
	#cv2.destroyAllWindows()
	return captured_image

def detect_test_tubes(image):
    if image is None:
        print(f"Không thể đọc ảnh từ camera.")
        return None, None

    # Hiển thị ảnh gốc
    cv2.imshow('Original Image', image)
    #cv2.waitKey(0)

    # Cắt ảnh từ y=110 đến y=165
    image_cropped = image

    # Hiển thị ảnh đã cắt
    cv2.imshow('Cropped Image', image_cropped)
    #cv2.waitKey(0)

    # Chuyển đổi ảnh sang thang độ xám
    gray = cv2.cvtColor(image_cropped, cv2.COLOR_BGR2GRAY)

    # Hiển thị ảnh thang độ xám
    cv2.imshow('Gray Image', gray)
    #cv2.waitKey(0)

    # Áp dụng Gaussian Blur để làm mờ ảnh và giảm nhiễu
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Hiển thị ảnh đã làm mờ
    cv2.imshow('Blurred Image', blurred)
    #cv2.waitKey(0)

    # Sử dụng thresholding để chuyển đổi ảnh thành nhị phân
    _, thresh = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY_INV)

    # Hiển thị ảnh nhị phân
    cv2.imshow('Thresholded Image', thresh)
    #cv2.waitKey(0)

    # Đảo ngược giá trị nhị phân trong ảnh nhị phân
    inverted_thresh = cv2.bitwise_not(thresh)

    # Hiển thị ảnh nhị phân đảo ngược
    cv2.imshow('Inverted Thresholded Image', inverted_thresh)
    #cv2.waitKey(0)

    # Tìm các contours trong ảnh nhị phân đảo ngược
    contours, _ = cv2.findContours(inverted_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Lọc các contours để lấy các ống nghiệm
    test_tubes = []
    for contour in contours:
        # Bỏ qua các contour nhỏ hoặc không phải hình chữ nhật
        if cv2.contourArea(contour) < 100:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = h / w
        if 2 <= aspect_ratio <= 5:  # Giả sử ống nghiệm có tỉ lệ khung hình nằm trong khoảng này
            test_tubes.append((x, y, w, h))

    # Trích xuất màu trung bình của mỗi ống nghiệm
    colors = []
    for (x, y, w, h) in test_tubes:
        roi = image_cropped[y:y+h, x:x+w]
        avg_color = cv2.mean(roi)[:3]  # Trích xuất giá trị màu trung bình
        colors.append(avg_color)
        # Vẽ hình chữ nhật quanh ống nghiệm
        cv2.rectangle(image, (x, y+110), (x+w, y+h+110), (0, 255, 0), 2)

    # Hiển thị ảnh với các ống nghiệm được phát hiện
    cv2.imshow('Detected Test Tubes', image)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    return len(test_tubes), colors

# Chụp ảnh từ camera
captured_image = capture_image_from_camera()
crop_image = captured_image[200:280, 100:500]
# Phát hiện ống nghiệm trên ảnh chụp từ camera
num_test_tubes, test_tube_colors = detect_test_tubes(crop_image)

if num_test_tubes is not None:
    print(f'Tìm thấy {num_test_tubes} ống nghiệm.')
    for i, color in enumerate(test_tube_colors):
        print(f'Ống nghiệm {i + 1}: RGB{color}')
else:
    print('Đã xảy ra lỗi trong quá trình xử lý ảnh.')

cv2.waitKey(0)
cv2.destroyAllWindows()
