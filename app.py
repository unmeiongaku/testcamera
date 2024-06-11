import os
import datetime
import pandas as pd
from time import sleep
from threading import Thread
import cv2
from flask import Flask, render_template, request, redirect, url_for, jsonify
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
from picamera import PiCamera
from picamera.array import PiRGBArray

# Hàm phát hiện ống nghiệm và trích xuất màu RGB
def detect_test_tubes(image):
    if image is None:
        print("Không thể đọc ảnh. Hãy kiểm tra lại.")
        return None, None

    image = image[200:280, 100:500]
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
    cap = PiCamera()
    cap.resolution = (640, 480)
    cap.framerate = 32
    raw_capture = PiRGBArray(cap, size=(640, 480))
    cap.capture(raw_capture, format='bgr')
    captured_image = raw_capture.array

    cap.close()
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

            # Lưu ảnh vào thư mục
            image_path = os.path.join(image_dir, f'test_tube_{timestamp.strftime("%Y%m%d_%H%M%S")}.jpg')
            cv2.imwrite(image_path, image)

        sleep(interval)

# Hàm chạy thread chụp ảnh
def start_capture_thread(interval=20):
    capture_thread = Thread(target=capture_and_save, args=(interval,))
    capture_thread.daemon = True
    capture_thread.start()

app = Flask(__name__)
dash_app = dash.Dash(__name__, server=app, url_base_pathname='/graphs/')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        interval = int(request.form.get('interval', 20))
        start_capture_thread(interval)
        return redirect(url_for('index'))

    return render_template('index.html')

@dash_app.callback(
    Output('graphs', 'children'),
    [Input('interval-component', 'n_intervals')]
)
def update_graphs(n_intervals):
    df = pd.read_csv('test_tube_rgb_values.csv')
    fig = make_subplots(rows=5, cols=2, subplot_titles=[f'Tube {i}' for i in range(1, 11)])
    for i in range(1, 11):
        if f'Tube_{i}_R' in df.columns:
            fig.add_trace(go.Scatter(x=df['Timestamp'], y=df[f'Tube_{i}_R'], mode='lines', name=f'Tube {i} R'), row=(i-1)//2+1, col=(i-1)%2+1)
            fig.add_trace(go.Scatter(x=df['Timestamp'], y=df[f'Tube_{i}_G'], mode='lines', name=f'Tube {i} G'), row=(i-1)//2+1, col=(i-1)%2+1)
            fig.add_trace(go.Scatter(x=df['Timestamp'], y=df[f'Tube_{i}_B'], mode='lines', name=f'Tube {i} B'), row=(i-1)//2+1, col=(i-1)%2+1)

    fig.update_layout(height=1000, showlegend=False)
    return [dcc.Graph(figure=fig)]

dash_app.layout = html.Div([
    dcc.Interval(id='interval-component', interval=20*1000, n_intervals=0),
    html.Div(id='graphs')
])

if __name__ == '__main__':
    app.run(debug=True)

