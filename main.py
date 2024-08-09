import os
import cv2
import pandas as pd
from ultralytics import YOLO
from Records import Entry 
import datetime
import requests
import json
from flask import Flask, Response, jsonify

app = Flask(__name__)
cap = cv2.VideoCapture('final.mp4')

model = YOLO('yolov8s.pt')
CLASS_LIST = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'van']
tracker = Entry()
PURPLE_LINE_Y = 270
PINK_LINE_Y = 300
LINE_OFFSET = 6
down_objects = {}
up_objects = {}
entry_objects = []
exit_objects = []
entry_count = 0
exit_count = 0
available_slots = 0
total_lots = 0

# Retrieve carpark data
def retrieve_carpark_data():
    global available_slots, total_lots
    API_URL = "https://api.data.gov.sg/v1/transport/carpark-availability"
    date_time = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    response = requests.get(API_URL, params={"date_time": date_time})
    if response.status_code == 200:
        records = response.json()
        for carpark_data in records.get('items', [])[0].get('carpark_data', []):
            if carpark_data.get('carpark_number') == 'JB2':
                for info in carpark_data.get('carpark_info', []):
                    if info.get('lot_type') == 'C':
                        available_slots = int(info.get('lots_available'))
                        total_lots = int(info.get('total_lots'))
                        break

retrieve_carpark_data()

def publish_availability(total_slots, available_slots, incoming_cars, outgoing_cars):
    API_ENDPOINT = 'https://lotlocate1.onrender.com/add_record'
    data = {
        "total_slots": total_slots,
        "total_available": available_slots,
        "incoming_car": incoming_cars,
        "outgoing_car": outgoing_cars
    }
    headers = {'Content-Type': 'application/json'}
    requests.post(API_ENDPOINT, data=json.dumps(data), headers=headers)

def process_frame(frame):
    global entry_objects, exit_objects, entry_count, exit_count, available_slots

    frame = cv2.resize(frame, (400, 400))
    results = model.predict(frame)
    bboxes = extract_bounding_boxes(results[0].boxes.data)
    bbox_ids = tracker.update(bboxes)

    for bbox in bbox_ids:
        x3, y3, x4, y4, obj_id = bbox
        cx = (x3 + x4) // 2
        cy = (y3 + y4) // 2

        update_counting_lines(obj_id, cx, cy)

    draw_frame_annotations(frame)
    return frame

def extract_bounding_boxes(data):
    bbox_data = data.detach().cpu().numpy()
    px = pd.DataFrame(bbox_data).astype("float")
    bboxes = []

    for _, row in px.iterrows():
        x1, y1, x2, y2, _, class_id = map(int, row)
        if CLASS_LIST[class_id] in ['car', 'motorcycle', 'truck', 'van']:
            bboxes.append([x1, y1, x2, y2])
    return bboxes

def update_counting_lines(obj_id, cx, cy):
    global entry_count, exit_count, available_slots

    if PURPLE_LINE_Y < (cy + LINE_OFFSET) and PURPLE_LINE_Y > (cy - LINE_OFFSET):
        down_objects[obj_id] = cy
    if obj_id in down_objects:
        if PINK_LINE_Y < (cy + LINE_OFFSET) and PINK_LINE_Y > (cy - LINE_OFFSET):
            if obj_id not in entry_objects:
                entry_objects.append(obj_id)
                entry_count += 1
                available_slots -= 1
                publish_availability(total_lots, available_slots, entry_count, exit_count)

    if PINK_LINE_Y < (cy + LINE_OFFSET) and PINK_LINE_Y > (cy - LINE_OFFSET):
        up_objects[obj_id] = cy
    if obj_id in up_objects:
        if PURPLE_LINE_Y < (cy + LINE_OFFSET) and PURPLE_LINE_Y > (cy - LINE_OFFSET):
            if obj_id not in exit_objects:
                exit_objects.append(obj_id)
                exit_count += 1
                available_slots += 1
                publish_availability(total_lots, available_slots, entry_count, exit_count)

def draw_frame_annotations(frame):
    text_color = (0, 0, 0)
    light_purple_color = (255, 182, 193)
    light_pink_color = (221, 160, 221)

    cv2.line(frame, (20, PURPLE_LINE_Y), (400, PURPLE_LINE_Y), light_purple_color, 2)
    cv2.line(frame, (20, PINK_LINE_Y), (400, PINK_LINE_Y), light_pink_color, 2)
    cv2.putText(frame, 'Purple Line', (20, PURPLE_LINE_Y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)
    cv2.putText(frame, 'Pink Line', (20, PINK_LINE_Y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)
    cv2.putText(frame, 'Entry - ' + str(len(entry_objects)), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)
    cv2.putText(frame, 'Exit - ' + str(len(exit_objects)), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)

def generate_frames():
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = process_frame(frame)
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def home():
    return 'Welcome to LotLocate'

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

cap.release()
