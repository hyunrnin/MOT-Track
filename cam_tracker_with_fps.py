import cv2
import numpy as np
import time
from ultralytics import YOLO
from for_Byte.byte_tracker import BYTETracker

model = YOLO('yolov9c.pt')

byte_tracker = BYTETracker(args={'track_thresh': 0.5, 'match_thresh': 0.8, 'track_buffer': 30, 'mot20': False})

cap = cv2.VideoCapture(0)

prev_time = time.time()
fps = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("카메라에서 프레임을 읽을 수 없습니다.")
        break

    start_time = time.time()

    results = model(frame)

    detections = []
    for box in results[0].boxes:
        class_id = int(box.cls[0])
        if class_id == 0:
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            detections.append([x1, y1, x2, y2, confidence])

    detection_list = np.array(detections) if detections else np.empty((0, 5))

    tracks, bbox_list = byte_tracker.update(detection_list, frame)

    for track, bbox in zip(tracks, bbox_list):
        track_id = track.track_id
        color = (0, 255, 0)

        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)

        cv2.putText(frame, f"ID: {track_id}", (int(bbox[0]), int(bbox[1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

    end_time = time.time()
    frame_time = end_time - start_time
    fps = 0.9 * fps + 0.1 * (1 / frame_time)

    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    cv2.imshow('YOLO Real-Time Tracking', frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
