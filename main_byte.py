import os
import cv2
import numpy as np
from for_Byte.byte_tracker import BYTETracker
import util.module_result as module_result
#gpt version
def process_each_frame(detections, frame_id, byte_tracker, frame):
    if frame_id in detections:
        detection_list = np.array(detections[frame_id])
    else:
        detection_list = np.array([])

    tracks, bbox_list = byte_tracker.update(detection_list, frame)

    bbox_list = bbox_list if bbox_list is not None else []

    visualize_results1(frame, tracks, bbox_list)


"""
def process_each_frame(detections, frame_id, byte_tracker, frame):
        if frame_id in detections:
            detection_list = []
            for det in detections[frame_id]:
                x1, y1, x2, y2, conf = det
                    # DeepSORT expects: [left,top,w,h,confidence]
                detection_list.append([x1, y1, x2-x1, y2-y1, conf])
                    # 트래커 업데이트
                
            detection_list = np.array(detection_list)    
            tracks = byte_tracker.update(detection_list, frame)
        else:
                # CPU 처리
            detection_list = []
            for det in detections[frame_id]:
                x1, y1, x2, y2, conf = det
                detection_list.append(([x1, y1, x2-x1, y2-y1, conf]))
            tracks, bbox_list = byte_tracker.update(detection_list, frame=frame)
            
            # Score to bbox_score
           # Draw results
        visualize_results1(frame, tracks, bbox_list)   
"""
@staticmethod
def visualize_results1(frame, tracks, bbox_list):
    for track, bbox in zip(tracks, bbox_list):
        track_id = track.track_id
        color = (0, 255, 0)

        cv2.rectangle(frame, 
                      (int(bbox[0]), int(bbox[1])), 
                      (int(bbox[2]), int(bbox[3])), 
                      color, 2)

        cv2.putText(frame, f"ID: {track_id}", 
                    (int(bbox[0]), int(bbox[1]) - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, 
                    (255, 255, 255), 2)

    return frame

sequence_path = "./Venice-2/img1"
detection_file = "./Venice-2/det/det.txt"
detections = []

detections = module_result.save_detection(detection_file, sort_ver='byte')

image_files = sorted(os.listdir(sequence_path))

byte_tracker = BYTETracker(args={'track_thresh': 0.5, 'match_thresh': 0.8, 'track_buffer': 30, 'mot20': None})

for frame_id in sorted(detections.keys()):
    img_path = os.path.join(sequence_path, f"{frame_id:06d}.jpg")
    frame = cv2.imread(img_path)
    process_each_frame(detections, frame_id, byte_tracker, frame)
    cv2.imshow('ByteTracking', frame)
    if cv2.waitKey(30) & 0xFF == 27:
        break


"""

# === 프레임 단위로 객체 추적 ===
for img_file, det_file in zip(image_files, detections):
    img_path = os.path.join(sequence_path, img_file)
    det_path = os.path.join(detections, det_file)

    # 이미지 로드
    frame = cv2.imread(img_path)
    if frame is None:
        print(f"이미지 로드 실패: {img_path}")
        continue

    # Detection 파일 로드 (각 줄: x1, y1, x2, y2, conf, class_id)
    detections = []
    with open(det_path, "r") as f:
        for line in f.readlines():
            data = list(map(float, line.strip().split()))
            x1, y1, x2, y2, conf, class_id = data
            detections.append([x1, y1, x2, y2, conf])

    # numpy 배열로 변환
    detections = np.array(detections)

    # ByteTrack을 통해 객체 추적
    if len(detections) > 0:
        tracked_objects = byte_tracker.update(detections)

        # 시각화
        module_result.visualize_results(frame, tracked_objects)

    # 화면 출력
    cv2.imshow("ByteTrack Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
"""