import cv2

@staticmethod
def save_detection(detection_file, deepsort_ver = False):
    detections = {}
    # detection_file에서 frame_id 별로 detection 결과를 읽어 detections에 저장
    with open(detection_file, 'r') as f:
        for line in f:
            frame_id, _, x, y, w, h, conf, _, _, _ = map(float, line.strip().split(','))
            frame_id = int(frame_id)
            if conf > 0.3:  # confidence threshold
                if frame_id not in detections:
                    detections[frame_id] = []
                if deepsort_ver: #deepsort의 경우 deepsort_ver=True가 되어 해당 코드 실행 
                    detections[frame_id].append([x, y, x+w, y+h, conf])
                else:
                    detections[frame_id].append([x, y, x+w, y+h])
    return detections

@staticmethod
def visualize_results(frame, tracks):
    for track in tracks:
        if hasattr(track, 'is_confirmed') and not track.is_confirmed():
            continue

        track_id = track.track_id if hasattr(track, 'track_id') else track[4]
        ltrb = track.to_ltrb() if hasattr(track, 'to_ltrb') else track[:4]

        # Draw bounding box
        cv2.rectangle(frame, 
                      (int(ltrb[0]), int(ltrb[1])), 
                      (int(ltrb[2]), int(ltrb[3])), 
                      (0, 255, 0), 2)

        # Put track ID in the bounding box
        cv2.putText(frame, 
                   f"ID: {track_id}",
                   (int(ltrb[0]), int(ltrb[1]) - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 
                   0.75, 
                   (255, 255, 255), 
                   2)
    return frame