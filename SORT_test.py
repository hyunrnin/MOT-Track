import cv2
import numpy as np
from sort import Sort
import os

def main():
    # 입력 비디오 파일 경로
    sequence_path = "./Venice-2/img1"
    detection_file = "./Venice-2/det/det.txt"
    detections = {}
    with open(detection_file, 'r') as f:
        for line in f:
            frame_id, _, x, y, w, h, conf, _, _, _ = map(float, line.strip().split(','))
            frame_id = int(frame_id)
            if conf > 0.3:  # confidence threshold
                if frame_id not in detections:
                    detections[frame_id] = []
                detections[frame_id].append([x, y, x+w, y+h])

    # SORT 객체 생성
    sort_tracker = Sort()

    # 각 프레임에 대해 처리
    for frame_id in sorted(detections.keys()):
        # 이미지 로드
        img_path = os.path.join(sequence_path, f"{frame_id:06d}.jpg")
        frame = cv2.imread(img_path)
        
        # 트래커 업데이트
        trackers = sort_tracker.update(np.array(detections[frame_id]))
    
        # 결과 시각화
        for track in trackers:
            track_id = track[4]
            
            cv2.rectangle(frame, 
                        (int(track[0]), int(track[1])), 
                        (int(track[2]), int(track[3])), 
                        (0, 255, 0), 2)
            cv2.putText(frame, 
                       str(track_id),
                       (int(track[0]), int(track[1])-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       0.9, 
                       (0, 255, 0), 
                       2)

        # 결과 표시
        cv2.imshow('MOT Tracking', frame)
        if cv2.waitKey(30) & 0xFF == 27:  # ESC 키로 종료
            break
            
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()