import cv2
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
import os
import torch

def main():
    # CUDA 사용 가능 여부 확인
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    tracker = DeepSort(
        max_age=30,
        n_init=3,
        nms_max_overlap=1.0,
        max_cosine_distance=0.3,
        nn_budget=None,
        override_track_class=None,
        embedder="mobilenet",
        half=True,  # FP16 사용으로 GPU 메모리 절약
        bgr=True,
        embedder_gpu=True,  # GPU 사용
        embedder_model_name=None,
        embedder_wts=None,
        polygon=False,
        today=None
    )

    sequence_path = "./Venice-2/img1"  # 입력 비디오 파일 경로
    detection_file = "./Venice-2/det/det.txt"  # detection 결과 파일
    detections = {}
    try:
        with open(detection_file, 'r') as f:
            for line in f:
                frame_id, _, x, y, w, h, conf, _, _, _ = map(float, line.strip().split(','))
                frame_id = int(frame_id)
                if conf > 0.3:  # confidence threshold
                    if frame_id not in detections:
                        detections[frame_id] = []
                    detections[frame_id].append([x, y, x+w, y+h, conf])
    except IOError as e:
        print(f"detection 파일 읽기 오류: {"e"}")
        return 
    
    if torch.cuda.is_available():
        stream = torch.cuda.Stream()
        torch.cuda.synchronize()

    # Process each frame
    for frame_id in sorted(detections.keys()):
        # 이미지 로드
        img_path = os.path.join(sequence_path, f"{frame_id:06d}.jpg")
        frame = cv2.imread(img_path)
        
        if frame_id in detections:
            # CUDA 사용 시 비동기 처리
            if torch.cuda.is_available():
                with torch.cuda.stream(stream):
                    # DeepSORT 형식에 맞게 detection 데이터 변환
                    detection_list = []
                    for det in detections[frame_id]:
                        x1, y1, x2, y2, conf = det
                        # DeepSORT expects: [left,top,w,h,confidence]
                        detection_list.append(([x1, y1, x2-x1, y2-y1], conf, None))

                    # 트래커 업데이트
                    tracks = tracker.update_tracks(detection_list, frame=frame)
            else:
                # CPU 처리
                detection_list = []
                for det in detections[frame_id]:
                    x1, y1, x2, y2, conf = det
                    detection_list.append(([x1, y1, x2-x1, y2-y1], conf, None))
                tracks = tracker.update_tracks(detection_list, frame=frame)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            # Score to bbox_score
            
            
            # Draw results
            for track in tracks:
                if not track.is_confirmed():
                    continue
                
                track_id = track.track_id
                ltrb = track.to_ltrb()
                
                # draw bounding box
                cv2.rectangle(frame, 
                            (int(ltrb[0]), int(ltrb[1])), 
                            (int(ltrb[2]), int(ltrb[3])), 
                            (0, 255, 0), 2)
                
                # put ID in the bounding box
                cv2.putText(frame, 
                           str(track_id),
                           (int(ltrb[0]), int(ltrb[1])-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 
                           0.9, 
                           (0, 255, 0), 
                           2)

        # 결과 표시
        cv2.imshow('DeepSORT Tracking', frame)
        if cv2.waitKey(30) & 0xFF == 27:  # ESC 키로 종료
            break
            
    cv2.destroyAllWindows()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()