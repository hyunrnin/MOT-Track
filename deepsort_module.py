from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2
import os
import torch
import module_test

@staticmethod
def deepsort1(
        detections,sequence_path, stream,
        max_age=30, n_init=3, nms_max_overlap=1.0, nn_budget=None, max_cosine_distance=0.3, 
        override_track_class=None, embedder="mobilenet", half=True, bgr=True, embedder_gpu=True,
        embedder_model_name=None, embedder_wts=None,polygon=False,today=None, 
        ):
    
    tracker = DeepSort(
        max_age=max_age,
        n_init=n_init,
        nms_max_overlap=nms_max_overlap,
        max_cosine_distance=max_cosine_distance,
        nn_budget=nn_budget,
        override_track_class=override_track_class,
        embedder=embedder,
        half=half,  # FP16 사용으로 GPU 메모리 절약
        bgr=bgr,
        embedder_gpu=embedder_gpu,  # GPU 사용
        embedder_model_name=embedder_model_name,
        embedder_wts=embedder_wts,
        polygon=polygon,
        today=today
        
    )
    for frame_id in sorted(detections.keys()):
        # 이미지 로드
        img_path = os.path.join(sequence_path, f"{frame_id:06d}.jpg")
        frame = cv2.imread(img_path)
        process_each_frame(detections, stream, frame_id, tracker, frame)
        cv2.imshow('DeepSORT Tracking', frame)
        if cv2.waitKey(30) & 0xFF == 27:  # ESC 키로 종료
            break

def process_each_frame(detections, stream, frame_id, tracker, frame):
    
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
            module_test.visualize_results(frame, tracks)   

        # 결과 표시
