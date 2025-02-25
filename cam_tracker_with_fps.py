import cv2
import numpy as np
import time
from ultralytics import YOLO
from for_Byte.byte_tracker import BYTETracker  # ByteTrack 모듈 (설치 필요)

# YOLOv9 모델 로드
model = YOLO('yolov9c.pt')

# BYTETracker 초기화 (올바른 파라미터 설정 방식 적용)
byte_tracker = BYTETracker(args={'track_thresh': 0.5, 'match_thresh': 0.8, 'track_buffer': 30, 'mot20': False})

# 카메라 열기 (0: 기본 웹캠)
cap = cv2.VideoCapture(0)

# FPS 측정을 위한 변수
prev_time = time.time()
fps = 0  # 초기 FPS 값 설정

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("카메라에서 프레임을 읽을 수 없습니다.")
        break

    # 현재 시간 기록 (프레임 처리 시작 시간)
    start_time = time.time()

    # YOLO 객체 탐지 수행
    results = model(frame)

    # 탐지된 객체 정보 추출
    detections = []
    for box in results[0].boxes:
        class_id = int(box.cls[0])  # 클래스 ID
        if class_id == 0:  # 특정 클래스(0)만 추적
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())  # 좌상단(x1, y1) & 우하단(x2, y2)

            # YOLO 결과를 BYTETracker가 이해할 수 있는 형식으로 변환 (x1, y1, x2, y2, confidence)
            detections.append([x1, y1, x2, y2, confidence])

    # numpy 배열로 변환 (예외 처리 추가)
    detection_list = np.array(detections) if detections else np.empty((0, 5))

    # BYTETracker 업데이트 (반환값 수정)
    tracks, bbox_list = byte_tracker.update(detection_list, frame)

    # 실시간 Tracking 결과 시각화
    for track, bbox in zip(tracks, bbox_list):
        track_id = track.track_id
        color = (0, 255, 0)  # 초록색

        # 바운딩 박스 그리기
        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)

        # 트래킹 ID 표시
        cv2.putText(frame, f"ID: {track_id}", (int(bbox[0]), int(bbox[1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

    # FPS 계산 (부드럽게 보정)
    end_time = time.time()
    frame_time = end_time - start_time
    fps = 0.9 * fps + 0.1 * (1 / frame_time)  # 지수 평균 적용하여 부드럽게 표시

    # FPS 화면에 표시
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # 결과 화면에 출력
    cv2.imshow('YOLO Real-Time Tracking', frame)

    # ESC 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == 27:
        break

# 자원 해제
cap.release()
cv2.destroyAllWindows()
