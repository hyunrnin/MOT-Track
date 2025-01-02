import cv2
import numpy as np
from sort.tracker import SortTracker

def main():
    # 입력 비디오 파일 경로
    video_path = 'C:/Users/user/Downloads/MOT20-01-raw.webm'
    cap = cv2.VideoCapture(video_path)

    # SORT 객체 생성
    sort_tracker = SortTracker()

    # 비디오 출력 설정
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter('./output_tracking_result.avi', fourcc, 30, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 임의의 검출값 생성 (실제로는 객체 검출 알고리즘 사용 필요)
        # 예: xmin, ymin, xmax, ymax, confidence
        detections = np.array([[50, 100, 150, 200, 0.9, 1], [200, 250, 300, 350, 0.8, 1]])  # 예시 값
        if detections.size == 0:
            detections = np.empty((0, 5))  # 비어 있는 감지 배열

        # SORT 알고리즘을 통해 객체 추적
        trackers = sort_tracker.update(detections, frame)
        print(trackers)
        # 추적 결과 화면에 표시
        for tracker in trackers:
            x1, y1, x2, y2, track_id = tracker
            cv2.putText(frame, f'ID: {int(track_id)}', (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

        # 결과 비디오에 저장
        out.write(frame)

        # 화면에 출력
        cv2.imshow('Tracking Result', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 자원 해제
    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
