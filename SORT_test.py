import os
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter

# Kalman Tracker 클래스 정의
class KalmanTracker:
    def __init__(self):
        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        self.kf.F = np.array([[1, 0, 1, 0], 
                               [0, 1, 0, 1], 
                               [0, 0, 1, 0], 
                               [0, 0, 0, 1]])
        self.kf.H = np.array([[1, 0, 0, 0], 
                               [0, 1, 0, 0]])
        self.kf.P *= 1000.
        self.kf.R = np.eye(2) * 0.1
        self.kf.Q = np.eye(4) * 0.1
        self.track_id = None
        self.age = 0

    def update(self, measurement):
        self.kf.update(measurement)

    def predict(self):
        self.kf.predict()
        return self.kf.x[:2]  # Only return x, y position

# SORT 알고리즘 클래스 정의
class Sort:
    def __init__(self, max_age=5, min_hits=3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.trackers = []
        self.track_id = 1

    def update(self, detections):
        if len(detections) == 0:
            for tracker in self.trackers:
                tracker.age += 1
            return []

        tracker_positions = np.array([tracker.predict() for tracker in self.trackers])
        cost_matrix = np.linalg.norm(tracker_positions[:, np.newaxis] - detections, axis=2)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        matches = []
        unmatched_trackers = []
        unmatched_detections = []

        for t, d in zip(row_ind, col_ind):
            if cost_matrix[t, d] < 0.5:
                matches.append((t, d))
            else:
                unmatched_detections.append(d)

        for t in range(len(self.trackers)):
            if t not in row_ind:
                unmatched_trackers.append(t)

        for t, d in matches:
            self.trackers[t].update(detections[d])

        for t in unmatched_trackers:
            self.trackers[t].age += 1

        new_trackers = []
        for d in unmatched_detections:
            tracker = KalmanTracker()
            tracker.update(detections[d])
            tracker.track_id = self.track_id
            self.track_id += 1
            new_trackers.append(tracker)

        self.trackers = [t for t in self.trackers if t.age < self.max_age]
        self.trackers.extend(new_trackers)

        return [(t.kf.x[0], t.kf.x[1], t.track_id) for t in self.trackers if t.age > self.min_hits]

# MOT15 데이터셋의 감지 정보를 읽는 함수 (임시 예시)
def load_mot15_annotations(annotation_file):
    # 실제로는 MOT15의 annotation 파일을 파싱하여 객체 위치 정보로 변환해야 합니다.
    detections = []
    with open(annotation_file, 'r') as file:
        for line in file:
            values = list(map(float, line.strip().split(',')))
            # (x, y, width, height, track_id 등)
            x, y, w, h = values[2], values[3], values[4], values[5]  # 예시로 x, y, w, h만 사용
            detections.append([x, y, w, h])
    return detections

def main():
    image_folder = 'path/to/MOT15/images'  # 이미지 폴더 경로
    annotation_folder = 'path/to/MOT15/annotations'  # annotation 폴더 경로

    image_files = sorted('path/to/MOT15/images')  # 이미지 파일을 정렬하여 읽기
    sort = Sort()

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output_tracking_result.avi', fourcc, 30, (1920, 1080))  # 출력 비디오 파일 설정

    for image_file in image_files:
        if not image_file.endswith('.jpg'):  # 이미지 파일 필터링
            continue

        image_path = os.path.join(image_folder, image_file)
        annotation_file = os.path.join(annotation_folder, image_file.replace('.jpg', '.txt'))  # 예시 annotation 경로

        frame = cv2.imread(image_path)

        # MOT15의 annotation 파일을 사용하여 객체의 위치를 얻음
        detections = load_mot15_annotations(annotation_file)

        # SORT 알고리즘을 통해 객체 추적
        trackers = sort.update(detections)

        # 추적 결과 화면에 표시
        for tracker in trackers:
            x, y, track_id = tracker
            cv2.putText(frame, f'ID: {int(track_id)}', (int(x), int(y)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.rectangle(frame, (int(x), int(y)), (int(x) + 50, int(y) + 50), (0, 255, 0), 2)  # 객체를 사각형으로 표시

        # 결과 비디오에 저장
        out.write(frame)

        # 화면에 출력
        cv2.imshow('Tracking Result', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 비디오 처리 종료
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
