import cv2
import numpy as np
from dir_SORT.sort import Sort
import os
import util.module_result as module_result
def main():

#-------------------------------------------------------------------------------
    sequence_path = "./Venice-2/img1" # 입력 비디오 파일 경로
    detection_file = "./Venice-2/det/det.txt" # detection 결과 파일
    detections = {}
    # detection_file에서 frame_id 별로 detection 결과를 읽어 detections에 저장
    detections = module_result.save_detection(detection_file, sort_ver='sort')
#-------------------------------------------------------------------------------

    # SORT 객체 생성
    sort_tracker = Sort(max_age=4)

#-------------------------------------------------------------------------------
    # 각 프레임에 대해 처리
    for frame_id in sorted(detections.keys()):
        # 이미지 로드
        img_path = os.path.join(sequence_path, f"{frame_id:06d}.jpg")
        frame = cv2.imread(img_path)
        # 트래커 업데이트
        trackers = sort_tracker.update(np.array(detections[frame_id]))

#-------------------------------------------------------------------------------
        # 결과 시각화
        module_result.visualize_results(frame, trackers)    
#--------------------------------------------------------------------------------        
        # 결과 표시
        cv2.imshow('MOT Tracking', frame)
        if cv2.waitKey(30) & 0xFF == 27:  # ESC 키로 종료
            break
            
    cv2.destroyAllWindows()

#-------------------------------------------------------------------------------
if __name__ == "__main__":
    main()