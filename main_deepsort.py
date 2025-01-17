import cv2
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
import os
import torch
import module_test
import deepsort_module
def main():
    # CUDA 사용 가능 여부 확인
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    sequence_path = "./Venice-2/img1"  # 입력 비디오 파일 경로
    detection_file = "./Venice-2/det/det.txt"  # detection 결과 파일
    detections = {}
    try:
          detections = module_test.save_detection(detection_file, deepsort_ver=True)
    except IOError as e:
        print(f"detection 파일 읽기 오류: {"e"}")
        return 
    
    if torch.cuda.is_available():
        stream = torch.cuda.Stream()
        torch.cuda.synchronize()
#------------------------------------------------------------------------------
    deepsort_module.deepsort1(detections=detections, sequence_path=sequence_path, stream=stream)
    # Process each frame

 #------------------------------------------------------------------------------            
            
    cv2.destroyAllWindows()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()