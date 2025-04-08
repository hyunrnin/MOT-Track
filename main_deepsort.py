import cv2
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
import os
import torch
import util.module_result as module_result
import util.deepsort_module as deepsort_module

def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    sequence_path = "./Venice-2/img1"
    detection_file = "./Venice-2/det/det.txt"
    detections = {}
    try:
          detections = module_result.save_detection(detection_file, sort_ver='deepsort')
    except IOError as e:
        print(f"detection 파일 읽기 오류: {"e"}")
        return 
    
    if torch.cuda.is_available():
        stream = torch.cuda.Stream()
        torch.cuda.synchronize()

    deepsort_module.deepsort1(detections=detections, sequence_path=sequence_path, stream=stream)           
            
    cv2.destroyAllWindows()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()