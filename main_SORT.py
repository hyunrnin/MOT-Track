import cv2
import numpy as np
from dir_SORT.sort import Sort
import os
import util.module_result as module_result
def main():

    sequence_path = "./Venice-2/img1"
    detection_file = "./Venice-2/det/det.txt"
    detections = {}
    detections = module_result.save_detection(detection_file, sort_ver='sort')

    sort_tracker = Sort(max_age=4)

    for frame_id in sorted(detections.keys()):
        img_path = os.path.join(sequence_path, f"{frame_id:06d}.jpg")
        frame = cv2.imread(img_path)
        trackers = sort_tracker.update(np.array(detections[frame_id]))

        module_result.visualize_results(frame, trackers)    
 
        cv2.imshow('MOT Tracking', frame)
        if cv2.waitKey(30) & 0xFF == 27:
            break
            
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()