import cv2
import numpy as np
import os
from deep_sort.application_util import preprocessing
from deep_sort.deep_sort import nn_matching
from deep_sort.deep_sort.detection import Detection
from deep_sort.deep_sort.tracker import Tracker
from deep_sort.tools import generate_detections as gdet

def main():
    sequence_path = "C:/Users/user/pj/dev-tracking/Venice-2/img1"
    detection_file = "C:/Users/user/pj/dev-tracking/Venice-2/det/det.txt"
    
    # Initialize deep sort
    max_cosine_distance = 0.3
    nn_budget = None
    
    # Deep SORT 초기화
    model_filename = 'C:/Users/user/pj/dev-tracking/deep_sort/networks/mars-small128.pb'  # deep sort에서 제공하는 모델
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

    # Load detections
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
    
    # Process each frame
    for frame_id in sorted(detections.keys()):
        # Load image
        img_path = os.path.join(sequence_path, f"{frame_id:06d}.jpg")
        frame = cv2.imread(img_path)
        
        if frame_id in detections:
            dets = np.array(detections[frame_id])
            boxes = dets[:, :4]
            scores = dets[:, 4]
            
            # Generate features for each box
            features = encoder(frame, boxes)
            
            # Score to bbox_score
            detections_obj = [Detection(bbox, score, feature) for bbox, score, feature 
                            in zip(boxes, scores, features)]
            
            # Run non-maxima suppression
            boxes = np.array([d.tlwh for d in detections_obj])
            pick = [b for b in boxes]
            scores = np.array([d.confidence for d in detections_obj])
            #indices = preprocessing.non_max_suppression(boxes, 1.0, scores)
            #detections_obj = [detections_obj[i] for i in pick]
            
            # Update tracker
            tracker.predict()
            tracker.update(detections_obj)
            
            # Draw results
            for track in tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue
                
                bbox = track.to_tlbr()
                # Draw bounding box
                cv2.rectangle(frame, 
                            (int(bbox[0]), int(bbox[1])), 
                            (int(bbox[2]), int(bbox[3])), 
                            (0, 255, 0), 2)
                # Put ID
                cv2.putText(frame, 
                           str(track.track_id),
                           (int(bbox[0]), int(bbox[1])-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 
                           0.9, 
                           (0, 255, 0), 
                           2)

        # Show results
        cv2.imshow('Deep SORT Tracking', frame)
        if cv2.waitKey(30) & 0xFF == 27:  # ESC key to stop
            break
            
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()