import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

model = YOLO("best.pt")
def yolo_segment(image: np.ndarray, l1_label):
    try:
        results = model.predict(image)
        segments = []
        width, height, _ = image.shape
        for result in results:
            boxes = result.boxes  # Bounding boxes
            xyxy = boxes.xyxy.cpu().numpy()  # Coordinates in (x1, y1, x2, y2) format
            xywh = boxes.xywh.cpu().numpy()
            confs = boxes.conf
            
            names = result.names
            cls = boxes.cls.cpu().numpy()  # Class indices
            
            if l1_label in [0, 1]:
                for i, box in enumerate(xyxy):
                    # try:
                        x1, y1, x2, y2 = map(int, box)
                        segment = image[y1:y2, x1:x2]
                        cv2.imwrite(f"segs/seg{i}.jpg", segment)
                        segments.append(segment)  # Append each cropped segment as np.ndarray
                
            else:
                bounding_threshold = 0.50
                edge_threshold = 3
                
                for i, (box, conf) in enumerate(zip(xywh, confs)):
                    try:
                        if conf > bounding_threshold:
                            x, y, w, h = box
                            if x <= edge_threshold or x + w >= (width - edge_threshold):
                                pass

                            elif y <= edge_threshold or y + h >= (height - edge_threshold):
                                pass
                            else:
                                x1 = int(x)
                                y1 = int(y)
                                x2 = int(x+w)
                                y2 = int(y+h)
                                segment = image[y1:y2, x1:x2]
                                cv2.imwrite(f"seg{i}.jpg", segment)
                                segments.append(segment)
                    except Exception as e:
                        print(f"Error processing class 2: {e}")
                        
        return segments
    
    except Exception as e:
        print("Error processing images for section segmentation")
                                


# image_path = "test2.jpg"
# img = cv2.imread(image_path)
# results = yolo_segment(img, 2)
# print(len(results))