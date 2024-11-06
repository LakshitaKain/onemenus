import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

model = YOLO("../models/best.pt")

def yolo_segment(image: np.ndarray):

    try:
        results = model.predict(image)
        segments = []

        for result in results:
            boxes = result.boxes  # Bounding boxes
            xyxy = boxes.xyxy.cpu().numpy()  # Coordinates in (x1, y1, x2, y2) format
            names = result.names
            cls = boxes.cls.cpu().numpy()  # Class indices

            for i, box in enumerate(xyxy):
                try:
                    x1, y1, x2, y2 = map(int, box)
                    
                    segment = image[y1:y2, x1:x2]  
                    segments.append(segment)  # Append each cropped segment as np.ndarray

                    # # Display each segment
                    # plt.imshow(segment)
                    # plt.title(f"Segment {i} - Class: {names[int(cls[i])]}")
                    # plt.axis("off")
                    # plt.show()
                    
                except Exception as e:
                    print(f"Error processing segment {i}: {e}")
                    continue  # Move to the next segment if an error occurs
        return segments

    except Exception as e:
        print(f"Error during YOLO segmentation: {e}")
        return []
    

# #Example usage      
# image_path = "/Users/lakshita/Desktop/pipeline/yolo/testing/yolo/data/de-skewed-image_94be9b0c81a10c756fcafdc18894383d.jpg"
# img = cv2.imread(image_path)
# results = yolo_segment(img)
# print(results)