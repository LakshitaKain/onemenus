import numpy as np
import cv2
from ultralytics import YOLO
from config import Config

# Load the YOLO model
model = YOLO(Config.YOLO_MODEL_PATH)

def process_image_in_memory(image_array):
    try:
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
        img_height, img_width = img_rgb.shape[:2]

        # Run YOLO prediction
        results = model.predict(img_rgb)
        if results[0].masks is not None and results[0].boxes.conf is not None:
            masks = results[0].masks.data.detach().cpu().numpy()
            confidences = results[0].boxes.conf.detach().cpu().numpy()

            # Get mask with highest confidence
            best_mask_idx = np.argmax(confidences)
            best_mask = masks[best_mask_idx]
            best_mask = cv2.resize(best_mask, (img_width, img_height))
            best_mask = (best_mask * 255).astype(np.uint8)

            # Apply the mask to the image
            isolated_image = cv2.bitwise_and(img_rgb, img_rgb, mask=best_mask)

            return isolated_image

        return None

    except Exception as e:
        print(f"Error in processing the image: {e}")
        return None
