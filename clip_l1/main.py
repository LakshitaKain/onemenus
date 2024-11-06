from process import Image
from process import predict_image
import numpy as np
from PIL import Image
import cv2
def predict(image: np.ndarray):
    try:
        pil_image = Image.fromarray(image)
        predicted_class, confidence = predict_image(pil_image)
        return (int(predicted_class),float(confidence))
    except Exception as e:
        print(f"Error while processing l1 : {e}\n")
        return None      

# image_path =r"C:\Users\sunny\OneDrive\Desktop\Catogery 4\Screenshot 2024-10-14 010029.png"
# image = cv2.imread(image_path)