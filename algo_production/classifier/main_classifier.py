import numpy as np
from PIL import Image
import cv2
from classifier.process import predict_image
from classifier.process import Image
def predict(seg: np.ndarray):
    try:
        pil_image = Image.fromarray(seg)
        predicted_class, confidence = predict_image(pil_image)
        
            
        return [int(predicted_class),float(confidence),seg]
    except Exception as e:
        print(f"Error while processing l1 : {e}\n")
        return None
        
def process_segmented_images(seg):
    prediction_results = []
    for idx, image_array in enumerate(seg):
        segment_id = f"s{idx+1}"  
        result = predict(image_array)
        if result:
            result_dict = {
                "id": segment_id,
                "predicted_label": result[0],
                "confidence_score": result[1],
                "image_p": result[2]
            }
            prediction_results.append(result_dict)
    return prediction_results
