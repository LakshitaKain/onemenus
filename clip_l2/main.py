from clip_model import CLIPInfer
import cv2
from config import get_config


def predict_label(image_array,img_info):
    # model_type, head_path, hidden_dim, num_classes, device
    config = get_config()
    clip_inference = CLIPInfer(**config)
    predicted_label = clip_inference.process_image_and_predict_label(image_array)
    if predicted_label is not None:
        print(predicted_label)
        return int(predicted_label)
    
    
    return None
    
# img = cv2.imread("3a9cb4453f956f9fe98b3e09b5c353fb.jpg")
# predict_label(img, None)   