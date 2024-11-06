from clip_model import CLIPInfer
import cv2


def predict_label(image_array,img_info):
    clip_infer = CLIPInfer()
    predicted_label, confidence = clip_infer.predict_label(image_array)
    if predicted_label is not None:
        print(predicted_label)
        return int(predicted_label)
    
    
    return None
    
# img = cv2.imread("f739a19a02cb55903ce5a68aa316e7b8.jpg")
# predict_label(img, None)   