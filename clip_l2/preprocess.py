import numpy as np
import cv2
import torch

def preprocess_image(image_array, input_size):
    try:
        # Resize the image to the required input size
        img_array = cv2.resize(image_array, input_size)
        
        # Convert the image to a tensor and move the channels to the first dimension
        img_tensor = torch.tensor(img_array, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)

        # Repeat the grayscale channel to match the 3 channels (R, G, B)
        if img_tensor.shape[1] == 1:  # if grayscale, repeat channel
            img_tensor = img_tensor.repeat(1, 3, 1, 1)
        
        # Normalize the image
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1) * 255
        std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1) * 255
        img_tensor = (img_tensor - mean) / std
        
        return img_tensor
    
    except Exception as e:
        raise RuntimeError(f"Error in image preprocessing: {e}")


