import torch

import numpy as np
from PIL import Image
import cv2
from typing import Dict

from get_model import PerceptionDetectionModel
from get_predictions import predict_and_transform_image

# Hyperparameters
DEVICE = torch.device('cuda' if torch.cuda.is_available() else "cpu")
HIDDEN_DIM = 256
MODEL_PATH = "model_effv4.pt"


model = PerceptionDetectionModel(num_hidden=HIDDEN_DIM)
model.load_state_dict(torch.load(MODEL_PATH, weights_only=True, map_location=torch.device('cpu')))
model = model.to(device=DEVICE)


def deskew_and_orient_correction(image: np.ndarray, image_info: Dict) -> np.ndarray:
    try:
        # print("code started")
        _, processed_image = predict_and_transform_image(model=model, image=image)

        # cv2.imwrite("result.jpg", processed_image)
        result_image = cv2.cvtColor(processed_image, cv2.COLOR_32RGB)
        # print(rgb_image.shape)
        return result_image

    except Exception as e:
        component = "deskew_orient"
        # print(f"Error while processing {component} in google id: {image_info['google_id']} in link {image_info['link']} \n{e}\n")
        print(f"Error while processing {component}: {e}\n")
        return None

# img = cv2.imread("test_img.jpg")
# process_image(img, None)