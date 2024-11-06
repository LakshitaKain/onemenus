import torch
import torch.nn as nn

import numpy as np
from PIL import Image

from utils import denormalise, adjust_canvas_and_transform, perspective_transform
from get_transformations import transformations as transforms

DEVICE = torch.device('cuda' if torch.cuda.is_available() else "cpu")

def predict_and_transform_image(model: nn.Module, image: np.ndarray, device: torch.device = DEVICE):
    
    image, _ = perspective_transform(image)
    img = Image.fromarray(image)
    transformed_img = transforms(img).unsqueeze(0).to(device)
    # print(transformed_img.shape)
    perception_logits = model(transformed_img).flatten().tolist()
    # print(perception_logits)
    perception_logits = denormalise([perception_logits])
    
    perception_matrix = perception_logits.reshape(3, 3)
    numpy_matrix = perception_matrix.detach().cpu().numpy()

    # Apply perspective transformation using the inverse matrix
    inverse_matrix = np.linalg.inv(numpy_matrix)
    corrected_img = adjust_canvas_and_transform(image, inverse_matrix)
    return image, corrected_img