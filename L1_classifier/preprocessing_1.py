import numpy as np
import cv2
import torch
import requests
import yaml

# Load configuration from YAML file
def load_yaml_config(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

# Define the necessary preprocessing steps
def preprocess_image(image_url, CONFIG):
    try:
        # Fetch the image from the URL
        response = requests.get(image_url)
        if response.status_code != 200:
            raise ValueError(f"Image request failed with status code: {response.status_code}")

        img_array = np.array(cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_GRAYSCALE))
        
        if img_array is None:
            raise ValueError(f"Image not read correctly: {image_url}")
        
        img_array = cv2.resize(img_array, CONFIG["input_size"])
        img_tensor = torch.tensor(img_array, dtype=torch.float32)
        img_tensor = img_tensor.unsqueeze(0).repeat(1, 3, 1, 1)  # Convert grayscale to RGB

        mean = 255 * torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
        std = 255 * torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)

        img_tensor = (img_tensor - mean) / std
        return img_tensor.to(CONFIG['device'])
    except Exception as e:
        print(f"Error processing image {image_url}: {e}")
        return None