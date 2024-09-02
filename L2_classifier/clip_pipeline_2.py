import pandas as pd
import numpy as np
import torch
import clip
import torch.nn as nn
import requests
from io import BytesIO
import yaml
from preprocessing_2 import preprocess_image

class CLIPInfer:
    def __init__(self, config):
        self.clip_model, self.cls_head = self.load_model(config)

    def load_model(self, config):  
        # Load the CLIP model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        clip_model, _ = clip.load(config["clip_type"], device=device)

        # Define the classification head
        cls_head = nn.Sequential(
            nn.Linear(clip_model.visual.output_dim, config['hid_dim']),
            nn.ReLU(),  # Assuming ReLU, change as needed
            nn.Dropout(0.1),
            nn.Linear(config['hid_dim'], config['num_classes'])
        ).to(device)

        # Load the fine-tuned weights for the classification head
        cls_head.load_state_dict(torch.load(config['weight_path'], map_location=device))
        return clip_model, cls_head

    def predict_label(self, image_url):
        # Switch the model to evaluation mode
        self.clip_model.eval()
        self.cls_head.eval()

        try:
            with torch.no_grad():
                img_tensor = preprocess_image(image_url)
                if img_tensor is not None:
                    img_tensor = img_tensor.to(torch.float32)
                    self.clip_model.float()
                    self.cls_head.float()

                    features = self.clip_model.encode_image(img_tensor)
                    
                    if features.dim() != 2 or features.shape[0] != 1:
                        print(f"Unexpected features shape: {features.shape}")
                        return None

                    pred = self.cls_head(features)

                    if pred.dim() != 2 or pred.shape[0] != 1:
                        print(f"Unexpected prediction shape: {pred.shape}")
                        return None
                    
                    pred_label = pred.argmax(dim=1).item()
                    return pred_label
                else:
                    return None
        except Exception as e:
            print(f"Error during prediction: {e}")
            return None

# Load configuration from YAML file
def load_yaml_config(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

if __name__ == "__main__":
    config = load_yaml_config("config.yaml")['L2']  # Change 'L1' to 'L2' as needed
    clip_infer = CLIPInfer(config)