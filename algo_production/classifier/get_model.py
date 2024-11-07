import torch
import torch.nn as nn
from transformers import CLIPModel
import numpy as np
from PIL import Image
import cv2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def get_model(model_path=r"C:/Users/sunny/new_pro/last_production/classifier/model_weights/best_model.pth", num_classes=4, num_features=512):
    model_id = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(model_id)
    
    
    model.classifier = nn.Sequential(
        nn.Linear(num_features, 1024),
        nn.BatchNorm1d(1024),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(1024, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(512, num_classes)
    )

 
    model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    model.eval()

    return model
