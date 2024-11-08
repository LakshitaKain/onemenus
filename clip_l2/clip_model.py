import torch
import clip
import torch.nn as nn
from preprocess import preprocess_image
# from config import Config
    
class CLIPInfer(nn.Module):
    def __init__(self, model_type, head_path, input_size, hidden_dim, num_classes, device):
        super().__init__()
        self.device = device
        self.input_size = input_size
        self.clip, _ = clip.load(model_type, device)
        self.head = nn.Sequential(
            nn.Linear(self.clip.visual.output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes)
        ).to(device)

        self.head.load_state_dict(torch.load(head_path, map_location=device))

    def forward(self, image_array):
        return self.head(self.clip(image_array))
    
    def process_image_and_predict_label(self, image_array):
        self.clip.eval()
        self.head.eval()

        with torch.inference_mode():
            img_tensor = preprocess_image(image_array, input_size=self.input_size).to(self.device)
            if img_tensor is not None:
                features = self.clip.encode_image(img_tensor)
                pred = self.head(features)
                pred_label = pred.argmax(dim=1).item()
                return pred_label

            return None, None
