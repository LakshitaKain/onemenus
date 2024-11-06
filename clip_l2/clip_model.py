import torch
import clip
import torch.nn as nn
from preprocess import preprocess_image
from config import Config

class CLIPInfer:
    def __init__(self):
        self.clip_model, self.cls_head = self._load_model()

    def _load_model(self):
        clip_model, _ = clip.load(Config.CLIP_TYPE, device=Config.DEVICE)

        cls_head = nn.Sequential(
            nn.Linear(clip_model.visual.output_dim, Config.HID_DIM),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(Config.HID_DIM, Config.NUM_CLASSES)
        ).to(Config.DEVICE)

        cls_head.load_state_dict(torch.load(Config.WEIGHT_PATH, map_location=Config.DEVICE))
        return clip_model, cls_head

    def predict_label(self, image_array):
        self.clip_model.eval()
        self.cls_head.eval()

        img_tensor = preprocess_image(image_array)
        if img_tensor is not None:
            features = self.clip_model.encode_image(img_tensor)
            pred = self.cls_head(features)
            
            pred_probs = torch.nn.functional.softmax(pred, dim=1)
            pred_label = pred.argmax(dim=1).item()
            confidence = pred_probs[0, pred_label].item()
            return pred_label, confidence

        return None, None
