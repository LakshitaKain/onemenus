import torch
import clip
import torch.nn as nn
from preprocessing_2 import preprocess_image

class CLIPInfer:
    def __init__(self):
        # Define the configuration directly
        self.config = {
            "clip_type": "ViT-L/14@336px",
            "weight_path": "/root/riya/clip/clip_l2/weights/head_acc0.9049.pth",
            "input_size": [336, 336],
            "hid_dim": 512,
            "num_classes": 2,
            "device": "cpu"
        }
        self.clip_model, self.cls_head = self.load_model()

    def load_model(self):  
        # Load the CLIP model
        device = self.config["device"]
        clip_model, _ = clip.load(self.config["clip_type"], device=device)

        # Define the classification head
        cls_head = nn.Sequential(
            nn.Linear(clip_model.visual.output_dim, self.config['hid_dim']),
            nn.ReLU(),  # Assuming ReLU, change as needed
            nn.Dropout(0.1),
            nn.Linear(self.config['hid_dim'], self.config['num_classes'])
        ).to(device)

        # Load the fine-tuned weights for the classification head
        cls_head.load_state_dict(torch.load(self.config['weight_path'], map_location=device, weights_only=True))
        return clip_model, cls_head

    def predict_label(self, image_url):
        # Switch the model to evaluation mode
        self.clip_model.eval()
        self.cls_head.eval()

        img_tensor = preprocess_image(image_url, self.config)
        if img_tensor is not None:
            img_tensor = img_tensor.to(torch.float32)
            self.clip_model.float()
            self.cls_head.float()

            features = self.clip_model.encode_image(img_tensor)
            
            if features.dim() != 2 or features.shape[0] != 1:
                print(f"Unexpected features shape: {features.shape}")
                return None, None

            pred = self.cls_head(features)

            if pred.dim() != 2 or pred.shape[0] != 1:
                print(f"Unexpected prediction shape: {pred.shape}")
                return None, None
            
            pred_probs = torch.nn.functional.softmax(pred, dim=1)
            pred_label = pred.argmax(dim=1).item()
            confidence = pred_probs[0, pred_label].item()
            
            return pred_label, confidence
        return None, None
