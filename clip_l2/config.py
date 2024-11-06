import torch

class Config:
    CLIP_TYPE = "ViT-L/14@336px"
    WEIGHT_PATH = "head_acc0.9049.pth"
    INPUT_SIZE = (336, 336)
    HID_DIM = 512
    NUM_CLASSES = 2
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
