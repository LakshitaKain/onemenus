import torch
import torchvision.transforms.v2 as transforms
from torchvision.transforms import InterpolationMode

transformations = transforms.Compose([
    transforms.ToImage(),
    transforms.ToDtype(torch.float32, scale=True),  # Normalize expects float input
    transforms.Resize(size=[384, 384], interpolation=InterpolationMode.BILINEAR),
    transforms.CenterCrop(384),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])