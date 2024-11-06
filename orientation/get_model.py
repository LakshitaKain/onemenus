import torch.nn as nn
from torchvision.transforms import v2, InterpolationMode
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights

weights = EfficientNet_V2_S_Weights.DEFAULT
eff_net = efficientnet_v2_s(weights)

class PerceptionDetectionModel(nn.Module):
    def __init__(self, num_hidden):
        super().__init__()
        self.model = eff_net
        
        self.model.classifier = nn.Sequential(
            nn.Linear(1280, 512),
            nn.ReLU(),
            nn.Linear(512, num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, 9)
        )
    
    def forward(self, image):
        logits = self.model(image)
        
        return logits