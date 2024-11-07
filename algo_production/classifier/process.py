import torch
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from PIL import Image
from classifier.get_model import get_model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = get_model()
model.to(device)


size = 224
mean = [0.48145466, 0.4578275, 0.40821073]
std = [0.26862954, 0.26130258, 0.27577711]
transform = Compose([
    Resize((size, size)),
    ToTensor(),
    Normalize(mean=mean, std=std)
])

def predict_image(image: Image.Image) -> tuple:
    
    image = transform(image).unsqueeze(0)
    image = image.to(device)
    
    
    with torch.no_grad():
        image_features = model.get_image_features(image)
        outputs = model.classifier(image_features)
        _, predicted = torch.max(outputs, 1)
        confidence = torch.softmax(outputs, dim=1)[0][predicted].item() 
    
    return predicted.item(), confidence
