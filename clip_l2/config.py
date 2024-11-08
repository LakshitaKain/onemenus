import torch

def get_config():

    return {
        "model_type" : "ViT-L/14@336px",
        "head_path" : "head_acc0.9049 (3).pth",
        "input_size" : (336, 336),
        "hidden_dim" : 512,
        "num_classes" : 2,
        "device" : torch.device("cuda" if torch.cuda.is_available() else "cpu")
    }
