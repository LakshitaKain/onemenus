import yaml
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from clip_pipeline_1 import CLIPInfer

# Load configuration from YAML file
def load_yaml_config(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

# Create an instance of the APIRouter
router_l1 = APIRouter()

# Load configuration
config = load_yaml_config("/root/ocr/CLIP/config.yaml")['L1']  # Change 'L1' to 'L2' as needed
clip_infer = CLIPInfer(config)

# @router_l1.post("/predict")
# async def predict_label(image_url):
#     try:
#         predicted_label = clip_infer.predict_label(image_url)
#         if predicted_label is not None:
#             return {"predicted_label": predicted_label}
#         else:
#             raise HTTPException(status_code=400, detail="Prediction failed.")
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))


from pydantic import BaseModel

class PredictRequest(BaseModel):
   image_url: str

@router_l1.post("/predict")
async def predict_label(request: PredictRequest):
   try:
       predicted_label = clip_infer.predict_label(request.image_url)
       if predicted_label is not None:
           return int(predicted_label)
       else:
           raise HTTPException(status_code=400, detail="Prediction failed.")
   except Exception as e:
       raise HTTPException(status_code=500, detail=str(e))