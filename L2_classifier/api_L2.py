## api_L2.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from clip_pipeline_2 import CLIPInfer

# Create an instance of the APIRouter
router_l2 = APIRouter()

# Initialize the CLIPInfer class
clip_infer = CLIPInfer()

class PredictRequest(BaseModel):
    image_url: str

@router_l2.post("/predict_L2")
async def predict_label(request: PredictRequest):
    try:
        predicted_label, confidence = clip_infer.predict_label(request.image_url)
        if predicted_label is not None:
            return {
                "image_url": request.image_url,
                "predicted_label": int(predicted_label),
                "confidence": confidence
            }
        else:
            return {"detail": "Prediction failed."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
