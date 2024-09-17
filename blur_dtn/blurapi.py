from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from blur_detection import BlurDetector
import numpy as np
import requests
from PIL import Image
import io

router = APIRouter()

# Initialize blur detector
blur_detector = BlurDetector()

class ImageRequest(BaseModel):
    img_url: str

@router.post("/process_image/")
async def process_image(request: ImageRequest):
    image_url = request.img_url

    try:
        # Fetch image from the URL
        response = requests.get(image_url)
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail="Error fetching image from URL")

        # Open image and convert to numpy array
        image = Image.open(io.BytesIO(response.content)).convert('RGB')
        image_np = np.array(image)

        # Compute blur score
        blur_score, is_blurry = blur_detector.compute_blur_score(image_np)

        # Return results
        return {
            "img_url": image_url,
            "blur_score": blur_score,
            "is_blurry": is_blurry
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
