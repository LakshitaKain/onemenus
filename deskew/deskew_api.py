from fastapi import FastAPI, HTTPException
from fastapi import APIRouter
import requests
from io import BytesIO
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as torchvision_T
import boto3
import hashlib
import os
from pymongo import MongoClient
import gc
import cv2
import base64

# Initialize FastAPI and router
app = FastAPI()
router = APIRouter()

# MongoDB Configuration
MONGO_URI = "mongodb://ReadOnly:iXfZs4HVhN@213.199.54.177:27017/"
DB_NAME = "topmenus_product"
db = MongoClient(MONGO_URI)[DB_NAME]

# Wasabi S3 Configuration
BUCKET_NAME = 'onemenus'
S3_DIRECTORY = 'deskew'

# Load the model function
def load_model(num_classes=2, model_name="mbv3", device=torch.device("cpu")):
    from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large
    model = deeplabv3_mobilenet_v3_large(num_classes=num_classes, aux_loss=True)
    checkpoint_path = os.path.join(os.getcwd(), "model_mbv3_iou_mix_2C049.pth")

    model.to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint, strict=False)
    model.eval()
    return model

# Image processing function
def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    pts = np.array(pts)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect.astype("int").tolist()

def find_dest(pts):
    (tl, tr, br, bl) = pts
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    destination_corners = [[0, 0], [maxWidth, 0], [maxWidth, maxHeight], [0, maxHeight]]
    return order_points(destination_corners)

def scan(image_true, trained_model, image_size=384, BUFFER=10):
    IMAGE_SIZE = image_size
    half = IMAGE_SIZE // 2

    imH, imW, C = image_true.shape
    image_model = cv2.resize(image_true, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_NEAREST)

    scale_x = imW / IMAGE_SIZE
    scale_y = imH / IMAGE_SIZE

    # Normalize and prepare input for the model
    preprocess_transforms = torchvision_T.Compose([
        torchvision_T.ToTensor(),
        torchvision_T.Normalize(mean=(0.4611, 0.4359, 0.3905), std=(0.2193, 0.2150, 0.2109)),
    ])
    image_model = preprocess_transforms(image_model)
    image_model = torch.unsqueeze(image_model, dim=0)

    with torch.no_grad():
        out = trained_model(image_model)["out"].cpu()

    del image_model
    gc.collect()

    out = torch.argmax(out, dim=1, keepdims=True).permute(0, 2, 3, 1)[0].numpy().squeeze().astype(np.int32)
    r_H, r_W = out.shape

    _out_extended = np.zeros((IMAGE_SIZE + r_H, IMAGE_SIZE + r_W), dtype=out.dtype)
    _out_extended[half:half + IMAGE_SIZE, half:half + IMAGE_SIZE] = out * 255
    out = _out_extended.copy()

    del _out_extended
    gc.collect()

    canny = cv2.Canny(out.astype(np.uint8), 225, 255)
    canny = cv2.dilate(canny, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    contours, _ = cv2.findContours(canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    page = sorted(contours, key=cv2.contourArea, reverse=True)[0]

    epsilon = 0.02 * cv2.arcLength(page, True)
    corners = cv2.approxPolyDP(page, epsilon, True)
    corners = np.concatenate(corners).astype(np.float32)

    corners[:, 0] -= half
    corners[:, 1] -= half
    corners[:, 0] *= scale_x
    corners[:, 1] *= scale_y

    if not (np.all(corners.min(axis=0) >= (0, 0)) and np.all(corners.max(axis=0) <= (imW, imH))):
        left_pad, top_pad, right_pad, bottom_pad = 0, 0, 0, 0
        rect = cv2.minAreaRect(corners.reshape((-1, 1, 2)))
        box = cv2.boxPoints(rect)
        box_corners = np.int32(box)

        box_x_min = np.min(box_corners[:, 0])
        box_x_max = np.max(box_corners[:, 0])
        box_y_min = np.min(box_corners[:, 1])
        box_y_max = np.max(box_corners[:, 1])

        if box_x_min <= 0:
            left_pad = abs(box_x_min) + BUFFER
        if box_x_max >= imW:
            right_pad = (box_x_max - imW) + BUFFER
        if box_y_min <= 0:
            top_pad = abs(box_y_min) + BUFFER
        if box_y_max >= imH:
            bottom_pad = (box_y_max - imH) + BUFFER

        image_extended = np.zeros((top_pad + bottom_pad + imH, left_pad + right_pad + imW, C), dtype=image_true.dtype)
        image_extended[top_pad:top_pad + imH, left_pad:left_pad + imW, :] = image_true
        image_extended = image_extended.astype(np.float32)

        box_corners[:, 0] += left_pad
        box_corners[:, 1] += top_pad
        corners = box_corners
        image_true = image_extended

    corners = sorted(corners.tolist())
    corners = order_points(corners)
    destination_corners = find_dest(corners)
    M = cv2.getPerspectiveTransform(np.float32(corners), np.float32(destination_corners))

    final = cv2.warpPerspective(image_true, M, (destination_corners[2][0], destination_corners[2][1]), flags=cv2.INTER_LANCZOS4)
    final = np.clip(final, a_min=0, a_max=255)
    final = final.astype(np.uint8)

    return final

# Function to upload the image to Wasabi S3
def upload_image_to_s3(image, bucket_name, s3_directory, original_url):
    session = boto3.Session(
        aws_access_key_id='9AICS9 A69JOMDFPVVELX',
        aws_secret_access_key='JKm0CX0QSE0mbfS1y0V8siWTUOESS2msT50LytAU',
        region_name='us-east-1'
    )

    s3 = session.resource('s3', endpoint_url="https://s3.us-east-1.wasabisys.com")
    unique_filename = generate_unique_filename(original_url)
    s3_object_name = f'{s3_directory}/de-skewed-image_{unique_filename}.jpg'

    buffer = BytesIO()
    image.save(buffer, format='JPEG', quality=100)
    buffer.seek(0)

    s3.Bucket(bucket_name).put_object(
        Key=s3_object_name,
        Body=buffer,
        ACL='public-read',
        ContentType='image/jpeg',
        ContentDisposition='inline'
    )

    public_url = f'https://{bucket_name}.s3.wasabisys.com/{s3_object_name}'
    return public_url

# Function to save the image URL in MongoDB
def save_url_to_mongo(image_data):
    collection = db['trail_bin']
    collection.insert_one(image_data)

# Function to generate a unique filename using MD5 hash
def generate_unique_filename(original_url):
    hash_object = hashlib.md5(original_url.encode())
    return hash_object.hexdigest()

# API Endpoint to process the image
@router.get("/deskew/")
async def deskew_image(image_url: str):
    try:
        # Load the image from the URL
        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content)).convert('RGB')
        image = np.array(image)

        # Load the model
        model = load_model()

        # Apply the de-skewing scan
        final = scan(image_true=image, trained_model=model)

        if final is None:
            raise HTTPException(status_code=500, detail="Error processing the image.")

        # Convert the result to an image and upload to Wasabi S3
        result = Image.fromarray(final[:, :, ::-1])
        public_url = upload_image_to_s3(result, BUCKET_NAME, S3_DIRECTORY, image_url)

        # Encode image to Base64
        buffered = BytesIO()
        result.save(buffered, format="JPEG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()

        # Prepare data for MongoDB
        image_data = {
            'original_url': image_url,
            'deskew_url': public_url,
            'image_base64': img_base64
        }

        # Save the image data to MongoDB
        save_url_to_mongo(image_data)

        return {"original_url": image_url, "deskew_url": public_url, "image_base64": img_base64}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=2236)