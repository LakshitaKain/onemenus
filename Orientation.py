from fastapi import FastAPI, HTTPException, APIRouter
import requests
import numpy as np
import cv2
import boto3
import hashlib
from io import BytesIO
from PIL import Image
import json  # Import JSON module to structure error responses
import pytesseract
from pytesseract import Output

# Initialize FastAPI and router
app = FastAPI()
router = APIRouter()

# Wasabi S3 Configuration
BUCKET_NAME = 'onemenus'
S3_DIRECTORY = 'orientation'

# Function to download an image from a URL
def download_image(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an error for bad responses
        image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        return image
    except Exception as e:
        print(f"Error downloading image from {url}: {e}")
        return None

# Function to align and rotate the image
def align_and_rotate_image(image):
    if image is None or image.size == 0:
        print("Could not read image.")
        return None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        print("No contours found in the image.")
        return image

    c = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(c)
    box = np.int32(cv2.boxPoints(rect))

    width, height = rect[1]
    angle = rect[2]

    if width < height:
        angle = 90 + angle
    if angle < -45:
        angle += 90
    elif angle < 0:
        angle += 180

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    aligned = cv2.warpAffine(image, M, (w, h))
    rotated = cv2.rotate(aligned, cv2.ROTATE_90_CLOCKWISE)

    return rotated

def add_padding(image, padding_color=(0, 0, 0)):
    """
    Add padding to the image to allow rotation without cropping content.
    :param image: The input image.
    :param padding_color: The color of the padding (default is black).
    :return: Image with added padding.
    """
    # Calculate the desired padding (10% of the height of the image)
    height, width = image.shape[:2]
    padding = int(height * 0.20)  # Ensure padding is an integer
    # Create a new image with padding added
    padded_image = cv2.copyMakeBorder(
        image,
        padding, padding, padding, padding,
        cv2.BORDER_CONSTANT,
        value=padding_color
    )
    return padded_image

# Function to upload the image to Wasabi S3
def upload_image_to_s3(image, bucket_name, s3_directory, original_url):
    try:
        # Initialize a session using boto3
        session = boto3.Session(
            aws_access_key_id='9AICS9 A69JOMDFPVVELX',  # Replace with environment variables or secrets manager for security
            aws_secret_access_key='JKm0CX0QSE0mbfS1y0V8siWTUOESS2msT50LytAU',
            region_name='us-east-1'
        )
    except Exception as e:
        return json.dumps({"status": "error", "step": "Initializing boto3 session", "message": str(e)})

    try:
        # Initialize S3 resource
        s3 = session.resource('s3', endpoint_url="https://s3.us-east-1.wasabisys.com")
    except Exception as e:
        return json.dumps({"status": "error", "step": "Connecting to S3 resource", "message": str(e)})

    try:
        # Generate a unique filename for the image
        unique_filename = generate_unique_filename(original_url)
        s3_object_name = f'{s3_directory}/aligned_rotated_image_{unique_filename}.jpg'

        # Convert the image to a buffer for upload
        buffer = BytesIO()
        image.save(buffer, format='JPEG', quality=100)
        buffer.seek(0)
    except Exception as e:
        return json.dumps({"status": "error", "step": "Preparing image for upload", "message": str(e)})

    try:
        # Upload the image to S3
        s3.Bucket(bucket_name).put_object(
            Key=s3_object_name,
            Body=buffer,
            ACL='public-read',
            ContentType='image/jpeg',
            ContentDisposition='inline'
        )
        
        # Construct the public URL
        public_url = f'https://{bucket_name}.s3.wasabisys.com/{s3_object_name}'
        return json.dumps({"status": "success", "public_url": public_url})
    except Exception as e:
        return json.dumps({"status": "error", "step": "Uploading image to S3", "message": str(e)})

# Function to generate a unique filename using MD5 hash
def generate_unique_filename(original_url):
    hash_object = hashlib.md5(original_url.encode())
    return hash_object.hexdigest()

# API Endpoint to process the image
@router.get("/align-and-rotate/")
async def align_and_rotate_image_endpoint(image_url: str):
    try:
        # Download the image from the given URL
        image = download_image(image_url)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Error downloading the image.")

        # Align and rotate the image
        padding_img = add_padding(image)
        aligned_rotated_image = align_and_rotate_image(padding_img)

        if aligned_rotated_image is None:
            raise HTTPException(status_code=500, detail="Error processing the image.")

        # Attempt OCR-based orientation detection
        try:
            # Convert to grayscale for OCR processing
            gray = cv2.cvtColor(aligned_rotated_image, cv2.COLOR_BGR2GRAY)
            
            # Apply thresholding to preprocess the image (Optional)
            _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Attempt orientation detection without specifying DPI first
            data = pytesseract.image_to_osd(thresh, output_type=Output.DICT)
            rotation_angle = data['rotate']

            # If orientation detected, rotate accordingly
            if rotation_angle == 90:
                aligned_rotated_image = cv2.rotate(aligned_rotated_image, cv2.ROTATE_90_CLOCKWISE)
            elif rotation_angle == 270:
                aligned_rotated_image = cv2.rotate(aligned_rotated_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        except pytesseract.TesseractError as e:
            print(f"Error with default configuration for OCR: {e}")
            # Attempt orientation detection with DPI configuration
            try:
                # Set a reasonable DPI by resizing (example 300 DPI)
                height, width = aligned_rotated_image.shape[:2]
                resized_image = cv2.resize(aligned_rotated_image, (width * 2, height * 2))  # Resize to improve OCR

                # Convert the resized image to grayscale for OCR processing
                gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

                # Apply adaptive thresholding to preprocess the image
                thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

                # Retry with DPI configuration
                data = pytesseract.image_to_osd(thresh, config='--dpi 300', output_type=Output.DICT)
                rotation_angle = data['rotate']

                # If orientation detected, rotate accordingly
                if rotation_angle == 90:
                    aligned_rotated_image = cv2.rotate(aligned_rotated_image, cv2.ROTATE_90_CLOCKWISE)
                elif rotation_angle == 270:
                    aligned_rotated_image = cv2.rotate(aligned_rotated_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
            
            except pytesseract.TesseractError as e:
                print(f"Error processing with DPI configuration: {e}")
                # Fall back to the aligned and rotated image without further changes
           
            except Exception as e:
                print(f"General error during OCR retry: {e}")
                # Fall back to the aligned and rotated image without further changes

        except Exception as e:
            print(f"General error during OCR: {e}")
            # Fall back to the aligned and rotated image without further changes

        # Convert the result to a PIL Image and upload to Wasabi S3
        result_image = Image.fromarray(cv2.cvtColor(aligned_rotated_image, cv2.COLOR_BGR2RGB))
        public_url = upload_image_to_s3(result_image, BUCKET_NAME, S3_DIRECTORY, image_url)

        return {"original_url": image_url, "aligned_rotated_url": public_url}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=1111)

