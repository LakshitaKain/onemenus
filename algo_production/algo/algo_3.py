import cv2
import numpy as np
import easyocr
import torch
import json
from craft_text.craft import CRAFT

def is_uppercase(text):
    return text.isupper()

def calculate_letter_edge_height(bbox, image):
    (top_left, top_right, bottom_right, bottom_left) = bbox
    x_min = int(min(top_left[0], bottom_left[0]))
    x_max = int(max(top_right[0], bottom_right[0]))
    y_min = int(min(top_left[1], top_right[1]))
    y_max = int(max(bottom_left[1], bottom_right[1]))
    roi = image[y_min:y_max, x_min:x_max]
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, binary_roi = cv2.threshold(gray_roi, 150, 255, cv2.THRESH_BINARY_INV)
    vertical_projection = np.sum(binary_roi, axis=1)
    top_edge = np.where(vertical_projection > 0)[0][0]
    bottom_edge = np.where(vertical_projection > 0)[0][-1]
    letter_edge_height = bottom_edge - top_edge
    return letter_edge_height

def perform_ocr_with_craft_3(image, resize_factor=1.0, dpi=300, use_gpu=True):
    craft = CRAFT(pretrained=True)
    if image is None:
        raise ValueError(f"Failed to load image from {image_path}")
    if resize_factor != 1.0:
        image_resized = cv2.resize(image, (0, 0), fx=resize_factor, fy=resize_factor)
    else:
        image_resized = image
    image_tensor = torch.from_numpy(image_resized).permute(2, 0, 1).float().div(255.0).unsqueeze(0)
    if use_gpu:
        image_tensor = image_tensor.cuda()
        craft = craft.cuda()

    # Pass image through CRAFT model
    with torch.no_grad():
        result = craft.forward(image_tensor)

    # Check if result is a tuple and handle accordingly
    if isinstance(result, tuple):
        boxes = result[0]  # Assume first element is boxes; adjust if necessary
    else:
        boxes = result.get('boxes', None)
    if boxes.size == 0:
        raise ValueError("No text detected by CRAFT.")
    
    reader = easyocr.Reader(['en'], gpu=use_gpu)
    try:
        if use_gpu:
            torch.cuda.empty_cache()
        results = reader.readtext(image, batch_size=1)
        if not results:
            raise ValueError("No text detected by EasyOCR.")
        
        results_sorted = sorted(results, key=lambda r: (r[0][0][1], r[0][0][0]))
        
        # Variables to store dish type and names
        dish_type = None
        dish_names = []
        
        for i, (bbox, text, prob) in enumerate(results_sorted):
            letter_edge_height = calculate_letter_edge_height(bbox, image_resized)
            height_in_inches = letter_edge_height / dpi
            font_size_in_points = height_in_inches * 72
            
            # Assign text to dish type or dish names
            if i == 0:
                dish_type = text
            else:
                dish_names.append(text)

        # Prepare JSON output
        output_data = [{"Type": dish_type, "Name": name,"Price":"","Calories":"","Add-On":"","DO":""} for name in dish_names]
        return output_data

    except Exception as e:
        print("Error: ", str(e))



