import cv2
import numpy as np
import easyocr
import torch
import re
from craft_text.craft import CRAFT
import json
import os


def expand_box(x_expand: float, y_expand: float, box: np.ndarray, img_w: int, img_h: int) -> np.ndarray:
    x_center = np.mean(box[:, 0])
    y_center = np.mean(box[:, 1])
    expanded_box = np.copy(box)
    for i in range(4):
        expanded_box[i, 0] = x_center + (box[i, 0] - x_center) * (1.0 + x_expand)  # Expand x
        expanded_box[i, 1] = y_center + (box[i, 1] - y_center) * (1.0 + y_expand)  # Expand y
    expanded_box[:, 0] = np.clip(expanded_box[:, 0], 0, img_w)  # Clip x coordinates
    expanded_box[:, 1] = np.clip(expanded_box[:, 1], 0, img_h)  # Clip y coordinates
    return expanded_box

def remove_last_numeric_expression(text):
    return re.sub(r'\s*\d+(\.\d+)?(?:-\d+(\.\d+)?)*$', '', text)

def extract_dish_data(lines, dish_type, dish_type_description):
    dishes = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Find price (line that ends with a numeric value)
        price_match = re.search(r'(\$?\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?(?:-\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?)?)$', line)
        if price_match:
            price = price_match.group(1)
            dish_name = line[:price_match.start()].strip()

            # Check if the next line(s) are part of the description
            description = ""
            i += 1
            while i < len(lines) and not re.search(r'\s*\d+(\.\d+)?(?:-\d+(\.\d+)?)*$', lines[i].strip()):
                description_line = lines[i].strip()
                if not description_line:
                    i += 1
                    continue
                if description_line.isupper():
                    break
                description += description_line + " "
                i += 1

            description = description.strip()

            if dish_name and price:
                dishes.append({
                    'Type': dish_type,
                    'Type_Desc': dish_type_description,
                    'Name': dish_name,
                    'Price': price,
                    'Desc': description if description else "",
                    'Add_On':"",
                    'Calories':"",
                    'DO':"",
                })
        else:
            i += 1

    return dishes

def perform_ocr_with_craft_2(image, resize_factor=1.0, dpi=300, use_gpu=True, json_output_path="output_dishes.json"):
    craft = CRAFT()
    #image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image from {image_path}")
    img_h, img_w = image.shape[:2]

    if resize_factor != 1.0:
        image_resized = cv2.resize(image, (0, 0), fx=resize_factor, fy=resize_factor)
    else:
        image_resized = image


    tensor_image = torch.from_numpy(image_resized).permute(2, 0, 1) / 255.0
    result = craft(tensor_image.unsqueeze(0))[0]
    boxes = result.detach().cpu().numpy()
    
    if boxes.size == 0:
        raise ValueError("No text detected by CRAFT.")

    expanded_boxes = [expand_box(0.1, 0.1, box, img_w, img_h) for box in boxes]

    reader = easyocr.Reader(['en'], gpu=use_gpu)

    try:
        if use_gpu:
            torch.cuda.empty_cache()

        results = reader.readtext(image_path, batch_size=1)

        if not results:
            raise ValueError("No text detected by EasyOCR.")

        results_sorted = sorted(results, key=lambda r: (r[0][0][1], r[0][0][0]))

        grouped_results = []
        current_line = []
        current_y = results_sorted[0][0][0][1]
        tolerance = 10

        for result in results_sorted:
            bbox, text, prob = result
            y_center = (bbox[0][1] + bbox[2][1]) / 2

            if abs(y_center - current_y) <= tolerance:
                current_line.append(result)
            else:
                current_line = sorted(current_line, key=lambda r: r[0][0][0])
                line_text = ' '.join([r[1] for r in current_line])
                if line_text.strip():
                    grouped_results.append(line_text)
                current_line = [result]
                current_y = y_center

        if current_line:
            current_line = sorted(current_line, key=lambda r: r[0][0][0])
            line_text = ' '.join([r[1] for r in current_line])
            if line_text.strip():
                grouped_results.append(line_text)

        temp = grouped_results.copy()

        if len(temp) > 0:
            dish_type = temp[0].strip()
            temp.pop(0)

            dish_type_description = ""
            lines_to_remove = []
            for i, line in enumerate(temp):
                if re.search(r'\d+(\.\d+)?(?:-\d+(\.\d+)?)*$', line.strip()):
                    break
                dish_type_description += line + " "
                lines_to_remove.append(i)

            for i in sorted(lines_to_remove, reverse=True):
                temp.pop(i)

            dish_type_description = dish_type_description.strip()
            if dish_type_description:
                dish_type_description = remove_last_numeric_expression(dish_type_description)
        else:
            dish_type = ""
            dish_type_description = ""

        dishes = extract_dish_data(temp, dish_type, dish_type_description)

        print(dishes)

    except MemoryError as e:
        print("Memory Error: ", str(e))
        print("Try using CPU mode or a smaller image size.")
    except Exception as e:
        print("Error: ", str(e))

