import os
import easyocr
import numpy as np
from PIL import Image
from typing import List, Tuple, Dict

reader = easyocr.Reader(['en'])  

def extract_text_with_confidence_easyocr(image: np.ndarray) -> Tuple[str, float, int]:
    try:
        results = reader.readtext(image, detail=1, paragraph=False)
        texts = []
        confidences = []
        for res in results:
            
            texts.append(res[1])
            confidences.append(res[2])
        combined_text = ' '.join(texts)
        average_confidence = np.mean(confidences) if confidences else 0
        total_words = len(texts)
        return combined_text, average_confidence, total_words
    except Exception as e:
        print(f"Error processing with EasyOCR: {e}")
        return "", 0, 0

def extract_texts_from_restaurant_images_easyocr(restaurant_images: List[Tuple[np.ndarray, str]]) ->Tuple[Dict, Dict, Dict, Dict] :
    texts = {}
    confidences = {}
    word_counts = {}
    images = {}
    for i, (image, link) in enumerate(restaurant_images):

        text, avg_conf, word_count = extract_text_with_confidence_easyocr(image)
        img_name = link
        texts[img_name] = text
        images[img_name] = image
        confidences[img_name] = avg_conf
        word_counts[img_name] = word_count
    return images, texts, confidences, word_counts