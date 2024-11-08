import os
import cv2
import numpy as np
from typing import List, Tuple

from get_embeddings import create_tfidf_matrix
from ocr import extract_texts_from_restaurant_images_easyocr
from generate_clusters import get_clusters_using_kmeans
from get_best_images import select_best_images_from_clusters

def process_restaurant(restaurant: List[Tuple[np.ndarray, str]]) -> List[Tuple[np.ndarray, str]]:
    
    """Takes in the list of numpy arrays resembling images of a restaurant and returns the best images while minimizing information loss.
    This function is used for cost reduction
    

    Returns:
        List[numpy arrays]: List of best images
    """
    
    try:
        images, texts, confidences, word_counts = extract_texts_from_restaurant_images_easyocr(restaurant)

        image_names = list(texts.keys())
        
        # Step 2: Create TF-IDF matrix
        tfidf_matrix, _ = create_tfidf_matrix(texts, image_names)

        
        # Step 3: Cluster images
        clusters = get_clusters_using_kmeans(tfidf_matrix, images)
        
        # Step 4: Select best images
        best_images = select_best_images_from_clusters(clusters, images, confidences, word_counts)
        # for i, img in enumerate(best_images):
        #     cv2.imwrite(f"image_{i}.jpg", img)

        return best_images
    except Exception as e:
        print(f"Error while forming clusters", e)
        return restaurant
        

