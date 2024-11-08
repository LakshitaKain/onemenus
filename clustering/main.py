from typing import List, Tuple
import numpy as np
import cv2

from process_restaurant import process_restaurant

def generate_best_images_from_google_id(images: List[Tuple[np.ndarray, str]]) -> List[Tuple[np.ndarray, str]]:
    """Takes a list of image arrays, forms them into clusters and return the best images

    Args:
        images (List[Tuple[np.ndarray, str]]): Input arrays and links

    Returns:
        List[Tuple[np.ndarray, str]]: best images and their links
    """
    best_images = process_restaurant(restaurant=images)

    return best_images
    

# img_path1 = "../testing/small/0x808f8f000d2e6cff:0x3caec3fe126aaf7b_processed/clusters/cluster_0/921c087ea3968f592782f80266e816db.jpg.jpg"
# img_path2 = "../testing/small/0x808f8f000d2e6cff:0x3caec3fe126aaf7b_processed/clusters/cluster_0/b965940e17dd6d0572d03e76671b062f.jpg.jpg"
# img_path3 = "../testing/small/0x808f8f000d2e6cff:0x3caec3fe126aaf7b_processed/clusters/cluster_1/6ebb4665e36199df5167d712fa369d5d.jpg.jpg"
# img_path4 = "../testing/small/0x808f8f000d2e6cff:0x3caec3fe126aaf7b_processed/clusters/cluster_1/73775ba92c5bf46f8cf23a66d97cd7d0.jpg.jpg"
# img1 = cv2.imread(img_path1)
# img2 = cv2.imread(img_path2)
# img3 = cv2.imread(img_path3)
# img4 = cv2.imread(img_path4)

# print(generate_best_images_from_google_id([(img1, img_path1), (img2, img_path2), (img3, img_path3), (img4, img_path4)]))