# from yolo_robo.main import yolo_bg
# from orientation.main import deskew_and_orient_correction

# def process_label_zero_image(image_np):
#     """
#     Processes an image with label 0 by applying yolo_bg and deskew_and_orient_correction.
    
#     Parameters:
#         image_np (np.ndarray): The original image as a NumPy array.
    
#     Returns:
#         np.ndarray or None: The processed image as a NumPy array if successful, else None.
#     """
#     # Apply yolo_bg
#     yolo_image_np = yolo_bg(image_np)
#     if yolo_image_np is None:
#         print("    yolo_bg failed. Skipping this image.")
#         return None
#     print("    Applied yolo_bg.")
    
#     # Apply deskew_and_orient_correction
#     deskew_image_np = deskew_and_orient_correction(yolo_image_np)
#     if deskew_image_np is None:
#         print("    deskew_and_orient_correction failed. Skipping this image.")
#         return None
#     print("    Applied deskew_and_orient_correction.")
    
#     return deskew_image_np

# process_l1_label_0_class.py

# process_l1_label_0_class.py

from yolo_robo.main import yolo_bg
from orientation.main import deskew_and_orient_correction

def process_label_zero_image(image_np):
    """
    Processes an image with label 0 by applying yolo_bg and deskew_and_orient_correction.
    
    Parameters:
        image_np (np.ndarray): The original image as a NumPy array.
    
    Returns:
        np.ndarray or None: The processed image as a NumPy array if successful, else None.
    """
    # Apply yolo_bg
    yolo_image_np = yolo_bg(image_np)
    if yolo_image_np is None:
        print("    yolo_bg failed. Skipping this image.")
        return None
    print("    Applied yolo_bg.")
    
    # Apply deskew_and_orient_correction
    deskew_image_np = deskew_and_orient_correction(yolo_image_np)
    if deskew_image_np is None:
        print("    deskew_and_orient_correction failed. Skipping this image.")
        return None
    print("    Applied deskew_and_orient_correction.")
    
    return deskew_image_np

