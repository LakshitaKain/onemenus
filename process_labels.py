from clip_l1.main import predict as clip1_predict
from clip_l2.main import predict_label as clip2_predict_label
from preprocessing_l1_0 import process_label_zero_image

def process_image_labels(image_np):
    """
    Processes an image by applying label predictions and necessary image processing.

    Parameters:
        image_np (np.ndarray): The original image as a NumPy array.

    Returns:
        tuple or None:
            - If l2_label == 1 and l1_label == 0: Returns (processed_image_np).
            - If l2_label == 1 and l1_label in [1, 2]: Returns (image_np).
            - Otherwise: Returns None.
    """
    # Step 2: Apply predict from clip_l1.main
    l1_label = clip1_predict(image_np)
    print(f"    clip_l1.predict label: {l1_label}")

    # Check if label is 0, 1, or 2
    if l1_label in [0, 1, 2]:
        # Step 3: Apply predict_label from clip_l2.main
        l2_label = clip2_predict_label(image_np)
        print(f"    clip_l2.predict_label: {l2_label}")

        # Combined condition 
        if l2_label == 1 and l1_label == 0:
            # Process label 0 images using the standalone function
            deskew_image_np = process_label_zero_image(image_np)
            if deskew_image_np is not None:
                print("    Image added to clustering list (deskew_image_np).")
                return (deskew_image_np)
            else:
                print("    Image processing failed. Skipping this image.")
                return None  # Skip if processing failed

        elif l2_label == 1 and l1_label in [1, 2]:
            # For label 1 and 2, add the original image
            print(f"    Image added to clustering list (image_np) for label {l1_label}.")
            return (image_np)

        else:
            print("    l2_label is not 1. Skipping this image.")
            return None  # Skip this image if l2_label != 1
    else:
        print("    l1_label not in [0, 1, 2]. Skipping this image.")
        return None  # Skip labels 3 and above
