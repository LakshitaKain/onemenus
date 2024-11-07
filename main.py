import pandas as pd

# Importing functions from your modules
from menu_data_fetch.main import main as fetch_menu_data
from clip_l1.main import predict as clip1_predict
from clip_l2.main import predict_label as clip2_predict_label
from yolo_robo.main import yolo_bg
from orientation.main import deskew_and_orient_correction
from cluster.main import generate_best_images_from_google_id
from image_utils import url_to_ndarray

def main():
    # Step 1: Fetch the dataset
    df_menu = fetch_menu_data()
    
    # Ensure that the necessary columns exist
    if 'google_id' not in df_menu.columns or 'wasabi_url' not in df_menu.columns:
        print("Dataset must contain 'google_id' and 'wasabi_url' columns.")
        return
    
    # Group the dataframe by 'google_id' to process one google_id at a time
    grouped = df_menu.groupby('google_id')
    
    for google_id, group in grouped:
        print(f"Processing Google ID: {google_id}")
        
        # Initialize a list to collect images for clustering
        store_corrected_image_for_clustering = []
        
        # Iterate over each wasabi_url for the current google_id
        for idx, row in group.iterrows():
            url = row['wasabi_url']
            print(f"  Processing image URL: {url}")
            
            # Step 1: Convert URL to ndarray using the imported function
            image_np = url_to_ndarray(url)
            if image_np is None:
                print("    Skipping due to image loading error.")
                continue  # Skip this image if there's an error
            
            # Step 2: Apply predict from clip_l1.main
            l1_label = clip1_predict(image_np)
            print(f"    clip_l1.predict label: {l1_label}")
            
            # Check if label is 0, 1, or 2
            if l1_label in [0, 1, 2]:
                # Step 3: Apply predict_label from clip_l2.main
                l2_label = clip2_predict_label(image_np)
                print(f"    clip_l2.predict_label: {l2_label}")
                
                if l2_label == 1:
                    if l1_label == 0:
                        # Apply yolo_bg
                        yolo_image_np = yolo_bg(image_np)
                        if yolo_image_np is None:
                            print("    yolo_bg failed. Skipping this image.")
                            continue  # Skip if yolo_bg fails
                        print("    Applied yolo_bg.")
                        
                        # Apply process_image
                        deskew_image_np = deskew_and_orient_correction(yolo_image_np)
                        if deskew_image_np is None:
                            print("    process_image failed. Skipping this image.")
                            continue  # Skip if process_image fails
                        print("    Applied process_image.")
                        
                        # Add the processed image to the clustering list
                        store_corrected_image_for_clustering.append(deskew_image_np)
                        print("    Image added to clustering list (deskew_image_np).")
                    
                    elif l1_label in [1, 2]:
                        if l2_label == 1:                     
                        # Add the original image to the clustering list
                            store_corrected_image_for_clustering.append(image_np)
                            print(f"    Image added to clustering list (image_np) for label {l1_label}.")
                else:
                    print("    l2_label is 0. Skipping this image.")
                    continue  # Skip this image if l2_label != 1
            else:
                print("    l1_label not in [0, 1, 2]. Skipping this image.")
                continue  # Skip labels 3 and above
        
        # After processing all images for the current google_id
        if store_corrected_image_for_clustering:
            print(f"  Generating best images for Google ID: {google_id}")
            best_images = generate_best_images_from_google_id(store_corrected_image_for_clustering)
            print(f"    Generated {len(best_images)} best images.")
            
        else:
            print("  No images to cluster for this Google ID.")
        
        print(f"Finished processing Google ID: {google_id}\n")

if __name__ == "__main__":
    main()

