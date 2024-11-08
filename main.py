import pandas as pd

# Importing functions from your modules
from menu_data_fetch.main import main as fetch_menu_data
from cluster.main import generate_best_images_from_google_id
from algo_production.main import main  # Assuming this is needed
from image_utils import url_to_ndarray
from process_labels import process_image_labels  # Import the new function

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

            # Use the extracted function to process labels and images
            result = process_image_labels(image_np, url)

            if result is not None:
                # Append the returned tuple to the clustering list
                store_corrected_image_for_clustering.append(result)
            else:
                # Image was skipped due to failed processing
                continue

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
