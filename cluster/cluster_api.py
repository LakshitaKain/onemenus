from fastapi import FastAPI, HTTPException, Query
from fastapi import APIRouter
import requests
import cv2
import pytesseract
import re
from nltk.corpus import stopwords
import os
from sentence_transformers import SentenceTransformer, util
import numpy as np

app = FastAPI()
router = APIRouter()

# This endpoint will handle list of image URLs
@router.post("/process_images/")
def process_images(image_urls: list[str]):
    cleaned_texts = []

    for url in image_urls:
        # Step 1: Download the image
        response = requests.get(url)

        # Check if the request was successful
        if response.status_code == 200:
            # Write the response content to a temporary file
            with open("temp_image.jpg", "wb") as f:
                f.write(response.content)

            # Step 2: Read the image using OpenCV
            image = cv2.imread("temp_image.jpg")

            # Step 3: Use pytesseract to perform OCR on the image
            text = pytesseract.image_to_string(image).lower()

            # Step 4: Remove digits and special characters
            text = re.sub(r'[^a-zA-Z\s]', '', text)  # Keep only letters and whitespace

            # Step 5: Remove stop words
            stop_words = set(stopwords.words('english'))
            words = text.split()

            filtered_text = [word for word in words if word.lower() not in stop_words and len(word) > 2]

            filtered_text = ' '.join(filtered_text)
            num_words = len(filtered_text.split())
            cleaned_texts.append((filtered_text, num_words))

            # Clean up: remove the temporary file
            os.remove("temp_image.jpg")
        else:
            print(f"Error: Unable to download image from {url}, status code {response.status_code}.")

    # Initialize the SentenceTransformer model
    model_sentence = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

    # Generate embeddings for the cleaned texts
    embeddings = model_sentence.encode([text[0] for text in cleaned_texts])

    # Create a similarity matrix
    similarity_matrix = util.pytorch_cos_sim(embeddings, embeddings)

    # Set a threshold
    threshold = 0.70

    # Dictionary to hold clusters
    clusters = {}
    visited = set()

    for i in range(len(cleaned_texts)):
        if i in visited:
            continue

        cluster_name = f"Cluster_{len(clusters) + 1}"
        clusters[cluster_name] = [image_urls[i]]
        visited.add(i)

        for j in range(i + 1, len(cleaned_texts)):
            if similarity_matrix[i][j] >= threshold:
                clusters[cluster_name].append(image_urls[j])
                visited.add(j)

    final_output = []

    for cluster_name, urls in clusters.items():
        indices = [image_urls.index(url) for url in urls]

        if len(urls) > 1:
            best_index = max(indices, key=lambda index: cleaned_texts[index][1])  # index with max number of words
            best_url = image_urls[best_index]
        else:
            best_url = urls[0]

        output_entry = {
            'cluster_id': cluster_name,
            'best_url': best_url
        }

        final_output.append(output_entry)

    return final_output


app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=2237)  # Adjust the host and port as needed.

# from fastapi import FastAPI, HTTPException, Query
# from fastapi import APIRouter
# from pydantic import BaseModel, HttpUrl
# import requests
# import cv2
# import pytesseract
# import re
# from nltk.corpus import stopwords
# import os
# from sentence_transformers import SentenceTransformer, util
# import numpy as np

# app = FastAPI()
# router = APIRouter()

# # Define Pydantic model for request body
# class ImageURLRequest(BaseModel):
#     image_urls: list[HttpUrl]

# # This endpoint will handle list of image URLs
# @router.post("/process_images/")
# def process_images(request: ImageURLRequest):
#     cleaned_texts = []
#     image_urls = request.image_urls

#     for url in image_urls:
#         # Step 1: Download the image
#         response = requests.get(url)

#         # Check if the request was successful
#         if response.status_code == 200:
#             # Write the response content to a temporary file
#             with open("temp_image.jpg", "wb") as f:
#                 f.write(response.content)

#             # Step 2: Read the image using OpenCV
#             image = cv2.imread("temp_image.jpg")

#             # Step 3: Use pytesseract to perform OCR on the image
#             text = pytesseract.image_to_string(image).lower()

#             # Step 4: Remove digits and special characters
#             text = re.sub(r'[^a-zA-Z\s]', '', text)  # Keep only letters and whitespace

#             # Step 5: Remove stop words
#             stop_words = set(stopwords.words('english'))
#             words = text.split()

#             filtered_text = [word for word in words if word.lower() not in stop_words and len(word) > 2]

#             filtered_text = ' '.join(filtered_text)
#             num_words = len(filtered_text.split())
#             cleaned_texts.append((filtered_text, num_words))

#             # Clean up: remove the temporary file
#             os.remove("temp_image.jpg")
#         else:
#             print(f"Error: Unable to download image from {url}, status code {response.status_code}.")

#     # Initialize the SentenceTransformer model
#     model_sentence = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

#     # Generate embeddings for the cleaned texts
#     embeddings = model_sentence.encode([text[0] for text in cleaned_texts])

#     # Create a similarity matrix
#     similarity_matrix = util.pytorch_cos_sim(embeddings, embeddings)

#     # Set a threshold
#     threshold = 0.70

#     # Dictionary to hold clusters
#     clusters = {}
#     visited = set()

#     for i in range(len(cleaned_texts)):
#         if i in visited:
#             continue

#         cluster_name = f"Cluster_{len(clusters) + 1}"
#         clusters[cluster_name] = [image_urls[i]]
#         visited.add(i)

#         for j in range(i + 1, len(cleaned_texts)):
#             if similarity_matrix[i][j] >= threshold:
#                 clusters[cluster_name].append(image_urls[j])
#                 visited.add(j)

#     final_output = []

#     for cluster_name, urls in clusters.items():
#         indices = [image_urls.index(url) for url in urls]

#         if len(urls) > 1:
#             best_index = max(indices, key=lambda index: cleaned_texts[index][1])  # index with max number of words
#             best_url = image_urls[best_index]
#         else:
#             best_url = urls[0]

#         output_entry = {
#             'cluster_id': cluster_name,
#             'best_url': best_url
#         }

#         final_output.append(output_entry)

#     return final_output


# app.include_router(router)

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=2237)  # Adjust the host and port as needed.
