import requests
from io import BytesIO
from PIL import Image
import numpy as np

def url_to_ndarray(url):
    """
    Converts an image URL to a NumPy ndarray.
    
    Parameters:
        url (str): The URL of the image to fetch and convert.
    
    Returns:
        np.ndarray or None: The image as a NumPy array if successful, else None.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()  # Check for HTTP errors
        img = Image.open(BytesIO(response.content)).convert('RGB')
        return np.array(img)
    except Exception as e:
        print(f"Error fetching image from {url}: {e}")
        return None
