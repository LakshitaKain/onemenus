import torch

import cv2
import numpy as np

def generate_inference_matrix():

    return np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ], dtype=np.float32)

def perspective_transform(image, margin_ratio=0.1, fixed_position=(100, 100)):

    height, width = image.shape[:2]

    # Generate random transformation matrix
    matrix = generate_inference_matrix()

    # Calculate the bounding box of the transformed image
    corners = np.array([
        [0, 0, 1],
        [width, 0, 1],
        [width, height, 1],
        [0, height, 1]
    ]).T
    transformed_corners = matrix @ corners
    transformed_corners = transformed_corners[:2] / transformed_corners[2]

    min_x, min_y = np.min(transformed_corners, axis=1)
    max_x, max_y = np.max(transformed_corners, axis=1)

    transformed_width = max_x - min_x
    transformed_height = max_y - min_y

    # Calculate a dynamic margin based on the size of the transformed image
    margin_x = int(transformed_width * margin_ratio)
    margin_y = int(transformed_height * margin_ratio)

    # Ensure that the canvas size is large enough to fit the transformed image
    canvas_width = max(int(transformed_width + 2 * margin_x), width)
    canvas_height = max(int(transformed_height + 2 * margin_y), height)

    # Create a black canvas with the calculated dimensions
    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

    # Set a fixed position for the image within the canvas
    fixed_x, fixed_y = fixed_position

    # Apply the translation to the transformation matrix to position the image
    matrix[0, 2] = fixed_x - min_x  # Fixed x position
    matrix[1, 2] = fixed_y - min_y  # Fixed y position

    # Ensure the translation values don't go out of bounds (clamping)
    matrix[0, 2] = np.clip(matrix[0, 2], 0, canvas_width - width)
    matrix[1, 2] = np.clip(matrix[1, 2], 0, canvas_height - height)

    # Apply the perspective transformation onto the dynamically sized canvas
    transformed_image = cv2.warpPerspective(image, matrix, (canvas_width, canvas_height), dst=canvas, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

    return transformed_image, matrix

def get_transformed_image_bounds(image: np.ndarray, matrix: np.ndarray):
    height, width = image.shape[:2]

    # Define the four corners of the original image
    corners = np.array([
        [0, 0],           # Top-left
        [width, 0],       # Top-right
        [width, height],  # Bottom-right
        [0, height]       # Bottom-left
    ], dtype='float32')

    # Convert to homogeneous coordinates for the perspective transformation
    ones = np.ones((4, 1))
    corners_homogeneous = np.hstack([corners, ones])

    # Apply the transformation matrix to the corners
    transformed_corners = matrix @ corners_homogeneous.T
    transformed_corners /= transformed_corners[2]  # Normalize to convert back from homogeneous coords

    # Get the min and max x, y coordinates to determine the bounding box
    min_x, max_x = transformed_corners[0].min(), transformed_corners[0].max()
    min_y, max_y = transformed_corners[1].min(), transformed_corners[1].max()

    return min_x, max_x, min_y, max_y

def adjust_canvas_and_transform(image: np.ndarray, matrix: np.ndarray):

    # Get the bounding box of the transformed image
    min_x, max_x, min_y, max_y = get_transformed_image_bounds(image, matrix)

    # Calculate the size of the new canvas
    new_width = int(np.ceil(max_x - min_x))
    new_height = int(np.ceil(max_y - min_y))

    # Adjust the transformation matrix to shift the image into the visible area
    translation_matrix = np.array([[1, 0, -min_x], 
                                   [0, 1, -min_y], 
                                   [0, 0, 1]])

    adjusted_matrix = translation_matrix @ matrix

    # Apply the adjusted transformation
    transformed_image = cv2.warpPerspective(image, adjusted_matrix, (new_width, new_height))

    return transformed_image

def denormalise(all_matrix_values):
    denormalised_matrix = []
    for mat_vals in all_matrix_values:
        mat_vals[0] /= 100
        mat_vals[1] /= 1000
        mat_vals[2] /= 1
        mat_vals[3] /= 1000
        mat_vals[4] /= 100
        mat_vals[5] /= 1
        mat_vals[6] /= 1e6
        mat_vals[7] /= 1e6
        mat_vals[8] /= 100
        
    denormalised_matrix.append(mat_vals)
    
    return torch.tensor(denormalised_matrix).squeeze()