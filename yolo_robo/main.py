import cv2
from image_processor import process_image_in_memory

def yolo_bg(image_array):
    try:
        # Process the image
        processed_image = process_image_in_memory(image_array)
        
        # Save the processed image if needed
        # cv2.imwrite("processed_image.jpg", processed_image)
        
        # Convert the image from BGR to RGB format
        result_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)

        return result_image

    except Exception as e:
        print(f"Error: {e}")

# Test the function with a sample image
# img = cv2.imread("a3afe960d6f2ee1eef8911d9069dad98.jpg")
# yolo_bg(img)


