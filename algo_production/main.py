import cv2
from yolo.scripts.inference_new import yolo_segment
from classifier.main_classifier import process_segmented_images
from algo.algo_1 import perform_ocr_with_craft_1
from algo.algo_2 import perform_ocr_with_craft_2
from algo.algo_3 import perform_ocr_with_craft_3
from algo.gpt import perform_gpt

def main(image):
    #image = cv2.imread(image_path)
    segmented_images = yolo_segment(image)
    results = process_segmented_images(segmented_images)
    results_dict = {}

    for result in results:
        image_path = result['image_p']
        predicted_class = result['predicted_label']
        id_label = result['id']
        confidence_score = result['confidence_score']
        entry = [predicted_class, confidence_score]

        if predicted_class == 0:
            ocr_result = perform_ocr_with_craft_1(image_path, resize_factor=1.0, dpi=300, use_gpu=False)
            entry.append(ocr_result)
        elif predicted_class == 1:
            ocr_result = perform_ocr_with_craft_2(image_path, resize_factor=1.0, dpi=300, use_gpu=False)
            entry.append(ocr_result)
        elif predicted_class == 2:
            ocr_result = perform_ocr_with_craft_3(image_path, resize_factor=1.0, dpi=300, use_gpu=False)
            entry.append(ocr_result)
        elif predicted_class == 3:
            ocr_result = perform_gpt(image_path)
            entry.append(ocr_result)

        results_dict[id_label] = entry

    # print(results_dict)

if __name__ == "__main__":
    # image_path = r"C:\Users\sunny\new_pro\last_production\image_1405_segment_1.jpg"
    main(image)
