import cv2
from yolo.scripts.inference_new import yolo_segment
from classifier.main_classifier import process_segmented_images
from algo.algo_1 import perform_ocr_with_craft_1
from algo.algo_2 import perform_ocr_with_craft_2
from algo.algo_3 import perform_ocr_with_craft_3
from algo.gpt import perform_gpt
from cleaner_function.algo_cleaner import correct_dish_name_in_json
from cleaner_function.algo_cleaner import correct_dish_name_in_json_v2
from cleaner_function.algo_cleaner import correct_price_in_json
import openai

def main(image_path):
    image = cv2.imread(image_path)
    segmented_images = yolo_segment(image)
    results = process_segmented_images(segmented_images)
    results_dict = {}

    for result in results:
        image_path = result['image_p']
        predicted_class = result['predicted_label']
        id_label = result['id']
        confidence_score = result['confidence_score']
        entry = [predicted_class, confidence_score]

        try:
            if predicted_class == 0:
                ocr_result = perform_ocr_with_craft_1(image_path, resize_factor=1.0, dpi=300, use_gpu=False)
            elif predicted_class == 1:
                ocr_result = perform_ocr_with_craft_2(image_path, resize_factor=1.0, dpi=300, use_gpu=False)
            elif predicted_class == 2:
                ocr_result = perform_ocr_with_craft_3(image_path, resize_factor=1.0, dpi=300, use_gpu=False)
            elif predicted_class == 3:
                ocr_result = perform_gpt_with_handling(image_path)
            entry.append(ocr_result)
        except openai.error.RateLimitError as e:
            print(f"Rate limit error: {e}")
            entry.append({"error": "Rate limit exceeded"})
        except Exception as e:
            print(f"An error occurred: {e}")
            entry.append({"error": str(e)})

        results_dict[id_label] = entry

    final_data = []

    for value in results_dict.values():
        predicted_class = value[0]
        json_data = value[2]  

        if predicted_class in [0, 1]:
            correct_dish_name_in_json(json_data)
            correct_dish_name_in_json_v2(json_data)
            correct_price_in_json(json_data)
            final_data.extend(json_data)
        elif predicted_class == 2:
            final_data.extend(json_data)
        elif predicted_class == 3:
            correct_dish_name_in_json(json_data)
            final_data.extend(json_data)        

    # print(final_data)

def perform_gpt_with_handling(image_path):
    try:
        return perform_gpt(image_path)
    except openai.error.RateLimitError as e:
        print(f"OpenAI RateLimitError: {e}")
        return {"error": "Rate limit exceeded"}
    except Exception as e:
        print(f"Error in GPT processing: {e}")
        return {"error": str(e)}

if __name__ == "__main__":
    # image_path = r"C:\Users\sunny\new_pro\72342548b29bb77aca16074cae952fa5.jpg"
    main(image_path)
