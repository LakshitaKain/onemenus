import cv2
import base64
import openai
import json
import os

from dotenv import load_dotenv
load_dotenv()

openai.api_key = os.getenv("api_key")

def encode_image(image_path):
    return image_path.tobytes()
def perform_gpt(image_path):
    
        encoded_image = encode_image(image_path)
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text",
                         "text": """You are tasked with extracting specific features from a given menu image. Your goal is to analyze the image and extract the following information for each dish:
1. dish_name
2. dish_description
3. dish_price
4. menu_section
5. dish_quantity
6. dish_dietary_option
7. dish_calories
8. dish_add_ons
To complete this task, carefully examine the menu image provided:
<menu_image>
{{MENU_IMAGE}}
</menu_image>
Follow these steps to extract the required information:
1. Analyze the overall structure of the menu, identifying different sections if present.
2. For each dish in the menu:
   a. Identify and extract the dish name.
   b. Locate and extract the dish description, if available.
   c. Find and extract the dish price. Pay attention to any special symbols or notations used for prices (e.g., $, €, £). If a dish has multiple price options (e.g., small/large sizes, chicken/beef option ), include all options in the dish_price field, separated by a forward slash (/).
   d. Determine the menu section the dish belongs to, if applicable.
   e. Look for any quantity information (e.g., "serves 2") and extract it.
   f. Identify any dietary options (e.g., vegetarian, gluten-free) and extract them.
   g. If calorie information is provided, extract it.
   h. Look for any add-ons or customization options and extract them.
3. Organize the extracted information into a JSON dictionary format.
Guidelines for handling missing information:
- If any of the required information is not available or unclear for a particular dish, use "N/A" (Not Available) for that field.
- If the menu image does not contain information for a specific feature across all dishes, you may omit that key from the JSON dictionary.
Your output should be a JSON dictionary with the following structure:
<output_format>
{
  "dishes": [
    {
      "dish_name": "String",
      "dish_description": "String or null",
      "dish_price": "String or null",
      "menu_section": "String or null",
      "dish_quantity": "String or null",
      "dish_dietary_option": "String or null",
      "dish_calories": "String or null",
      "dish_add_ons": "String or null"
    },
    // Additional dishes...
  ]
}
</output_format>
Ensure that your output is properly formatted as a valid JSON dictionary. Include all extracted information, using null for any missing values. If you encounter any ambiguities or difficulties in extracting certain information, make your best judgment based on the available data in the image."""
                        },
                        {"type": "text", "text": "Output format: JSON"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{encoded_image}",
                            },
                        },
                    ],
                }
            ],
            temperature=0,
            max_tokens=4096
        )
       
        return response.choices[0].message.content
    