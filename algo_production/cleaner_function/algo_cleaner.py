import re
import json
from jsonpath_ng.ext import parse
price_jsonpath_expr = parse('$..Price')
name_jsonpath_expr = parse('$..Name')

def remove_unnecessary_characters(dish_name):
    if not isinstance(dish_name, str):
        return dish_name
    dish_name = re.sub(r'^[^\w\s]+', '', dish_name)

    
    tokens = dish_name.split()

    
    new_tokens = []
    for index, token in enumerate(tokens):
        letters = re.findall(r'[A-Za-z]', token)
        digits = re.findall(r'\d', token)
        if len(letters) >= 2 and not digits:
            
            new_tokens = tokens[index:]
            break
    else:
        
        return ''

    
    dish_name = ' '.join(new_tokens)

    
    dish_name = dish_name.rstrip('_')

    
    dish_name = dish_name.rstrip('sS').strip()

    return dish_name.strip()

def correct_dish_name_in_json(json_data):
    for match in name_jsonpath_expr.find(json_data):
        dish_name = match.value
        if isinstance(dish_name, str):
            corrected_name = remove_unnecessary_characters(dish_name)
            match.context.value[match.path.fields[0]] = corrected_name

def correct_dish_name_in_json_v2(json_data):

    price_at_end_pattern = re.compile(
        r'[:\s]*'  # Optional colon or spaces
        r'(?:[Ss]?\d{1,3}(?:,\d{3})*(?:\.\d{2})?\s*)+'  
        r'[Ss]?$',  
        re.VERBOSE
    )

    for item in json_data:
        dish_name = item.get("Name", "")
        existing_price = item.get("Price", "").strip()
        prices = []

        if isinstance(dish_name, str):
            
            cleaned_dish_name = remove_unnecessary_characters(dish_name)

    
            match_number_comma = re.search(r'(\d+),$', cleaned_dish_name)
            if match_number_comma:
                
                number = match_number_comma.group(1)
                
                dish_name = cleaned_dish_name[:match_number_comma.start()].strip()
                
                existing_price_numeric = existing_price.strip('$')
                new_price = f'${number},{existing_price_numeric}'
                item["Price"] = new_price
                
                dish_name = dish_name.rstrip('sS').strip()
            
                dish_name = remove_unnecessary_characters(dish_name)
                item["Name"] = dish_name
                continue  
            else:
                
                match = re.search(price_at_end_pattern, cleaned_dish_name)
                if match:
                    prices_str = match.group()
                    
                    dish_name = cleaned_dish_name[:match.start()].strip()
                    
                    price_matches = re.findall(r'[Ss]?\d{1,3}(?:,\d{3})*(?:\.\d{2})?', prices_str)
                    for price_match in price_matches:
                        
                        price_numeric = price_match.replace(',', '')
                        standardized_price = re.sub(r'^[Ss]', '$', price_numeric)
                        if not standardized_price.startswith('$'):
                            standardized_price = f'${standardized_price}'
                        prices.append(standardized_price)
                    
                    dish_name = dish_name.rstrip('sS').strip()
                    dish_name = ' '.join(dish_name.split())
                    dish_name = remove_unnecessary_characters(dish_name)
                    item["Name"] = dish_name

            
                    if existing_price:
                        if not existing_price.startswith('$'):
                            existing_price = f'${existing_price}'
                        if existing_price not in prices:
                            prices.append(existing_price)

                    
                    unique_prices = list(dict.fromkeys(prices))
                    if unique_prices:
                        item["Price"] = ' | '.join(unique_prices)
                    else:
                        item["Price"] = existing_price if existing_price else item.pop("Price", None)
                else:
                    
                    if existing_price and not existing_price.startswith('$'):
                        existing_price = f'${existing_price}'
                        item["Price"] = existing_price
                    elif not existing_price:
                        item.pop("Price", None)
                
                    dish_name = cleaned_dish_name.rstrip('sS').strip()
                    dish_name = ' '.join(dish_name.split())
                    dish_name = remove_unnecessary_characters(dish_name)
                    item["Name"] = dish_name
        else:
            pass

def correct_price_in_json(json_data):
    pass


