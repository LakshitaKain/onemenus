import pandas as pd
import hashlib
import re
from tqdm import tqdm
import time
from datetime import datetime
from pytz import timezone
from config import Config

tqdm.pandas()

def preprocess_data(data):
    start_time = time.time()
    
    data = data.explode('menu_images')
    data.dropna(subset=['menu_images'], inplace=True)
    data.drop_duplicates(subset='menu_images', inplace=True)
    data.reset_index(drop=True, inplace=True)

    data['bus_ind'] = data['_id'].astype('str')
    data['bus_ind'] = data['bus_ind'] + "_" + data.groupby('name').cumcount().add(1).astype(str)

    # Apply tqdm to track progress in these operations
    data['review_img_big'] = data['menu_images'].progress_apply(lambda x: x['menu_img'])
    data['url_hash'] = data['review_img_big'].progress_apply(lambda x: hashlib.md5(x.encode()).hexdigest())
    data['review_img_big'] = data['review_img_big'].progress_apply(lambda x: x.split('=')[0])
    data['review_img_big'] = data['review_img_big'].progress_apply(lambda x: x + "=w2048-h2048")
    data['img_date'] = data['menu_images'].progress_apply(lambda x: x['img_date'])

    # Using list comprehension with tqdm progress tracking
    data['source'] = [
        'menu' if isinstance(menu_image['source'], str) and re.search(r'menu', menu_image['source'], re.IGNORECASE) 
        else menu_image['source'] 
        for menu_image in tqdm(data['menu_images'], desc="Processing sources")
    ]

    data['label'] = data['source'].progress_apply(lambda x: 'menu' if x == 'menu' else 'non-menu')
    data['scores'] = data['source'].progress_apply(lambda x: 1 if x == 'menu' else 0)
    data['mnm'] = data['source'].progress_apply(lambda x: 'menu' if x == 'menu' else 'non-menu')   
    data['wasabi_url'] = Config.S3_BASE_URL + data['url_hash'] + ".jpg"
        
    end_time = time.time()
    time_taken_mnm = (end_time - start_time) / 3600
    
    return data, time_taken_mnm