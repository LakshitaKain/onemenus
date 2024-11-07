import pandas as pd
from data_loader import load_data
from data_preprocessing import preprocess_data
from mongo_operations import mongo_cluster_analysis
import time
from datetime import datetime
from pytz import timezone

def main():
    with open('text_file20_05.txt', 'r') as file:
        google_id = [line.strip() for line in file]
    
    google_id = google_id[:]
    data = load_data(google_id)
    df, time_taken = preprocess_data(data)


    created_on = datetime.now(timezone("Asia/Kolkata"))
    df['created_on'] = created_on
    df = df[['name', 'google_id', 'category', 'description', 'img_url', 'address', 'lat', 'long', 'rating', 
             'total_reviews_count', 'plus_code', 'phone', 'popular_times', 'opening_hours', 'amenties', 
             'website', 'review_img_big','wasabi_url', 'img_date', 'bus_ind', 'label', 'scores', 'mnm', 
            'created_on', 'source']]

    # Insert into MongoDB
    json_data_df = df.to_dict(orient='records')
    mongo_cluster_analysis(json_data_df)

    # Removing duplicates from the data frame
    df_menu = df[df['mnm'] == 'menu'].drop_duplicates(subset='wasabi_url').reset_index(drop=True)
    
    # Save df_menu to CSV
    df_menu.to_csv("df_menu_new.csv", index=False)

    return df_menu

if __name__ == "__main__":
    df_menu = main()
