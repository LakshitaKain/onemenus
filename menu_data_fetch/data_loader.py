import pymongo
import pandas as pd
from config import Config

def load_data(google_ids):
    conn = pymongo.MongoClient(Config.MONGO_URI)
    db = conn[Config.DB_NAME_TOPMENUS]
    collection = db[Config.COLLECTION_NAME_TOPMENUS]
    data = pd.DataFrame(collection.find({'google_id': {"$in": google_ids}}))
    return data
