import pymongo
from config import Config

def mongo_cluster_analysis(data):
    conn = pymongo.MongoClient(Config.MONGO_URI)
    db = conn[Config.DB_NAME_PRODUCT]
    collection = db[Config.COLLECTION_NAME_PRODUCT]
    collection.insert_many(data)



