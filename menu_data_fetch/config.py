from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

class Config:
    MONGO_URI = os.getenv("MONGO_URI")
    DB_NAME_TOPMENUS = os.getenv("DB_NAME_TOPMENUS")
    COLLECTION_NAME_TOPMENUS = os.getenv("COLLECTION_NAME_TOPMENUS")
    DB_NAME_PRODUCT = os.getenv("DB_NAME_PRODUCT")
    COLLECTION_NAME_PRODUCT = os.getenv("COLLECTION_NAME_PRODUCT")
    S3_BASE_URL = os.getenv("S3_BASE_URL")
