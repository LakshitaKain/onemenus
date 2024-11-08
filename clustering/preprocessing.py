from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import re
from nltk.tokenize import word_tokenize

stemmer = PorterStemmer()
stop_words = stopwords.words("english")

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and numbers
    text = re.sub(r'[^a-z\s]', '', text)
    # Tokenize
    words = word_tokenize(text)
    # Remove stopwords and apply stemming
    filtered_words = [stemmer.stem(word) for word in words if word not in stop_words]
    return filtered_words
