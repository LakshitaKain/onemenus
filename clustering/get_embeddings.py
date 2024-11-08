from sklearn.feature_extraction.text import TfidfVectorizer

from preprocessing import preprocess_text

def create_tfidf_matrix(texts, image_names):
    corpus = [' '.join(preprocess_text(texts[image_name])) for image_name in image_names]
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),   
        min_df=2,             # Ignore terms that appear in less than 2 documents
        # max_df=0.9,         # Ignore terms that appear in more than 80% of the documents
        # max_features=1000     # Limit to top x features
    )
    tfidf_matrix = vectorizer.fit_transform(corpus)
    return tfidf_matrix, vectorizer