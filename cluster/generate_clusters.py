from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def are_all_images_unique(tfidf_matrix, threshold=0.7):

    similarity_matrix = cosine_similarity(tfidf_matrix)
    np.fill_diagonal(similarity_matrix, 0)  # Exclude self-similarity
    max_similarities = similarity_matrix.max(axis=1)
    return np.all(max_similarities < threshold)

def get_clusters_using_kmeans(tfidf_matrix, images, max_k=20, similarity_threshold=0.7):

    # Check if all images are unique
    unique = are_all_images_unique(tfidf_matrix, threshold=similarity_threshold)
    
    if unique:
        print("All images are unique. Assigning each image to its own cluster.")
        clusters = {i: [image] for i, image in enumerate(images)}
    else:
        silhouettes = []
        K = range(2, min(max_k, len(images)))
        for k in K:
            kmeans = KMeans(n_clusters=k, random_state=42)
            labels = kmeans.fit_predict(tfidf_matrix)
            silhouette_avg = silhouette_score(tfidf_matrix, labels)
            silhouettes.append(silhouette_avg)
        
        if silhouettes:
            best_k = K[np.argmax(silhouettes)]
        else:
            best_k = 2  # Default to 2 clusters if silhouette scores are unavailable
                
        # Perform K-Means with best_k
        kmeans = KMeans(n_clusters=best_k, random_state=42)
        labels = kmeans.fit_predict(tfidf_matrix)
        
        # Form clusters
        clusters = {}
        for image, label in zip(images, labels):
            clusters.setdefault(label, []).append((image, images[image]))
    
    return clusters

