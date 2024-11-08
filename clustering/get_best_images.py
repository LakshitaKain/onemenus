def calculate_composite_quality(word_count, avg_confidence, weight_word=1.0, weight_conf=0.5):
 
    # return (weight_word * word_count) + (weight_conf * avg_confidence * 100)  
    return word_count

def select_best_images_from_clusters(clusters, images, confidences, word_counts):
    best_images = []
    best_image_clusters = {}  # To store the cluster label of each best image

    for cluster_label, image_meta_data in clusters.items():
        image_quality_scores = {}
        for (image_name, image) in image_meta_data:
            # Calculate quality score for each image
            avg_confidence = confidences.get(image_name, 0)
            word_count = word_counts.get(image_name, 0)
            quality_score = calculate_composite_quality(word_count, avg_confidence)
            image_quality_scores[image_name] = quality_score

        # Identify the best image based on the highest quality score
        best_image = max(image_quality_scores, key=image_quality_scores.get)
        best_images.append((images[best_image], best_image))
        
        best_image_clusters[best_image] = cluster_label  # Map best image to its cluster label

    return best_images