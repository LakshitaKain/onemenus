import cv2
import numpy as np
from skimage.filters import laplace, sobel, roberts

class BlurDetector:
    def __init__(self, threshold=0.006):
        self.threshold = threshold

    def compute_blur_score(self, image_np):
        image_gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

        # Compute features
        lap_feat = laplace(image_gray)
        sob_feat = sobel(image_gray)
        rob_feat = roberts(image_gray)

        # Calculate variances for Laplace, Sobel, and Roberts filters
        lap_var = float(lap_feat.var())
        sob_var = float(sob_feat.var())
        rob_var = float(rob_feat.var())

        # Compute total variance
        total_variance = lap_var + sob_var + rob_var

        # Classification based on threshold
        is_blurry = bool(total_variance <= self.threshold)

        return total_variance, is_blurry
