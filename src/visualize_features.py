import os
import random
import numpy as np
import matplotlib.pyplot as plt
import cv2
from FeatureExtractors.feature_extractor import (
    HOGFeatureExtractor,
    LBPFeatureExtractor,
    GaborExtractor,
    SIFTFeatureExtractor,
)


def load_random_image():
    # Get all test directories
    test_dir = "Dataset/Data/test"
    categories = os.listdir(test_dir)

    # Randomly select a category
    category = random.choice(categories)
    category_path = os.path.join(test_dir, category)

    # Get all images in the selected category
    images = [
        f for f in os.listdir(category_path) if f.endswith((".png", ".jpg", ".jpeg"))
    ]

    # Randomly select an image
    image_name = random.choice(images)
    image_path = os.path.join(category_path, image_name)

    # Read the image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img, image_name


def visualize_features():
    # Load a random image
    img, img_name = load_random_image()

    # Initialize feature extractors
    hog_extractor = HOGFeatureExtractor()
    lbp_extractor = LBPFeatureExtractor()
    gabor_extractor = GaborExtractor()
    sift_extractor = SIFTFeatureExtractor()

    # Convert image to grayscale for feature extraction
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Extract features
    hog_features = hog_extractor.extract(gray_img)
    lbp_features = lbp_extractor.extract(gray_img)
    gabor_features = gabor_extractor.extract(gray_img)
    sift_features = sift_extractor.extract(gray_img)

    # Create visualization
    plt.figure(figsize=(15, 10))

    # Original image
    plt.subplot(2, 3, 1)
    plt.imshow(img)
    plt.title("Original Image")
    plt.axis("off")

    # HOG visualization
    plt.subplot(2, 3, 2)
    plt.plot(hog_features)
    plt.title("HOG Features")

    # LBP visualization
    plt.subplot(2, 3, 3)
    plt.plot(lbp_features)
    plt.title("LBP Features")

    # Gabor visualization
    plt.subplot(2, 3, 4)
    plt.plot(gabor_features)
    plt.title("Gabor Features")

    # SIFT visualization
    plt.subplot(2, 3, 5)
    plt.plot(sift_features)
    plt.title("SIFT Features")

    plt.tight_layout()
    plt.savefig(f"feature_visualization_{img_name}.png")
    plt.close()


if __name__ == "__main__":
    visualize_features()
