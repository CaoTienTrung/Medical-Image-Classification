import os
import random
import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.feature import local_binary_pattern, hog
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


def visualize_hog(img, hog_extractor):
    """Trực quan hóa HOG features."""
    # Sử dụng visualize=True để lấy ảnh HOG
    _, hog_image = hog(
        img,
        orientations=hog_extractor.params["orientations"],
        pixels_per_cell=hog_extractor.params["pixels_per_cell"],
        cells_per_block=hog_extractor.params["cells_per_block"],
        block_norm=hog_extractor.params["block_norm"],
        visualize=True,
        feature_vector=hog_extractor.params["feature_vector"],
    )
    # Chuẩn hóa ảnh HOG về [0, 255]
    hog_image = (
        (hog_image - hog_image.min())
        / (hog_image.max() - hog_image.min() + 1e-10)
        * 255
    )
    return hog_image.astype(np.uint8)


def visualize_lbp(img, lbp_extractor):
    """Trực quan hóa LBP features."""
    lbp = local_binary_pattern(
        img,
        P=lbp_extractor.params["P"],
        R=lbp_extractor.params["R"],
        method=lbp_extractor.params["method"],
    )
    lbp_image = (lbp - lbp.min()) / (lbp.max() - lbp.min() + 1e-10) * 255
    return lbp_image.astype(np.uint8)


def visualize_gabor(img, gabor_extractor):
    """Trực quan hóa Gabor features."""
    params = gabor_extractor.params
    kernel = cv2.getGaborKernel(
        ksize=params["ksize"],
        sigma=params["sigmas"][0],
        theta=params["thetas"][0],
        lambd=params["lambdas"][0],
        gamma=params["gamma"],
        psi=params["psi"],
        ktype=cv2.CV_32F,
    )
    gabor_image = cv2.filter2D(img, cv2.CV_32F, kernel)
    gabor_image = (
        (gabor_image - gabor_image.min())
        / (gabor_image.max() - gabor_image.min() + 1e-10)
        * 255
    )
    return gabor_image.astype(np.uint8)


def visualize_sift(img, sift_extractor):
    """Trực quan hóa SIFT keypoints."""
    keypoints, descriptors = sift_extractor.sift.detectAndCompute(img, None)
    sift_image = cv2.drawKeypoints(
        img, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
    return sift_image


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

    # Extract and visualize features
    hog_image = visualize_hog(gray_img, hog_extractor)
    lbp_image = visualize_lbp(gray_img, lbp_extractor)
    gabor_image = visualize_gabor(gray_img, gabor_extractor)
    sift_image = visualize_sift(gray_img, sift_extractor)

    # Create visualization
    plt.figure(figsize=(15, 10))

    # Original image
    plt.subplot(2, 3, 1)
    plt.imshow(img)
    plt.title("Original Image")
    plt.axis("off")

    # HOG visualization
    plt.subplot(2, 3, 2)
    plt.imshow(hog_image, cmap="gray")
    plt.title("HOG Features")
    plt.axis("off")

    # LBP visualization
    plt.subplot(2, 3, 3)
    plt.imshow(lbp_image, cmap="gray")
    plt.title("LBP Features")
    plt.axis("off")

    # Gabor visualization
    plt.subplot(2, 3, 4)
    plt.imshow(gabor_image, cmap="gray")
    plt.title("Gabor Features")
    plt.axis("off")

    # SIFT visualization
    plt.subplot(2, 3, 5)
    plt.imshow(sift_image, cmap="gray")
    plt.title("SIFT Keypoints")
    plt.axis("off")

    plt.tight_layout()
    output_path = f"feature_visualization_{img_name}.png"
    plt.savefig(output_path)
    plt.close()
    print(f"Đã lưu hình ảnh trực quan tại: {output_path}")


if __name__ == "__main__":
    visualize_features()
