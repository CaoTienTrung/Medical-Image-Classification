import os
import random
import numpy as np
import cv2
import matplotlib.pyplot as plt
from src.FeatureExtractors.feature_extractor import (
    HOGFeatureExtractor,
    LBPFeatureExtractor,
    GaborExtractor,
    SIFTFeatureExtractor,
)


def load_random_image(data_dir):
    """Tải ngẫu nhiên một ảnh từ thư mục."""
    image_files = [
        f for f in os.listdir(data_dir) if f.endswith((".png", ".jpg", ".jpeg"))
    ]
    if not image_files:
        raise ValueError("Không tìm thấy ảnh trong thư mục.")

    random_image_path = os.path.join(data_dir, random.choice(image_files))
    img = cv2.imread(random_image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Không thể tải ảnh: {random_image_path}")
    return img


def visualize_hog(img, hog_extractor):
    """Trực quan hóa HOG features."""
    hog_features = hog_extractor.extract(img)
    # HOG features là vector 1D, cần reshape lại để trực quan hóa
    # Tính số cell và tạo lưới hiển thị
    cell_size = hog_extractor.params["pixels_per_cell"][0]
    num_cells_x = img.shape[1] // cell_size
    num_cells_y = img.shape[0] // cell_size
    hog_image = hog_features.reshape((num_cells_y, num_cells_x, -1)).mean(axis=2)
    hog_image = (
        (hog_image - hog_image.min()) / (hog_image.max() - hog_image.min()) * 255
    )
    return hog_image.astype(np.uint8)


def visualize_lbp(img, lbp_extractor):
    """Trực quan hóa LBP features."""
    lbp = lbp_extractor.extract(img)
    # LBP trả về histogram, ta sử dụng ảnh LBP trực tiếp
    lbp_image = local_binary_pattern(
        img,
        P=lbp_extractor.params["P"],
        R=lbp_extractor.params["R"],
        method=lbp_extractor.params["method"],
    )
    lbp_image = (
        (lbp_image - lbp_image.min()) / (lbp_image.max() - lbp_image.min()) * 255
    )
    return lbp_image.astype(np.uint8)


def visualize_gabor(img, gabor_extractor):
    """Trực quan hóa Gabor features."""
    feats = gabor_extractor.extract(img)
    # Tạo một ảnh đại diện bằng cách áp dụng một kernel Gabor
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
        / (gabor_image.max() - gabor_image.min())
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
    # Đường dẫn đến thư mục chứa ảnh
    data_dir = "Dataset/Data/train"

    # Khởi tạo các bộ trích xuất đặc trưng
    hog_extractor = HOGFeatureExtractor()
    lbp_extractor = LBPFeatureExtractor()
    gabor_extractor = GaborExtractor()
    sift_extractor = SIFTFeatureExtractor(max_keypoints=500)

    # Tải ảnh ngẫu nhiên
    img = load_random_image(data_dir)

    # Trích xuất và trực quan hóa các đặc trưng
    hog_image = visualize_hog(img, hog_extractor)
    lbp_image = visualize_lbp(img, lbp_extractor)
    gabor_image = visualize_gabor(img, gabor_extractor)
    sift_image = visualize_sift(img, sift_extractor)

    # Tạo figure để hiển thị
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 3, 1)
    plt.imshow(img, cmap="gray")
    plt.title("Ảnh gốc")
    plt.axis("off")

    plt.subplot(2, 3, 2)
    plt.imshow(hog_image, cmap="gray")
    plt.title("HOG Features")
    plt.axis("off")

    plt.subplot(2, 3, 3)
    plt.imshow(lbp_image, cmap="gray")
    plt.title("LBP Features")
    plt.axis("off")

    plt.subplot(2, 3, 4)
    plt.imshow(gabor_image, cmap="gray")
    plt.title("Gabor Features")
    plt.axis("off")

    plt.subplot(2, 3, 5)
    plt.imshow(sift_image, cmap="gray")
    plt.title("SIFT Keypoints")
    plt.axis("off")

    plt.tight_layout()
    output_path = "feature_visualization.png"
    plt.savefig(output_path)
    plt.close()
    print(f"Đã lưu hình ảnh trực quan tại: {output_path}")


if __name__ == "__main__":
    visualize_features()
