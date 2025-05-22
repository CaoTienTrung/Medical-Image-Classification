import numpy as np
import sys
import os
from sklearn.metrics import accuracy_score
import joblib
from tqdm import tqdm
from itertools import product

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.FeatureExtractors.feature_extractor import HOGFeatureExtractor
from src.models.models import KNNClassifier
from src.custom_dataset import CustomImageDataset

import warnings

warnings.filterwarnings("ignore")


if __name__ == "__main__":
    print("Loading datasets...")
    data_directory = "Dataset/Data"
    train_data = CustomImageDataset(
        directory=os.path.join(data_directory, "train"),
        label_mode="int",
        color_mode="grayscale",
        image_size=(224, 224),
        interpolation="bilinear",
    )
    test_data = CustomImageDataset(
        directory=os.path.join(data_directory, "test"),
        label_mode="int",
        color_mode="grayscale",
        image_size=(224, 224),
        interpolation="bilinear",
    )

    print("\nPreparing training data...")
    X_train, y_train = [], []
    for img, label in tqdm(train_data, desc="Processing training images"):
        img_np = img.numpy().transpose(1, 2, 0)
        X_train.append(img_np)
        y_train.append(label)
    X_train = np.array(X_train)
    y_train = np.array(y_train)

    # Define parameter grid
    hog_params = {
        "orientations": [8, 9, 10],
        "pixels_per_cell": [(8, 8), (16, 16)],
        "cells_per_block": [(2, 2), (3, 3)],
    }
    knn_params = {
        "n_neighbors": [3, 5, 7, 9],
        "weights": ["uniform", "distance"],
        "metric": ["euclidean", "manhattan", "minkowski"],
    }

    # Generate all parameter combinations
    param_combinations = []
    for hog_vals in product(*hog_params.values()):
        hog_dict = dict(zip(hog_params.keys(), hog_vals))
        for knn_vals in product(*knn_params.values()):
            knn_dict = dict(zip(knn_params.keys(), knn_vals))
            param_combinations.append((hog_dict, knn_dict))

    print(f"\nTotal parameter combinations to try: {len(param_combinations)}")

    best_score = 0
    best_params = None
    best_model = None

    # Manual grid search
    for i, (hog_dict, knn_dict) in enumerate(
        tqdm(param_combinations, desc="Grid Search Progress")
    ):
        # Create feature extractor with current HOG parameters
        feature_extractor = HOGFeatureExtractor()
        feature_extractor.params.update(hog_dict)

        # Create KNN classifier with current parameters
        model = KNNClassifier(
            n_neighbors=knn_dict["n_neighbors"],
            weights=knn_dict["weights"],
            feature_extractor=feature_extractor,
        )

        # Train and evaluate
        model.train(train_data)
        y_pred, metrics = model.predict(test_data)

        # Update best parameters if current model is better
        if metrics["accuracy"] > best_score:
            best_score = metrics["accuracy"]
            best_params = {**hog_dict, **knn_dict}
            best_model = model

        print(f"\nCombination {i+1}/{len(param_combinations)}")
        print(f"Parameters: {hog_dict}, {knn_dict}")
        print(f"Accuracy: {metrics['accuracy']:.4f}")

    print("\nBest parameters found:")
    print(best_params)
    print("\nBest accuracy:", best_score)

    # Save best parameters
    print("\nSaving results...")
    with open("best_hog_knn_params.txt", "w") as f:
        for param, value in best_params.items():
            f.write(f"{param}: {value}\n")
    print("Best parameters saved to 'best_hog_knn_params.txt'")

    # Save best model
    joblib.dump(best_model, "best_hog_knn_model.joblib")
    print("Best model saved as 'best_hog_knn_model.joblib'")
