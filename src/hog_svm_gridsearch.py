import numpy as np
from sklearn.model_selection import GridSearchCV
import sys
import os
from sklearn.metrics import accuracy_score
import joblib
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.FeatureExtractors.feature_extractor import HOGFeatureExtractor
from src.models.models import SVMClassifier
from src.custom_dataset import CustomImageDataset


if __name__ == "__main__":
    print("Loading datasets...")
    data_directory = "../Dataset/Data"
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

    param_grid = {
        "feature_extractor__params__orientations": [8, 9, 10],
        "feature_extractor__params__pixels_per_cell": [(8, 8), (16, 16)],
        "feature_extractor__params__cells_per_block": [(2, 2), (3, 3)],
        "C": [0.1, 1, 10],
        "kernel": ["rbf", "linear"],
        "degree": [2, 3, 4],
    }

    base_model = SVMClassifier()

    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=5,
        n_jobs=-1,
        verbose=2,
        scoring="accuracy",
    )

    print("\nPreparing training data...")
    X_train, y_train = [], []
    for img, label in tqdm(train_data, desc="Processing training images"):
        img_np = img.numpy().transpose(1, 2, 0)
        X_train.append(img_np)
        y_train.append(label)
    X_train = np.array(X_train)
    y_train = np.array(y_train)

    print("\nStarting GridSearchCV...")
    grid_search.fit(X_train, y_train)

    print("\nBest parameters found:")
    print(grid_search.best_params_)
    print("\nBest cross-validation score:")
    print(grid_search.best_score_)

    print("\nPreparing test data...")
    X_test, y_test = [], []
    for img, label in tqdm(test_data, desc="Processing test images"):
        img_np = img.numpy().transpose(1, 2, 0)
        X_test.append(img_np)
        y_test.append(label)
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    print("\nEvaluating best model on test set...")
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    test_score = accuracy_score(y_test, y_pred)
    print("\nTest set score:", test_score)

    print("\nSaving results...")
    best_params = grid_search.best_params_
    with open("best_hog_svm_params.txt", "w") as f:
        for param, value in best_params.items():
            f.write(f"{param}: {value}\n")
    print("Best parameters saved to 'best_hog_svm_params.txt'")

    joblib.dump(best_model, "best_hog_svm_model.joblib")
    print("Best model saved as 'best_hog_svm_model.joblib'")
