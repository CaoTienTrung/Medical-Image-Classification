import numpy as np
import pickle
import os
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from tqdm import tqdm

from FeatureExtractors.feature_extractor import *


class SVMClassifier:
    def __init__(
        self,
        C=1,
        kernel="rbf",
        degree=3,
        random_state=42,
        feature_extractor=HOGFeatureExtractor(),
        model_path="svm_model.pkl",
    ):
        self.model = SVC(C=C, kernel=kernel, degree=degree, random_state=random_state)
        self.feature_extractor = feature_extractor
        self.model_path = model_path

    def train(self, dataset):
        X, y = [], []
        for img, label in dataset:
            img_np = img.numpy().transpose(1, 2, 0)
            feat = self.feature_extractor.extract(img_np)
            X.append(feat)
            y.append(label)
        X = np.array(X)
        y = np.array(y)
        self.model.fit(X, y)

        with open(self.model_path, "wb") as f:
            pickle.dump(self.model, f)
        print(f"[INFO] Model saved to {self.model_path}")

    def predict(self, dataset):
        with open(self.model_path, "rb") as f:
            self.model = pickle.load(f)
        print(f"[INFO] Model loaded from {self.model_path}")

        X, y = [], []
        for img, label in dataset:
            img_np = img.numpy().transpose(1, 2, 0)
            feat = self.feature_extractor.extract(img_np)
            X.append(feat)
            y.append(label)
        X = np.array(X)
        y = np.array(y)
        y_pred = self.model.predict(X)
        metrics = None
        metrics = {
            "accuracy": accuracy_score(y, y_pred),
            "precision": precision_score(y, y_pred, average="macro"),
            "recall": recall_score(y, y_pred, average="macro"),
            "f1": f1_score(y, y_pred, average="macro"),
        }
        return y_pred, metrics


class KNNClassifier:
    def __init__(self, n_neighbors=5, weights="uniform", feature_extractor=None):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.feature_extractor = feature_extractor
        # metric is defaulted to euclidean
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights)

    def train(self, train_data):
        X_train, y_train = [], []
        for img, label in train_data:
            features = self.feature_extractor.extract(img.numpy().transpose(1, 2, 0))
            X_train.append(features)
            y_train.append(label)

        X_train = np.array(X_train)
        y_train = np.array(y_train)

        self.model.fit(X_train, y_train)

    def predict(self, test_data):
        X_test, y_true = [], []
        for img, label in test_data:
            features = self.feature_extractor.extract(img.numpy().transpose(1, 2, 0))
            X_test.append(features)
            y_true.append(label)

        X_test = np.array(X_test)
        y_true = np.array(y_true)

        y_pred = self.model.predict(X_test)

        metrics = {"accuracy": accuracy_score(y_true, y_pred)}

        return y_pred, metrics
