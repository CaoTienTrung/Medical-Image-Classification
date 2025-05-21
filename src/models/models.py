import numpy as np
import pickle
import os
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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
