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

import numpy as np
import pickle
import os
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
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
            img_np = np.squeeze(img_np, axis=-1)
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
            img_np = np.squeeze(img_np, axis=-1)
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


class RFClassifier:
    def __init__(
        self,
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        criterion="gini",
        random_state=42,
        feature_extractor=HOGFeatureExtractor(),
        model_path="rf_model.pkl",
    ):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            criterion=criterion,
            random_state=random_state,
        )
        self.feature_extractor = feature_extractor
        self.model_path = model_path

    def train(self, dataset):
        X, y = [], []
        for img, label in dataset:
            img_np = img.numpy().transpose(1, 2, 0)
            img_np = np.squeeze(img_np, axis=-1)
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
            img_np = np.squeeze(img_np, axis=-1)
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


class LRClassifier:
    def __init__(
        self,
        C=1.0,
        solver="lbfgs",
        max_iter=1000,
        multi_class="multinomial",
        random_state=42,
        feature_extractor=HOGFeatureExtractor(),
        model_path="lr_model.pkl",
    ):
        self.model = LogisticRegression(
            C=C,
            solver=solver,
            max_iter=max_iter,
            multi_class=multi_class,
            random_state=random_state,
        )
        self.feature_extractor = feature_extractor
        self.model_path = model_path

    def train(self, dataset):
        X, y = [], []
        for img, label in dataset:
            img_np = img.numpy().transpose(1, 2, 0)
            img_np = np.squeeze(img_np, axis=-1)
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
            img_np = np.squeeze(img_np, axis=-1)
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


class XGBoostClassifier:
    def __init__(
        self,
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        random_state=42,
        objective="multi:softmax",
        num_class=4,
        feature_extractor=HOGFeatureExtractor(),
        model_path="xgb_model.pkl",
    ):
        self.model = XGBClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=random_state,
            objective=objective,
            num_class=num_class,
        )
        self.feature_extractor = feature_extractor
        self.model_path = model_path

    def train(self, dataset):
        X, y = [], []
        for img, label in dataset:
            img_np = img.numpy().transpose(1, 2, 0)
            img_np = np.squeeze(img_np, axis=-1)
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
            img_np = np.squeeze(img_np, axis=-1)
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
    def __init__(
        self,
        n_neighbors=5,
        weights="uniform",
        feature_extractor=None,
        model_path="knn_model.pkl",
    ):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.feature_extractor = feature_extractor
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights)
        self.model_path = model_path

    def train(self, dataset):
        X, y = [], []
        for img, label in tqdm(dataset, desc="Extracting features"):
            img_np = img.numpy().transpose(1, 2, 0)
            feat = self.feature_extractor.extract(img_np)
            X.append(feat)
            y.append(label)

        X = np.array(X)
        y = np.array(y)

        self.model.fit(X, y)

        # Save model
        with open(self.model_path, "wb") as f:
            pickle.dump(self.model, f)
        print(f"[INFO] Model saved to {self.model_path}")

    def predict(self, dataset):
        # Load model
        with open(self.model_path, "rb") as f:
            self.model = pickle.load(f)
        print(f"[INFO] Model loaded from {self.model_path}")

        X, y = [], []
        for img, label in tqdm(dataset, desc="Extracting features"):
            img_np = img.numpy().transpose(1, 2, 0)
            feat = self.feature_extractor.extract(img_np)
            X.append(feat)
            y.append(label)

        X = np.array(X)
        y = np.array(y)

        y_pred = self.model.predict(X)

        # Calculate metrics
        metrics = {
            "accuracy": accuracy_score(y, y_pred),
            "precision": precision_score(y, y_pred, average="macro"),
            "recall": recall_score(y, y_pred, average="macro"),
            "f1": f1_score(y, y_pred, average="macro"),
        }

        return y_pred, metrics
