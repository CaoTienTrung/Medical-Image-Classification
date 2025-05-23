import argparse
import yaml
from models import MIAFEx, SVMClassifier
from utils import chestCTforMIAFEx
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import torch.optim as optim
from utils import save_model, load_model_state
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description="Pipeline for ViT")
    parser.add_argument(
        "--config",
        type=str,
        default="mistral-7b-v1.0",
        help="Path to the config file",
    )

    return parser.parse_args()

def load_yaml(path):

    with open(path, 'r') as file:
        data = yaml.safe_load(file)
    return data

if __name__ == "__main__":
    args = parse_args()
    config = load_yaml(args.config)
    
    train_set = chestCTforMIAFEx(config["data"]["path"], "train", config["data"]["img_size"])
    dev_set = chestCTforMIAFEx(config["data"]["path"], "valid", config["data"]["img_size"])
    test_set = chestCTforMIAFEx(config["data"]["path"], "test", config["data"]["img_size"])
    
    train_loader = DataLoader(train_set, batch_size=config["data"]["batch_size"], shuffle=True)
    dev_loader = DataLoader(dev_set, batch_size=config["data"]["batch_size"], shuffle=False)
    test_loader = DataLoader(test_set, batch_size=config["data"]["batch_size"], shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    viT = MIAFEx(
        d_model = config["model"]["d_model"],
        encoder_layers= config["model"]["encoder_layers"],
        patch_size= config["model"]["patch_size"],
        num_classes= config["model"]["num_classes"],
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    num_epochs = config["model"]["epochs"]

    optimizer = optim.NAdam(viT.parameters(), lr=config["adam"]["lr"])

    model, optimizer = load_model_state(viT, optimizer, config["model"]["save_path"])


    model.eval()
    features_train, labels_train = model.features_from_loader(train_loader)
    features_dev, labels_dev = model.features_from_loader(dev_loader)

    # Concatenate train + dev
    features_train_dev = np.concatenate([features_train, features_dev], axis=0)
    labels_train_dev = np.concatenate([labels_train, labels_dev], axis=0)

    # Fit SVM on train + dev
    svm = SVMClassifier(C=1, kernel='rbf', degree=3, model_path=config["svm"]["model_path"])
    svm.model.fit(features_train_dev, labels_train_dev)

    # Evaluate on test set
    features_test, labels_test = model.features_from_loader(test_loader)
    y_pred = svm.model.predict(features_test)

    # Calculate metrics
    accuracy = accuracy_score(labels_test, y_pred)
    precision = precision_score(labels_test, y_pred, average='macro')
    recall = recall_score(labels_test, y_pred, average='macro')
    f1 = f1_score(labels_test, y_pred, average='macro')
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

