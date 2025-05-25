import argparse
import yaml
from models import MIAFEx, SVMClassifier, LRClassifier, RFClassifier, XGBoostClassifier
from utils import chestCTforMIAFEx, chestCTforViT
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
    
    train_set = chestCTforViT(config["data"]["path"], "train", config["data"]["img_size"])
    dev_set = chestCTforViT(config["data"]["path"], "valid", config["data"]["img_size"])
    test_set = chestCTforViT(config["data"]["path"], "test", config["data"]["img_size"])
    
    train_loader = DataLoader(train_set, batch_size=config["data"]["batch_size"], shuffle=True)
    dev_loader = DataLoader(dev_set, batch_size=config["data"]["batch_size"], shuffle=False)
    test_loader = DataLoader(test_set, batch_size=config["data"]["batch_size"], shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    viT = MIAFEx(
        image_size = 224,
        patch_size = 16,
        num_classes = 4,
        dim = 1024,
        depth = 6,
        heads = 16,
        mlp_dim = 2048,
        dropout = 0.3,
        emb_dropout = 0.1
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    num_epochs = config["model"]["epochs"]

    optimizer = optim.Adam(viT.parameters(), lr=config["adam"]["lr"])

    vit, optimizer = load_model_state(viT, optimizer, config["model"]["save_path"])


    vit.eval()
    features_train, labels_train = vit.features_from_loader(train_loader)
    features_dev, labels_dev = vit.features_from_loader(dev_loader)

    # Concatenate train + dev
    features_train_dev = np.concatenate([features_train, features_dev], axis=0)
    labels_train_dev = np.concatenate([labels_train, labels_dev], axis=0)

    print("Unique train+dev labels:", np.unique(labels_train_dev))

    # Fit SVM on train + dev
    svm = SVMClassifier(C=config["svm"]["C"], kernel=config["svm"]["kernel"], degree=config["svm"]["degree"], model_path=config["svm"]["model_path"])
    lr = LRClassifier(C = config["lr"]["C"],solver = config["lr"]["solver"], max_iter = config["lr"]["max_iter"], random_state=config["lr"]["random_state"], model_path=config["lr"]["model_path"])
    rf = RFClassifier(n_estimators=config["rf"]["n_estimators"], criterion=config["rf"]["criterion"],random_state=config["rf"]["random_state"], model_path=config["rf"]["model_path"])
    xgb = XGBoostClassifier(n_estimators=config["xgb"]["n_estimators"], max_depth=config["xgb"]["max_depth"], learning_rate=config["xgb"]["lr"], model_path=config["xgb"]["model_path"], num_class=config["xgb"]["num_classes"])

    list_of_models = [svm, lr, rf, xgb]
    results = []
    for model in list_of_models:
        model.model.fit(features_train_dev, labels_train_dev)

        # Evaluate on test set
        features_test, labels_test = vit.features_from_loader(test_loader)
        y_pred = model.model.predict(features_test)

        # Calculate metrics
        accuracy = accuracy_score(labels_test, y_pred)
        precision = precision_score(labels_test, y_pred, average='macro')
        recall = recall_score(labels_test, y_pred, average='macro')
        f1 = f1_score(labels_test, y_pred, average='macro')
        result = {}
        result[model.__class__.__name__] = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }
        results.append(result)

        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
    
    # save results

    import json

    with open(config["results"]["save_path"], 'w') as file:
        json.dump(results, file)

