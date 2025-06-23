import argparse
import yaml
from models import MIAFEx, MIAFExTF
from utils import chestCTforMIAFEx, chestCTforViT, CustomImageDataset, custom_image_dataset_from_directory
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import torch.optim as optim
from utils import save_model, load_model_state
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.utils.class_weight import compute_class_weight
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

def save_json(data, path):
    import json
    with open(path, 'w') as file:
        json.dump(data, file, indent=4)

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        CE_loss = nn.functional.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-CE_loss)
        focal_loss = (1 - pt) ** self.gamma * CE_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
        
import os
from torchvision import transforms

def train(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits, probs, _, _ = model(images)
        # print(logits.shape)
        # print(labels.shape)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

            
        _, preds = torch.max(probs, dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)
        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    print(f'Loss: {epoch_loss} - Accc: {epoch_acc}')
    return epoch_loss, epoch_acc


def evaluate(model, loader, device, criterion):
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0
    with torch.no_grad():
        for images, labels in tqdm(loader):
            images = images.to(device)
            labels = labels.to(device)
            logits, probs, _, _ = model(images)
            loss = criterion(logits, labels)
            _, preds = torch.max(probs, dim=1)
            correct += (preds == labels).sum().item()
            total += images.size(0)
            running_loss += loss.item() * images.size(0)
    epoch_loss = running_loss / total
    epoch_acc = correct / total

    return epoch_loss, epoch_acc


def main():
    args = parse_args()
    config = load_yaml(args.config)
    
    train_set = chestCTforViT(config["data"]["path"], "train", config["data"]["img_size"])
    dev_set = chestCTforViT(config["data"]["path"], "valid", config["data"]["img_size"])
    test_set = chestCTforViT(config["data"]["path"], "test", config["data"]["img_size"])
    
    train_loader = DataLoader(train_set, batch_size=config["data"]["batch_size"], shuffle=True)
    dev_loader = DataLoader(dev_set, batch_size=config["data"]["batch_size"], shuffle=False)
    test_loader = DataLoader(test_set, batch_size=config["data"]["batch_size"], shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # model = MIAFEx(
    # ).to(device)

    model = MIAFExTF(vit_model_name='vit_base_patch16_224', num_classes=config["model"]["num_classes"], freeze_backbone=True)
    model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.NAdam(filter(lambda p: p.requires_grad, model.parameters()), lr=config["adam"]["lr"])

    epochs = config["model"]["epochs"]
    training_res = {}
    # Training loop

    test_acc_max = 0.0
    for epoch in range(1, epochs+1):
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, device, criterion)
        print(f'Epoch {epoch}/{epochs} | Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}')
        training_res[epoch] = {
            "train_loss": train_loss,
            "train_acc": train_acc,
            "test_acc": test_acc,
            "test_loss": test_loss
        }
        if test_acc > test_acc_max:
            test_acc_max = test_acc
            save_model(model, optimizer, config["model"]["save_path"])
            print(f"Model saved at epoch {epoch} with accuracy {test_acc:.4f}")

    save_json(training_res, config["model"]["save_json"])


    # Final evaluation
    test_loss, final_acc = evaluate(model, test_loader, device, criterion)
    print(f'Final Test Accuracy: {final_acc:.4f}')

    # save_model(model, optimizer, config["model"]["save_path"])

if __name__ == '__main__':
    main()