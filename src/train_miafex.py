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

if __name__ == "__main__":
    args = parse_args()
    config = load_yaml(args.config)
    
    data_dir = '/home/anhkhoa/Medical-Image-Classification/Dataset/Data'
    num_classes = len(os.listdir(os.path.join(data_dir, 'train')))
    batch_size = 8
    epochs = 10
    lr = 1e-4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Data loaders
    train_dir = os.path.join(data_dir, 'train')
    test_dir  = os.path.join(data_dir, 'test')

    # You can add data augmentation / normalization transforms here
    transform = transforms.Compose([
        # transforms.Resize((224, 224)),
        # transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # train_dataset = CustomImageDataset(train_dir, labels='inferred', label_mode='int', transform=transform)
    # test_dataset  = CustomImageDataset(test_dir,  labels='inferred', label_mode='int', class_names=train_dataset.class_names, transform=transform)

    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=4)
    # test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=4)
    train_loader = custom_image_dataset_from_directory(
        os.path.join(data_dir, 'train'),
        label_mode='int',
        color_mode='rgb',
        batch_size=batch_size,
        image_size=(224,224),
        interpolation='bilinear',
        shuffle=True,
        seed=None,
        num_workers=2,
        transform=transform
    )
    test_loader = custom_image_dataset_from_directory(
        os.path.join(data_dir, 'test'),
        label_mode='int',
        color_mode='rgb',
        batch_size=batch_size,
        image_size=(224,224),
        interpolation='bilinear',
        shuffle=True,
        seed=None,
        num_workers=2,
        transform=transform
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # all_train_labels = [int(label) for _, label in train_set]

    # # Bây giờ tính class weights an toàn
    # classes = np.unique(all_train_labels)
    # class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=all_train_labels)
    # class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
    # viT = MIAFEx(
    #     image_size = 224,
    #     patch_size = 16,
    #     num_classes = 4,
    #     dim = 1024,
    #     depth = 6,
    #     heads = 16,
    #     mlp_dim = 2048,
    #     dropout = 0.1,
    #     emb_dropout = 0.1,
    #     pool='mean'
    # ).to(device)

    viT = MIAFExTF(
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    num_epochs = config["model"]["epochs"]

    optimizer = optim.Adam(viT.parameters(), lr=config["adam"]["lr"])



    
    for epoch in range(num_epochs):
        viT.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        progress_bar = tqdm(train_loader, 
                        desc=f'Epoch {epoch+1}/{num_epochs}', 
                        unit='batch')
        
        for batch_id, (inputs, labels) in enumerate(progress_bar):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = viT(inputs)

            loss = criterion(outputs, labels.long())
            loss.backward()
            optimizer.step()

            # Đếm số đúng
            _, preds = torch.max(outputs.data, 1)
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)

            running_loss += loss.item()
            progress_bar.set_postfix({
                'loss': f'{running_loss/(batch_id+1):.4f}',
                'acc': f'{100 * correct_train / total_train:.2f}%'
            })

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct_train / total_train
        print(f'Epoch {epoch+1} - Avg Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.2f}%')

        viT.eval()
        correct_pred = 0
        total_pred = 0
        with torch.no_grad():
            test_bar = tqdm(test_loader, desc='Testing', unit='batch')
            for inputs, labels in test_bar:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = viT(inputs)
                _, pred = torch.max(outputs.data, 1)
                total_pred += labels.size(0)
                correct_pred += (pred == labels).sum().item()
                test_bar.set_postfix({'acc': f'{100*correct_pred/total_pred:.2f}%'})


        


    viT.eval()  # Set model to evaluation mode
    correct_pred = 0
    total_pred = 0

    y_true = []
    y_pred = []
    with torch.no_grad():
        test_bar = tqdm(test_loader, desc='Testing', unit='batch')
        for inputs, labels in test_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = viT(inputs)
            _, pred = torch.max(outputs.data, 1)
            total_pred += labels.size(0)
            correct_pred += (pred == labels).sum().item()
            test_bar.set_postfix({'acc': f'{100*correct_pred/total_pred:.2f}%'})

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(pred.cpu().numpy())



    
    print(f'\n Test Accuracy: {100 * correct_pred / total_pred:.2f}%')

    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(cm)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues')
    plt.show()

    # save the model
    save_model(viT, optimizer, config["model"]["save_path"])