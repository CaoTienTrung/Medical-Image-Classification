import os
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from custom_dataset import *  # assuming your dataset file is custom_dataset.py
from miafex import MIAFEx              # updated model file
from tqdm import tqdm

def train(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits, probs, _ = model(images)
        # print(logits.shape)
        # print(labels.shape)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(probs, dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    print(f'Loss: {epoch_loss} - Accc: {epoch_acc}')
    return epoch_loss, epoch_acc


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(loader):
            images = images.to(device)
            labels = labels.to(device)
            logits, probs, _ = model(images)
            _, preds = torch.max(probs, dim=1)
            correct += (preds == labels).sum().item()
            total += images.size(0)

    return correct / total


def main():
    # Config
    data_dir = 'F:\Studies\Third_year\Computer_vision\Project\ProjectCode\Dataset\Data'
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

    model = MIAFEx(vit_model_name='vit_base_patch16_224', num_classes=num_classes, freeze_backbone=True)
    model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.NAdam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    # Training loop
    for epoch in range(1, epochs+1):
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
        test_acc = evaluate(model, test_loader, device)
        print(f'Epoch {epoch}/{epochs} | Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}')

        checkpoint_path = f'miafex2_epoch_{epoch}.pth'
        torch.save(model.state_dict(), checkpoint_path)

    # Final evaluation
    final_acc = evaluate(model, test_loader, device)
    print(f'Final Test Accuracy: {final_acc:.4f}')


if __name__ == '__main__':
    main()
