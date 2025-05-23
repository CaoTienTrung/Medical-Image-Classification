import argparse
import yaml
from models import MIAFEx
from utils import chestCTforMIAFEx, chestCTforViT
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import torch.optim as optim
from utils import save_model, load_model_state
from vit_pytorch import ViT

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

    viT = ViT(
        image_size = 224,
        patch_size = 16,
        num_classes = 4,
        dim = 1024,
        depth = 6,
        heads = 16,
        mlp_dim = 2048,
        dropout = 0.1,
        emb_dropout = 0.1
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    num_epochs = config["model"]["epochs"]

    optimizer = optim.NAdam(viT.parameters(), lr=config["adam"]["lr"])


    viT.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        progress_bar = tqdm(train_loader, 
                        desc=f'Epoch {epoch+1}/{num_epochs}', 
                        unit='batch')
        
        for batch_id, (inputs, labels) in enumerate(progress_bar):
            inputs, labels = inputs.to(device), labels.to(device)
        
            optimizer.zero_grad()
            outputs = viT(inputs)
            labels = labels.type_as(outputs)
            loss = criterion(outputs, labels.long())
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            progress_bar.set_postfix({'loss': running_loss/(batch_id+1)})
        
        print(f'Epoch {epoch+1} - Avg Loss: {running_loss/len(train_loader):.4f}')



    viT.eval()  # Set model to evaluation mode
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

    print(f'\n Test Accuracy: {100 * correct_pred / total_pred:.2f}%')


    # save the model
    save_model(viT, optimizer, config["model"]["save_path"])