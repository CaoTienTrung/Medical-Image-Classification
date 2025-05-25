import torch
import yaml
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from model.models import MIAFExTF, SVMClassifier
import argparse


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_models(config):
    # Load ViT model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vit_model = MIAFExTF(
        vit_model_name="vit_base_patch16_224",
        num_classes=config["model"]["num_classes"],
        freeze_backbone=True,
    ).to(device)

    # Load SVM model
    svm = SVMClassifier(
        C=config["svm"]["C"],
        kernel=config["svm"]["kernel"],
        degree=config["svm"]["degree"],
        model_path=config["svm"]["model_path"],
    )
    svm.load(config["svm"]["model_path"])

    return vit_model, svm, device


def preprocess_image(image_path, img_size=(224, 224)):
    # Load and preprocess image
    transform = transforms.Compose(
        [
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image


def predict(image_path, config_path):
    # Load config
    config = load_config(config_path)

    # Load models
    vit_model, svm, device = load_models(config)

    # Preprocess image
    image = preprocess_image(image_path, tuple(config["data"]["img_size"]))
    image = image.to(device)

    # Extract features using ViT
    with torch.no_grad():
        features = vit_model.extract_features(image)
        features = features.cpu().numpy()

    # Predict using SVM
    prediction = svm.model.predict(features)

    # Map prediction to class names
    class_names = {
        0: "adenocarcinoma",
        1: "large.cell.carcinoma",
        2: "squamous.cell.carcinoma",
        3: "normal",
    }

    predicted_class = class_names[prediction[0]]
    return predicted_class


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict single image")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument(
        "--config", type=str, default="model/configs.yaml", help="Path to config file"
    )

    args = parser.parse_args()

    result = predict(args.image, args.config)
    print(f"Predicted class: {result}")
