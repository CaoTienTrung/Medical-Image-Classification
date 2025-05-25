import torch

from torch.utils.data import Dataset
import os
import cv2
import torch
import numpy as np


class chestCTforMIAFEx(Dataset):
    def __init__(self, datapath, load_type="train", img_size=(224, 224)):
        self.datapath = datapath
        self.label_dic = {
            "adenocarcinoma": 0,
            "large.cell.carcinoma": 1,
            "squamous.cell.carcinoma": 2,
            "normal": 3,
        }
        self.load_type = load_type
        self.data = self.load_data()
        self.img_size = tuple(img_size)

    def load_data(self):
        data = []
        try:
            for folder in os.listdir(os.path.join(self.datapath, self.load_type)):
                # for name, label in self.label_dic.items():
                #     if name in url:
                #         item = {
                #             "url": url,
                #             "label": label,
                #         }
                #         data.append(item)
                for file in os.listdir(
                    os.path.join(self.datapath, self.load_type, folder)
                ):
                    for name, label in self.label_dic.items():
                        if name in folder:
                            item = {
                                "url": os.path.join(folder, file),
                                "label": label,
                            }
                            data.append(item)
        except FileNotFoundError:
            print(f"Directory {os.path.join(self.datapath, self.load_type)} not found.")
            raise
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        url = os.path.join(self.datapath, self.load_type, item["url"])

        img = cv2.imread(url)

        # Resize ảnh về kích thước chuẩn
        img = cv2.resize(img, self.img_size)

        # Xử lý nếu ảnh không đủ 3 kênh
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        # Chuyển từ BGR -> RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Normalize về [0,1]
        img = img.astype(np.float32) / 255.0

        # Standardize bằng ImageNet mean/std
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std  # Broadcasting theo từng channel

        img = torch.from_numpy(img)  # Chuyển từ HWC -> CHW

        label = torch.tensor(item["label"], dtype=torch.long)

        return img, label


class chestCTforViT(Dataset):
    def __init__(self, datapath, load_type="train", img_size=(224, 224)):
        self.datapath = datapath
        self.label_dic = {
            "adenocarcinoma": 0,
            "large.cell.carcinoma": 1,
            "squamous.cell.carcinoma": 2,
            "normal": 3,
        }
        self.load_type = load_type
        self.data = self.load_data()
        self.img_size = tuple(img_size)

    def load_data(self):
        data = []
        try:
            for folder in os.listdir(os.path.join(self.datapath, self.load_type)):
                # for name, label in self.label_dic.items():
                #     if name in url:
                #         item = {
                #             "url": url,
                #             "label": label,
                #         }
                #         data.append(item)
                for file in os.listdir(
                    os.path.join(self.datapath, self.load_type, folder)
                ):
                    for name, label in self.label_dic.items():
                        if name in folder:
                            item = {
                                "url": os.path.join(folder, file),
                                "label": label,
                            }
                            data.append(item)
        except FileNotFoundError:
            print(f"Directory {os.path.join(self.datapath, self.load_type)} not found.")
            raise
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        url = os.path.join(self.datapath, self.load_type, item["url"])

        img = cv2.imread(url)

        # Resize ảnh về kích thước chuẩn
        img = cv2.resize(img, self.img_size)

        # Xử lý nếu ảnh không đủ 3 kênh
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        # Chuyển từ BGR -> RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = img.astype(np.float32) / 255.0

        # # Standardize bằng ImageNet mean/std
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std  # Broadcasting theo từng channel

        img = torch.from_numpy(img).permute(2, 0, 1)  # Chuyển từ HWC -> CHW

        label = torch.tensor(item["label"], dtype=torch.long)

        return img, label


import random
from torch.utils.data import DataLoader
import numpy as np


class CustomImageDataset(Dataset):
    def __init__(
        self,
        directory,
        label_mode="int",  # 'int', 'binary', 'categorical', or None
        color_mode="rgb",  # 'rgb' or 'grayscale'
        image_size=(256, 256),
        interpolation="bilinear",  # 'nearest' or 'bilinear'
        transform=None,
    ):
        # Check valid
        if label_mode not in ("int", "binary", "categorical", None):
            raise ValueError(
                "label_mode must be one of 'int', 'binary', 'categorical', or None."
            )
        if color_mode not in ("rgb", "grayscale"):
            raise ValueError("color_mode must be 'rgb' or 'grayscale'.")
        if interpolation not in ("nearest", "bilinear"):
            raise ValueError("interpolation must be 'nearest' or 'bilinear'.")

        self.directory = directory

        self.label_mode = label_mode
        self.color_mode = color_mode
        self.image_size = image_size
        self.interpolation = (
            cv2.INTER_LINEAR if interpolation == "bilinear" else cv2.INTER_NEAREST
        )

        self.transform = transform

        # Determine class names
        self.class_names = sorted(
            entry.name for entry in os.scandir(directory) if entry.is_dir()
        )
        self.class_to_index = {name: idx for idx, name in enumerate(self.class_names)}

        # Take samples
        self.samples = []
        for root, _, files in os.walk(directory):
            if root == directory:
                continue
            for fname in files:
                if fname.lower().endswith(("png", "jpg")):
                    path = os.path.join(root, fname)
                    label = self.class_to_index[os.path.basename(root)]
                    self.samples.append((path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = cv2.imread(path)

        if img is None:
            raise ValueError(f"Cannot read image: {path}")

        if self.color_mode == "grayscale":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = cv2.resize(img, self.image_size, interpolation=self.interpolation)
        img = img.astype(np.float32) / 255.0

        if self.color_mode == "grayscale":
            img = np.expand_dims(img, -1)  # (height, width, channel)
        img = np.transpose(img, (2, 0, 1))  # (channel, height, width)
        tensor = torch.from_numpy(img)
        if self.transform:
            tensor = self.transform(tensor)

        if self.label_mode == "int":
            return tensor, label
        if self.label_mode == "binary":
            return tensor, torch.tensor(label, dtype=torch.float32)
        if self.label_mode == "categorical":
            one_hot = torch.zeros(len(self.class_names), dtype=torch.float32)
            one_hot[label] = 1.0
            return tensor, one_hot


def custom_image_dataset_from_directory(
    directory,
    label_mode="int",
    color_mode="rgb",
    batch_size=32,
    image_size=(224, 224),
    interpolation="bilinear",
    shuffle=True,
    seed=None,
    num_workers=0,
    transform=None,
):
    dataset = CustomImageDataset(
        directory,
        label_mode=label_mode,
        color_mode=color_mode,
        image_size=image_size,
        interpolation=("bilinear" if interpolation == "bilinear" else "nearest"),
        transform=transform,
    )
    if seed is not None:
        random.seed(seed)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )
    return loader


def save_model(model, optimizer, save_path):
    """
    Save the model and optimizer state to a specified path.

    Args:
        model (torch.nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer to save.
        save_path (str): Path to save the model file.
    """
    try:
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            save_path,
        )
        print(f"Model saved successfully to {save_path}")
    except Exception as e:
        print(f"Error saving model: {e}")


def load_model_state(model, optimizer, model_path):
    """
    Load the model state from a specified path.

    Args:
        model (torch.nn.Module): The model to load the state into.
        model_path (str): Path to the saved model file.

    Returns:
        torch.nn.Module: The model with loaded state.
    """
    try:
        checkpoint = torch.load(
            model_path,
            map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print(f"Model state loaded successfully from {model_path}")
    except Exception as e:
        print(f"Error loading model state: {e}")

    return model, optimizer
