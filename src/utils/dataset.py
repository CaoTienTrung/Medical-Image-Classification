from torch.utils.data import Dataset
import os 
import cv2
import torch
import numpy as np


class chestCTforMIAFEx(Dataset):
    def __init__(self, datapath, load_type = "train", img_size = (224, 224)):
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
                for file in os.listdir(os.path.join(self.datapath, self.load_type, folder)):
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
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std  # Broadcasting theo từng channel

        img = torch.from_numpy(img)  # Chuyển từ HWC -> CHW

        label = torch.tensor(item["label"], dtype=torch.long)

        return img, label




class chestCTforViT(Dataset):
    def __init__(self, datapath, load_type = "train", img_size = (224, 224)):
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
                for file in os.listdir(os.path.join(self.datapath, self.load_type, folder)):
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
        img = img.astype(np.float32) / 255.0  # Normalize về [0, 1]
        img = (img - img.mean()) / (img.std() + 1e-8)  # Standardize riêng từng ảnh

        img = torch.from_numpy(img).permute(2,0,1)  # Chuyển từ HWC -> CHW

        label = torch.tensor(item["label"], dtype=torch.long)

        return img, label