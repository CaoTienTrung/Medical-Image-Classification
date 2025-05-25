import numpy as np
import pickle
import os
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
from FeatureExtractors.feature_extractor import *
from xgboost import XGBClassifier

class SVMClassifier:
    def __init__(self, C=1, kernel='rbf', degree=3, random_state=42, feature_extractor=HOGFeatureExtractor(), model_path="svm_model.pkl",):
        self.model = SVC(
            C=C, 
            kernel=kernel, 
            degree=degree, 
            random_state=random_state
        )
        self.feature_extractor = feature_extractor
        self.model_path = model_path

    def train(self, dataset):
        X, y = [], []
        for img, label in dataset:
            img_np = img.numpy().transpose(1, 2, 0)
            feat = self.feature_extractor.extract(img_np)
            X.append(feat)
            y.append(label)
        X = np.array(X)
        y = np.array(y)
        self.model.fit(X, y)

        with open(self.model_path, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"[INFO] Model saved to {self.model_path}")


    def predict(self, dataset):
        with open(self.model_path, 'rb') as f:
            self.model = pickle.load(f)
        print(f"[INFO] Model loaded from {self.model_path}")

        X, y = [], []
        for img, label in dataset:
            img_np = img.numpy().transpose(1, 2, 0)
            feat = self.feature_extractor.extract(img_np)
            X.append(feat)
            y.append(label)
        X = np.array(X)
        y = np.array(y)
        y_pred = self.model.predict(X)
        metrics = None
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, average='macro'),
            'recall': recall_score(y, y_pred, average='macro'),
            'f1': f1_score(y, y_pred, average='macro')
        }
        return y_pred, metrics

class RFClassifier:
    def __init__(self, n_estimators=100, criterion='gini', random_state=42, feature_extractor=HOGFeatureExtractor(), model_path="rf_model.pkl",):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators, 
            criterion=criterion, 
            random_state=random_state
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

        with open(self.model_path, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"[INFO] Model saved to {self.model_path}")


    def predict(self, dataset):
        with open(self.model_path, 'rb') as f:
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
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, average='macro'),
            'recall': recall_score(y, y_pred, average='macro'),
            'f1': f1_score(y, y_pred, average='macro')
        }
        return y_pred, metrics
    
class LRClassifier:
    def __init__(self, C=1.0, solver='lbfgs', max_iter=1000, multi_class='multinomial', random_state=42, feature_extractor=HOGFeatureExtractor(), model_path="lr_model.pkl",):
        self.model = LogisticRegression(
            C=C,
            solver=solver,
            max_iter=max_iter,
            multi_class=multi_class,
            random_state=random_state
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

        with open(self.model_path, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"[INFO] Model saved to {self.model_path}")


    def predict(self, dataset):
        with open(self.model_path, 'rb') as f:
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
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, average='macro'),
            'recall': recall_score(y, y_pred, average='macro'),
            'f1': f1_score(y, y_pred, average='macro')
        }
        return y_pred, metrics
    
class XGBoostClassifier:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42, objective='multi:softmax', num_class=4, feature_extractor=HOGFeatureExtractor(), model_path="xgb_model.pkl",):
        self.model = XGBClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=random_state,
            objective=objective,
            num_class=num_class
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

        with open(self.model_path, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"[INFO] Model saved to {self.model_path}")


    def predict(self, dataset):
        with open(self.model_path, 'rb') as f:
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
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, average='macro'),
            'recall': recall_score(y, y_pred, average='macro'),
            'f1': f1_score(y, y_pred, average='macro')
        }
        return y_pred, metrics

import torch
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        max_len = int(max_len)
        d_model = int(d_model)
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        ).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return self.pe[:, : x.size(1)]


class Attention(nn.Module):
    def __init__(self, attention_dropout=0.1):
        super(Attention, self).__init__()
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = 1.0 / math.sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        return V.contiguous()


class AttentionLayer(nn.Module):
    def __init__(self, d_model=768, n_heads=8):
        super(AttentionLayer, self).__init__()

        d_keys = d_model // n_heads
        d_values = d_model // n_heads

        self.inner_attention = Attention()

        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)

        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out = self.inner_attention(queries, keys, values)
        out = out.view(B, L, -1)

        return self.out_projection(out)


class EncoderLayer(nn.Module):
    def __init__(self, d_model, dropout=0.1, n_head=8):
        super(EncoderLayer, self).__init__()

        d_ff = 4 * d_model
        self.attention = AttentionLayer(d_model=d_model, n_heads=n_head)

        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu

    def forward(self, x):
        new_x = self.attention(x, x, x)
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y)


class Encoder(nn.Module):
    def __init__(self, encoder_layers, d_model, norm_layer):
        super(Encoder, self).__init__()

        self.attn_layers = nn.ModuleList(
            [EncoderLayer(d_model=d_model) for _ in range(encoder_layers)]
        )
        self.norm = norm_layer

    def forward(self, x):
        for attn_layer in self.attn_layers:
            x = attn_layer(x)

        return self.norm(x)


# class MIAFEx(nn.Module):
#     def __init__(self, d_model=768, encoder_layers=3, patch_size=16, num_classes=4):
#         super(MIAFEx, self).__init__()

#         self.d_model = d_model


#         self.patch_size = patch_size
#         self.num_patches = (224**2 // patch_size**2) 

#         # self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

#         self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

#         self.PE = PositionalEmbedding(d_model, max_len=self.num_patches + 1)

#         self.dropout = nn.Dropout(0.1)
#         self.patch_embed = nn.Conv2d(
#             in_channels=3,
#             out_channels=d_model,
#             kernel_size=patch_size,
#             stride=patch_size,
#         )

#         self.encoder = Encoder(
#             encoder_layers=encoder_layers,
#             d_model=d_model,
#             norm_layer=nn.LayerNorm(d_model),
#         )

#         self.w_refine = nn.Parameter(torch.randn(d_model))

#         self.softmax = nn.Sequential(
#             nn.LayerNorm(d_model),
#             nn.Linear(d_model, num_classes),
#             nn.Softmax(dim=-1),
#         )

#         self.classifier = nn.Sequential(
#             nn.LayerNorm(d_model),
#             nn.Linear(d_model, num_classes),
#         )

#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     def forward(self, x):
#         # (B, 224, 224, 3) - > (B, 3, 224, 224)
#         x = x.permute(0, 3, 1, 2)
        
#         B, C, L, L = x.shape

#         # Patch embedding: (B, 3, 224, 224) -> (B, d_model, 16, 16)
#         x = self.patch_embed(x)

#         # Flatten -> (B, num_patches, d_model)
#         x = x.flatten(2).transpose(1, 2)

#         cls_tokens = self.cls_token.expand(B, -1, -1)
#         x = torch.cat((cls_tokens, x), dim=1)

#         # PE
#         x = x + self.PE(x)
#         x = self.dropout(x)

#         x = self.encoder(x)

#         # CLS Token
#         cls_output = x[:, 0]

#         refined_output = cls_output * self.w_refine

#         output = self.classifier(refined_output)

#         return output

#     def extract_features(self, x):
#         # (B, 224, 224, 3) - > (B, 3, 224, 224)
#         x = x.permute(0, 3, 1, 2)
        
#         B, C, L, L = x.shape

#         # Patch embedding: (B, 3, 224, 224) -> (B, d_model, 16, 16)
#         x = self.patch_embed(x)

#         # Flatten -> (B, num_patches, d_model)
#         # x = x.flatten(2).transpose(1, 2)

#         cls_tokens = self.cls_token.repeat(B, 1, 1)
#         x = torch.cat((cls_tokens, x), dim=1)

#         # PE
#         x = x + self.PE(x)
#         x = self.dropout(x)

#         x = self.encoder(x)

#         # CLS Token
#         cls_output = x[:, 0]

#         refined_output = cls_output * self.w_refine
#         return refined_output

#     def features_from_loader(self, loader):
#         features = []
#         labels = []
#         for batch in tqdm(loader, desc = 'Extracting features'):
#             x, y = batch
#             x = x.to(self.device)
#             with torch.no_grad():
#                 feature = self.extract_features(x)
#                 features.append(feature.cpu().numpy())
#                 labels.append(y.cpu().numpy())
#         features = np.concatenate(features, axis=0)
#         labels = np.concatenate(labels, axis=0)
#         return features, labels
    

class BasicBlock(nn.Module):
    """The Residual Block"""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, downsample: bool = False) -> None:
        """
        Create the Residual Block

        Args:
            in_channels (int): number of input channels
            out_channels (int): number of output channels
            stride (int): stride of first 3x3 convolution layer
            downsample (bool): whether to adjust for spatial dimensions due to downsampling via stride=2
        """
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # For downsampling, the skip connection will pass through the 1x1 conv layer with stride of 2 to
        # match the spatial dimension of the downsampled feature maps and channels for the add operation.
        #
        # More specifically, the 'downsample block' is used for layer 2, 3, 4 of ResNet18 where the first conv2d
        # layer of the BasicBlock uses a stride of 2 instead of 1 to downsample feature maps for a larger
        # receptive field.
        # This is why we need to carefully craft our 'downsample block' to make sure spatial dimensions are
        # not disrupted when we add the skip connection in these residual blocks.
        self.downsample = None
        if downsample:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x.clone()
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))

        if self.downsample:  # if layer not None
            identity = self.downsample(identity)

        x += identity
        o = self.relu(x)

        return o


class ResNet18(nn.Module):
    """The ResNet-18 Model"""

    def __init__(self, n_classes: int = 10) -> None:
        """
        Create the ResNet-18 Model

        Args:
            n_classes (int, optional): The number of output classes we predict for. Defaults to 10.
        """
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=(3, 3), bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = nn.Sequential(
            BasicBlock(64, 64),
            BasicBlock(64, 64),
        )
        self.layer2 = nn.Sequential(
            BasicBlock(64, 128, stride=2, downsample=True),
            BasicBlock(128, 128),
        )
        self.layer3 = nn.Sequential(
            BasicBlock(128, 256, stride=2, downsample=True),
            BasicBlock(256, 256),
        )
        self.layer4 = nn.Sequential(
            BasicBlock(256, 512, stride=2, downsample=True),
            BasicBlock(512, 512),
        )
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        # our fully connected layer will be different to accomodate for CIFAR-10
        self.fc = nn.Linear(in_features=512, out_features=n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)  # [bs, 512, 1, 1]

        x = torch.squeeze(x)  # reshape to [bs, 512]
        o = self.fc(x)
        
        return o
    
import timm

class ViTTransferLearning(nn.Module):
    def __init__(self, num_classes, pretrained=True, freeze_backbone=False):
        super().__init__()
        # Load pretrained ViT trên ImageNet-21k
        
        self.model = timm.create_model('vit_base_patch16_224_in21k', pretrained=pretrained)
        
        # Thay lớp cuối cho phù hợp với số lớp mới
        in_features = self.model.head.in_features
        self.model.head = nn.Linear(in_features, num_classes)
        
        if freeze_backbone:
            # Freeze toàn bộ backbone, chỉ train lớp head
            for param in self.model.parameters():
                param.requires_grad = False
            for param in self.model.head.parameters():
                param.requires_grad = True

    def forward(self, x):
        return self.model(x)

class MIAFExTF(nn.Module):
    def __init__(self,
                 vit_model_name='vit_base_patch16_224_in21k',
                 num_classes=4,
                 freeze_backbone: bool = False):
        super(MIAFExTF, self).__init__()

        self.backbone = timm.create_model('vit_base_patch16_224_in21k', pretrained=True)
        embed_dim = self.backbone.num_features  # typically 768

        # Optional: freeze ViT backbone parameters
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.w_refine = nn.Parameter(torch.ones(embed_dim), requires_grad=True)
        # Fully connected classification head
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        # Extract CLS token feature from ViT backbone
        h_out = self.backbone.forward_features(x)  # (B, D)

        cls_token = h_out[:, 0, :]                 # (B, D)

        # Refinement
        refined = cls_token * self.w_refine        # (B, D)
        # Classification
        logits = self.classifier(refined)          # (B, num_classes)
        return logits

    def extract_features(self, x):
        # (B, 224, 224, 3) - > (B, 3, 224, 224)
        x = x.permute(0, 3, 1, 2)
        
        cls_output = self.backbone.forward_features(x)

        refined_output = cls_output * self.w_refine
        return refined_output

    def features_from_loader(self, loader):
        features = []
        labels = []
        for batch in tqdm(loader, desc = 'Extracting features'):
            x, y = batch
            x = x.to(self.device)
            with torch.no_grad():
                feature = self.extract_features(x)
                features.append(feature.cpu().numpy())
                labels.append(y.cpu().numpy())
        features = np.concatenate(features, axis=0)
        labels = np.concatenate(labels, axis=0)
        return features, labels


import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)

class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Linear(dim, num_classes)

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)


class MIAFEx(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Linear(dim, num_classes)

        self.w_refine = nn.Parameter(torch.ones(dim), requires_grad=True)

        self.classifier = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Dropout(dropout),
            nn.Linear(dim, num_classes),
        )


        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        cls_output = x
        refined_output = cls_output * self.w_refine
        
        refined_output = self.to_latent(refined_output)
        
        return self.classifier(refined_output)

    def extract_features(self, x):
        x = self.to_patch_embedding(x)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        cls_output = x
        refined_output = cls_output * self.w_refine
        return refined_output

    def features_from_loader(self, loader):
        features = []
        labels = []
        for batch in tqdm(loader, desc = 'Extracting features'):
            x, y = batch
            x = x.to(self.device)
            with torch.no_grad():
                feature = self.extract_features(x)
                features.append(feature.cpu().numpy())
                labels.append(y.cpu().numpy())
        features = np.concatenate(features, axis=0)
        labels = np.concatenate(labels, axis=0)
        return features, labels


