import torch
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
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

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

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
    def __init__(self, encoder_layers, d_model):
        super(Encoder, self).__init__()

        self.attn_layers = nn.ModuleList(
            [EncoderLayer(d_model=d_model) for _ in range(encoder_layers)]
        )

    def forward(self, x):
        for attn_layer in self.attn_layers:
            x = attn_layer(x)

        return x


class ViT(nn.Module):
    def __init__(self, d_model=768, encoder_layers=3, patch_size=16, num_classes=4):
        super(ViT, self).__init__()

        self.d_model = d_model
        self.encoder = Encoder(encoder_layers=encoder_layers, d_model=d_model)

        self.patch_size = patch_size
        self.num_patches = (256 // patch_size) ** 2

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        self.PE = PositionalEmbedding(d_model, max_len=self.num_patches + 1)

        self.dropout = nn.Dropout(0.1)
        self.patch_embed = nn.Conv2d(
            in_channels=3,
            out_channels=d_model,
            kernel_size=patch_size,
            stride=patch_size,
        )

        self.encoder = Encoder(
            encoder_layers=encoder_layers,
            d_model=d_model,
            norm_layer=nn.LayerNorm(d_model),
        )

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, num_classes)
        )

    def forward(self, x):
        # (B, 256, 256, 3) - > (B, 3, 255, 255)
        x = x.permute(0, 3, 1, 2)

        B, C, L, L = x.shape

        # Patch embedding: (B, 3, 256, 256) -> (B, d_model, 16, 16)
        x = self.patch_embed(x)

        # Flatten -> (B, num_patches, d_model)
        x = x.flatten(2).transpose(1, 2)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # PE
        x = x + self.PE(x)
        x = self.dropout(x)

        x = self.encoder(x)

        # CLS Token
        cls_output = x[:, 0]

        output = self.mlp_head(cls_output)  # (B, num_classes)

        return output
