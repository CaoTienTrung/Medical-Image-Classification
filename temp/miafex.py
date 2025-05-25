import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class MIAFEx(nn.Module):
    """
    Combines pretrained ViT backbone, refinement mechanism (Sec. 3.3), and classification head.
    """
    def __init__(self,
                 vit_model_name='vit_base_patch16_224_in21k',
                 num_classes=1000,
                 freeze_backbone: bool = False):
        super(MIAFEx, self).__init__()

        self.backbone = timm.create_model(vit_model_name, pretrained=True)
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
        # print(logits.shape)
        probs = F.softmax(logits, dim=-1)          # (B, num_classes)
        return logits, probs, refined

    @torch.no_grad()
    def extract_refined_features(self, x):
        """
        Inference step: use trained model to compute RF for downstream classifier
        """
        self.eval()
        _, _, refined = self.forward(x)
        return refined