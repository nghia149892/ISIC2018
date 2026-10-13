from ..config import cfg
import torch.nn as nn
import torch
from torchvision.models import resnet18, ResNet18_Weights


def StudentModel(num_classes=cfg.NUM_CLASSES, pretrained=True):
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    return model

class DINOv2Classifier(nn.Module):
    def __init__(self, num_classes = cfg.NUM_CLASSES):
        super().__init__()
        # Load backbone DINOv2
        self.backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        self.embed_dim = self.backbone.embed_dim
        # Lớp phân loại
        self.classifier = nn.Sequential(
            nn.Linear(self.embed_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # DINOv2 trả về đặc trưng qua forward_features
        features = self.backbone.forward_features(x)
        # Sử dụng 'x_norm_clstoken' làm đại diện cho hình ảnh
        cls_token = features['x_norm_clstoken']
        return self.classifier(cls_token)

