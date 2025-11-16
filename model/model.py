import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights

class AgeGenderModel(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        backbone = models.resnet50(weights=weights)
    
        self.feature_dim = backbone.fc.in_features

        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        self.pool = nn.AdaptiveAvgPool2d(1)

        # Age head (regression)
        self.age_head = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 1)
        )

        # Gender head (binary classification)
        self.gender_head = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        f = self.backbone(x)
        f = self.pool(f).view(x.size(0), -1)

        age_pred = self.age_head(f).squeeze(1)
        gender_logit = self.gender_head(f).squeeze(1)

        return age_pred, gender_logit
