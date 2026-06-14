from __future__ import annotations

import torch.nn as nn
from torchvision import models


def build_model(model_name: str, num_classes: int):
    if model_name.lower() != "resnet18":
        raise ValueError(f"Unsupported model_name: {model_name}")
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
