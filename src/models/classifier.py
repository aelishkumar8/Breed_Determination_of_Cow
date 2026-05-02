"""
Classifier module for Cow Breed Detection.

Provides a configurable image classifier using pretrained
torchvision backbones with a custom classification head.
"""

import torch
import torch.nn as nn
from torchvision import models


# Supported architectures and their feature dimensions
ARCHITECTURES = {
    "resnet18": (models.resnet18, 512),
    "resnet50": (models.resnet50, 2048),
    "efficientnet_b0": (models.efficientnet_b0, 1280),
}


class CowBreedClassifier(nn.Module):
    """
    Transfer-learning classifier for cow breeds.

    Uses a pretrained backbone with a custom fully-connected head.
    """

    def __init__(self, architecture: str = "resnet50",
                 num_classes: int = 10,
                 pretrained: bool = True,
                 dropout: float = 0.5):
        super().__init__()

        if architecture not in ARCHITECTURES:
            raise ValueError(
                f"Unsupported architecture '{architecture}'. "
                f"Choose from: {list(ARCHITECTURES.keys())}"
            )

        model_fn, feature_dim = ARCHITECTURES[architecture]
        weights = "IMAGENET1K_V1" if pretrained else None
        self.backbone = model_fn(weights=weights)

        # Remove the original classification head
        if architecture.startswith("resnet"):
            self.backbone.fc = nn.Identity()  # type: ignore[assignment]
        elif architecture.startswith("efficientnet"):
            self.backbone.classifier = nn.Identity()  # type: ignore[assignment]

        # Custom classification head
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout / 2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.classifier(features)


def build_model(config: dict) -> CowBreedClassifier:
    """Instantiate a classifier from the project config."""
    model_cfg = config["model"]
    return CowBreedClassifier(
        architecture=model_cfg["architecture"],
        num_classes=model_cfg["num_classes"],
        pretrained=model_cfg["pretrained"],
        dropout=model_cfg["dropout"],
    )
