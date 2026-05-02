"""
Helper utilities for Cow Breed Detection.

Common functions used across the project.
"""

import os
import random
from pathlib import Path

import numpy as np
import torch
import yaml


def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_parameters(model: torch.nn.Module) -> int:
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_device() -> torch.device:
    """Return the best available device (CUDA > CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_dir(path: str):
    """Create directory if it doesn't exist."""
    Path(path).mkdir(parents=True, exist_ok=True)
