"""
Training module for Cow Breed Detection.

Provides a complete training loop with validation,
early stopping, model checkpointing, and logging.
"""

import os
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from tqdm import tqdm

from src.data.dataset import get_dataloaders, load_config
from src.models.classifier import build_model


def get_optimizer(model, config: dict):
    """Create optimizer from config."""
    train_cfg = config["training"]
    lr = train_cfg["learning_rate"]
    wd = train_cfg["weight_decay"]
    name = train_cfg["optimizer"].lower()

    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    elif name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    elif name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr,
                               momentum=0.9, weight_decay=wd)
    else:
        raise ValueError(f"Unsupported optimizer: {name}")


def get_scheduler(optimizer, config: dict):
    """Create learning-rate scheduler from config."""
    train_cfg = config["training"]
    name = train_cfg["scheduler"].lower()

    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=train_cfg["epochs"]
        )
    elif name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=10, gamma=0.1
        )
    elif name == "none":
        return None
    else:
        raise ValueError(f"Unsupported scheduler: {name}")


def train_one_epoch(model, loader, criterion, optimizer, device):
    """Run a single training epoch. Returns (avg_loss, accuracy)."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="Training", leave=False):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    avg_loss = running_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


@torch.no_grad()
def validate(model, loader, criterion, device):
    """Run validation. Returns (avg_loss, accuracy)."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="Validating", leave=False):
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    avg_loss = running_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


def train(config_path: str = "config/config.yaml"):
    """Full training pipeline."""
    config = load_config(config_path)
    train_cfg = config["training"]
    paths_cfg = config["paths"]

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data
    train_loader, val_loader, _ = get_dataloaders(config)
    train_count = len(train_loader.dataset)  # type: ignore[arg-type]
    val_count = len(val_loader.dataset)  # type: ignore[arg-type]
    print(f"Training samples: {train_count}")
    print(f"Validation samples: {val_count}")

    # Model
    model = build_model(config).to(device)
    print(f"Model: {config['model']['architecture']}")

    # Loss, optimizer, scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(model, config)
    scheduler = get_scheduler(optimizer, config)

    # Training loop
    best_val_acc = 0.0
    patience_counter = 0
    weights_dir = Path(paths_cfg["model_weights"])
    weights_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, train_cfg["epochs"] + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{train_cfg['epochs']}")
        print(f"{'='*60}")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        if scheduler is not None:
            scheduler.step()

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.2f}%")

        # Checkpointing
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            if train_cfg["save_best_model"]:
                save_path = weights_dir / "best_model.pth"
                torch.save(model.state_dict(), save_path)
                print(f"✓ Saved best model (val_acc={val_acc:.2f}%)")
        else:
            patience_counter += 1
            print(f"No improvement ({patience_counter}/{train_cfg['early_stopping_patience']})")

        # Early stopping
        if patience_counter >= train_cfg["early_stopping_patience"]:
            print("\nEarly stopping triggered.")
            break

    print(f"\nTraining complete. Best validation accuracy: {best_val_acc:.2f}%")


if __name__ == "__main__":
    train()
