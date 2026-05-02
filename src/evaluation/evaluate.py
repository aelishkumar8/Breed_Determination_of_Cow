"""
Evaluation module for Cow Breed Detection.

Provides functions to evaluate a trained model on the test set
and generate classification reports, confusion matrices, etc.
"""

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from tqdm import tqdm

from src.data.dataset import get_dataloaders, load_config, CowBreedDataset
from src.models.classifier import build_model


@torch.no_grad()
def evaluate_model(model, loader, device):
    """Run inference and collect predictions + ground truths."""
    model.eval()
    all_preds = []
    all_labels = []

    for images, labels in tqdm(loader, desc="Evaluating"):
        images = images.to(device)
        outputs = model(images)
        _, predicted = outputs.max(1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())

    return np.array(all_preds), np.array(all_labels)


def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None):
    """Generate and optionally save a confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))  # type: ignore[arg-type]
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix — Cow Breed Classification")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Confusion matrix saved to {save_path}")
    plt.show()


def run_evaluation(config_path: str = "config/config.yaml",
                   weights_path: Optional[str] = None):
    """Full evaluation pipeline."""
    config = load_config(config_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    _, _, test_loader = get_dataloaders(config)
    test_count = len(test_loader.dataset)  # type: ignore[arg-type]
    print(f"Test samples: {test_count}")

    # Load model
    model = build_model(config).to(device)
    if weights_path is None:
        weights_path = str(Path(config["paths"]["model_weights"]) / "best_model.pth")
    model.load_state_dict(torch.load(weights_path, map_location=device))
    print(f"Loaded weights from {weights_path}")

    # Evaluate
    preds, labels = evaluate_model(model, test_loader, device)

    # Get class names
    dataset = CowBreedDataset(root_dir=config["data"]["raw_dir"])
    class_names = dataset.classes

    # Report
    acc = accuracy_score(labels, preds)
    print(f"\nTest Accuracy: {acc * 100:.2f}%\n")
    print(classification_report(labels, preds, target_names=class_names))

    # Confusion matrix
    output_dir = Path(config["paths"]["outputs"])
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_confusion_matrix(
        labels, preds, class_names,
        save_path=output_dir / "confusion_matrix.png" 
    )


if __name__ == "__main__":
    run_evaluation()
