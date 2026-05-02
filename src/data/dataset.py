"""
Dataset module for Cow Breed Detection.

Provides a PyTorch Dataset class for loading cow breed images
with configurable transforms and train/val/test splitting.
"""

import os
from pathlib import Path

import yaml
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms


def load_config(config_path: str = "config/config.yaml") -> dict:
    """Load project configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


class CowBreedDataset(Dataset):
    """
    Custom Dataset for cow breed images.

    Expects images organized in subdirectories by breed:
        data/raw/
        ├── breed_1/
        │   ├── img_001.jpg
        │   └── ...
        ├── breed_2/
        └── ...
    """

    def __init__(self, root_dir: str, transform=None):
        """
        Args:
            root_dir:  Path to the root data directory.
            transform: Optional torchvision transforms to apply.
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.classes = sorted(
            [d for d in os.listdir(root_dir) if (self.root_dir / d).is_dir()]
        )
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.samples = self._load_samples()

    def _load_samples(self):
        """Walk subdirectories and collect (image_path, label) pairs."""
        samples = []
        valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        for cls in self.classes:
            cls_dir = self.root_dir / cls
            for img_name in os.listdir(cls_dir):
                if Path(img_name).suffix.lower() in valid_extensions:
                    samples.append((cls_dir / img_name, self.class_to_idx[cls]))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img_path, label = self.samples[index]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


def get_transforms(image_size: tuple = (224, 224), is_training: bool = True):
    """Return torchvision transforms for training or evaluation."""
    if is_training:
        return transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(30),               # ±30° handles diagonal cows
            transforms.RandomAffine(
                degrees=0,                                # rotation already handled above
                translate=(0.1, 0.1),                     # shift up to 10% — cow not always centered
                scale=(0.85, 1.15),                       # zoom in/out — cow at different distances
                shear=10,                                 # perspective skew — camera angle variation
            ),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.3),  # 3D viewpoint changes
            transforms.ColorJitter(brightness=0.3, contrast=0.3,
                                   saturation=0.3, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])


def get_dataloaders(config: dict):
    """
    Create train, validation, and test DataLoaders from config.

    Returns:
        (train_loader, val_loader, test_loader)
    """
    data_cfg = config["data"]
    image_size = tuple(data_cfg["image_size"])
    batch_size = data_cfg["batch_size"]
    num_workers = data_cfg["num_workers"]

    # Full dataset with training transforms (will split later)
    full_dataset = CowBreedDataset(
        root_dir=data_cfg["raw_dir"],
        transform=get_transforms(image_size, is_training=True),
    )

    # Split sizes
    total = len(full_dataset)
    train_size = int(total * data_cfg["train_split"])
    val_size = int(total * data_cfg["val_split"])
    test_size = total - train_size - val_size

    train_set, val_set, test_set = random_split(
        full_dataset, [train_size, val_size, test_size]
    )

    train_loader = DataLoader(train_set, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size,
                             shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader
