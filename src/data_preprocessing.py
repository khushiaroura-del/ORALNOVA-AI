"""
src/data_preprocessing.py
Dataset loading, augmentation, and preprocessing transforms.
"""

import os
from pathlib import Path
from typing import Tuple, Optional

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from PIL import Image

# ImageNet normalisation statistics used because MobileNetV3 was
# pre-trained on ImageNet. These must be applied to all images.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# Input resolution expected by MobileNetV3-Small
INPUT_SIZE = 224


def get_train_transforms() -> transforms.Compose:
    """Return augmentation + normalisation transforms for training."""
    return transforms.Compose([
        transforms.Resize((INPUT_SIZE + 32, INPUT_SIZE + 32)),
        transforms.RandomCrop(INPUT_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.ColorJitter(brightness=0.3, contrast=0.3,
                               saturation=0.2, hue=0.05),
        transforms.RandomRotation(degrees=15),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_eval_transforms() -> transforms.Compose:
    """Return deterministic transforms for validation / test / inference."""
    return transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_inference_transforms() -> transforms.Compose:
    """Same as eval transforms - kept separate for clarity."""
    return get_eval_transforms()


def load_datasets(
    data_dir: str,
    batch_size: int = 16,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader, list]:
    """
    Load train/val/test ImageFolder datasets.

    Args:
        data_dir:    Root directory containing train/, val/, test/ subfolders.
        batch_size:  Samples per batch.
        num_workers: DataLoader worker processes.

    Returns:
        (train_loader, val_loader, test_loader, class_names)
    """
    data_dir = Path(data_dir)
    train_dir = data_dir / "train"
    val_dir   = data_dir / "val"
    test_dir  = data_dir / "test"

    for split_dir in [train_dir, val_dir, test_dir]:
        if not split_dir.exists():
            raise FileNotFoundError(
                f"Split directory not found: {split_dir}\n"
                "Please create data/train, data/val, data/test with class subfolders."
            )

    train_dataset = datasets.ImageFolder(str(train_dir), transform=get_train_transforms())
    val_dataset   = datasets.ImageFolder(str(val_dir),   transform=get_eval_transforms())
    test_dataset  = datasets.ImageFolder(str(test_dir),  transform=get_eval_transforms())

    class_names = train_dataset.classes

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True,  num_workers=num_workers, pin_memory=False)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size,
                              shuffle=False, num_workers=num_workers, pin_memory=False)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size,
                              shuffle=False, num_workers=num_workers, pin_memory=False)

    return train_loader, val_loader, test_loader, class_names


def preprocess_image_for_inference(image: Image.Image) -> torch.Tensor:
    """
    Preprocess a PIL image for model inference.

    Args:
        image: PIL Image (RGB).

    Returns:
        Tensor of shape [1, 3, 224, 224].
    """
    if image.mode != "RGB":
        image = image.convert("RGB")
    transform = get_inference_transforms()
    tensor = transform(image)           # [3, H, W]
    return tensor.unsqueeze(0)          # [1, 3, H, W]
