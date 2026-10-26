"""
src/model.py
MobileNetV3-Small architecture for oral lesion classification.
"""

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import MobileNet_V3_Small_Weights
from src import NUM_CLASSES


def build_model(num_classes: int = NUM_CLASSES, pretrained: bool = True) -> nn.Module:
    """
    Build MobileNetV3-Small with a custom classification head.

    Args:
        num_classes: Number of output classes.
        pretrained:  If True, load ImageNet weights for the backbone.

    Returns:
        nn.Module - the modified model.
    """
    weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.mobilenet_v3_small(weights=weights)

    # Replace the classifier head
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)

    return model


def load_checkpoint(model: nn.Module, checkpoint_path: str, device: torch.device) -> dict:
    """
    Load a saved checkpoint into the model.

    Args:
        model:           The model instance.
        checkpoint_path: Path to the .pth file.
        device:          Target device.

    Returns:
        dict with 'epoch', 'val_acc', 'val_loss' metadata (if saved).
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    meta = {
        "epoch": checkpoint.get("epoch", "unknown"),
        "val_acc": checkpoint.get("val_acc", None),
        "val_loss": checkpoint.get("val_loss", None),
        "classes": checkpoint.get("classes", None),
    }
    return meta


def get_model_info(model: nn.Module) -> dict:
    """Return parameter count and model name."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "architecture": "MobileNetV3-Small",
        "total_params": total,
        "trainable_params": trainable,
        "total_params_m": round(total / 1_000_000, 2),
        "trainable_params_m": round(trainable / 1_000_000, 2),
    }
