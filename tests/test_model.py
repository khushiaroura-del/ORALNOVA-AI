"""
tests/test_model.py
Unit tests for model architecture, checkpoints, and metadata.
"""

import os
import sys
import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import NUM_CLASSES, CLASS_NAMES
from src.model import build_model, load_checkpoint, get_model_info
from src.train import save_checkpoint


class TestModelArchitecture:
    def test_build_model_default_classes(self):
        model = build_model(num_classes=NUM_CLASSES, pretrained=False)
        assert isinstance(model, nn.Module)
        assert model.classifier[3].out_features == NUM_CLASSES

    def test_build_model_custom_classes(self):
        model = build_model(num_classes=3, pretrained=False)
        assert model.classifier[3].out_features == 3

    def test_model_info(self):
        model = build_model(num_classes=NUM_CLASSES, pretrained=False)
        info = get_model_info(model)
        assert info["architecture"] == "MobileNetV3-Small"
        assert info["total_params"] > 0
        assert info["trainable_params"] > 0
        assert "total_params_m" in info


class TestCheckpoints:
    def test_save_and_load_checkpoint(self):
        model = build_model(num_classes=NUM_CLASSES, pretrained=False)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        with tempfile.TemporaryDirectory() as tmp_dir:
            ckpt_path = Path(tmp_dir) / "test_model.pth"
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=5,
                val_acc=0.88,
                val_loss=0.32,
                path=ckpt_path,
                classes=CLASS_NAMES,
            )
            assert ckpt_path.exists()

            new_model = build_model(num_classes=NUM_CLASSES, pretrained=False)
            meta = load_checkpoint(new_model, str(ckpt_path), device=torch.device("cpu"))

            assert meta["epoch"] == 5
            assert meta["val_acc"] == 0.88
            assert meta["val_loss"] == 0.32
            assert meta["classes"] == CLASS_NAMES
