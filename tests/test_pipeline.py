"""
tests/test_pipeline.py
Unit tests for OralLesionScreeningPipeline and Grad-CAM integration.
"""

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import CLASS_NAMES, NUM_CLASSES
from src.model import build_model
from src.train import save_checkpoint
from src.screening_pipeline import OralLesionScreeningPipeline
from src.gradcam import get_gradcam_for_mobilenetv3


def create_dummy_image(width: int = 300, height: int = 300) -> Image.Image:
    arr = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


class TestScreeningPipeline:
    def test_pipeline_missing_checkpoint(self):
        pipeline = OralLesionScreeningPipeline("models/non_existent.pth")
        assert not pipeline.is_loaded
        assert "not found" in pipeline.load_error.lower()

        img = create_dummy_image()
        result = pipeline.run(img)
        assert not result.success
        assert "checkpoint not found" in result.error_message.lower()
        assert result.quality_report is not None

    def test_pipeline_with_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            ckpt_path = Path(tmp_dir) / "oralnova_best.pth"
            model = build_model(num_classes=NUM_CLASSES, pretrained=False)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=1,
                val_acc=0.9,
                val_loss=0.2,
                path=ckpt_path,
                classes=CLASS_NAMES,
            )

            pipeline = OralLesionScreeningPipeline(str(ckpt_path), device="cpu")
            assert pipeline.is_loaded

            img = create_dummy_image(300, 300)
            result = pipeline.run(img)

            assert result.success
            assert result.predicted_class in CLASS_NAMES
            assert 0.0 <= result.confidence <= 1.0
            assert len(result.probabilities) == NUM_CLASSES
            assert result.gradcam_heatmap is not None
            assert result.gradcam_overlay is not None
            assert isinstance(result.gradcam_overlay, Image.Image)

    def test_pipeline_binary_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            ckpt_path = Path(tmp_dir) / "oralnova_binary_best.pth"
            model = build_model(num_classes=2, pretrained=False)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            binary_classes = ["Benign", "Malignant"]

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=2,
                val_acc=0.85,
                val_loss=0.35,
                path=ckpt_path,
                classes=binary_classes,
            )

            pipeline = OralLesionScreeningPipeline(str(ckpt_path), device="cpu")
            assert pipeline.is_loaded
            assert pipeline.class_names == binary_classes

            img = create_dummy_image(224, 224)
            result = pipeline.run(img)
            assert result.success
            assert result.predicted_class in binary_classes
            assert len(result.probabilities) == 2
            assert "Benign" in result.probabilities
            assert "Malignant" in result.probabilities


class TestGradCAM:
    def test_gradcam_hooks_and_cleanup(self):
        model = build_model(num_classes=NUM_CLASSES, pretrained=False)
        gradcam = get_gradcam_for_mobilenetv3(model)

        assert gradcam._fwd_handle is not None
        assert gradcam._bwd_handle is not None

        # Clean up
        gradcam.cleanup()
        assert gradcam._fwd_handle is None
        assert gradcam._bwd_handle is None
