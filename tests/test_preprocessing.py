"""
tests/test_preprocessing.py
Unit tests for data_preprocessing module.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch
import numpy as np
from PIL import Image

from src.data_preprocessing import (
    get_train_transforms,
    get_eval_transforms,
    get_inference_transforms,
    preprocess_image_for_inference,
    INPUT_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
)


def make_test_image(width: int = 300, height: int = 300, mode: str = "RGB") -> Image.Image:
    """Create a synthetic PIL image for testing."""
    arr = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    return Image.fromarray(arr, mode=mode)


class TestTransforms:
    def test_eval_transforms_output_shape(self):
        img = make_test_image(300, 300)
        transform = get_eval_transforms()
        tensor = transform(img)
        assert tensor.shape == (3, INPUT_SIZE, INPUT_SIZE), (
            f"Expected (3, {INPUT_SIZE}, {INPUT_SIZE}), got {tensor.shape}"
        )

    def test_train_transforms_output_shape(self):
        img = make_test_image(300, 300)
        transform = get_train_transforms()
        tensor = transform(img)
        assert tensor.shape == (3, INPUT_SIZE, INPUT_SIZE)

    def test_inference_transforms_output_shape(self):
        img = make_test_image(512, 512)
        transform = get_inference_transforms()
        tensor = transform(img)
        assert tensor.shape == (3, INPUT_SIZE, INPUT_SIZE)

    def test_eval_transforms_dtype(self):
        img = make_test_image()
        tensor = get_eval_transforms()(img)
        assert tensor.dtype == torch.float32

    def test_eval_transforms_range(self):
        """After normalisation the range may exceed [0,1] - just check it is float."""
        img = make_test_image()
        tensor = get_eval_transforms()(img)
        assert tensor.min().item() < 1.0  # some values normalised below 1

    def test_transforms_are_deterministic(self):
        """Eval transforms must be deterministic (no random augmentation)."""
        img = make_test_image(300, 300)
        t1 = get_eval_transforms()(img)
        t2 = get_eval_transforms()(img)
        assert torch.allclose(t1, t2), "Eval transforms should be deterministic"


class TestPreprocessImageForInference:
    def test_output_shape(self):
        img = make_test_image(400, 400)
        tensor = preprocess_image_for_inference(img)
        assert tensor.shape == (1, 3, INPUT_SIZE, INPUT_SIZE), (
            f"Expected (1, 3, {INPUT_SIZE}, {INPUT_SIZE}), got {tensor.shape}"
        )

    def test_output_dtype(self):
        img = make_test_image()
        tensor = preprocess_image_for_inference(img)
        assert tensor.dtype == torch.float32

    def test_rgba_conversion(self):
        """RGBA images should be auto-converted to RGB."""
        arr = np.random.randint(0, 255, (300, 300, 4), dtype=np.uint8)
        img = Image.fromarray(arr, mode="RGBA")
        tensor = preprocess_image_for_inference(img)
        assert tensor.shape == (1, 3, INPUT_SIZE, INPUT_SIZE)

    def test_grayscale_conversion(self):
        """Grayscale images should be auto-converted to RGB."""
        arr = np.random.randint(0, 255, (300, 300), dtype=np.uint8)
        img = Image.fromarray(arr, mode="L")
        tensor = preprocess_image_for_inference(img)
        assert tensor.shape == (1, 3, INPUT_SIZE, INPUT_SIZE)

    def test_small_image(self):
        """Images smaller than INPUT_SIZE should be upscaled."""
        img = make_test_image(50, 50)
        tensor = preprocess_image_for_inference(img)
        assert tensor.shape == (1, 3, INPUT_SIZE, INPUT_SIZE)

    def test_large_image(self):
        """Large images should be downscaled."""
        img = make_test_image(1920, 1080)
        tensor = preprocess_image_for_inference(img)
        assert tensor.shape == (1, 3, INPUT_SIZE, INPUT_SIZE)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
