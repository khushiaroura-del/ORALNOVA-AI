"""
tests/test_image_quality.py
Unit tests for image_quality module.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
from PIL import Image

from src.image_quality import (
    assess_quality,
    compute_blur_score,
    compute_brightness,
    compute_contrast,
    quality_report_to_dict,
    BLUR_THRESHOLD,
    MIN_BRIGHTNESS,
    MAX_BRIGHTNESS,
    MIN_CONTRAST,
    MIN_WIDTH,
    MIN_HEIGHT,
)


def make_image(width: int = 400, height: int = 400,
               value: int = 128, noise: int = 50) -> Image.Image:
    """Create a synthetic PIL image with controllable brightness."""
    arr = np.clip(
        np.full((height, width, 3), value, dtype=np.int32) +
        np.random.randint(-noise, noise, (height, width, 3)),
        0, 255
    ).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def make_sharp_image(size: int = 400) -> Image.Image:
    """Image with high-frequency content = sharp."""
    arr = np.zeros((size, size, 3), dtype=np.uint8)
    arr[::2, ::2] = 255
    return Image.fromarray(arr, mode="RGB")


def make_blurry_image(size: int = 400) -> Image.Image:
    """A uniform solid colour image = blurry (zero Laplacian variance)."""
    arr = np.full((size, size, 3), 150, dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


class TestComputeBlurScore:
    def test_sharp_image_high_score(self):
        img = make_sharp_image()
        score = compute_blur_score(img)
        assert score > BLUR_THRESHOLD, f"Sharp image should have blur score > {BLUR_THRESHOLD}, got {score}"

    def test_blurry_image_low_score(self):
        img = make_blurry_image()
        score = compute_blur_score(img)
        assert score < BLUR_THRESHOLD, f"Blurry image should have blur score < {BLUR_THRESHOLD}, got {score}"

    def test_returns_float(self):
        img = make_image()
        assert isinstance(compute_blur_score(img), float)


class TestComputeBrightness:
    def test_bright_image(self):
        img = make_image(value=230, noise=5)
        b = compute_brightness(img)
        assert b > 200

    def test_dark_image(self):
        img = make_image(value=20, noise=5)
        b = compute_brightness(img)
        assert b < 50

    def test_returns_float(self):
        assert isinstance(compute_brightness(make_image()), float)


class TestComputeContrast:
    def test_high_contrast(self):
        img = make_sharp_image()
        c = compute_contrast(img)
        assert c > MIN_CONTRAST

    def test_low_contrast(self):
        """Uniform solid colour has near-zero contrast."""
        img = make_blurry_image()
        c = compute_contrast(img)
        assert c < MIN_CONTRAST

    def test_returns_float(self):
        assert isinstance(compute_contrast(make_image()), float)


class TestAssessQuality:
    def test_good_image_passes(self):
        img = make_sharp_image(500)
        report = assess_quality(img)
        # Sharp image with mid-brightness should pass blur check
        assert not report.is_blurry
        assert not report.is_low_res

    def test_low_res_image(self):
        img = make_image(width=100, height=100)
        report = assess_quality(img)
        assert report.is_low_res
        assert not report.overall_pass
        assert len(report.issues) > 0

    def test_dark_image(self):
        img = make_image(value=10, noise=3)
        report = assess_quality(img)
        assert report.is_too_dark
        assert not report.overall_pass

    def test_bright_image(self):
        img = make_image(value=245, noise=3)
        report = assess_quality(img)
        assert report.is_too_bright
        assert not report.overall_pass

    def test_quality_score_range(self):
        img = make_image()
        report = assess_quality(img)
        assert 0.0 <= report.quality_score <= 100.0

    def test_grade_is_letter(self):
        img = make_image()
        report = assess_quality(img)
        assert report.grade in ("A", "B", "C", "D", "F")

    def test_report_dimensions(self):
        img = make_image(width=640, height=480)
        report = assess_quality(img)
        assert report.width == 640
        assert report.height == 480

    def test_blurry_image_flagged(self):
        img = make_blurry_image()
        report = assess_quality(img)
        assert report.is_blurry


class TestQualityReportToDict:
    def test_returns_dict(self):
        img = make_image()
        report = assess_quality(img)
        d = quality_report_to_dict(report)
        assert isinstance(d, dict)

    def test_required_keys(self):
        img = make_image()
        report = assess_quality(img)
        d = quality_report_to_dict(report)
        for key in ["blur_score", "brightness", "contrast", "quality_score", "grade",
                    "overall_pass", "issues", "warnings"]:
            assert key in d, f"Missing key: {key}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
