"""
src/image_quality.py
Image quality assessment for oral photographs.
Checks: blur, brightness, contrast, resolution.
"""

from dataclasses import dataclass, field
from typing import Tuple

import cv2
import numpy as np
from PIL import Image


# Thresholds - tuned for clinical oral photographs
BLUR_THRESHOLD         = 80.0    # Laplacian variance; below = blurry
MIN_BRIGHTNESS         = 40.0    # Mean pixel brightness (0-255)
MAX_BRIGHTNESS         = 220.0
MIN_CONTRAST           = 30.0    # Standard deviation of pixel values
MIN_WIDTH              = 224
MIN_HEIGHT             = 224
IDEAL_MIN_WIDTH        = 400
IDEAL_MIN_HEIGHT       = 400


@dataclass
class QualityReport:
    """Summary of image quality assessment results."""
    blur_score: float       = 0.0
    is_blurry: bool         = False
    brightness: float       = 0.0
    is_too_dark: bool       = False
    is_too_bright: bool     = False
    contrast: float         = 0.0
    is_low_contrast: bool   = False
    width: int              = 0
    height: int             = 0
    is_low_res: bool        = False
    is_ideal_res: bool      = False
    overall_pass: bool      = False
    issues: list = field(default_factory=list)
    warnings: list = field(default_factory=list)

    @property
    def quality_score(self) -> float:
        """Composite quality score 0-100."""
        score = 100.0
        if self.is_blurry:       score -= 35.0
        if self.is_too_dark:     score -= 20.0
        if self.is_too_bright:   score -= 15.0
        if self.is_low_contrast: score -= 20.0
        if self.is_low_res:      score -= 25.0
        return max(0.0, score)

    @property
    def grade(self) -> str:
        """Letter grade for quality score."""
        s = self.quality_score
        if s >= 85: return "A"
        if s >= 70: return "B"
        if s >= 55: return "C"
        if s >= 40: return "D"
        return "F"


def pil_to_cv2(image: Image.Image) -> np.ndarray:
    """Convert PIL RGB image to OpenCV BGR uint8 array."""
    return cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)


def compute_blur_score(image: Image.Image) -> float:
    """
    Compute Laplacian variance as a blur measure.
    Higher = sharper.
    """
    gray = cv2.cvtColor(pil_to_cv2(image), cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def compute_brightness(image: Image.Image) -> float:
    """Mean luminance (0-255) of the image."""
    gray = cv2.cvtColor(pil_to_cv2(image), cv2.COLOR_BGR2GRAY)
    return float(gray.mean())


def compute_contrast(image: Image.Image) -> float:
    """Standard deviation of grayscale pixel values."""
    gray = cv2.cvtColor(pil_to_cv2(image), cv2.COLOR_BGR2GRAY)
    return float(gray.std())


def assess_quality(image: Image.Image) -> QualityReport:
    """
    Run full image quality assessment.

    Args:
        image: PIL Image.

    Returns:
        QualityReport dataclass instance.
    """
    report = QualityReport()

    # --- Blur ---
    report.blur_score = compute_blur_score(image)
    report.is_blurry  = report.blur_score < BLUR_THRESHOLD
    if report.is_blurry:
        report.issues.append(
            f"Image appears blurry (sharpness score: {report.blur_score:.1f} < {BLUR_THRESHOLD}). "
            "Please retake the photograph with better focus."
        )

    # --- Brightness ---
    report.brightness    = compute_brightness(image)
    report.is_too_dark   = report.brightness < MIN_BRIGHTNESS
    report.is_too_bright = report.brightness > MAX_BRIGHTNESS
    if report.is_too_dark:
        report.issues.append(
            f"Image is too dark (brightness: {report.brightness:.1f}). "
            "Improve lighting before capturing."
        )
    if report.is_too_bright:
        report.issues.append(
            f"Image is overexposed (brightness: {report.brightness:.1f}). "
            "Reduce flash / ambient light."
        )

    # --- Contrast ---
    report.contrast         = compute_contrast(image)
    report.is_low_contrast  = report.contrast < MIN_CONTRAST
    if report.is_low_contrast:
        report.issues.append(
            f"Low contrast detected (std: {report.contrast:.1f}). "
            "Ensure subject is well-lit and camera focused."
        )

    # --- Resolution ---
    report.width, report.height = image.size
    report.is_low_res  = report.width < MIN_WIDTH or report.height < MIN_HEIGHT
    report.is_ideal_res = report.width >= IDEAL_MIN_WIDTH and report.height >= IDEAL_MIN_HEIGHT

    if report.is_low_res:
        report.issues.append(
            f"Resolution too low ({report.width}x{report.height}). "
            f"Minimum required: {MIN_WIDTH}x{MIN_HEIGHT}."
        )
    elif not report.is_ideal_res:
        report.warnings.append(
            f"Image resolution ({report.width}x{report.height}) is acceptable but "
            f"ideally should be at least {IDEAL_MIN_WIDTH}x{IDEAL_MIN_HEIGHT}."
        )

    report.overall_pass = len(report.issues) == 0
    return report


def quality_report_to_dict(report: QualityReport) -> dict:
    """Convert QualityReport to a plain dict for JSON serialisation."""
    return {
        "blur_score":      round(report.blur_score, 2),
        "is_blurry":       report.is_blurry,
        "brightness":      round(report.brightness, 2),
        "is_too_dark":     report.is_too_dark,
        "is_too_bright":   report.is_too_bright,
        "contrast":        round(report.contrast, 2),
        "is_low_contrast": report.is_low_contrast,
        "width":           report.width,
        "height":          report.height,
        "is_low_res":      report.is_low_res,
        "quality_score":   round(report.quality_score, 1),
        "grade":           report.grade,
        "overall_pass":    report.overall_pass,
        "issues":          report.issues,
        "warnings":        report.warnings,
    }
