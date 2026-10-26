"""
src/screening_pipeline.py
End-to-end oral lesion screening pipeline:
  1. Load model
  2. Preprocess image
  3. Run inference
  4. Generate Grad-CAM
  5. Return structured result
"""

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from src import CLASS_NAMES, CLASS_DESCRIPTIONS, CLASS_RISK, NUM_CLASSES
from src.model import build_model, load_checkpoint, get_model_info
from src.data_preprocessing import preprocess_image_for_inference
from src.image_quality import assess_quality, QualityReport
from src.gradcam import get_gradcam_for_mobilenetv3


@dataclass
class ScreeningResult:
    """Full result from a screening pipeline run."""
    # Prediction
    predicted_class: str = ""
    predicted_idx: int = 0
    confidence: float = 0.0
    probabilities: Dict[str, float] = field(default_factory=dict)
    class_description: str = ""
    risk_level: str = ""
    # Timing
    inference_time_ms: float = 0.0
    # Quality
    quality_report: Optional[QualityReport] = None
    # Grad-CAM
    gradcam_heatmap: Optional[np.ndarray] = None
    gradcam_overlay: Optional[Image.Image] = None
    # Status
    success: bool = False
    error_message: str = ""
    # Model metadata
    model_info: Dict = field(default_factory=dict)
    checkpoint_meta: Dict = field(default_factory=dict)


class OralLesionScreeningPipeline:
    """
    Complete screening pipeline for oral lesion classification.

    Example:
        pipeline = OralLesionScreeningPipeline("models/oralnova_best.pth")
        result = pipeline.run(pil_image)
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        device: Optional[str] = None,
    ):
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = None
        self.checkpoint_meta: Dict = {}
        self.model_info: Dict = {}
        self.checkpoint_path = checkpoint_path
        self.is_loaded = False
        self.load_error: str = ""

        if checkpoint_path:
            self._load_model(checkpoint_path)

    def _load_model(self, checkpoint_path: str):
        """Attempt to load model from checkpoint."""
        path = Path(checkpoint_path)
        if not path.exists():
            self.load_error = (
                f"Checkpoint not found: {checkpoint_path}\n\n"
                "To generate a checkpoint, train the model first:\n"
                "  python -m src.train --data_dir data\n\n"
                "A real dataset of clinical oral photographs is required for training."
            )
            return

        try:
            temp_checkpoint = torch.load(checkpoint_path, map_location="cpu")
            classes = temp_checkpoint.get("classes") or CLASS_NAMES
            self.class_names = list(classes)
            num_classes = len(self.class_names)

            self.model = build_model(num_classes=num_classes, pretrained=False)
            self.checkpoint_meta = load_checkpoint(
                self.model, checkpoint_path, self.device
            )
            self.model.to(self.device)
            self.model.eval()
            self.model_info = get_model_info(self.model)
            self.is_loaded = True
        except Exception as e:
            self.load_error = f"Error loading checkpoint: {e}"
            self.model = None

    def run(self, image: Image.Image) -> ScreeningResult:
        """
        Run the full screening pipeline on a PIL image.

        Args:
            image: PIL Image (any mode - converted internally).

        Returns:
            ScreeningResult dataclass.
        """
        result = ScreeningResult(
            model_info=self.model_info,
            checkpoint_meta=self.checkpoint_meta,
        )

        # --- Image quality check ---
        result.quality_report = assess_quality(image)

        # --- Model check ---
        if not self.is_loaded:
            result.success = False
            result.error_message = self.load_error or "No model loaded."
            return result

        try:
            # --- Preprocessing ---
            input_tensor = preprocess_image_for_inference(image).to(self.device)

            # --- Inference ---
            t_start = time.perf_counter()
            with torch.no_grad():
                logits = self.model(input_tensor)
                probs  = F.softmax(logits, dim=1)[0]
            t_end = time.perf_counter()

            result.inference_time_ms = (t_end - t_start) * 1000

            probs_np = probs.cpu().numpy()
            pred_idx = int(np.argmax(probs_np))

            class_names = getattr(self, "class_names", CLASS_NAMES)
            result.predicted_idx   = pred_idx
            result.predicted_class = class_names[pred_idx]
            result.confidence      = float(probs_np[pred_idx])
            result.probabilities   = {
                cls: float(probs_np[i]) for i, cls in enumerate(class_names)
            }
            result.class_description = CLASS_DESCRIPTIONS.get(
                result.predicted_class, "Clinical description unavailable."
            )
            result.risk_level = CLASS_RISK.get(result.predicted_class, "low")

            # --- Grad-CAM ---
            try:
                gradcam = get_gradcam_for_mobilenetv3(self.model)
                input_for_cam = preprocess_image_for_inference(image).to(self.device)
                heatmap = gradcam(input_for_cam, class_idx=pred_idx)

                result.gradcam_heatmap = heatmap
                result.gradcam_overlay = gradcam.overlay(image, heatmap, alpha=0.45)
                gradcam.cleanup()
            except Exception as cam_err:
                result.gradcam_heatmap = None
                result.gradcam_overlay = None

            result.success = True

        except Exception as e:
            result.success = False
            result.error_message = f"Inference error: {e}"

        return result
