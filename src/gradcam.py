"""
src/gradcam.py
Grad-CAM (Gradient-weighted Class Activation Mapping) for MobileNetV3-Small.
Produces a heatmap overlay that highlights regions influencing the prediction.
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from typing import Optional, Tuple


class GradCAM:
    """
    Grad-CAM implementation for any CNN model.

    Usage:
        cam = GradCAM(model, target_layer=model.features[-1])
        heatmap = cam(input_tensor, class_idx)
        overlay = cam.overlay(original_image, heatmap)
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self._gradients: Optional[torch.Tensor] = None
        self._activations: Optional[torch.Tensor] = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self._activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self._gradients = grad_output[0].detach()

        self._fwd_handle = self.target_layer.register_forward_hook(forward_hook)
        self._bwd_handle = self.target_layer.register_full_backward_hook(backward_hook)

    def __call__(
        self,
        input_tensor: torch.Tensor,
        class_idx: Optional[int] = None,
    ) -> np.ndarray:
        """
        Compute Grad-CAM heatmap.

        Args:
            input_tensor: [1, 3, H, W] preprocessed image tensor.
            class_idx:    Target class index. If None, uses top prediction.

        Returns:
            Heatmap as a float32 numpy array in [0, 1] with shape [H, W].
        """
        self.model.eval()
        input_tensor = input_tensor.requires_grad_(True)

        logits = self.model(input_tensor)

        if class_idx is None:
            class_idx = int(logits.argmax(dim=1).item())

        self.model.zero_grad()
        score = logits[0, class_idx]
        score.backward()

        # Gradients: [1, C, H, W]
        gradients = self._gradients   # [1, C, H, W]
        activations = self._activations  # [1, C, H, W]

        # Global average pooling of gradients
        weights = gradients.mean(dim=(2, 3), keepdim=True)  # [1, C, 1, 1]

        # Weighted combination of activation maps
        cam = (weights * activations).sum(dim=1, keepdim=True)  # [1, 1, H, W]
        cam = F.relu(cam)

        # Normalise to [0, 1]
        cam = cam.squeeze().cpu().numpy()
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        else:
            cam = np.zeros_like(cam)

        return cam.astype(np.float32)

    def overlay(
        self,
        original_image: Image.Image,
        heatmap: np.ndarray,
        alpha: float = 0.4,
        colormap: int = cv2.COLORMAP_JET,
    ) -> Image.Image:
        """
        Overlay the Grad-CAM heatmap on the original image.

        Args:
            original_image: PIL image (RGB).
            heatmap:        Grad-CAM heatmap [H, W] in [0, 1].
            alpha:          Heatmap transparency (0=invisible, 1=opaque).
            colormap:       OpenCV colormap.

        Returns:
            PIL Image with heatmap overlay.
        """
        orig = np.array(original_image.convert("RGB"))
        h, w = orig.shape[:2]

        # Resize heatmap to original image size
        heatmap_resized = cv2.resize(heatmap, (w, h))
        heatmap_uint8 = (heatmap_resized * 255).astype(np.uint8)
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)
        heatmap_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

        # Blend
        blended = cv2.addWeighted(orig, 1 - alpha, heatmap_rgb, alpha, 0)
        return Image.fromarray(blended.astype(np.uint8))

    def cleanup(self):
        """Remove hooks to free memory and prevent accumulation."""
        if hasattr(self, "_fwd_handle") and self._fwd_handle is not None:
            self._fwd_handle.remove()
            self._fwd_handle = None
        if hasattr(self, "_bwd_handle") and self._bwd_handle is not None:
            self._bwd_handle.remove()
            self._bwd_handle = None
        self._gradients = None
        self._activations = None


def get_gradcam_for_mobilenetv3(model: nn.Module) -> GradCAM:
    """
    Create GradCAM for MobileNetV3-Small, targeting the last conv block.

    Args:
        model: MobileNetV3-Small nn.Module.

    Returns:
        Configured GradCAM instance.
    """
    # MobileNetV3-Small: model.features[-1] is the last ConvBNActivation block
    target_layer = model.features[-1]
    return GradCAM(model, target_layer)
