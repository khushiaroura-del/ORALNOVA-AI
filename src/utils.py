"""
src/utils.py
Shared utility functions for ORALNOVA-AI.
"""

import io
import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def format_confidence(prob: float) -> str:
    """Format a probability as a percentage string."""
    return f"{prob * 100:.1f}%"


def format_ms(ms: float) -> str:
    """Format milliseconds."""
    return f"{ms:.1f} ms"


def risk_level_color(risk: str) -> str:
    """Return a hex colour for the risk level."""
    return {
        "low":    "#4CAF50",
        "medium": "#FF9800",
        "high":   "#F44336",
    }.get(risk.lower(), "#9E9E9E")


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _get_pyplot():
    """Lazily import and configure matplotlib.pyplot."""
    import matplotlib
    matplotlib.use("Agg")   # non-interactive backend for server use
    import matplotlib.pyplot as plt
    return plt


def plot_probability_bar(
    probabilities: Dict[str, float],
    predicted_class: str,
) -> Image.Image:
    """
    Create a horizontal bar chart of class probabilities.

    Returns:
        PIL Image of the chart.
    """
    plt = _get_pyplot()
    classes = list(probabilities.keys())
    probs   = [probabilities[c] * 100 for c in classes]

    colors = []
    for cls in classes:
        if cls == predicted_class:
            colors.append("#72E4F2")   # icy cyan
        else:
            colors.append("#3BA6A0")   # muted teal

    fig, ax = plt.subplots(figsize=(7, 3.5))
    fig.patch.set_facecolor("#101820")
    ax.set_facecolor("#101820")

    bars = ax.barh(classes, probs, color=colors, edgecolor="none", height=0.55)

    # Labels inside / outside bars
    for bar, prob in zip(bars, probs):
        x = bar.get_width()
        ax.text(
            x + 0.5, bar.get_y() + bar.get_height() / 2,
            f"{prob:.1f}%",
            va="center", ha="left",
            color="#F4F5F2", fontsize=9, fontweight="bold",
        )

    ax.set_xlim(0, 105)
    ax.set_xlabel("Probability (%)", color="#F4F5F2", fontsize=10)
    ax.tick_params(colors="#F4F5F2", labelsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#3BA6A0")
    ax.spines["bottom"].set_color("#3BA6A0")
    ax.xaxis.label.set_color("#F4F5F2")
    ax.yaxis.label.set_color("#F4F5F2")

    plt.tight_layout(pad=0.5)

    buf = io.BytesIO()
    plt.savefig(buf, format="png", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).copy()


def plot_confusion_matrix(
    cm: list,
    class_names: list,
) -> Image.Image:
    """
    Plot a confusion matrix as a PIL Image.
    """
    plt = _get_pyplot()
    cm_arr = np.array(cm, dtype=float)
    # Normalise rows
    row_sums = cm_arr.sum(axis=1, keepdims=True)
    cm_norm  = np.divide(cm_arr, row_sums, out=np.zeros_like(cm_arr), where=row_sums != 0)

    fig, ax = plt.subplots(figsize=(6, 5))
    fig.patch.set_facecolor("#101820")
    ax.set_facecolor("#101820")

    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(colors="#F4F5F2")

    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(class_names, rotation=30, ha="right",
                       color="#F4F5F2", fontsize=8)
    ax.set_yticklabels(class_names, color="#F4F5F2", fontsize=8)
    ax.set_xlabel("Predicted", color="#F4F5F2")
    ax.set_ylabel("Actual", color="#F4F5F2")
    ax.tick_params(colors="#F4F5F2")

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, f"{int(cm_arr[i,j])}",
                    ha="center", va="center",
                    color="white" if cm_norm[i,j] > 0.5 else "#72E4F2",
                    fontsize=9)

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).copy()


# ---------------------------------------------------------------------------
# File helpers
# ---------------------------------------------------------------------------

def load_evaluation_results(reports_dir: str = "reports", filename: str = "evaluation_results.json") -> Optional[Dict[str, Any]]:
    """Load evaluation results JSON if it exists."""
    path = Path(reports_dir) / filename
    if path.exists():
        with open(path, "r") as f:
            return json.load(f)
    return None


def ensure_dir(path: str):
    """Create directory if it does not exist."""
    Path(path).mkdir(parents=True, exist_ok=True)
