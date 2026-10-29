"""
app.py
ORALNOVA-AI -- Lightweight Deep Learning for Oral Lesion Screening
Streamlit Dashboard
"""

import io
import os
import sys
import time
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).parent))

from src import CLASS_NAMES, CLASS_DESCRIPTIONS, CLASS_RISK, NUM_CLASSES
from src.image_quality import assess_quality, quality_report_to_dict
from src.model import build_model, get_model_info
from src.screening_pipeline import OralLesionScreeningPipeline
from src.utils import (
    format_confidence,
    format_ms,
    risk_level_color,
    plot_probability_bar,
    load_evaluation_results,
)

# ---------------------------------------------------------------------------
# Page configuration -- MUST be first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="ORALNOVA-AI | Oral Lesion Screening",
    page_icon="\U0001f9b7",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS -- Premium dark theme
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@400;500;600;700&display=swap');

:root {
  --navy:    #07111F;
  --charcoal:#101820;
  --ivory:   #F4F5F2;
  --cyan:    #72E4F2;
  --teal:    #3BA6A0;
  --teal-dim:#2A7A75;
  --surface: #0D1B2A;
  --card:    #132030;
  --border:  #1E3045;
  --warn:    #FF9800;
  --danger:  #EF5350;
  --success: #4CAF50;
  --font-main: 'Inter', sans-serif;
  --font-head: 'Space Grotesk', sans-serif;
}

html, body, [class*="css"] {
  font-family: var(--font-main);
  background-color: var(--navy) !important;
  color: var(--ivory) !important;
}

.stApp {
  background: linear-gradient(135deg, #07111F 0%, #0D1E30 60%, #07111F 100%) !important;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem !important; }

section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #0A1828 0%, #07111F 100%) !important;
  border-right: 1px solid var(--border) !important;
}

.oralnova-hero {
  background: linear-gradient(135deg, #0D1E30 0%, #0F2640 50%, #0A1D32 100%);
  border: 1px solid var(--border);
  border-radius: 20px;
  padding: 2.5rem 2rem 2rem 2rem;
  margin-bottom: 1.5rem;
  position: relative;
  overflow: hidden;
}

.hero-title {
  font-family: var(--font-head);
  font-size: 2.6rem;
  font-weight: 700;
  background: linear-gradient(90deg, #72E4F2 20%, #3BA6A0 80%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  margin: 0 0 0.2rem 0;
}

.hero-tagline {
  color: rgba(244,245,242,0.65);
  font-size: 1rem;
  margin: 0 0 1.2rem 0;
}

.hero-pills { display: flex; flex-wrap: wrap; gap: 0.5rem; }
.pill {
  background: rgba(59,166,160,0.15);
  border: 1px solid rgba(59,166,160,0.35);
  color: #72E4F2;
  border-radius: 20px;
  padding: 0.3rem 0.8rem;
  font-size: 0.78rem;
  font-weight: 500;
}

.nova-card {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 1.4rem;
  margin-bottom: 1rem;
}

.nova-card-title {
  font-family: var(--font-head);
  color: var(--cyan);
  font-size: 0.85rem;
  font-weight: 600;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  margin-bottom: 0.8rem;
}

.stat-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
  gap: 0.75rem;
  margin: 0.75rem 0;
}

.stat-box {
  background: linear-gradient(135deg, #0D1E30, #132030);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 1rem 0.8rem;
  text-align: center;
}

.stat-value {
  font-family: var(--font-head);
  font-size: 1.55rem;
  font-weight: 700;
  color: var(--cyan);
  line-height: 1.1;
}

.stat-label {
  font-size: 0.72rem;
  color: rgba(244,245,242,0.55);
  text-transform: uppercase;
  letter-spacing: 0.06em;
  margin-top: 0.25rem;
}

.risk-badge {
  display: inline-block;
  border-radius: 8px;
  padding: 0.3rem 0.8rem;
  font-size: 0.8rem;
  font-weight: 600;
  text-transform: uppercase;
}

.risk-low    { background: rgba(76,175,80,0.15);  color: #4CAF50; border: 1px solid rgba(76,175,80,0.4); }
.risk-medium { background: rgba(255,152,0,0.15);  color: #FF9800; border: 1px solid rgba(255,152,0,0.4); }
.risk-high   { background: rgba(239,83,80,0.15);  color: #EF5350; border: 1px solid rgba(239,83,80,0.4); }

.quality-bar-track {
  background: var(--border);
  border-radius: 6px;
  height: 8px;
  overflow: hidden;
  margin-top: 4px;
}
.quality-bar-fill {
  height: 100%;
  border-radius: 6px;
}

.section-div { border: none; border-top: 1px solid var(--border); margin: 1.2rem 0; }

.disclaimer-box {
  background: rgba(239,83,80,0.07);
  border: 1px solid rgba(239,83,80,0.25);
  border-radius: 12px;
  padding: 1rem 1.2rem;
  color: rgba(244,245,242,0.7);
  font-size: 0.82rem;
  line-height: 1.55;
}

[data-testid="stFileUploadDropzone"] {
  background: rgba(19,32,48,0.6) !important;
  border: 2px dashed #2A7A75 !important;
  border-radius: 14px !important;
}

.stButton > button {
  background: linear-gradient(135deg, #3BA6A0, #2A7A75) !important;
  color: #F4F5F2 !important;
  border: none !important;
  border-radius: 10px !important;
  font-weight: 600 !important;
  padding: 0.55rem 1.5rem !important;
  box-shadow: 0 2px 12px rgba(59,166,160,0.25) !important;
}

[data-testid="stMetricLabel"] { color: rgba(244,245,242,0.6) !important; font-size: 0.78rem !important; }
[data-testid="stMetricValue"] { color: #72E4F2 !important; }
.stTabs [data-baseweb="tab"] { color: rgba(244,245,242,0.55) !important; }
.stTabs [aria-selected="true"] { color: #72E4F2 !important; border-bottom-color: #72E4F2 !important; }
.stProgress > div > div { background: linear-gradient(90deg, #3BA6A0, #72E4F2) !important; }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_CHECKPOINT = "models/oralnova_binary_best.pth"

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "pipeline" not in st.session_state:
    st.session_state.pipeline = None
if "pipeline_loaded_path" not in st.session_state:
    st.session_state.pipeline_loaded_path = None

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        "<div style='text-align:center;padding:0.5rem 0 1.2rem 0;'>"
        "<div style='font-size:1.3rem;font-weight:700;color:#72E4F2;'>"
        "&#x1F9B7; ORALNOVA-AI</div>"
        "<div style='color:rgba(244,245,242,0.45);font-size:0.72rem;margin-top:0.2rem;'>"
        "v1.0.0 -- Research Prototype</div></div>"
        "<hr style='border-color:#1E3045;margin:0 0 1rem 0;'>",
        unsafe_allow_html=True,
    )

    st.markdown("### Model Settings")
    checkpoint_path = st.text_input(
        "Checkpoint Path",
        value=DEFAULT_CHECKPOINT,
        help="Path to a trained .pth file.",
    )

    load_btn = st.button("Reload Model", use_container_width=True)

    st.markdown("<hr style='border-color:#1E3045;'>", unsafe_allow_html=True)
    st.markdown("### About")
    st.markdown(
        "<div style='color:rgba(244,245,242,0.6);font-size:0.80rem;line-height:1.6;'>"
        "<b>Model:</b> MobileNetV3-Small<br>"
        "<b>Classes:</b> 2 output classes (Benign / Malignant)<br>"
        "<b>Explainability:</b> Grad-CAM<br>"
        "<b>Device:</b> CPU<br>"
        "<b>Modality:</b> Clinical photographs</div>",
        unsafe_allow_html=True,
    )

    st.markdown("<hr style='border-color:#1E3045;'>", unsafe_allow_html=True)
    st.markdown("### Class Reference")
    for cls, desc in CLASS_DESCRIPTIONS.items():
        risk = CLASS_RISK[cls]
        st.markdown(
            f"<div style='margin-bottom:0.6rem;'>"
            f"<span class='risk-badge risk-{risk}'>{risk.upper()}</span> "
            f"<b style='color:#F4F5F2;'>{cls}</b><br>"
            f"<span style='color:rgba(244,245,242,0.5);font-size:0.75rem;'>{desc}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown("<hr style='border-color:#1E3045;'>", unsafe_allow_html=True)
    st.markdown(
        "<div class='disclaimer-box'>"
        "<b>Medical Disclaimer</b><br>"
        "This is an educational AI prototype. NOT a medical device. "
        "Always consult a qualified healthcare provider.</div>",
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Hero banner
# ---------------------------------------------------------------------------
st.markdown(
    "<div class='oralnova-hero'>"
    "<div class='hero-title'>&#x1F9B7; ORALNOVA-AI</div>"
    "<p class='hero-tagline'>"
    "Lightweight Deep Learning for Oral Lesion Screening &middot; "
    "AI-Assisted Analysis with Explainable Predictions</p>"
    "<div class='hero-pills'>"
    "<span class='pill'>&#x1F9E0; MobileNetV3-Small</span>"
    "<span class='pill'>&#x1F525; Grad-CAM XAI</span>"
    "<span class='pill'>&#x1F4CA; 2 Output Classes</span>"
    "<span class='pill'>&#x1F5BC; Quality Analysis</span>"
    "<span class='pill'>&#x26A1; Real-Time Inference</span>"
    "<span class='pill'>&#x1F393; Educational Prototype</span>"
    "</div></div>",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Load / reload pipeline
# ---------------------------------------------------------------------------
def get_pipeline(path: str) -> OralLesionScreeningPipeline:
    if (
        st.session_state.pipeline is None
        or st.session_state.pipeline_loaded_path != path
    ):
        with st.spinner("Loading model pipeline..."):
            st.session_state.pipeline = OralLesionScreeningPipeline(
                checkpoint_path=path
            )
            st.session_state.pipeline_loaded_path = path
    return st.session_state.pipeline


if load_btn:
    st.session_state.pipeline = None

pipeline = get_pipeline(checkpoint_path)

# ---------------------------------------------------------------------------
# Model status strip
# ---------------------------------------------------------------------------
mc1, mc2, mc3, mc4 = st.columns(4)

if pipeline.is_loaded:
    meta = pipeline.checkpoint_meta
    info = pipeline.model_info
    with mc1: st.metric("Model Status", "Loaded")
    with mc2: st.metric("Architecture", info.get("architecture", "MobileNetV3"))
    with mc3: st.metric("Checkpoint Epoch", str(meta.get("epoch", "-")))
    with mc4:
        va = meta.get("val_acc")
        st.metric("Val Accuracy", f"{va*100:.1f}%" if va else "-")
else:
    with mc1: st.metric("Model Status", "Not Loaded")
    with mc2: st.metric("Architecture", "MobileNetV3-Small")
    with mc3: st.metric("Parameters", "~2.54 M")
    with mc4: st.metric("Device", "CPU")

    st.warning(
        f"**No trained checkpoint found at `{checkpoint_path}`.**\n\n"
        "Train the model first:\n```\npython -m src.train --data_dir data\n```\n\n"
        "A real dataset of clinical oral photographs is required. "
        "Image quality analysis works without a checkpoint."
    )

st.markdown("<hr class='section-div'>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Main layout
# ---------------------------------------------------------------------------
left_col, right_col = st.columns([1, 1.6], gap="large")

with left_col:
    st.markdown("<div class='nova-card'>", unsafe_allow_html=True)
    st.markdown("<div class='nova-card-title'>Image Upload</div>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Upload oral photograph",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        label_visibility="collapsed",
    )
    st.markdown("</div>", unsafe_allow_html=True)

    if uploaded_file is not None:
        image = Image.open(io.BytesIO(uploaded_file.read()))

        # Preview
        st.markdown("<div class='nova-card'>", unsafe_allow_html=True)
        st.markdown("<div class='nova-card-title'>Image Preview</div>", unsafe_allow_html=True)
        st.image(image, use_container_width=True, caption=uploaded_file.name)
        w, h = image.size
        st.markdown(
            f"<div style='color:rgba(244,245,242,0.5);font-size:0.78rem;text-align:center;'>"
            f"{w} x {h} px &middot; {uploaded_file.type}</div>",
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

        # Quality Analysis
        st.markdown("<div class='nova-card'>", unsafe_allow_html=True)
        st.markdown("<div class='nova-card-title'>Image Quality Analysis</div>", unsafe_allow_html=True)

        with st.spinner("Analysing image quality..."):
            qr = assess_quality(image)

        qs = qr.quality_score
        grade_color = {
            "A": "#4CAF50", "B": "#8BC34A", "C": "#FF9800",
            "D": "#FF5722", "F": "#F44336",
        }.get(qr.grade, "#9E9E9E")

        st.markdown(
            f"<div style='display:flex;align-items:center;gap:1rem;margin-bottom:0.8rem;'>"
            f"<div style='font-size:2.4rem;font-weight:800;color:{grade_color};'>{qr.grade}</div>"
            f"<div>"
            f"  <div style='font-size:1rem;color:#F4F5F2;font-weight:600;'>Quality Score: {qs:.0f}/100</div>"
            f"  <div style='font-size:0.78rem;color:rgba(244,245,242,0.5);'>"
            f"    {'Acceptable for analysis' if qr.overall_pass else 'Issues detected'}</div>"
            f"</div></div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<div class='quality-bar-track'>"
            f"<div class='quality-bar-fill' style='width:{qs}%;background:{grade_color};'></div>"
            f"</div>",
            unsafe_allow_html=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)
        qc1, qc2 = st.columns(2)
        with qc1:
            st.markdown(
                f"{'OK' if not qr.is_blurry else 'WARN'} **Sharpness**: {qr.blur_score:.0f}",
                unsafe_allow_html=False,
            )
            st.markdown(
                f"{'OK' if not qr.is_too_dark and not qr.is_too_bright else 'WARN'} **Brightness**: {qr.brightness:.0f}/255"
            )
        with qc2:
            st.markdown(
                f"{'OK' if not qr.is_low_contrast else 'WARN'} **Contrast**: {qr.contrast:.0f}"
            )
            st.markdown(
                f"{'OK' if not qr.is_low_res else 'WARN'} **Resolution**: {qr.width}x{qr.height}"
            )

        for issue in qr.issues:
            st.error(issue)
        for warn in qr.warnings:
            st.warning(warn)

        st.markdown("</div>", unsafe_allow_html=True)

with right_col:
    if uploaded_file is not None:
        st.markdown("<div class='nova-card'>", unsafe_allow_html=True)
        st.markdown("<div class='nova-card-title'>AI Screening Analysis</div>", unsafe_allow_html=True)
        run_btn = st.button("Run Screening Analysis", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        if run_btn:
            with st.spinner("Running inference pipeline..."):
                result = pipeline.run(image)
            st.session_state["last_result"] = result
            st.session_state["last_image"] = image

    if "last_result" in st.session_state:
        result = st.session_state["last_result"]
        orig_image = st.session_state.get("last_image")

        if not result.success:
            st.markdown(
                f"<div style='background:rgba(239,83,80,0.10);border:1px solid rgba(239,83,80,0.35);"
                f"border-radius:12px;padding:1.2rem;'>"
                f"<b style='color:#EF5350;'>Inference Unavailable</b><br><br>"
                f"<span style='color:rgba(244,245,242,0.75);font-size:0.85rem;white-space:pre-line;'>"
                f"{result.error_message}</span></div>",
                unsafe_allow_html=True,
            )
        else:
            pred_cls = result.predicted_class
            confidence = result.confidence
            risk = result.risk_level

            st.markdown(
                f"<div class='nova-card' style='border-color:rgba(114,228,242,0.3);'>"
                f"<div class='nova-card-title'>Prediction Result</div>"
                f"<div style='display:flex;align-items:flex-start;gap:1.2rem;flex-wrap:wrap;'>"
                f"  <div>"
                f"    <div style='font-size:1.8rem;font-weight:700;color:#72E4F2;'>{pred_cls}</div>"
                f"    <span class='risk-badge risk-{risk}'>{risk.upper()} RISK</span>"
                f"  </div>"
                f"  <div style='flex:1;'>"
                f"    <div style='color:rgba(244,245,242,0.55);font-size:0.78rem;text-transform:uppercase;'>Confidence</div>"
                f"    <div style='font-size:2rem;font-weight:700;color:#F4F5F2;'>{confidence*100:.1f}%</div>"
                f"    <div style='font-size:0.78rem;color:rgba(244,245,242,0.45);'>"
                f"    Inference: {result.inference_time_ms:.1f} ms</div>"
                f"  </div></div>"
                f"<div style='margin-top:1rem;padding:0.75rem;background:rgba(19,32,48,0.8);"
                f"border-radius:8px;color:rgba(244,245,242,0.7);font-size:0.83rem;'>"
                f"<b>Clinical Note:</b> {result.class_description}</div></div>",
                unsafe_allow_html=True,
            )

            # Probability bars
            st.markdown("<div class='nova-card'>", unsafe_allow_html=True)
            st.markdown("<div class='nova-card-title'>Class Probabilities</div>", unsafe_allow_html=True)
            for cls_name, prob in sorted(
                result.probabilities.items(), key=lambda x: x[1], reverse=True
            ):
                is_pred = cls_name == result.predicted_class
                ca, cb = st.columns([3, 1])
                with ca:
                    st.markdown(
                        f"<div style='display:flex;align-items:center;gap:0.5rem;margin-bottom:0.3rem;'>"
                        f"  <span style='color:{'#F4F5F2' if is_pred else 'rgba(244,245,242,0.6)'};font-size:0.85rem;"
                        f"        font-weight:{'600' if is_pred else '400'};min-width:130px;'>{cls_name}"
                        f"    {'*' if is_pred else ''}</span>"
                        f"  <div style='flex:1;background:#1E3045;border-radius:4px;height:10px;'>"
                        f"    <div style='width:{prob*100:.1f}%;height:100%;border-radius:4px;"
                        f"         background:{'linear-gradient(90deg,#3BA6A0,#72E4F2)' if is_pred else '#2A7A75'};"
                        f"         '></div></div></div>",
                        unsafe_allow_html=True,
                    )
                with cb:
                    st.markdown(
                        f"<div style='text-align:right;color:#72E4F2;font-weight:600;font-size:0.85rem;'>"
                        f"{prob*100:.1f}%</div>",
                        unsafe_allow_html=True,
                    )
            st.markdown("</div>", unsafe_allow_html=True)

            # Grad-CAM
            if result.gradcam_overlay is not None:
                st.markdown("<div class='nova-card'>", unsafe_allow_html=True)
                st.markdown(
                    "<div class='nova-card-title'>Grad-CAM Explainability</div>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    "<div style='color:rgba(244,245,242,0.55);font-size:0.80rem;margin-bottom:0.7rem;'>"
                    "Heatmap highlights regions most influential in the prediction. "
                    "Warm colours = high relevance.</div>",
                    unsafe_allow_html=True,
                )
                gc1, gc2 = st.columns(2)
                with gc1:
                    st.markdown("<div style='text-align:center;font-size:0.78rem;color:rgba(244,245,242,0.5);'>Original</div>", unsafe_allow_html=True)
                    if orig_image:
                        st.image(orig_image, use_container_width=True)
                with gc2:
                    st.markdown("<div style='text-align:center;font-size:0.78rem;color:rgba(244,245,242,0.5);'>Grad-CAM Overlay</div>", unsafe_allow_html=True)
                    st.image(result.gradcam_overlay, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

    elif uploaded_file is None:
        st.markdown(
            "<div class='nova-card' style='text-align:center;padding:3rem 2rem;'>"
            "<div style='font-size:3.5rem;margin-bottom:1rem;'>&#x1F9B7;</div>"
            "<div style='font-size:1.1rem;color:rgba(244,245,242,0.6);margin-bottom:0.5rem;'>"
            "Upload an oral photograph to begin</div>"
            "<div style='font-size:0.80rem;color:rgba(244,245,242,0.35);'>"
            "Supports JPG, PNG, BMP, WebP</div></div>",
            unsafe_allow_html=True,
        )

# ---------------------------------------------------------------------------
# Tabs: Model Info / Evaluation / Dataset
# ---------------------------------------------------------------------------
st.markdown("<hr class='section-div'>", unsafe_allow_html=True)
tab1, tab2, tab3 = st.tabs(["Model Architecture", "Evaluation Results", "Dataset Info"])

with tab1:
    tc1, tc2 = st.columns(2)
    output_classes = pipeline.class_names if pipeline.is_loaded else CLASS_NAMES
    info = pipeline.model_info if pipeline.is_loaded else {}

    with tc1:
        st.markdown("<div class='nova-card'>", unsafe_allow_html=True)
        st.markdown("<div class='nova-card-title'>Architecture Details</div>", unsafe_allow_html=True)

        total_params = info.get("total_params_m")
        param_text = f"{total_params:.2f} M" if total_params is not None else "Unavailable"
        architecture = info.get("architecture", "Not loaded")
        device_text = str(pipeline.device) if pipeline.is_loaded else "Not loaded"

        st.markdown(
            f"| Property | Value |\n"
            f"|---|---|\n"
            f"| Architecture | {architecture} |\n"
            f"| Input | 224 x 224 x 3 (RGB) |\n"
            f"| Total Params | {param_text} |\n"
            f"| Output Classes | {len(output_classes)} |\n"
            f"| Head | Checkpoint-defined output layer |\n"
            f"| Device | {device_text} |"
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with tc2:
        st.markdown("<div class='nova-card'>", unsafe_allow_html=True)
        st.markdown("<div class='nova-card-title'>Output Classes</div>", unsafe_allow_html=True)

        for i, cls in enumerate(output_classes):
            risk = CLASS_RISK.get(cls, "unknown")
            st.markdown(
                f"<div style='display:flex;align-items:center;gap:0.6rem;padding:0.45rem 0;"
                f"border-bottom:1px solid #1E3045;'>"
                f"<div style='background:#1E3045;color:#72E4F2;font-size:0.75rem;font-weight:600;"
                f"padding:0.15rem 0.45rem;border-radius:6px;'>{i}</div>"
                f"<div style='flex:1;color:#F4F5F2;font-size:0.85rem;font-weight:500;'>{cls}</div>"
                f"<span class='risk-badge risk-{risk}'>{risk.upper()}</span></div>",
                unsafe_allow_html=True,
            )

        st.markdown("</div>", unsafe_allow_html=True)

with tab2:
    eval_results = load_evaluation_results("reports")
    if eval_results:
        acc = eval_results.get("accuracy", 0)
        report = eval_results.get("classification_report", {})
        macro = report.get("macro avg", {})
        colA, colB, colC, colD = st.columns(4)
        with colA: st.metric("Test Accuracy", f"{acc*100:.1f}%")
        with colB: st.metric("Macro Precision", f"{macro.get('precision',0):.3f}")
        with colC: st.metric("Macro Recall", f"{macro.get('recall',0):.3f}")
        with colD: st.metric("Macro F1", f"{macro.get('f1-score',0):.3f}")

        cm = eval_results.get("confusion_matrix")
        classes = eval_results.get("classes", CLASS_NAMES)
        if cm:
            try:
                from src.utils import plot_confusion_matrix
                cm_img = plot_confusion_matrix(cm, classes)
                st.image(cm_img, caption="Normalised Confusion Matrix", width=500)
            except Exception:
                st.info("Confusion matrix plot unavailable.")
    else:
        st.info(
            "No evaluation results found. "
            "Run: `python -m src.evaluate --data_dir data` after training."
        )

with tab3:
    st.markdown("""
### Dataset Requirements

| Property | Requirement |
|---|---|
| Modality | Clinical oral photographs (RGB) |
| Format | JPG or PNG |
| Min resolution | 224 x 224 px (400 x 400 recommended) |
| Min per class (train) | ~200 images |

### Folder Structure

```
data/
  train/
    Normal/
    AphthousUlcer/
    Leukoplakia/
    OralCancer/
    Gingivitis/
  val/   (same structure)
  test/  (same structure)
```

### Finding a Dataset

Search Kaggle for: `oral cancer images` or `oral lesion dataset`

> **Important:** Use clinical oral photographs only. Histopathology slides
> are a different modality and NOT compatible with this pipeline.
    """)

# ---------------------------------------------------------------------------
# Footer disclaimer
# ---------------------------------------------------------------------------
st.markdown("<hr class='section-div'>", unsafe_allow_html=True)
st.markdown(
    "<div class='disclaimer-box' style='margin-bottom:2rem;'>"
    "<b>IMPORTANT MEDICAL DISCLAIMER:</b> ORALNOVA-AI is an <b>educational AI research prototype</b> "
    "developed for learning and demonstration purposes only. It is <b>NOT</b> a medical device, "
    "clinical diagnostic tool, or substitute for professional evaluation. Predictions made by this "
    "system must not be used for any clinical, diagnostic, or treatment decisions. Always consult a "
    "qualified dental professional or healthcare provider for any oral health concerns. Results without "
    "real trained weights are not meaningful and must not be interpreted as clinical predictions."
    "</div>",
    unsafe_allow_html=True,
)

