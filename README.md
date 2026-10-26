# ORALNOVA-AI — Lightweight Deep Learning for Oral Lesion Screening

> **AI-Assisted Oral Image Analysis with Explainable Predictions**

ORALNOVA-AI is a professional, educational AI research prototype for oral image
screening using deep learning, image-quality checks, class probability displays,
and Grad-CAM explainability visualizations.

---

## Features

| Feature | Details |
|---|---|
| Model | MobileNetV3-Small (pre-trained backbone) |
| Modality | Clinical oral photographs (RGB) |
| Classes | Normal, Aphthous Ulcer, Leukoplakia, Oral Cancer, Gingivitis |
| Explainability | Grad-CAM heatmap overlays |
| Quality Check | Blur, brightness, contrast, resolution analysis |
| Interface | Streamlit web dashboard |
| Inference | CPU-optimized, real-time predictions |

---

## Quick Start (Windows)

```powershell
# 1. Clone / download the project
cd C:\Users\DELL\Desktop\ORALNOVA-AI

# 2. Create virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install streamlit Pillow opencv-python-headless numpy pandas scikit-learn matplotlib tqdm scipy pytest

# 4. Launch the app
streamlit run app.py
```

---

## Dataset Instructions

ORALNOVA-AI uses **clinical oral photographs** (not histopathology slides).

### Recommended Free Dataset
- **Oral Cancer & Lesion Image Dataset** — Kaggle
  - URL: https://www.kaggle.com/
  - Search: "oral cancer images", "oral lesion dataset"
  - License: Verify CC BY / research-use before downloading

### Folder Structure Required
```
data/
  train/
    Normal/          ← healthy oral cavity images
    AphthousUlcer/   ← canker sore images
    Leukoplakia/     ← white patch lesion images
    OralCancer/      ← malignant lesion images
    Gingivitis/      ← gum disease images
  val/
    (same subfolders)
  test/
    (same subfolders)
```

> ⚠️ Do NOT mix histopathology slides with clinical photographs. ORALNOVA-AI
> is trained on and designed for **clinical oral RGB photographs only**.

---

## Training

```powershell
# With venv activated
python -m src.train --data_dir data --epochs 30 --batch_size 16 --lr 0.001
```

Training will:
- Fine-tune MobileNetV3-Small with transfer learning
- Save best checkpoint to `models/oralnova_best.pth`
- Log loss / accuracy curves to `reports/`

---

## Evaluation

```powershell
python -m src.evaluate --data_dir data --checkpoint models/oralnova_best.pth
```

Outputs: accuracy, precision, recall, F1-score, confusion matrix → `reports/`

---

## Running the Application

```powershell
# Activate venv first
.venv\Scripts\Activate.ps1

# Launch dashboard
streamlit run app.py
```

Visit: http://localhost:8501

---

## Project Structure

```
ORALNOVA-AI/
├── app.py                    # Streamlit dashboard
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── .gitignore
├── data/                     # Dataset (not included, see above)
├── models/                   # Saved checkpoints
├── src/
│   ├── model.py              # MobileNetV3-Small architecture
│   ├── data_preprocessing.py # Dataset loader and transforms
│   ├── image_quality.py      # Blur / brightness / contrast checks
│   ├── screening_pipeline.py # End-to-end inference pipeline
│   ├── gradcam.py            # Grad-CAM explainability
│   ├── train.py              # Training script
│   ├── evaluate.py           # Evaluation script
│   └── utils.py              # Shared utilities
├── tests/                    # Unit tests
└── reports/                  # Metrics, plots, evaluation outputs
```

---

## Troubleshooting

| Problem | Solution |
|---|---|
| "No trained model found" | Train first: `python -m src.train` |
| CUDA out of memory | Use CPU: `--device cpu` (default) |
| Slow inference | MobileNetV3 is CPU-optimized; normal on first run |
| Import errors | Ensure venv is activated |
| Streamlit port in use | `streamlit run app.py --server.port 8502` |

---

## Limitations

- **Not a medical device.** For research and educational purposes only.
- Accuracy depends entirely on dataset quality and size.
- No trained weights are bundled. You must train with a real dataset.
- Predictions without real trained weights are not meaningful.

---

## Medical Research Disclaimer

> This application is an **educational AI prototype** and is **NOT** a medical
> device, diagnostic tool, or substitute for professional clinical assessment.
> Never use this tool to make clinical decisions. Consult a qualified healthcare
> provider for any medical concerns.

---

## License

MIT License — Educational / Research Use Only

