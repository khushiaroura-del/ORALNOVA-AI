"""
src/evaluate.py
Evaluation script: generates accuracy, precision, recall, F1, confusion matrix.

Usage:
    python -m src.evaluate --data_dir data --checkpoint models/oralnova_best.pth
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)

from src import CLASS_NAMES, NUM_CLASSES
from src.model import build_model, load_checkpoint
from src.data_preprocessing import load_datasets


@torch.no_grad()
def evaluate_model(model, test_loader, device, class_names):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    for images, labels in test_loader:
        images = images.to(device)
        logits = model(images)
        probs  = F.softmax(logits, dim=1).cpu().numpy()
        preds  = np.argmax(probs, axis=1)
        all_probs.extend(probs.tolist())
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.numpy().tolist())

    acc = accuracy_score(all_labels, all_preds)
    report = classification_report(
        all_labels, all_preds,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(all_labels, all_preds)

    binary_metrics = {}
    if len(class_names) == 2:
        # Index 0: Benign (Negative), Index 1: Malignant (Positive)
        tn, fp = cm[0][0], cm[0][1]
        fn, tp = cm[1][0], cm[1][1]
        sens = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        f1   = float(2 * (prec * sens) / (prec + sens)) if (prec + sens) > 0 else 0.0
        binary_metrics = {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
            "sensitivity": sens,
            "specificity": spec,
            "precision": prec,
            "f1_score": f1,
        }

    return {
        "accuracy": acc,
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "binary_metrics": binary_metrics,
        "predictions": all_preds,
        "labels": all_labels,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate ORALNOVA-AI model")
    parser.add_argument("--data_dir",   type=str, default="data")
    parser.add_argument("--checkpoint", type=str, default="models/oralnova_best.pth")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device",     type=str, default="cpu")
    parser.add_argument("--output",     type=str, default=None)
    args = parser.parse_args()

    device = torch.device(args.device)

    # --- Data ---
    _, _, test_loader, class_names = load_datasets(args.data_dir, args.batch_size)
    print(f"Classes: {class_names}")

    # --- Model ---
    model = build_model(num_classes=len(class_names), pretrained=False)
    meta  = load_checkpoint(model, args.checkpoint, device)
    model.to(device)
    print(f"Loaded checkpoint (epoch={meta['epoch']}, val_acc={meta['val_acc']})")

    # --- Evaluate ---
    results = evaluate_model(model, test_loader, device, class_names)

    print(f"\nTest Accuracy: {results['accuracy']:.4f}")
    if results.get("binary_metrics"):
        bm = results["binary_metrics"]
        print(f"Sensitivity (Recall of Malignant): {bm['sensitivity']:.4f}")
        print(f"Specificity (Recall of Benign):    {bm['specificity']:.4f}")
        print(f"Precision (PPV of Malignant):      {bm['precision']:.4f}")
        print(f"F1-Score:                          {bm['f1_score']:.4f}")
        print(f"Counts: TP={bm['true_positives']}, TN={bm['true_negatives']}, FP={bm['false_positives']}, FN={bm['false_negatives']}")

    print("\nClassification Report:")
    report = results["classification_report"]
    for cls in class_names:
        row = report.get(cls, {})
        print(
            f"  {cls:<20} precision={row.get('precision',0):.3f} "
            f"recall={row.get('recall',0):.3f} f1={row.get('f1-score',0):.3f}"
        )

    # --- Save ---
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    if args.output:
        save_path = Path(args.output)
    else:
        is_binary = "binary" in args.checkpoint or "binary" in args.data_dir or len(class_names) == 2
        save_path = reports_dir / ("evaluation_results_binary.json" if is_binary else "evaluation_results.json")

    save_data = {
        "accuracy": results["accuracy"],
        "classification_report": results["classification_report"],
        "confusion_matrix": results["confusion_matrix"],
        "binary_metrics": results.get("binary_metrics", {}),
        "checkpoint": args.checkpoint,
        "classes": class_names,
        "sample_counts": {cls: sum(1 for label_idx in results["labels"] if class_names[label_idx] == cls) for cls in class_names},
        "limitations": "Notice: Patient identifiers were not supplied in the source dataset. Patient-level leakage cannot be ruled out.",
    }
    with open(save_path, "w") as f:
        json.dump(save_data, f, indent=2)

    print(f"\nResults saved to {save_path}")


if __name__ == "__main__":
    main()
