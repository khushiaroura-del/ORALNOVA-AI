"""
src/train.py
Training script for ORALNOVA-AI.

Usage:
    python -m src.train --data_dir data --epochs 30 --batch_size 16 --lr 0.001
"""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from src import NUM_CLASSES, CLASS_NAMES
from src.model import build_model, get_model_info
from src.data_preprocessing import load_datasets


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += images.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += images.size(0)
    return total_loss / total, correct / total


def save_checkpoint(model, optimizer, epoch, val_acc, val_loss, path, classes=None):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_acc": val_acc,
        "val_loss": val_loss,
        "classes": classes or CLASS_NAMES,
    }, path)


def main():
    parser = argparse.ArgumentParser(description="Train ORALNOVA-AI model")
    parser.add_argument("--data_dir",   type=str, default="data",
                        help="Root directory with train/val/test subfolders")
    parser.add_argument("--epochs",     type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--device",     type=str, default="cpu")
    parser.add_argument("--num_workers",type=int, default=0)
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Using device: {device}")

    # --- Data ---
    train_loader, val_loader, _, class_names = load_datasets(
        args.data_dir, args.batch_size, args.num_workers
    )
    print(f"Classes detected: {class_names}")
    if len(class_names) != NUM_CLASSES:
        print(f"WARNING: Expected {NUM_CLASSES} classes, found {len(class_names)}.")

    # --- Model ---
    model = build_model(num_classes=len(class_names), pretrained=True).to(device)
    info = get_model_info(model)
    print(f"Model: {info['architecture']}, {info['trainable_params_m']}M trainable params")

    # --- Training setup ---
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_acc = 0.0
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    prefix = "oralnova_binary" if ("binary" in args.data_dir or len(class_names) == 2) else "oralnova"
    best_ckpt_path = models_dir / f"{prefix}_best.pth"
    last_ckpt_path = models_dir / f"{prefix}_last.pth"
    history_path = reports_dir / f"{prefix}_training_history.json"

    print("\n--- Training Started ---")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        vl_loss, vl_acc = eval_epoch(model, val_loader, criterion, device)
        scheduler.step()
        elapsed = time.time() - t0

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(vl_loss)
        history["val_acc"].append(vl_acc)

        print(
            f"Epoch [{epoch:03d}/{args.epochs}] "
            f"train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} "
            f"val_loss={vl_loss:.4f} val_acc={vl_acc:.4f} "
            f"({elapsed:.1f}s)"
        )

        # Save best
        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            save_checkpoint(model, optimizer, epoch, vl_acc, vl_loss,
                            best_ckpt_path, classes=class_names)
            print(f"  >> Best model saved to {best_ckpt_path} (val_acc={vl_acc:.4f})")

    # Save last
    save_checkpoint(model, optimizer, args.epochs, vl_acc, vl_loss,
                    last_ckpt_path, classes=class_names)

    # Save training history
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining complete. Best val_acc = {best_val_acc:.4f}")
    print(f"Checkpoints: {best_ckpt_path}, {last_ckpt_path}")
    print(f"History:     {history_path}")


if __name__ == "__main__":
    main()
