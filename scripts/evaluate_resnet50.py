"""
scripts/evaluate_resnet50.py
────────────────────────────
Generates full evaluation metrics for the trained ResNet50 waste classifier:

  1. Confusion Matrix        →  reports/resnet50_confusion_matrix.png
  2. Normalised Confusion Matrix (%) → reports/resnet50_confusion_matrix_norm.png
  3. Per-class Accuracy Bar   →  reports/resnet50_per_class_accuracy.png
  4. Per-class Precision / Recall / F1 Bar → reports/resnet50_prf1.png
  5. Confidence Distribution  →  reports/resnet50_confidence_dist.png
  6. Top-K Accuracy (K=1,2,3) →  reports/resnet50_topk_accuracy.png
  7. Full text report         →  reports/resnet50_report.txt

Usage (venv active, run from project root):
    python scripts/evaluate_resnet50.py
    python scripts/evaluate_resnet50.py --split test
    python scripts/evaluate_resnet50.py --split valid
"""

import argparse
import os
import sys
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image, ImageFile
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — no display needed
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.metrics import (
    confusion_matrix, classification_report,
    precision_recall_fscore_support, top_k_accuracy_score,
)

ImageFile.LOAD_TRUNCATED_IMAGES = True

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "YOLOV8M" / "datasets" / "balanced_v3"
MODEL_PATH  = BASE_DIR / "models" / "resnet50_waste.pth"
REPORT_DIR  = BASE_DIR / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = [
    "plastic",        # 0
    "metal",          # 1
    "glass",          # 2
    "e_waste",        # 3
    "medical_waste",  # 4
    "paper",          # 5
    "cardboard",      # 6
    "organic_waste",  # 7
]
NUM_CLASSES = len(CLASS_NAMES)

# Colour palette — one per class (matches app.py hex colours)
CLASS_COLORS = [
    "#0E7490",   # plastic      — teal
    "#B45309",   # metal        — amber
    "#0F766E",   # glass        — teal-green
    "#7C2D12",   # e_waste      — dark red
    "#B91C1C",   # medical_waste— red
    "#1D4ED8",   # paper        — blue
    "#78350F",   # cardboard    — brown
    "#15803D",   # organic_waste— green
]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ─────────────────────────────────────────────────────────────────────────────
# Dataset  (same logic as train_resnet50.py — crop dominant bbox)
# ─────────────────────────────────────────────────────────────────────────────
class WasteDataset(Dataset):
    def __init__(self, split: str, transform=None):
        self.transform = transform
        self.samples: list[tuple] = []

        img_dir = DATASET_DIR / split / "images"
        lbl_dir = DATASET_DIR / split / "labels"

        for img_path in sorted(img_dir.glob("*.[jJpP][pPnN][gG]*")):
            lbl_path = lbl_dir / (img_path.stem + ".txt")
            if not lbl_path.exists():
                continue
            lines = lbl_path.read_text(encoding="utf-8", errors="ignore").strip().splitlines()
            if not lines:
                continue

            best_area, best_cls, best_bbox = -1.0, -1, None
            for line in lines:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cid = int(parts[0])
                cx, cy, bw, bh = map(float, parts[1:5])
                if bw * bh > best_area:
                    best_area, best_cls, best_bbox = bw * bh, cid, (cx, cy, bw, bh)

            if best_cls < 0 or best_cls >= NUM_CLASSES:
                continue
            self.samples.append((img_path, best_cls, best_bbox))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, class_id, (cx, cy, bw, bh) = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        W, H = img.size
        pad = 10
        x1 = max(0,     int((cx - bw/2) * W) - pad)
        y1 = max(0,     int((cy - bh/2) * H) - pad)
        x2 = min(W - 1, int((cx + bw/2) * W) + pad)
        y2 = min(H - 1, int((cy + bh/2) * H) + pad)
        crop = img.crop((x1, y1, x2, y2)) if (x2 > x1 and y2 > y1) else img
        if self.transform:
            crop = self.transform(crop)
        return crop, class_id


# ─────────────────────────────────────────────────────────────────────────────
# Model builder  (must match train_resnet50.py architecture exactly)
# ─────────────────────────────────────────────────────────────────────────────
def build_model() -> nn.Module:
    net = models.resnet50(weights=None)
    net.fc = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(net.fc.in_features, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.3),
        nn.Linear(512, NUM_CLASSES),
    )
    return net


# ─────────────────────────────────────────────────────────────────────────────
# Plot helpers
# ─────────────────────────────────────────────────────────────────────────────
TITLE_FONT  = dict(fontsize=14, fontweight="bold", color="#14213D")
LABEL_FONT  = dict(fontsize=11, color="#263241")
TICK_FONT   = dict(fontsize=9, color="#5B6B7C")
FIG_BG      = "#F5F6F8"
CARD_BG     = "#FFFFFF"
GRID_COLOR  = "#E5E7EB"


def _save(fig, name: str):
    path = REPORT_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=FIG_BG)
    plt.close(fig)
    print(f"  Saved → {path}")


def plot_confusion_matrix(cm: np.ndarray, normalised: bool = False):
    fig, ax = plt.subplots(figsize=(10, 8), facecolor=FIG_BG)
    ax.set_facecolor(CARD_BG)

    if normalised:
        row_sums = cm.sum(axis=1, keepdims=True)
        data = np.where(row_sums == 0, 0, cm / row_sums * 100)
        fmt  = ".1f"
        cbar_label = "Recall (%)"
        title = "Normalised Confusion Matrix (row = true class, values = recall %)"
        fname = "resnet50_confusion_matrix_norm.png"
        cmap  = "Blues"
        vmax  = 100
    else:
        data = cm
        fmt  = "d"
        cbar_label = "Count"
        title = "Confusion Matrix (raw counts)"
        fname = "resnet50_confusion_matrix.png"
        cmap  = "YlOrBr"
        vmax  = None

    im = ax.imshow(data, interpolation="nearest", cmap=cmap,
                   vmin=0, vmax=vmax, aspect="auto")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label, fontsize=10, color="#5B6B7C")
    cbar.ax.tick_params(labelsize=8, colors="#5B6B7C")

    ticks = np.arange(NUM_CLASSES)
    short = [c.replace("_", "\n") for c in CLASS_NAMES]
    ax.set_xticks(ticks);  ax.set_xticklabels(short, **TICK_FONT)
    ax.set_yticks(ticks);  ax.set_yticklabels(short, **TICK_FONT)
    ax.set_xlabel("Predicted class", **LABEL_FONT)
    ax.set_ylabel("True class",      **LABEL_FONT)
    ax.set_title(title, **TITLE_FONT, pad=12)

    # Annotate cells
    thresh = (data.max() / 2.0) if not normalised else 50
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            val = data[i, j]
            txt = (f"{val:.1f}%" if normalised else f"{int(val)}")
            color = "white" if val > thresh else "#14213D"
            ax.text(j, i, txt, ha="center", va="center",
                    fontsize=7, color=color, fontweight="bold")

    fig.tight_layout()
    _save(fig, fname)


def plot_per_class_accuracy(per_class_acc: np.ndarray):
    fig, ax = plt.subplots(figsize=(10, 5), facecolor=FIG_BG)
    ax.set_facecolor(CARD_BG)

    x = np.arange(NUM_CLASSES)
    bars = ax.bar(x, per_class_acc * 100, color=CLASS_COLORS,
                  width=0.6, edgecolor="white", linewidth=0.8)

    # Value labels on bars
    for bar, val in zip(bars, per_class_acc * 100):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=9,
                color="#14213D", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_", "\n") for c in CLASS_NAMES], **TICK_FONT)
    ax.set_ylabel("Accuracy (%)", **LABEL_FONT)
    ax.set_title("Per-class Accuracy", **TITLE_FONT, pad=12)
    ax.set_ylim(0, 110)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax.axhline(per_class_acc.mean() * 100, color="#E17A1D", linewidth=1.5,
               linestyle="--", label=f"Mean {per_class_acc.mean()*100:.1f}%")
    ax.legend(fontsize=9)
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    _save(fig, "resnet50_per_class_accuracy.png")


def plot_prf1(precision, recall, f1):
    fig, ax = plt.subplots(figsize=(12, 5), facecolor=FIG_BG)
    ax.set_facecolor(CARD_BG)

    x     = np.arange(NUM_CLASSES)
    width = 0.26
    ax.bar(x - width, precision * 100, width, label="Precision",
           color="#0F4C81", alpha=0.85, edgecolor="white")
    ax.bar(x,          recall    * 100, width, label="Recall",
           color="#E17A1D", alpha=0.85, edgecolor="white")
    ax.bar(x + width,  f1        * 100, width, label="F1-Score",
           color="#15803D", alpha=0.85, edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_", "\n") for c in CLASS_NAMES], **TICK_FONT)
    ax.set_ylabel("Score (%)", **LABEL_FONT)
    ax.set_title("Per-class Precision / Recall / F1-Score", **TITLE_FONT, pad=12)
    ax.set_ylim(0, 115)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax.legend(fontsize=10, framealpha=0.9)
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    _save(fig, "resnet50_prf1.png")


def plot_confidence_distribution(all_probs: np.ndarray, all_labels: np.ndarray,
                                 all_preds: np.ndarray):
    correct_conf   = all_probs[all_preds == all_labels]
    incorrect_conf = all_probs[all_preds != all_labels]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor=FIG_BG)
    for ax in axes:
        ax.set_facecolor(CARD_BG)

    bins = np.linspace(0, 1, 26)

    # Left — correct vs incorrect confidence histogram
    axes[0].hist(correct_conf,   bins=bins, color="#15803D", alpha=0.75,
                 label=f"Correct  (n={len(correct_conf)})",   edgecolor="white")
    axes[0].hist(incorrect_conf, bins=bins, color="#B91C1C", alpha=0.75,
                 label=f"Incorrect (n={len(incorrect_conf)})", edgecolor="white")
    axes[0].set_xlabel("Confidence (softmax max prob)", **LABEL_FONT)
    axes[0].set_ylabel("Count", **LABEL_FONT)
    axes[0].set_title("Confidence Distribution", **TITLE_FONT, pad=10)
    axes[0].legend(fontsize=9)
    axes[0].grid(axis="y", color=GRID_COLOR, linewidth=0.8)
    axes[0].spines[["top", "right"]].set_visible(False)

    # Right — per-class mean confidence
    per_class_conf = []
    for cid in range(NUM_CLASSES):
        mask = all_labels == cid
        per_class_conf.append(all_probs[mask].mean() if mask.sum() > 0 else 0.0)
    per_class_conf = np.array(per_class_conf)

    x = np.arange(NUM_CLASSES)
    bars = axes[1].bar(x, per_class_conf * 100, color=CLASS_COLORS,
                       width=0.6, edgecolor="white")
    for bar, val in zip(bars, per_class_conf * 100):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                     f"{val:.1f}%", ha="center", va="bottom", fontsize=8,
                     color="#14213D", fontweight="bold")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([c.replace("_", "\n") for c in CLASS_NAMES], **TICK_FONT)
    axes[1].set_ylabel("Mean Confidence (%)", **LABEL_FONT)
    axes[1].set_title("Mean Confidence per Class", **TITLE_FONT, pad=10)
    axes[1].set_ylim(0, 115)
    axes[1].yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    axes[1].grid(axis="y", color=GRID_COLOR, linewidth=0.8)
    axes[1].spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    _save(fig, "resnet50_confidence_dist.png")


def plot_topk_accuracy(all_logits: np.ndarray, all_labels: np.ndarray):
    """Bar chart of Top-1, Top-2, Top-3 accuracy."""
    k_vals = [1, 2, 3]
    topk_accs = []
    for k in k_vals:
        acc = top_k_accuracy_score(all_labels, all_logits, k=k) * 100
        topk_accs.append(acc)

    fig, ax = plt.subplots(figsize=(6, 5), facecolor=FIG_BG)
    ax.set_facecolor(CARD_BG)

    colors = ["#0F4C81", "#E17A1D", "#15803D"]
    bars = ax.bar([f"Top-{k}" for k in k_vals], topk_accs,
                  color=colors, width=0.45, edgecolor="white")
    for bar, val in zip(bars, topk_accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{val:.2f}%", ha="center", va="bottom", fontsize=11,
                color="#14213D", fontweight="bold")

    ax.set_ylabel("Accuracy (%)", **LABEL_FONT)
    ax.set_title("Top-K Accuracy", **TITLE_FONT, pad=12)
    ax.set_ylim(0, 110)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    _save(fig, "resnet50_topk_accuracy.png")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Evaluate ResNet50 waste classifier")
    parser.add_argument("--split",  default="test",  choices=["train", "valid", "test"],
                        help="Dataset split to evaluate on (default: test)")
    parser.add_argument("--batch",  type=int, default=64)
    parser.add_argument("--no-cuda",action="store_true")
    args = parser.parse_args()

    # ── Device ───────────────────────────────────────────────────────────────
    device = torch.device("cpu")
    if not args.no_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.empty_cache()

    print(f"\n{'═'*62}")
    print(f"  ResNet50 Evaluation  —  split: {args.split}")
    print(f"{'═'*62}")
    print(f"  Model   : {MODEL_PATH}")
    print(f"  Device  : {device}")
    print(f"  Reports : {REPORT_DIR}\n")

    # ── Load model ───────────────────────────────────────────────────────────
    if not MODEL_PATH.exists():
        print(f"ERROR: {MODEL_PATH} not found.\nRun train_resnet50.py first.")
        sys.exit(1)

    net  = build_model()
    ckpt = torch.load(MODEL_PATH, map_location=device)
    net.load_state_dict(ckpt["model_state_dict"])
    net.to(device).eval()

    best_epoch = ckpt.get("epoch",   "?")
    best_acc   = ckpt.get("val_acc", 0) * 100
    print(f"  Checkpoint : epoch {best_epoch}  val_acc={best_acc:.2f}%\n")

    # ── Dataset ──────────────────────────────────────────────────────────────
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    ds = WasteDataset(args.split, transform=tf)
    print(f"  {args.split} samples : {len(ds)}")
    loader = DataLoader(ds, batch_size=args.batch, shuffle=False,
                        num_workers=0, pin_memory=(device.type == "cuda"))

    # ── Inference ────────────────────────────────────────────────────────────
    all_labels  = []
    all_preds   = []
    all_probs   = []    # max softmax probability (confidence)
    all_logits  = []    # full logit vectors for top-k

    print("  Running inference …")
    with torch.no_grad():
        for imgs, labels in loader:
            imgs   = imgs.to(device)
            logits = net(imgs)
            probs  = torch.softmax(logits, dim=1)
            preds  = probs.argmax(dim=1)

            all_labels.append(labels.numpy())
            all_preds.append(preds.cpu().numpy())
            all_probs.append(probs.max(dim=1).values.cpu().numpy())
            all_logits.append(probs.cpu().numpy())   # probs work for top_k

    all_labels = np.concatenate(all_labels)
    all_preds  = np.concatenate(all_preds)
    all_probs  = np.concatenate(all_probs)
    all_logits = np.concatenate(all_logits)

    # ── Metrics ──────────────────────────────────────────────────────────────
    overall_acc = (all_preds == all_labels).mean() * 100

    cm = confusion_matrix(all_labels, all_preds)

    per_class_acc = cm.diagonal() / cm.sum(axis=1)

    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, average=None, labels=list(range(NUM_CLASSES)), zero_division=0
    )
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="macro", zero_division=0
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="weighted", zero_division=0
    )

    top1 = top_k_accuracy_score(all_labels, all_logits, k=1) * 100
    top2 = top_k_accuracy_score(all_labels, all_logits, k=2) * 100
    top3 = top_k_accuracy_score(all_labels, all_logits, k=3) * 100

    # ── Console summary ──────────────────────────────────────────────────────
    print(f"\n{'─'*62}")
    print(f"  Overall Accuracy  : {overall_acc:.2f}%")
    print(f"  Top-1 / Top-2 / Top-3 : {top1:.2f}% / {top2:.2f}% / {top3:.2f}%")
    print(f"  Macro  P={macro_p*100:.2f}%  R={macro_r*100:.2f}%  F1={macro_f1*100:.2f}%")
    print(f"  Weighted P={weighted_p*100:.2f}%  R={weighted_r*100:.2f}%  F1={weighted_f1*100:.2f}%")
    print(f"{'─'*62}")
    print(f"  {'Class':<18} {'Prec':>7} {'Recall':>8} {'F1':>7} {'Acc':>7} {'Support':>9}")
    print(f"  {'─'*58}")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {name:<18} {precision[i]*100:>6.1f}%  {recall[i]*100:>6.1f}%  "
              f"{f1[i]*100:>6.1f}%  {per_class_acc[i]*100:>6.1f}%  {int(support[i]):>7}")
    print(f"{'─'*62}\n")

    # ── Save text report ──────────────────────────────────────────────────────
    report_txt  = REPORT_DIR / "resnet50_report.txt"
    clf_report  = classification_report(
        all_labels, all_preds,
        target_names=CLASS_NAMES, digits=4, zero_division=0
    )
    with open(report_txt, "w", encoding="utf-8") as f:
        f.write("ResNet50 Waste Classifier — Evaluation Report\n")
        f.write("=" * 62 + "\n")
        f.write(f"Split          : {args.split}\n")
        f.write(f"Checkpoint     : epoch {best_epoch}  val_acc={best_acc:.2f}%\n")
        f.write(f"Total samples  : {len(all_labels)}\n\n")
        f.write(f"Overall Accuracy  : {overall_acc:.4f}%\n")
        f.write(f"Top-1 Accuracy    : {top1:.4f}%\n")
        f.write(f"Top-2 Accuracy    : {top2:.4f}%\n")
        f.write(f"Top-3 Accuracy    : {top3:.4f}%\n\n")
        f.write(f"Macro  Precision  : {macro_p:.4f}\n")
        f.write(f"Macro  Recall     : {macro_r:.4f}\n")
        f.write(f"Macro  F1         : {macro_f1:.4f}\n\n")
        f.write(f"Weighted Precision: {weighted_p:.4f}\n")
        f.write(f"Weighted Recall   : {weighted_r:.4f}\n")
        f.write(f"Weighted F1       : {weighted_f1:.4f}\n\n")
        f.write("─" * 62 + "\n")
        f.write("Full Classification Report:\n\n")
        f.write(clf_report)
        f.write("\n" + "─" * 62 + "\n")
        f.write("Confusion Matrix (rows=true, cols=predicted):\n\n")
        header = "              " + "".join(f"{c[:8]:>10}" for c in CLASS_NAMES)
        f.write(header + "\n")
        for i, row in enumerate(cm):
            f.write(f"  {CLASS_NAMES[i]:<14}" + "".join(f"{v:>10}" for v in row) + "\n")
    print(f"  Saved → {report_txt}")

    # ── Generate all plots ────────────────────────────────────────────────────
    print("\n  Generating plots …")
    plot_confusion_matrix(cm, normalised=False)
    plot_confusion_matrix(cm, normalised=True)
    plot_per_class_accuracy(per_class_acc)
    plot_prf1(precision, recall, f1)
    plot_confidence_distribution(all_probs, all_labels, all_preds)
    plot_topk_accuracy(all_logits, all_labels)

    print(f"\n{'═'*62}")
    print(f"  All reports saved to:  {REPORT_DIR}")
    print(f"{'═'*62}")
    print(f"  resnet50_confusion_matrix.png")
    print(f"  resnet50_confusion_matrix_norm.png")
    print(f"  resnet50_per_class_accuracy.png")
    print(f"  resnet50_prf1.png")
    print(f"  resnet50_confidence_dist.png")
    print(f"  resnet50_topk_accuracy.png")
    print(f"  resnet50_report.txt")
    print(f"{'═'*62}\n")


if __name__ == "__main__":
    main()
