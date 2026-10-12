"""
scripts/train_yolov8m.py
────────────────────────
Train YOLOv8m on the 8-class waste detection dataset.
Optimised for NVIDIA RTX 3050 6 GB (Ampere).

Dataset layout (YOLO flat format):
    YOLOV8M/datasets/balanced_v3/
        train/  images/*.jpg   labels/*.txt
        valid/  images/        labels/
        test/   images/        labels/

Class mapping (confirmed from label files):
    0=plastic   1=metal   2=glass        3=e_waste
    4=medical_waste       5=paper        6=cardboard   7=organic_waste

Output (auto-copied after training):
    models/best.pt   ← best mAP checkpoint — what app.py loads
    models/last.pt   ← final epoch weights
    runs/detect/waste_yolov8m/  ← PR curve, confusion matrix, F1 curve

Usage (venv active, run from project root):
    python scripts/train_yolov8m.py
    python scripts/train_yolov8m.py --epochs 50
    python scripts/train_yolov8m.py --resume
"""

import argparse
import os
import shutil
import sys
import yaml
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "YOLOV8M" / "datasets" / "balanced_v3"
CONFIG_DIR  = BASE_DIR / "config"
MODEL_DIR   = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
CONFIG_DIR.mkdir(parents=True, exist_ok=True)

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


# ─────────────────────────────────────────────────────────────────────────────
# Step 1  —  write data.yaml with absolute paths (works from any cwd)
# ─────────────────────────────────────────────────────────────────────────────
def write_data_yaml() -> Path:
    yaml_path = CONFIG_DIR / "data_yolov8m.yaml"
    cfg = {
        "path":  str(DATASET_DIR),
        "train": "train/images",
        "val":   "valid/images",
        "test":  "test/images",
        "nc":    len(CLASS_NAMES),
        "names": CLASS_NAMES,
    }
    with open(yaml_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    print(f"  data.yaml → {yaml_path}")
    return yaml_path


# ─────────────────────────────────────────────────────────────────────────────
# Step 2  —  clean label files (polygon segments → bounding boxes)
#   Some source datasets mixed detection + segmentation annotations.
#   YOLO 8.1 silently drops segment lines but prints a warning. Converting
#   them to the tight bounding box prevents the warning and keeps every
#   annotation.
# ─────────────────────────────────────────────────────────────────────────────
def clean_labels() -> int:
    cleaned = 0
    for split in ("train", "valid", "test"):
        lbl_dir = DATASET_DIR / split / "labels"
        if not lbl_dir.exists():
            continue
        for lbl_file in lbl_dir.glob("*.txt"):
            lines   = lbl_file.read_text(encoding="utf-8", errors="ignore").splitlines()
            fixed, changed = [], False
            for line in lines:
                parts = line.strip().split()
                if not parts:
                    continue
                if len(parts) == 5:
                    fixed.append(line.strip())      # already a valid box line
                elif len(parts) > 5:                # polygon → convert to bbox
                    cid    = parts[0]
                    coords = list(map(float, parts[1:]))
                    xs = coords[0::2];  ys = coords[1::2]
                    cx = (min(xs) + max(xs)) / 2
                    cy = (min(ys) + max(ys)) / 2
                    bw = max(xs) - min(xs)
                    bh = max(ys) - min(ys)
                    fixed.append(f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                    changed = True
            if changed:
                lbl_file.write_text("\n".join(fixed) + "\n", encoding="utf-8")
                cleaned += 1
    return cleaned


# ─────────────────────────────────────────────────────────────────────────────
# Step 3  —  hyperparameters
#
#   batch=8   — with mosaic=1.0 each training step combines 4 images, so the
#               effective batch seen by the GPU is 8×4=32 images. batch=16
#               would require ~4.8 GB VRAM for the forward pass alone and OOMs
#               on the RTX 3050 6 GB.
#
#   workers=0 — on Windows, DataLoader workers are spawned as separate
#               processes. Spawning 4 workers for each epoch adds ~20-30 s of
#               overhead per epoch and caused the 11-minute epoch-1 issue.
#               workers=0 runs everything in the main process — slightly slower
#               throughput but no spawn overhead, so wall-clock time is faster.
#
#   cache=False — 10 GB RAM free but YOLO estimates 11.2 GB needed (it adds a
#               50% safety margin). Attempting to cache causes a warning and
#               falls back to disk anyway. Setting it to False skips the check
#               and goes straight to disk reads, which are fast enough for
#               640×640 JPEGs on a modern SSD.
#
#   amp=True  — FP16 mixed precision. Halves VRAM for activations and gives
#               ~1.5-2× speed on Ampere (RTX 30-series) via Tensor Cores.
#
#   cos_lr=True + patience=20 — cosine decay + early stopping. Training stops
#               automatically once mAP plateaus; no need to babysit.
# ─────────────────────────────────────────────────────────────────────────────
def get_hyps(args) -> dict:
    return dict(
        # ── core ─────────────────────────────────────────────────────────────
        imgsz           = 640,
        batch           = 8,        # safe for 6 GB VRAM; effective 32 with mosaic
        epochs          = args.epochs,
        # ── hardware ─────────────────────────────────────────────────────────
        device          = 0,        # GPU 0
        amp             = True,     # FP16 mixed precision (Ampere Tensor Cores)
        workers         = 0,        # 0 on Windows — avoids subprocess spawn overhead
        cache           = False,    # skip RAM cache (not enough free RAM)
        # ── optimizer ────────────────────────────────────────────────────────
        optimizer       = "AdamW",
        lr0             = 0.001,    # initial LR (lower than SGD default 0.01)
        lrf             = 0.01,     # final LR = lr0 × lrf (cosine target)
        momentum        = 0.937,
        weight_decay    = 0.0005,
        # ── warmup ───────────────────────────────────────────────────────────
        warmup_epochs   = 3,
        warmup_momentum = 0.8,
        warmup_bias_lr  = 0.1,
        # ── augmentation ─────────────────────────────────────────────────────
        mosaic          = 1.0,   # combine 4 images — biggest single accuracy boost
        mixup           = 0.05,  # very light blending — helps generalise
        copy_paste      = 0.0,   # OFF — heavy CPU work, slow on Windows w/ workers=0
        degrees         = 10.0,  # rotation
        translate       = 0.1,
        scale           = 0.5,
        shear           = 0.0,   # OFF — rarely helps detection tasks
        perspective     = 0.0,
        fliplr          = 0.5,
        flipud          = 0.05,
        hsv_h           = 0.015,
        hsv_s           = 0.7,
        hsv_v           = 0.4,
        # ── training stability ────────────────────────────────────────────────
        close_mosaic    = 10,       # disable mosaic for final 10 epochs
        label_smoothing = 0.05,
        cos_lr          = True,
        patience        = args.patience,
        # ── output ───────────────────────────────────────────────────────────
        val             = True,
        plots           = True,     # saves confusion matrix, PR curve, F1 curve
        save            = True,
        save_period     = -1,       # only best.pt + last.pt (no per-epoch saves)
        verbose         = True,
        exist_ok        = True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8m — RTX 3050 6 GB")
    parser.add_argument("--epochs",     type=int,   default=100,
                        help="Max epochs (early-stop will likely trigger before this)")
    parser.add_argument("--patience",   type=int,   default=20,
                        help="Early-stop: epochs without mAP improvement before stopping")
    parser.add_argument("--resume",     action="store_true",
                        help="Resume training from models/last.pt")
    parser.add_argument("--no-amp",     action="store_true",
                        help="Disable FP16 mixed precision")
    parser.add_argument("--pretrained", type=str,   default="yolov8m.pt",
                        help="Base weights: 'yolov8m.pt' (COCO) or path to a .pt file")
    args = parser.parse_args()

    # ── imports ───────────────────────────────────────────────────────────────
    try:
        from ultralytics import YOLO
        import torch
    except ImportError as e:
        print(f"ERROR: {e}\nActivate the venv first.")
        sys.exit(1)

    # ── GPU / RAM check ───────────────────────────────────────────────────────
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available — CPU training will be extremely slow.")
        device = "cpu"
    else:
        device = 0

    torch.cuda.empty_cache()   # free any leftover VRAM from previous runs

    gpu_name = torch.cuda.get_device_name(0) if device == 0 else "CPU"
    vram     = torch.cuda.get_device_properties(0).total_memory / 1024**3 if device == 0 else 0
    vram_free= (torch.cuda.get_device_properties(0).total_memory -
                torch.cuda.memory_allocated(0)) / 1024**3 if device == 0 else 0

    ram_free = ram_total = 0
    try:
        import psutil
        mem       = psutil.virtual_memory()
        ram_free  = mem.available / 1024**3
        ram_total = mem.total     / 1024**3
    except ImportError:
        pass

    print(f"\n{'═'*62}")
    print(f"  YOLOv8m Waste Detector — Training")
    print(f"{'═'*62}")
    print(f"  GPU            : {gpu_name}")
    print(f"  VRAM total     : {vram:.1f} GB   free: {vram_free:.1f} GB")
    if ram_total:
        print(f"  RAM            : {ram_total:.1f} GB total   {ram_free:.1f} GB free")

    # ── write data.yaml ───────────────────────────────────────────────────────
    data_yaml = write_data_yaml()

    # ── clean labels ──────────────────────────────────────────────────────────
    n = clean_labels()
    if n:
        print(f"  Labels cleaned : {n} file(s) had polygon segments → converted to boxes")
    else:
        print(f"  Labels         : all clean (no polygon segments found)")

    # ── hyperparams ───────────────────────────────────────────────────────────
    hyps = get_hyps(args)
    hyps["device"] = device
    if args.no_amp:
        hyps["amp"] = False

    print(f"\n  Weights        : {args.pretrained}")
    print(f"  Epochs         : {hyps['epochs']}  (early-stop patience: {hyps['patience']})")
    print(f"  Batch          : {hyps['batch']}  (effective ~{hyps['batch']*4} with mosaic)")
    print(f"  AMP            : {'ON  (FP16 Tensor Cores)' if hyps['amp'] else 'OFF'}")
    print(f"  Workers        : {hyps['workers']}  (main process, no spawn overhead)")
    print(f"  Cache          : {hyps['cache']}")
    print(f"  Optimizer      : {hyps['optimizer']}  lr0={hyps['lr0']}")
    print(f"  Augmentation   : mosaic={hyps['mosaic']}  mixup={hyps['mixup']}  "
          f"degrees={hyps['degrees']}  scale={hyps['scale']}")
    print(f"{'═'*62}\n")

    # ── load model + train ───────────────────────────────────────────────────
    if args.resume:
        # YOLO native resume: pass the last.pt path directly to resume=
        # This restores epoch number, optimizer state, scheduler — everything.
        last_pt = BASE_DIR / "runs" / "detect" / "waste_yolov8m" / "weights" / "last.pt"
        if not last_pt.exists():
            # fallback to models/last.pt if run dir missing
            last_pt = MODEL_DIR / "last.pt"

        if last_pt.exists():
            print(f"  Resuming from {last_pt}\n")
            net = YOLO(str(last_pt))
            net.train(resume=True)   # YOLO reads all settings from last.pt itself
        else:
            print(f"  --resume: last.pt not found — starting fresh.\n")
            net = YOLO(args.pretrained)
            net.train(
                data    = str(data_yaml),
                name    = "waste_yolov8m",
                project = str(BASE_DIR / "runs" / "detect"),
                **hyps,
            )
    else:
        net = YOLO(args.pretrained)
        net.train(
            data    = str(data_yaml),
            name    = "waste_yolov8m",
            project = str(BASE_DIR / "runs" / "detect"),
            **hyps,
        )

    # ── copy best/last → models/ ──────────────────────────────────────────────
    run_dir  = BASE_DIR / "runs" / "detect" / "waste_yolov8m"
    best_src = run_dir / "weights" / "best.pt"
    last_src = run_dir / "weights" / "last.pt"

    print()
    if best_src.exists():
        shutil.copy2(best_src, MODEL_DIR / "best.pt")
        print(f"  ✓ best.pt  →  {MODEL_DIR / 'best.pt'}")
    if last_src.exists():
        shutil.copy2(last_src, MODEL_DIR / "last.pt")
        print(f"  ✓ last.pt  →  {MODEL_DIR / 'last.pt'}")

    # ── final test-split evaluation ───────────────────────────────────────────
    print(f"\n{'─'*62}")
    print("  Final evaluation on TEST split …")
    print(f"{'─'*62}")

    torch.cuda.empty_cache()
    test_net = YOLO(str(MODEL_DIR / "best.pt"))
    metrics  = test_net.val(
        data    = str(data_yaml),
        split   = "test",
        imgsz   = 640,
        batch   = hyps["batch"],
        device  = device,
        workers = 0,
        verbose = True,
    )

    box = metrics.box
    print(f"\n{'═'*62}")
    print(f"  Training complete")
    print(f"{'═'*62}")
    print(f"  mAP50        : {box.map50*100:.2f}%")
    print(f"  mAP50-95     : {box.map*100:.2f}%")
    print(f"  Precision    : {box.mp*100:.2f}%")
    print(f"  Recall       : {box.mr*100:.2f}%")
    print(f"\n  Model        : {MODEL_DIR / 'best.pt'}")
    print(f"  Artefacts    : {run_dir}")
    print(f"{'═'*62}")
    print("\n  Run the app:  streamlit run app.py\n")


# ── Windows multiprocessing guard ────────────────────────────────────────────
if __name__ == "__main__":
    main()
