"""
scripts/train_resnet50.py
─────────────────────────
Fine-tune ResNet50 on the 8-class waste dataset.
Optimised for NVIDIA RTX 3050 6 GB (Ampere) — uses:
  • Mixed-precision training (torch.cuda.amp) for ~2× speed + ~40% less VRAM
  • Batch size 64 (fits comfortably in 6 GB with AMP)
  • 4 DataLoader workers (Windows-safe via __main__ guard)
  • cudnn.benchmark = True for fixed-size input speedup
  • Cosine-annealing LR with warm restarts

Dataset layout expected (YOLO flat format):
    YOLOV8M/datasets/balanced_v3/
        train/  images/*.jpg   labels/*.txt  (YOLO: "class_id cx cy w h")
        valid/  images/        labels/
        test/   images/        labels/

Output:
    models/resnet50_waste.pth        ← best val-acc checkpoint
    models/resnet50_waste_final.pth  ← final epoch weights

Usage (run from project root):
    python scripts/train_resnet50.py
    python scripts/train_resnet50.py --epochs 30 --batch 64 --lr 1e-4
"""

import argparse
import os
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

# ─────────────────────────────────────────────────────────────────────────────
# Paths & constants
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "YOLOV8M" / "datasets" / "balanced_v3"
MODEL_DIR   = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Must match app.py WASTE_CATEGORIES keys
CLASS_NAMES: dict[int, str] = {
    0: "plastic",
    1: "metal",
    2: "glass",
    3: "e_waste",
    4: "medical_waste",
    5: "paper",
    6: "cardboard",
    7: "organic_waste",
}
NUM_CLASSES = len(CLASS_NAMES)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ─────────────────────────────────────────────────────────────────────────────
# Dataset  —  YOLO flat layout → crop dominant bbox → classification sample
# ─────────────────────────────────────────────────────────────────────────────
class WasteDataset(Dataset):
    """
    Each YOLO label file may have several boxes.  We pick the one with the
    largest normalised area (dominant object) and crop it from the image.
    Single-object images are the common case — this handles both cleanly.
    """

    def __init__(self, split: str, transform=None):
        self.transform = transform
        self.samples: list[tuple[Path, int, tuple]] = []

        img_dir = DATASET_DIR / split / "images"
        lbl_dir = DATASET_DIR / split / "labels"

        if not img_dir.exists():
            raise FileNotFoundError(f"Missing: {img_dir}")
        if not lbl_dir.exists():
            raise FileNotFoundError(f"Missing: {lbl_dir}")

        skipped = 0
        for img_path in sorted(img_dir.glob("*.[jJpP][pPnN][gG]*")):
            lbl_path = lbl_dir / (img_path.stem + ".txt")
            if not lbl_path.exists():
                skipped += 1
                continue

            lines = lbl_path.read_text(encoding="utf-8", errors="ignore").strip().splitlines()
            if not lines:
                skipped += 1
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

            if best_cls < 0 or best_bbox is None or best_cls not in CLASS_NAMES:
                skipped += 1
                continue

            self.samples.append((img_path, best_cls, best_bbox))

        print(f"  [{split:5s}]  {len(self.samples):>5} samples  ({skipped} skipped)")
        dist = Counter(s[1] for s in self.samples)
        for cid in sorted(dist):
            print(f"           class {cid}  {CLASS_NAMES[cid]:15s}  {dist[cid]:>4}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, class_id, (cx, cy, bw, bh) = self.samples[idx]

        img = Image.open(img_path).convert("RGB")
        W, H = img.size

        pad = 10
        x1 = max(0,     int((cx - bw / 2) * W) - pad)
        y1 = max(0,     int((cy - bh / 2) * H) - pad)
        x2 = min(W - 1, int((cx + bw / 2) * W) + pad)
        y2 = min(H - 1, int((cy + bh / 2) * H) + pad)

        crop = img.crop((x1, y1, x2, y2)) if (x2 > x1 and y2 > y1) else img

        if self.transform:
            crop = self.transform(crop)
        return crop, class_id


# ─────────────────────────────────────────────────────────────────────────────
# Transforms
# ─────────────────────────────────────────────────────────────────────────────
def get_transforms(img_size: int = 224):
    train_tf = transforms.Compose([
        transforms.Resize((img_size + 32, img_size + 32)),
        transforms.RandomCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
        transforms.RandomRotation(15),
        transforms.RandomGrayscale(p=0.05),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_tf, val_tf


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────
def build_model(num_classes: int, freeze_backbone: bool) -> nn.Module:
    net = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    if freeze_backbone:
        for p in net.parameters():
            p.requires_grad = False
    # Replace head
    net.fc = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(net.fc.in_features, 512),   # 2048 → 512
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.3),
        nn.Linear(512, num_classes),
    )
    return net


# ─────────────────────────────────────────────────────────────────────────────
# One epoch  (AMP-aware)
# ─────────────────────────────────────────────────────────────────────────────
def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer | None,
    device: torch.device,
    scaler: GradScaler | None,
    is_train: bool,
    epoch_label: str = "",
) -> tuple[float, float]:
    model.train() if is_train else model.eval()

    total_loss = correct = total = 0
    use_amp = (scaler is not None) and (device.type == "cuda")

    n_batches = len(loader)
    t_start   = time.time()

    with torch.set_grad_enabled(is_train):
        for batch_idx, (imgs, labels) in enumerate(loader):
            imgs   = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if is_train:
                optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=use_amp):
                outputs = model(imgs)
                loss    = criterion(outputs, labels)

            if is_train:
                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

            bs          = imgs.size(0)
            total_loss += loss.item() * bs
            preds       = outputs.argmax(dim=1)
            correct    += (preds == labels).sum().item()
            total      += bs

            # Inline progress every 20 batches
            if is_train and (batch_idx + 1) % 20 == 0:
                elapsed = time.time() - t_start
                pct     = (batch_idx + 1) / n_batches * 100
                print(f"    {epoch_label}  [{batch_idx+1:3d}/{n_batches}  {pct:5.1f}%]  "
                      f"loss={total_loss/total:.4f}  acc={correct/total*100:.1f}%  "
                      f"[{elapsed:.0f}s]",
                      end="\r", flush=True)

    if is_train:
        print()  # newline after \r progress

    return total_loss / total, correct / total


# ─────────────────────────────────────────────────────────────────────────────
# Main training loop
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="ResNet50 waste classifier — RTX 3050 optimised")
    parser.add_argument("--epochs",    type=int,   default=25)
    parser.add_argument("--warmup",    type=int,   default=3,    help="Head-only warmup epochs")
    parser.add_argument("--batch",     type=int,   default=64,   help="Batch size (64 fits 6 GB with AMP)")
    parser.add_argument("--lr",        type=float, default=1e-4, help="Full fine-tune LR")
    parser.add_argument("--lr-warmup", type=float, default=5e-4, help="Head-only warmup LR")
    parser.add_argument("--img-size",  type=int,   default=224)
    parser.add_argument("--workers",   type=int,   default=4,    help="DataLoader workers")
    parser.add_argument("--no-amp",    action="store_true",      help="Disable mixed precision")
    parser.add_argument("--no-cuda",   action="store_true",      help="Force CPU")
    args = parser.parse_args()

    # ── Device ──────────────────────────────────────────────────────────────
    device = torch.device("cpu")
    if not args.no_cuda:
        if torch.cuda.is_available():
            device = torch.device("cuda")
            torch.backends.cudnn.benchmark = True   # Ampere speedup for fixed input size
        elif torch.backends.mps.is_available():
            device = torch.device("mps")

    use_amp = (device.type == "cuda") and (not args.no_amp)
    scaler  = GradScaler() if use_amp else None

    # Windows multiprocessing requires __main__ guard; we use persistent_workers
    # to avoid re-spawning workers every epoch (big win on Windows).
    num_workers       = args.workers if os.name != "nt" else args.workers
    persistent_workers = (num_workers > 0)

    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "N/A"

    print(f"\n{'═'*62}")
    print(f"  ResNet50 Waste Classifier  —  Training")
    print(f"{'═'*62}")
    print(f"  GPU          : {gpu_name}")
    print(f"  Device       : {device}")
    print(f"  Mixed prec.  : {'ON (AMP)' if use_amp else 'OFF'}")
    print(f"  Classes      : {NUM_CLASSES}")
    print(f"  Epochs       : {args.epochs}  (warmup: {args.warmup})")
    print(f"  Batch size   : {args.batch}")
    print(f"  LR warmup    : {args.lr_warmup}   →   full: {args.lr}")
    print(f"  Image size   : {args.img_size}px")
    print(f"  Workers      : {num_workers}")
    print(f"{'═'*62}\n")

    # ── Dataset ─────────────────────────────────────────────────────────────
    print("Loading datasets …")
    train_tf, val_tf = get_transforms(args.img_size)
    train_ds = WasteDataset("train", transform=train_tf)
    val_ds   = WasteDataset("valid", transform=val_tf)
    print()

    train_loader = DataLoader(
        train_ds, batch_size=args.batch, shuffle=True,
        num_workers=num_workers, pin_memory=(device.type == "cuda"),
        persistent_workers=persistent_workers,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch, shuffle=False,
        num_workers=num_workers, pin_memory=(device.type == "cuda"),
        persistent_workers=persistent_workers,
    )

    # ── Model ────────────────────────────────────────────────────────────────
    net = build_model(NUM_CLASSES, freeze_backbone=True).to(device)
    trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)
    total_p   = sum(p.numel() for p in net.parameters())
    print(f"ResNet50 loaded  (ImageNet-V2 pretrained)")
    print(f"  Total params    : {total_p:,}")
    print(f"  Trainable now   : {trainable:,}  (head only)\n")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    best_val_acc = 0.0
    best_ckpt    = MODEL_DIR / "resnet50_waste.pth"
    final_ckpt   = MODEL_DIR / "resnet50_waste_final.pth"

    def save_best(epoch, val_acc):
        torch.save({
            "epoch":            epoch,
            "model_state_dict": net.state_dict(),
            "val_acc":          val_acc,
            "class_names":      CLASS_NAMES,
            "img_size":         args.img_size,
        }, best_ckpt)

    # ── Phase 1 : Warmup — head only ────────────────────────────────────────
    print("── Phase 1 · Head-only warmup ──────────────────────────────────")
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, net.parameters()),
        lr=args.lr_warmup, weight_decay=1e-4,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.warmup)

    for epoch in range(1, args.warmup + 1):
        t0  = time.time()
        lbl = f"Warmup {epoch:2d}/{args.warmup}"
        tr_loss, tr_acc = run_epoch(net, train_loader, criterion, optimizer,
                                    device, scaler, is_train=True, epoch_label=lbl)
        vl_loss, vl_acc = run_epoch(net, val_loader,   criterion, None,
                                    device, scaler, is_train=False)
        scheduler.step()

        marker = ""
        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            save_best(epoch, vl_acc)
            marker = "  ✓ best"

        elapsed = time.time() - t0
        print(f"  Warmup {epoch:2d}/{args.warmup}  "
              f"loss={tr_loss:.4f}  acc={tr_acc*100:.1f}%  │  "
              f"val_loss={vl_loss:.4f}  val_acc={vl_acc*100:.1f}%  "
              f"[{elapsed:.0f}s]{marker}")

    # ── Phase 2 : Full fine-tuning — unfreeze backbone ──────────────────────
    print(f"\n── Phase 2 · Full fine-tuning  (backbone + head) ───────────────")
    for p in net.parameters():
        p.requires_grad = True

    trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"  Trainable params now: {trainable:,}\n")

    backbone_params = [p for n, p in net.named_parameters() if not n.startswith("fc")]
    head_params     = list(net.fc.parameters())
    optimizer = optim.AdamW([
        {"params": backbone_params, "lr": args.lr * 0.1,  "weight_decay": 1e-4},
        {"params": head_params,     "lr": args.lr,         "weight_decay": 1e-4},
    ])
    remaining = args.epochs - args.warmup
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=remaining, eta_min=1e-6)

    for epoch in range(args.warmup + 1, args.epochs + 1):
        t0  = time.time()
        lbl = f"Epoch  {epoch:2d}/{args.epochs}"
        tr_loss, tr_acc = run_epoch(net, train_loader, criterion, optimizer,
                                    device, scaler, is_train=True, epoch_label=lbl)
        vl_loss, vl_acc = run_epoch(net, val_loader,   criterion, None,
                                    device, scaler, is_train=False)
        scheduler.step()

        marker = ""
        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            save_best(epoch, vl_acc)
            marker = "  ✓ best"

        elapsed = time.time() - t0
        print(f"  Epoch  {epoch:2d}/{args.epochs}  "
              f"loss={tr_loss:.4f}  acc={tr_acc*100:.1f}%  │  "
              f"val_loss={vl_loss:.4f}  val_acc={vl_acc*100:.1f}%  "
              f"[{elapsed:.0f}s]{marker}")

    # ── Save final ───────────────────────────────────────────────────────────
    torch.save({
        "epoch":            args.epochs,
        "model_state_dict": net.state_dict(),
        "val_acc":          best_val_acc,
        "class_names":      CLASS_NAMES,
        "img_size":         args.img_size,
    }, final_ckpt)

    # ── VRAM report ──────────────────────────────────────────────────────────
    if device.type == "cuda":
        peak = torch.cuda.max_memory_allocated() / 1024**3
        print(f"\n  Peak VRAM used : {peak:.2f} GB  (of 6 GB available)")

    print(f"\n{'═'*62}")
    print(f"  Training complete")
    print(f"  Best val accuracy   : {best_val_acc*100:.2f}%")
    print(f"  Best checkpoint     : {best_ckpt}")
    print(f"  Final checkpoint    : {final_ckpt}")
    print(f"{'═'*62}\n")
    print("  Next step → run the app:")
    print("      streamlit run app.py\n")


# ── Windows multiprocessing safety ──────────────────────────────────────────
# DataLoader workers are spawned as child processes on Windows.
# Without this guard the worker code re-runs the script on import,
# causing a fork-bomb.  Always call main() inside __main__ on Windows.
if __name__ == "__main__":
    main()
