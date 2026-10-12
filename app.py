import os
import platform
import time
from collections import Counter
from datetime import datetime
import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageOps
import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from database import save_detections

# ─────────────────────────────────────────────
# Page configuration
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="SWMS — Smart Waste Management System | Government of India",
    page_icon="🇮🇳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
BASE_DIR         = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH       = os.path.join(BASE_DIR, "models", "best.pt")
RESNET_PATH      = os.path.join(BASE_DIR, "models", "resnet50_waste.pth")

# ─────────────────────────────────────────────
# ResNet50 — class mapping (must match train_resnet50.py)
# ─────────────────────────────────────────────
RESNET_CLASS_NAMES = {
    0: "plastic",
    1: "metal",
    2: "glass",
    3: "e_waste",
    4: "medical_waste",
    5: "paper",
    6: "cardboard",
    7: "organic_waste",
}
NUM_RESNET_CLASSES = len(RESNET_CLASS_NAMES)

# ImageNet normalisation for ResNet50 inference
_RESNET_TF = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# ─────────────────────────────────────────────
# Load YOLOv8m  (detection — bounding boxes only)
#   Cache key includes mtime so swapping best.pt is picked up automatically.
# ─────────────────────────────────────────────
@st.cache_resource
def load_model(path: str, mtime: float):
    if not os.path.exists(path):
        st.error(f"YOLO model file not found: {path}")
        return None
    try:
        return YOLO(path)
    except Exception as e:
        st.error(f"Error loading YOLO model: {e}")
        return None


model = load_model(
    MODEL_PATH,
    os.path.getmtime(MODEL_PATH) if os.path.exists(MODEL_PATH) else 0.0,
)

# ─────────────────────────────────────────────
# Load ResNet50  (classification — crop → class label)
# ─────────────────────────────────────────────
@st.cache_resource
def load_resnet(path: str, mtime: float):
    """
    Builds the same ResNet50 architecture used in train_resnet50.py,
    loads the saved weights, and returns the model in eval mode.
    Returns None (with a sidebar warning) if the checkpoint is absent.
    """
    if not os.path.exists(path):
        return None          # caller shows the warning

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = models.resnet50(weights=None)
    net.fc = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(net.fc.in_features, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.3),
        nn.Linear(512, NUM_RESNET_CLASSES),
    )
    try:
        ckpt = torch.load(path, map_location=device)
        net.load_state_dict(ckpt["model_state_dict"])
        net.to(device)
        net.eval()
        return net
    except Exception as e:
        st.error(f"Error loading ResNet50: {e}")
        return None


resnet_model = load_resnet(
    RESNET_PATH,
    os.path.getmtime(RESNET_PATH) if os.path.exists(RESNET_PATH) else 0.0,
)

_RESNET_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def classify_crop_resnet(crop_pil: Image.Image) -> tuple[str, float]:
    """
    Run ResNet50 on a PIL crop and return (class_name, confidence).
    Falls back to 'unknown' if ResNet50 is not loaded.
    """
    if resnet_model is None:
        return "unknown", 0.0
    tensor = _RESNET_TF(crop_pil).unsqueeze(0).to(_RESNET_DEVICE)
    with torch.no_grad():
        logits = resnet_model(tensor)
        probs  = torch.softmax(logits, dim=1)[0]
        top1   = int(probs.argmax())
        conf   = float(probs[top1])
    return RESNET_CLASS_NAMES.get(top1, "unknown"), conf

# ─────────────────────────────────────────────
# Waste categories — 8 classes matching the YOLOv8m model
# (removed 'can' and 'cable' from the old 10-class system —
#  'can' folds into 'metal', 'cable' folds into 'e_waste')
# ─────────────────────────────────────────────
WASTE_CATEGORIES = {
    'plastic':       {'color': (144, 116, 14),  'hex': '#0E7490', 'icon': '♻️',  'bin': 'Yellow Bin',  'schedule': 'Sch. I(a)', 'description': 'Recyclable plastic items',    'nature': 'Non-Biodegradable'},
    'metal':         {'color': (9, 83, 180),    'hex': '#B45309', 'icon': '🔩',  'bin': 'Blue Bin',    'schedule': 'Sch. I(b)', 'description': 'Metal cans, scrap and items',  'nature': 'Non-Biodegradable'},
    'glass':         {'color': (110, 118, 15),  'hex': '#0F766E', 'icon': '🍾',  'bin': 'Green Bin',   'schedule': 'Sch. I(c)', 'description': 'Glass bottles and containers', 'nature': 'Non-Biodegradable'},
    'e_waste':       {'color': (18, 45, 124),   'hex': '#7C2D12', 'icon': '💻',  'bin': 'E-Waste Bin', 'schedule': 'Sch. II(a)','description': 'Electronic waste and cables',  'nature': 'Non-Biodegradable'},
    'medical_waste': {'color': (28, 28, 185),   'hex': '#B91C1C', 'icon': '💉',  'bin': 'Red Bin',     'schedule': 'Sch. III',  'description': 'Medical and hazardous waste',  'nature': 'Hazardous'},
    'paper':         {'color': (216, 78, 29),   'hex': '#1D4ED8', 'icon': '📄',  'bin': 'Blue Bin',    'schedule': 'Sch. IV(a)','description': 'Paper and documents',          'nature': 'Biodegradable'},
    'cardboard':     {'color': (15, 53, 120),   'hex': '#78350F', 'icon': '📦',  'bin': 'Blue Bin',    'schedule': 'Sch. IV(b)','description': 'Cardboard boxes',              'nature': 'Biodegradable'},
    'organic_waste': {'color': (61, 128, 21),   'hex': '#15803D', 'icon': '🍃',  'bin': 'Green Bin',   'schedule': 'Sch. IV(c)','description': 'Biodegradable waste',          'nature': 'Biodegradable'},
}

DEFAULT_CATEGORY = {
    'color': (128, 114, 107), 'hex': '#6B7280',
    'icon': '🗑️', 'bin': 'General Bin', 'schedule': 'Sch. V', 'description': 'Unclassified waste type', 'nature': 'Unknown'
}

# Nature badge colours — light tints for a paper-white interface
NATURE_COLORS = {
    'Biodegradable':     {'bg': '#ECFDF3', 'text': '#15803D', 'border': '#86EFAC'},
    'Non-Biodegradable': {'bg': '#FFF7ED', 'text': '#C2410C', 'border': '#FDBA74'},
    'Hazardous':         {'bg': '#FEF2F2', 'text': '#B91C1C', 'border': '#FCA5A5'},
    'Unknown':           {'bg': '#F3F4F6', 'text': '#4B5563', 'border': '#D1D5DB'},
}


def check_model_classes(yolo_model):
    """
    Compares the classes inside the loaded model with the 8 classes this app
    knows about. Returns (ok, message). The app looks classes up by NAME, so
    a different order is harmless.
    """
    expected = set(WASTE_CATEGORIES.keys())
    names = [normalize_class(yolo_model.names[i]) for i in range(len(yolo_model.names))]
    missing = sorted(expected - set(names))
    extra   = sorted(set(names) - expected)
    if not missing and not extra:
        return True, ""
    msg = f"This model has {len(names)} classes, but the app expects {len(expected)}. "
    if missing:
        msg += f"Missing: {', '.join(missing)}. "
    if extra:
        msg += f"Not in the app's list: {', '.join(extra)}. "
    msg += "Make sure models/best.pt is the new YOLOv8m 8-class file."
    return False, msg


def conf_band(c: float) -> str:
    return "High" if c >= 0.50 else ("Medium" if c >= 0.25 else "Low")


def ref_number(seed: int, prefix: str = "WMS") -> str:
    """Generates a gazette-style reference number, e.g. WMS/2026/48213."""
    return f"{prefix}/2026/{(seed * 733 + 1009) % 90000 + 10000}"


def open_camera(index: int) -> cv2.VideoCapture:
    """
    Opens a camera with the correct backend for the current OS.
    cv2.CAP_DSHOW is Windows-only and silently fails to open a camera
    on Linux/Mac, so pick a sane backend per platform instead of
    hardcoding one.

    On Windows, CAP_DSHOW can hang or throw on some webcam drivers
    (especially when re-opening a different index right after another
    camera was released). If it fails to open, fall back to CAP_MSMF,
    and finally to the default backend, instead of letting the
    exception propagate and kill the whole Streamlit script run.
    """
    system = platform.system()

    if system == "Windows":
        backends_to_try = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    elif system == "Linux":
        backends_to_try = [cv2.CAP_V4L2, cv2.CAP_ANY]
    else:  # macOS and anything else
        backends_to_try = [cv2.CAP_ANY]

    for backend in backends_to_try:
        try:
            cap = cv2.VideoCapture(index, backend)
            if cap is not None and cap.isOpened():
                return cap
            if cap is not None:
                cap.release()
        except Exception:
            # Some backend/driver combinations raise instead of just
            # failing to open — keep trying the next backend rather
            # than letting this bubble up and crash the app.
            continue

    # Nothing worked — return a capture object that reports as not
    # opened, so the caller's existing `if not cap.isOpened():` check
    # handles it gracefully instead of raising.
    return cv2.VideoCapture()


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def normalize_class(name) -> str:
    """'Organic Waste' / 'organic-waste' / 'E-Waste' -> 'organic_waste' / 'e_waste'."""
    n = str(name).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {"ewaste": "e_waste", "e_waste": "e_waste", "organic": "organic_waste",
               "medical": "medical_waste", "biodegradable": "organic_waste"}
    return aliases.get(n, n)


def _extract(result, source_image: Image.Image | None = None):
    """
    Extract detections from a YOLO result.

    Two-stage pipeline:
      1. YOLOv8m provides bounding boxes (where the waste is).
      2. ResNet50 classifies each cropped region (what class it is).

    If ResNet50 is not loaded yet (e.g. model file missing), the class
    label from YOLO's own head is used as a fallback so the app still
    works before training is complete.

    source_image: the original PIL image, needed to crop boxes for ResNet50.
                  Pass None to skip ResNet50 classification (YOLO-only fallback).
    """
    dets = []

    # ── Classification model (yolov8-cls): no boxes, only class probs ──
    probs = getattr(result, "probs", None)
    if probs is not None:
        cname = normalize_class(result.names[int(probs.top1)])
        # Even for a cls model, try ResNet50 on the whole image
        if source_image is not None and resnet_model is not None:
            cname, conf = classify_crop_resnet(source_image)
        else:
            conf = float(probs.top1conf)
        dets.append({
            'class':         cname,
            'confidence':    conf,
            'category_info': WASTE_CATEGORIES.get(cname, DEFAULT_CATEGORY),
            'xyxy':          None,
        })
        return dets

    # ── Detection model: use YOLO boxes + ResNet50 classification ───
    if result.boxes is None or len(result.boxes) == 0:
        return dets

    W = H = None
    if source_image is not None:
        W, H = source_image.size

    for box in result.boxes:
        xyxy = [int(v) for v in box.xyxy[0].tolist()]

        # ── ResNet50 classification on the cropped region ────────────
        if source_image is not None and resnet_model is not None and W and H:
            x1, y1, x2, y2 = xyxy
            # 10px padding, clamped to image bounds
            pad = 10
            x1c = max(0,     x1 - pad)
            y1c = max(0,     y1 - pad)
            x2c = min(W - 1, x2 + pad)
            y2c = min(H - 1, y2 + pad)
            if x2c > x1c and y2c > y1c:
                crop  = source_image.crop((x1c, y1c, x2c, y2c))
                cname, conf = classify_crop_resnet(crop)
            else:
                cname = normalize_class(result.names.get(int(box.cls[0]), str(int(box.cls[0]))))
                conf  = float(box.conf[0])
        else:
            # Fallback: use YOLO's own class prediction
            cname = normalize_class(result.names.get(int(box.cls[0]), str(int(box.cls[0]))))
            conf  = float(box.conf[0])

        dets.append({
            'class':         cname,
            'confidence':    conf,
            'category_info': WASTE_CATEGORIES.get(cname, DEFAULT_CATEGORY),
            'xyxy':          xyxy,
        })
    return dets


def classify_image(image, conf_threshold=0.25):
    """Runs YOLO on a PIL image for boxes, then ResNet50 on each crop.
    If nothing is found, retries once at a lower confidence."""
    results = model(image, conf=conf_threshold, imgsz=640, iou=0.5, max_det=100, verbose=False)
    detections = [d for r in results for d in _extract(r, source_image=image)]
    if not detections and conf_threshold > 0.10:
        results = model(image, conf=0.10, imgsz=640, verbose=False)
        detections = [d for r in results for d in _extract(r, source_image=image)]
    return results, detections


def _iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def _nms(dets, iou_thr=0.5, keep=30):
    """Class-wise non-max suppression; keeps the strongest of overlapping same-class boxes."""
    dets = sorted(dets, key=lambda d: -d['confidence'])
    kept = []
    for d in dets:
        if d['xyxy'] is None:
            kept.append(d)
            continue
        if any(k['class'] == d['class'] and k['xyxy'] is not None
               and _iou(k['xyxy'], d['xyxy']) > iou_thr for k in kept):
            continue
        kept.append(d)
    return kept[:keep]


def detect_multiscale(image, conf):
    """
    Runs YOLO on the full photo AND on four overlapping 60%-size tiles,
    then classifies each crop with ResNet50 and merges via NMS.
    """
    W, H = image.size

    def run(img, ox, oy):
        """Run YOLO on img, offset boxes by (ox, oy), classify crops with ResNet50."""
        out = []
        pil_img = img if isinstance(img, Image.Image) else Image.fromarray(img)
        for r in model(pil_img, conf=conf, imgsz=640, iou=0.5, max_det=100, verbose=False):
            out += _extract(r, source_image=pil_img)
        # Shift box coordinates back to original image space
        for d in out:
            if d['xyxy'] is not None:
                x1, y1, x2, y2 = d['xyxy']
                d['xyxy'] = [x1 + ox, y1 + oy, x2 + ox, y2 + oy]
        return out

    dets = run(image, 0, 0)
    if getattr(model, "task", "detect") == "detect":
        tw, th = int(W * 0.6), int(H * 0.6)
        for ox in (0, W - tw):
            for oy in (0, H - th):
                tile = image.crop((ox, oy, ox + tw, oy + th))
                dets += run(tile, ox, oy)
    return _nms(dets)


def draw_detections(image, detections):
    """Draws every prediction on a copy of the photo. Returns an RGB numpy array."""
    import numpy as np
    canvas = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    for i, d in enumerate(detections):
        if d['xyxy'] is None:
            continue
        x1, y1, x2, y2 = d['xyxy']
        col = d['category_info']['color']
        cv2.rectangle(canvas, (x1, y1), (x2, y2), col, 2)
        label = f"{i+1}. {d['class'].replace('_',' ').title()} {d['confidence']*100:.0f}%"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        ty = max(th + 6, y1)
        cv2.rectangle(canvas, (x1, ty - th - 6), (x1 + tw + 6, ty), col, -1)
        cv2.putText(canvas, label, (x1 + 3, ty - 4), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)


def debug_predictions(image):
    """Raw top predictions at a very low confidence, to see what the model 'thinks'.
    Uses YOLO-only (no ResNet50) so the raw YOLO output is visible for diagnostics."""
    res = model(image, conf=0.001, imgsz=640, verbose=False)
    rows = []
    for r in res:
        for box in (r.boxes or []):
            cname = normalize_class(r.names.get(int(box.cls[0]), str(int(box.cls[0]))))
            rows.append((cname, round(float(box.conf[0]), 3)))
        probs = getattr(r, "probs", None)
        if probs is not None:
            cname = normalize_class(r.names[int(probs.top1)])
            rows.append((cname, round(float(probs.top1conf), 3)))
    rows.sort(key=lambda x: -x[1])
    return rows[:8]


def run_inference_on_frame(frame_bgr, conf_threshold):
    frame_rgb  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil_frame  = Image.fromarray(frame_rgb)
    results    = model(frame_rgb, conf=conf_threshold, imgsz=640, verbose=False)
    detections = [d for r in results for d in _extract(r, source_image=pil_frame)]

    for det in detections:
        cat = det['category_info']
        if det['xyxy'] is None:
            x1, y1, x2, y2 = 10, 40, 10, 40
        else:
            x1, y1, x2, y2 = det['xyxy']
            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), cat['color'], 2)
        label = f"{det['class'].replace('_',' ').title()} {det['confidence']*100:.0f}%"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(frame_bgr, (x1, y1 - th - 14), (x1 + tw + 10, y1), cat['color'], -1)
        cv2.putText(frame_bgr, label, (x1 + 5, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

    return frame_bgr, detections


# ─────────────────────────────────────────────
# Global CSS — official portal theme
#   Ink:        #14213D   (headings / primary text)
#   Ashoka Blue #0F4C81   (primary / links / header)
#   Saffron     #E17A1D   (accent / CTAs)
#   India Green #15803D   (biodegradable / success)
#   Paper       #F5F6F8   (page background)
#   Card        #FFFFFF
#   Border      #DCE1E8
#   Muted text  #5B6B7C
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600;9..144,700&family=Source+Sans+3:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Source Sans 3', sans-serif;
    background-color: #F5F6F8 !important;
    color: #263241;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 0 !important; padding-bottom: 2rem !important; max-width: 1180px; }

/* ── Tricolour strip ── */
.tricolour-strip { display: flex; width: 100%; height: 5px; margin-bottom: 0; }
.tricolour-strip div { flex: 1; }
.strip-saffron { background: #FF9933; }
.strip-white   { background: #FFFFFF; border-top: 1px solid #E5E7EB; border-bottom: 1px solid #E5E7EB; }
.strip-green   { background: #138808; }

/* ── Official header bar ── */
.gov-header {
    background: linear-gradient(180deg, #123A66 0%, #0F4C81 100%);
    padding: 0.85rem 2rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 0;
}
.gov-header-left { display: flex; align-items: center; gap: 0.9rem; }
.emblem-badge {
    width: 46px; height: 46px;
    border-radius: 50%;
    background: #FFFFFF;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.4rem;
    border: 2px solid #E17A1D;
    flex-shrink: 0;
}
.gov-header-title { color: #FFFFFF; }
.gov-header-eyebrow {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.66rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #B9D3EC;
    margin-bottom: 2px;
}
.gov-header-name {
    font-family: 'Fraunces', serif;
    font-size: 1.18rem;
    font-weight: 600;
    line-height: 1.25;
}
.gov-header-right { display: flex; align-items: center; gap: 0.6rem; }
.gov-pill {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.66rem;
    letter-spacing: 0.08em;
    color: #DCE9F7;
    border: 1px solid rgba(255,255,255,0.35);
    border-radius: 3px;
    padding: 4px 10px;
    text-transform: uppercase;
}

/* ── Breadcrumb ── */
.breadcrumb {
    background: #FFFFFF;
    border-bottom: 1px solid #DCE1E8;
    padding: 0.55rem 2rem;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    color: #5B6B7C;
    letter-spacing: 0.02em;
}
.breadcrumb span.current { color: #0F4C81; font-weight: 600; }

/* ── Hero / circular notice ── */
.hero-block {
    background: #FFFFFF;
    border: 1px solid #DCE1E8;
    border-top: 4px solid #E17A1D;
    border-radius: 4px;
    padding: 2.1rem 2.6rem;
    margin: 1.6rem 0 1.4rem;
}
.hero-ref {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #E17A1D;
    margin-bottom: 0.6rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.hero-ref::before { content: '§'; font-size: 0.9rem; }
.hero-title {
    font-family: 'Fraunces', serif;
    font-size: 2.15rem;
    font-weight: 600;
    color: #14213D;
    line-height: 1.2;
    margin-bottom: 0.55rem;
}
.hero-sub {
    font-size: 0.96rem;
    color: #5B6B7C;
    max-width: 640px;
    line-height: 1.65;
}
.hero-badge-row { display: flex; gap: 0.55rem; margin-top: 1.3rem; flex-wrap: wrap; }
.hero-badge {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.66rem;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    padding: 5px 11px;
    border-radius: 3px;
    border: 1px solid;
}
.hb-blue   { border-color: #0F4C81; color: #0F4C81; background: #EAF2FA; }
.hb-green  { border-color: #15803D; color: #15803D; background: #ECFDF3; }
.hb-amber  { border-color: #E17A1D; color: #B45309; background: #FFF7ED; }

/* ── Status bar ── */
.status-bar {
    background: #FFFFFF;
    border: 1px solid #DCE1E8;
    border-radius: 4px;
    padding: 0.7rem 1.3rem;
    display: flex;
    align-items: center;
    gap: 1.8rem;
    margin-bottom: 1.2rem;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    color: #5B6B7C;
    letter-spacing: 0.03em;
    flex-wrap: wrap;
}
.status-dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; margin-right: 6px; }
.dot-green { background: #15803D; box-shadow: 0 0 5px rgba(21,128,61,0.5); }
.dot-amber { background: #E17A1D; box-shadow: 0 0 5px rgba(225,122,29,0.5); }
.dot-grey  { background: #A9B4C0; }
.status-val { color: #0F4C81; font-weight: 600; }

/* ── Section label ── */
.section-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    color: #0F4C81;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    margin-bottom: 0.7rem;
    padding-bottom: 0.35rem;
    border-bottom: 1px solid #DCE1E8;
}

/* ── Gazette-style detection card ── */
.det-card {
    background: #FFFFFF;
    border: 1px solid #DCE1E8;
    border-left: 4px solid;
    border-radius: 3px;
    padding: 0.85rem 1.05rem;
    margin: 0.55rem 0;
}
.det-card-top { display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 3px; }
.det-card-title { font-family: 'Fraunces', serif; font-size: 1rem; font-weight: 600; color: #14213D; }
.det-card-ref { font-family: 'IBM Plex Mono', monospace; font-size: 0.62rem; color: #A0ACB9; letter-spacing: 0.04em; white-space: nowrap; }
.det-card-meta { font-size: 0.78rem; color: #5B6B7C; display: flex; align-items: center; gap: 0.7rem; flex-wrap: wrap; margin-top: 6px; }
.conf-pill { font-family: 'IBM Plex Mono', monospace; font-size: 0.7rem; padding: 2px 9px; border-radius: 3px; font-weight: 600; }
.nature-tag { font-family: 'IBM Plex Mono', monospace; font-size: 0.64rem; letter-spacing: 0.07em; padding: 2px 9px; border-radius: 3px; border: 1px solid; text-transform: uppercase; }
.bin-tag { background: #F5F6F8; border: 1px solid #DCE1E8; border-radius: 3px; padding: 2px 9px; font-size: 0.72rem; color: #445468; font-family: 'IBM Plex Mono', monospace; }
.sched-tag { font-family: 'IBM Plex Mono', monospace; font-size: 0.64rem; color: #8593A3; }

/* ── Info / disposal panel ── */
.info-panel { background: #FFFFFF; border: 1px solid #DCE1E8; border-radius: 4px; padding: 1rem 1.2rem; margin-bottom: 1rem; }
.info-panel-title { font-family: 'Fraunces', serif; font-size: 0.88rem; font-weight: 600; color: #14213D; margin-bottom: 0.6rem; border-bottom: 1px solid #DCE1E8; padding-bottom: 0.4rem; }

/* ── Sidebar "Schedule" table row ── */
.cat-row { display: flex; align-items: center; gap: 0.6rem; padding: 0.42rem 0; border-bottom: 1px dotted #DCE1E8; font-size: 0.78rem; }
.cat-dot { width: 9px; height: 9px; border-radius: 2px; flex-shrink: 0; }
.cat-name { color: #29394D; font-weight: 500; flex: 1; }
.cat-sched { font-family: 'IBM Plex Mono', monospace; font-size: 0.62rem; color: #A0ACB9; margin-right: 6px; }
.cat-bin { font-family: 'IBM Plex Mono', monospace; font-size: 0.64rem; color: #7A8A9C; }

/* ── Upload zone ── */
.upload-hint { border: 2px dashed #C6CEDA; border-radius: 6px; padding: 2.5rem; text-align: center; color: #8593A3; font-size: 0.85rem; margin-bottom: 1rem; background: #FFFFFF; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] { background: #FFFFFF !important; border-radius: 4px !important; border: 1px solid #DCE1E8 !important; padding: 4px !important; gap: 4px !important; }
.stTabs [data-baseweb="tab"] { background: transparent !important; border-radius: 3px !important; color: #5B6B7C !important; font-family: 'Fraunces', serif !important; font-weight: 600 !important; font-size: 0.88rem !important; padding: 0.5rem 1.2rem !important; border: none !important; transition: all 0.2s !important; }
.stTabs [aria-selected="true"] { background: #EAF2FA !important; color: #0F4C81 !important; border: 1px solid #0F4C81 !important; }

/* ── Buttons ── */
.stButton > button { font-family: 'Source Sans 3', sans-serif !important; font-weight: 600 !important; font-size: 0.85rem !important; border-radius: 4px !important; transition: all 0.2s !important; letter-spacing: 0.01em !important; }
.stButton > button[kind="primary"] { background: #E17A1D !important; color: #FFFFFF !important; border: none !important; box-shadow: 0 2px 8px rgba(225,122,29,0.28) !important; }
.stButton > button[kind="primary"]:hover { background: #C2670F !important; box-shadow: 0 3px 10px rgba(225,122,29,0.4) !important; }
.stButton > button[kind="primary"]:disabled { background: #C6CEDA !important; box-shadow: none !important; cursor: not-allowed !important; }
.stButton > button[kind="secondary"], .stButton > button:not([kind="primary"]) { background: #FFFFFF !important; color: #0F4C81 !important; border: 1px solid #C6CEDA !important; }
.stButton > button:not([kind="primary"]):hover { border-color: #0F4C81 !important; background: #EAF2FA !important; }

/* ── Slider ── */
.stSlider > div > div > div > div { background: #0F4C81 !important; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] { background: #FFFFFF !important; border-right: 1px solid #DCE1E8 !important; }
section[data-testid="stSidebar"] .stMarkdown h3, section[data-testid="stSidebar"] .stMarkdown h2 {
    font-family: 'Fraunces', serif !important; color: #14213D !important; font-size: 0.85rem !important;
    letter-spacing: 0.02em !important; text-transform: none !important; font-weight: 600 !important;
}

/* ── Number input ── */
.stNumberInput > div > div > input { background: #FFFFFF !important; border: 1px solid #C6CEDA !important; color: #263241 !important; border-radius: 4px !important; font-family: 'IBM Plex Mono', monospace !important; }

.stAlert { border-radius: 4px !important; }

/* ── Footer ── */
.gov-footer-wrap { background: #14213D; margin-top: 2.4rem; }
.gov-footer { max-width: 1180px; margin: 0 auto; padding: 1.8rem 2rem 1.2rem; display: grid; grid-template-columns: 1.4fr 1fr 1fr 1fr; gap: 1.6rem; }
.footer-col-title { font-family: 'IBM Plex Mono', monospace; font-size: 0.68rem; letter-spacing: 0.1em; text-transform: uppercase; color: #7DA8D6; margin-bottom: 0.7rem; }
.footer-col a, .footer-col div.flink { display: block; font-size: 0.8rem; color: #C3D2E3; margin-bottom: 0.45rem; text-decoration: none; }
.footer-brand { font-family: 'Fraunces', serif; font-size: 1.05rem; color: #FFFFFF; font-weight: 600; margin-bottom: 0.5rem; }
.footer-disclaimer { font-size: 0.76rem; color: #7E8EA3; line-height: 1.6; max-width: 480px; }
.gov-footer-bottom {
    border-top: 1px solid rgba(255,255,255,0.12);
    text-align: center;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.66rem;
    color: #6C7D93;
    letter-spacing: 0.06em;
    padding: 0.9rem 0 1.1rem;
    text-transform: uppercase;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Tricolour strip + official header
# ─────────────────────────────────────────────
st.markdown("""
<div class="tricolour-strip"><div class="strip-saffron"></div><div class="strip-white"></div><div class="strip-green"></div></div>
<div class="gov-header">
    <div class="gov-header-left">
        <div class="emblem-badge">🇮🇳</div>
        <div class="gov-header-title">
            <div class="gov-header-eyebrow">Government of India · Ministry of Environment, Forest &amp; Climate Change</div>
            <div class="gov-header-name">Smart Waste Management System</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Hero
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero-block">
    <div class="hero-ref">Circular No. SWMS/2026/AI-DET &nbsp;·&nbsp; Effective Immediately</div>
    <div class="hero-title">Classify, sort and recover municipal solid waste using computer vision</div>
    <div class="hero-sub">
        This portal identifies waste items from a live camera feed or an uploaded photograph, assigns each item
        to its statutory bin category, and records the observation for municipal reporting.
    </div>
    <div class="hero-badge-row">
        <span class="hero-badge hb-blue">YOLOv8m Detection + ResNet50 Classification</span>
        <span class="hero-badge hb-amber">8 Notified Waste Classes</span>
        <span class="hero-badge hb-green">Live Camera &amp; Photo Upload</span>
    </div>
</div>
""", unsafe_allow_html=True)

if model is None:
    st.warning("The detection model could not be loaded. Verify that `models/best.pt` exists before using this portal.")
else:
    _model_ok, _model_msg = check_model_classes(model)
    if not _model_ok:
        st.warning(_model_msg)

if resnet_model is None:
    st.info(
        "ℹ️ **ResNet50 classifier not loaded** — `models/resnet50_waste.pth` not found. "
        "YOLO's own class labels will be used as a fallback until you train and place the ResNet50 model. "
        "Run `python scripts/train_resnet50.py` to train it."
    )

# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Detection Settings")
    conf_threshold = st.slider("Confidence Threshold", 0.05, 1.0, 0.20, 0.05,
                               help="Lower = more detections but noisier results")
    show_all = st.checkbox(
        "Show ALL model predictions (photo upload)", value=True,
        help="Lists every prediction the model makes, even at 1% confidence. "
             "Very low-confidence items are likely wrong — check the confidence tag.")
    camera_index   = st.number_input("Camera Index", min_value=0, max_value=5, value=0, step=1,
                                     help="0 = default webcam. Try 1 or 2 if camera fails.")
    if model is not None:
        st.caption(f"YOLO task: {getattr(model, 'task', '?')} · {len(model.names)} classes")
    resnet_status = "✅ loaded" if resnet_model is not None else "⚠️ not found — run train_resnet50.py"
    st.caption(f"ResNet50: {resnet_status}")
    st.markdown("---")
    st.page_link("pages/dashboard.py", label="Analytics Dashboard", icon="📊")
    st.markdown("---")

    st.markdown("### Schedule I — Waste Classification Index")
    for category, info in WASTE_CATEGORIES.items():
        st.markdown(f"""
        <div class="cat-row">
            <div class="cat-dot" style="background:{info['hex']};"></div>
            <span class="cat-sched">{info['schedule']}</span>
            <span class="cat-name">{info['icon']} {category.replace('_',' ').title()}</span>
            <span class="cat-bin">{info['bin']}</span>
        </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────
if "webcam_running" not in st.session_state:
    st.session_state.webcam_running = False

tab1, tab2 = st.tabs(["Live Camera Detection", "Photo Upload & Analysis"])

# ══════════════════════════════════════════════
# TAB 2 — UPLOAD IMAGE
# ══════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-label">Image Analysis</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload an image to analyse",
        type=['png', 'jpg', 'jpeg', 'bmp', 'gif'],
        help="Supports JPG, PNG, BMP, GIF"
    )

    if uploaded_file is not None:
        # The live-camera loop blocks the script, so stop it when a new photo
        # arrives; otherwise the analysis below would never run.
        _seen = (uploaded_file.name, int(uploaded_file.size))
        if st.session_state.get("seen_upload") != _seen:
            st.session_state.seen_upload = _seen
            st.session_state.webcam_running = False
        image = Image.open(uploaded_file)
        image = ImageOps.exif_transpose(image).convert("RGB")
        col1, col2 = st.columns([1, 1], gap="large")

        with col1:
            st.markdown('<div class="section-label">Input Image</div>', unsafe_allow_html=True)
            st.image(image, width='stretch')

        with col2:
            st.markdown('<div class="section-label">Classification Report</div>', unsafe_allow_html=True)
            if model is not None:
                effective_conf = 0.001 if show_all else conf_threshold
                with st.spinner("Running AI analysis…"):
                    detections = detect_multiscale(image, effective_conf)
                detections = sorted(detections, key=lambda d: -d['confidence'])

                if detections:
                    # Streamlit re-runs this script on every click or slider move.
                    # Save to the database only once per (file, threshold) so the
                    # dashboard numbers are not inflated by repeated reruns.
                    _upload_key = (
                        str(getattr(uploaded_file, "file_id", uploaded_file.name)),
                        int(uploaded_file.size),
                        round(float(effective_conf), 2),
                    )
                    if st.session_state.get("last_saved_upload") != _upload_key:
                        save_detections(
                            detections, source="upload",
                            image_id=f"IMG-{datetime.now():%Y%m%d-%H%M%S}",
                            image_name=uploaded_file.name,
                        )
                        st.session_state.last_saved_upload = _upload_key

                    bio_count    = sum(1 for d in detections if d['category_info'].get('nature') == 'Biodegradable')
                    nonbio_count = sum(1 for d in detections if d['category_info'].get('nature') == 'Non-Biodegradable')
                    haz_count    = sum(1 for d in detections if d['category_info'].get('nature') == 'Hazardous')

                    st.markdown(f"""
                    <div class="status-bar" style="margin-bottom:1rem;">
                        <span>DETECTED: <span class="status-val">{len(detections)}</span></span>
                        <span style="color:#15803D;">BIO: {bio_count}</span>
                        <span style="color:#C2410C;">NON-BIO: {nonbio_count}</span>
                        <span style="color:#B91C1C;">HAZ: {haz_count}</span>
                    </div>""", unsafe_allow_html=True)

                    _counts = Counter(d['class'].replace('_', ' ').title() for d in detections)
                    st.markdown(
                        "**Items found in this image:** " +
                        ", ".join(f"{k} ×{v}" for k, v in _counts.most_common()))

                    for i, det in enumerate(detections):
                        c   = det['category_info']['hex']
                        nat = det['category_info'].get('nature', 'Unknown')
                        nc  = NATURE_COLORS.get(nat, NATURE_COLORS['Unknown'])
                        st.markdown(f"""
                        <div class="det-card" style="border-left-color:{c};">
                            <div class="det-card-top">
                                <div class="det-card-title">{det['category_info']['icon']} {det['class'].replace('_',' ').title()}</div>
                                <div class="det-card-ref">{ref_number(i, prefix="UPL")}</div>
                            </div>
                            <div class="det-card-meta">
                                <span class="conf-pill" style="background:{c}18;color:{c};">{det['confidence']*100:.1f}%</span>
                                <span class="bin-tag">{det['category_info']['bin']}</span>
                                <span class="sched-tag">{det['category_info']['schedule']} · {conf_band(det['confidence'])} confidence</span>
                                <span class="nature-tag" style="background:{nc['bg']};color:{nc['text']};border-color:{nc['border']};">{nat}</span>
                            </div>
                            <div style="font-size:0.75rem;color:#7A8A9C;margin-top:6px;">
                                {det['category_info']['description']}
                            </div>
                        </div>""", unsafe_allow_html=True)

                    st.markdown('<div class="section-label" style="margin-top:1.2rem;">Annotated Output</div>', unsafe_allow_html=True)
                    # results[0].plot() returns a BGR array, so tell Streamlit
                    # (otherwise red and blue colours appear swapped).
                    st.image(draw_detections(image, detections), channels="RGB", width='stretch')
                else:
                    st.markdown("""
                    <div style="background:#FFFFFF; border:1px solid #DCE1E8; border-radius:4px;
                                padding:2rem; text-align:center; color:#5B6B7C;">
                        <div style="font-size:1.7rem;">🔍</div>
                        <div style="font-family:'Fraunces',serif; font-weight:600; margin:0.5rem 0; color:#14213D;">
                            No waste items detected
                        </div>
                        <div style="font-size:0.78rem; font-family:'IBM Plex Mono',monospace; color:#8593A3;">
                            TRY LOWERING THE CONFIDENCE THRESHOLD IN THE SIDEBAR
                        </div>
                    </div>""", unsafe_allow_html=True)
                    with st.expander("Diagnostics — what the model sees (conf ≥ 0.1%)", expanded=True):
                        st.write(f"Model task: **{getattr(model, 'task', '?')}**")
                        st.write("Model classes:", [model.names[i] for i in range(len(model.names))])
                        _raw = debug_predictions(image)
                        if _raw:
                            st.table({"Class": [r[0] for r in _raw], "Confidence": [r[1] for r in _raw]})
                        else:
                            st.write("Model returned **nothing** even at 1% confidence.")
            else:
                st.error("Model not loaded. Please verify `models/best.pt` exists.")
    else:
        st.markdown("""
        <div class="upload-hint">
            <div style="font-size:1.9rem; margin-bottom:0.8rem;">📂</div>
            <div style="font-family:'Fraunces',serif; font-size:0.95rem; color:#14213D; font-weight:600;">
                Drop an image file above to begin analysis
            </div>
            <div style="font-size:0.72rem; margin-top:0.4rem; font-family:'IBM Plex Mono',monospace; letter-spacing:0.06em; color:#A0ACB9;">
                SUPPORTED FORMATS: JPG · PNG · BMP · GIF
            </div>
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════
# TAB 1 — LIVE WEBCAM
# ══════════════════════════════════════════════
with tab1:
    col_vid, col_info = st.columns([3, 1], gap="medium")

    with col_vid:
        st.markdown('<div class="section-label">Camera Feed</div>', unsafe_allow_html=True)
        btn_row = st.columns(2)
        with btn_row[0]:
            start_btn = st.button(
                "▶  Start Detection", type="primary", width='stretch',
                disabled=model is None,
            )
        with btn_row[1]:
            stop_btn = st.button("⏹  Stop", width='stretch')

        frame_placeholder  = st.empty()
        status_placeholder = st.empty()

    with col_info:
        st.markdown('<div class="section-label">Live Detections</div>', unsafe_allow_html=True)
        det_placeholder = st.empty()

        st.markdown("""
        <div class="info-panel">
            <div class="info-panel-title">Disposal Guide</div>
            <div style="font-size:0.78rem; color:#445468; line-height:1.85;">
                🟢 <b style="color:#15803D;">Green Bin</b> — Organic / Glass<br>
                🔵 <b style="color:#1D4ED8;">Blue Bin</b> — Paper / Cardboard / Metal<br>
                🟡 <b style="color:#B45309;">Yellow Bin</b> — Plastic<br>
                🔴 <b style="color:#B91C1C;">Red Bin</b> — Medical / Hazardous<br>
                🟣 <b style="color:#6D28D9;">E-Waste Bin</b> — Electronics
            </div>
        </div>
        """, unsafe_allow_html=True)

    if start_btn:
        st.session_state.webcam_running = True
    if stop_btn:
        st.session_state.webcam_running = False

    if st.session_state.webcam_running:
        cap = open_camera(int(camera_index))

        if not cap.isOpened():
            st.error(f"Could not open camera {int(camera_index)}. Try changing the Camera Index in the sidebar.")
            st.session_state.webcam_running = False
        else:
            try:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_FPS, 30)
                status_placeholder.markdown("""
                <div class="status-bar">
                    <span><span class="status-dot dot-green"></span>CAMERA ACTIVE</span>
                    <span><span class="status-dot dot-amber"></span>MODEL RUNNING</span>
                    <span>STREAM: <span class="status-val">LIVE</span></span>
                </div>""", unsafe_allow_html=True)

                SAVE_EVERY_SECONDS = 2.0
                last_save = 0.0
                frame_count = 0
                consecutive_failures = 0
                MAX_CONSECUTIVE_FAILURES = 20  # ~2s of bad reads before giving up

                while st.session_state.webcam_running:
                    try:
                        ret, frame = cap.read()
                        if not ret or frame is None:
                            consecutive_failures += 1
                            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                                status_placeholder.error(
                                    "Camera stopped responding. Stopping detection — "
                                    "try a different Camera Index in the sidebar."
                                )
                                st.session_state.webcam_running = False
                                break
                            status_placeholder.warning("Frame capture failed — retrying…")
                            time.sleep(0.1)
                            continue

                        consecutive_failures = 0

                        annotated, detections = run_inference_on_frame(frame, conf_threshold)

                        frame_count += 1
                        if detections and (time.time() - last_save) >= SAVE_EVERY_SECONDS:
                            save_detections(detections, source="webcam")
                            last_save = time.time()

                        display = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                        frame_placeholder.image(display, channels="RGB", width='stretch')

                        if detections:
                            det_html = ""
                            for i, det in enumerate(detections):
                                c    = det['category_info']['hex']
                                nat  = det['category_info'].get('nature', 'Unknown')
                                nc   = NATURE_COLORS.get(nat, NATURE_COLORS['Unknown'])
                                det_html += f"""
                                <div class="det-card" style="border-left-color:{c};">
                                    <div class="det-card-top">
                                        <div class="det-card-title">{det['category_info']['icon']} {det['class'].replace('_',' ').title()}</div>
                                        <div class="det-card-ref">{ref_number(frame_count + i)}</div>
                                    </div>
                                    <div class="det-card-meta">
                                        <span class="conf-pill" style="background:{c}18;color:{c};">{det['confidence']*100:.0f}%</span>
                                        <span class="bin-tag">{det['category_info']['bin']}</span>
                                        <span class="nature-tag" style="background:{nc['bg']};color:{nc['text']};border-color:{nc['border']};">{nat}</span>
                                    </div>
                                </div>"""
                            det_placeholder.markdown(det_html, unsafe_allow_html=True)
                        else:
                            det_placeholder.markdown("""
                            <div style="color:#A0ACB9; font-size:0.8rem; font-family:'IBM Plex Mono',monospace;
                                        padding:1.5rem 1rem; text-align:center; letter-spacing:0.06em; background:#FFFFFF;
                                        border:1px solid #DCE1E8; border-radius:4px;">
                                NO OBJECTS DETECTED<br><span style="font-size:0.65rem;">SCANNING…</span>
                            </div>""", unsafe_allow_html=True)

                        time.sleep(0.05)

                    except Exception as loop_err:
                        # Never let a single bad frame or inference error
                        # kill the whole Streamlit script run — surface it
                        # and stop the loop cleanly instead.
                        status_placeholder.error(f"Detection loop stopped due to an error: {loop_err}")
                        st.session_state.webcam_running = False
                        break
            finally:
                # Always release the camera, even if the loop exits via an
                # exception, so the device isn't left locked for other apps.
                cap.release()

            frame_placeholder.empty()
            status_placeholder.markdown("""
            <div class="status-bar">
                <span><span class="status-dot dot-grey"></span>CAMERA OFFLINE</span>
                <span>STREAM: <span style="color:#8593A3;">STOPPED</span></span>
            </div>""", unsafe_allow_html=True)
    else:
        frame_placeholder.markdown("""
        <div style="background:#FFFFFF; border:2px dashed #C6CEDA; border-radius:6px;
                    padding:4rem 2rem; text-align:center; color:#8593A3;">
            <div style="font-size:2.4rem; margin-bottom:1rem;">📷</div>
            <div style="font-family:'Fraunces',serif; font-size:1.05rem; color:#14213D; font-weight:600;">
                Camera feed inactive
            </div>
            <div style="font-size:0.78rem; margin-top:0.4rem; font-family:'IBM Plex Mono',monospace;
                        letter-spacing:0.06em; color:#A0ACB9;">
                CLICK START DETECTION TO BEGIN
            </div>
        </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────
st.markdown("""
<div class="gov-footer-wrap">
    <div class="gov-footer">
        <div class="footer-col">
            <div class="footer-brand">Smart Waste Management System</div>
            <div class="footer-disclaimer">
                This is a demonstration analytics tool. Bin assignments follow the Schedule I waste index
                shown in the sidebar and do not replace local municipal by-laws in force in your jurisdiction.
            </div>
        </div>
        <div class="footer-col">
            <div class="footer-col-title">Portal</div>
            <div class="flink">Detection Console</div>
            <div class="flink">Analytics Dashboard</div>
            <div class="flink">Waste Class Schedule</div>
        </div>
        <div class="footer-col">
            <div class="footer-col-title">Resources</div>
            <div class="flink">Accessibility Statement</div>
            <div class="flink">Terms of Use</div>
            <div class="flink">Right to Information</div>
        </div>
        <div class="footer-col">
            <div class="footer-col-title">Contact</div>
            <div class="flink">Urban Local Body Helpdesk</div>
            <div class="flink">Report a Malfunction</div>
            <div class="flink">Sitemap</div>
        </div>
    </div>
    <div class="gov-footer-bottom">
        Content Owned by Urban Local Body Intelligence Platform &nbsp;·&nbsp; Ministry of Environment, Forest &amp; Climate Change &nbsp;·&nbsp; Government of India
    </div>
</div>
""", unsafe_allow_html=True)
