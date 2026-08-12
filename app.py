import os
import platform
import time
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
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
# Load YOLO model
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(BASE_DIR, "models", "best.pt")
    try:
        return YOLO(MODEL_PATH)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# ─────────────────────────────────────────────
# Waste categories — colours tuned for a light,
# document-style interface (hex = CSS, color = BGR for cv2)
# ─────────────────────────────────────────────
WASTE_CATEGORIES = {
    'plastic':       {'color': (144, 116, 14),  'hex': '#0E7490', 'icon': '♻️',  'bin': 'Yellow Bin',  'schedule': 'Sch. I(a)', 'description': 'Recyclable plastic items',    'nature': 'Non-Biodegradable'},
    'metal':         {'color': (9, 83, 180),    'hex': '#B45309', 'icon': '🔩',  'bin': 'Blue Bin',    'schedule': 'Sch. I(b)', 'description': 'Metal cans and items',         'nature': 'Non-Biodegradable'},
    'glass':         {'color': (110, 118, 15),  'hex': '#0F766E', 'icon': '🍾',  'bin': 'Green Bin',   'schedule': 'Sch. I(c)', 'description': 'Glass bottles and containers', 'nature': 'Non-Biodegradable'},
    'can':           {'color': (14, 64, 146),   'hex': '#92400E', 'icon': '🥫',  'bin': 'Blue Bin',    'schedule': 'Sch. I(d)', 'description': 'Aluminum and tin cans',        'nature': 'Non-Biodegradable'},
    'cable':         {'color': (217, 40, 109),  'hex': '#6D28D9', 'icon': '🔌',  'bin': 'E-Waste Bin', 'schedule': 'Sch. II(a)','description': 'Electrical cables',            'nature': 'Non-Biodegradable'},
    'e_waste':       {'color': (18, 45, 124),   'hex': '#7C2D12', 'icon': '💻',  'bin': 'E-Waste Bin', 'schedule': 'Sch. II(b)','description': 'Electronic waste',             'nature': 'Non-Biodegradable'},
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
def classify_image(image, conf_threshold=0.25):
    results = model(image, conf=conf_threshold, verbose=False)
    detections = []
    for result in results:
        if result.boxes is not None and len(result.boxes) > 0:
            for box in result.boxes:
                class_id   = int(box.cls[0])
                confidence = float(box.conf[0])
                class_name = result.names.get(class_id, str(class_id))
                detections.append({
                    'class':         class_name,
                    'confidence':    confidence,
                    'category_info': WASTE_CATEGORIES.get(class_name, DEFAULT_CATEGORY),
                })
    return results, detections


def run_inference_on_frame(frame_bgr, conf_threshold):
    frame_rgb  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    results    = model(frame_rgb, conf=conf_threshold, verbose=False)
    detections = []

    for result in results:
        if result.boxes is not None and len(result.boxes) > 0:
            for box in result.boxes:
                class_id   = int(box.cls[0])
                confidence = float(box.conf[0])
                class_name = result.names.get(class_id, str(class_id))
                cat        = WASTE_CATEGORIES.get(class_name, DEFAULT_CATEGORY)
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]

                cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), cat['color'], 2)
                label = f"{class_name.replace('_',' ').title()} {confidence*100:.0f}%"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                cv2.rectangle(frame_bgr, (x1, y1 - th - 14), (x1 + tw + 10, y1), cat['color'], -1)
                cv2.putText(frame_bgr, label, (x1 + 5, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

                detections.append({
                    'class': class_name, 'confidence': confidence, 'category_info': cat,
                })

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
        <span class="hero-badge hb-blue">YOLOv8 Object Detection</span>
        <span class="hero-badge hb-amber">10 Notified Waste Classes</span>
        <span class="hero-badge hb-green">Live Camera &amp; Photo Upload</span>
    </div>
</div>
""", unsafe_allow_html=True)

if model is None:
    st.warning("The detection model could not be loaded. Verify that `models/best.pt` exists before using this portal.")

# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Detection Settings")
    conf_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05,
                               help="Lower = more detections but noisier results")
    camera_index   = st.number_input("Camera Index", min_value=0, max_value=5, value=0, step=1,
                                     help="0 = default webcam. Try 1 or 2 if camera fails.")
    st.markdown("---")
    st.page_link("pages/dashboard.py", label="📊 Analytics Dashboard", icon="📊")
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
tab1, tab2 = st.tabs(["Live Camera Detection", "Photo Upload & Analysis"])

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
                🔵 <b style="color:#1D4ED8;">Blue Bin</b> — Paper / Metal<br>
                🟡 <b style="color:#B45309;">Yellow Bin</b> — Plastic<br>
                🔴 <b style="color:#B91C1C;">Red Bin</b> — Medical / Hazardous<br>
                🟣 <b style="color:#6D28D9;">E-Waste Bin</b> — Electronics
            </div>
        </div>
        """, unsafe_allow_html=True)

    if "webcam_running" not in st.session_state:
        st.session_state.webcam_running = False
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

                SAVE_EVERY_N_FRAMES = 30
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
                        if detections and frame_count % SAVE_EVERY_N_FRAMES == 0:
                            save_detections(detections, source="webcam")

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
        image = Image.open(uploaded_file).convert("RGB")
        col1, col2 = st.columns([1, 1], gap="large")

        with col1:
            st.markdown('<div class="section-label">Input Image</div>', unsafe_allow_html=True)
            st.image(image, width='stretch')

        with col2:
            st.markdown('<div class="section-label">Classification Report</div>', unsafe_allow_html=True)
            if model is not None:
                with st.spinner("Running AI analysis…"):
                    results, detections = classify_image(image, conf_threshold)

                if detections:
                    save_detections(detections, source="upload")

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
                                <span class="sched-tag">{det['category_info']['schedule']}</span>
                                <span class="nature-tag" style="background:{nc['bg']};color:{nc['text']};border-color:{nc['border']};">{nat}</span>
                            </div>
                            <div style="font-size:0.75rem;color:#7A8A9C;margin-top:6px;">
                                {det['category_info']['description']}
                            </div>
                        </div>""", unsafe_allow_html=True)

                    st.markdown('<div class="section-label" style="margin-top:1.2rem;">Annotated Output</div>', unsafe_allow_html=True)
                    st.image(results[0].plot(), width='stretch')
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
