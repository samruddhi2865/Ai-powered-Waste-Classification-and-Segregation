import os
import time
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
from database import save_detections

# ─────────────────────────────────────────────
# Page configuration
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="SWMS — Smart Waste Management System",
    page_icon="♻️",
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
# Waste categories
# ─────────────────────────────────────────────
WASTE_CATEGORIES = {
    'plastic':       {'color': (0, 180, 166),  'hex': '#00B4A6', 'icon': '♻️',  'bin': 'Yellow Bin',  'description': 'Recyclable plastic items',      'nature': 'Non-Biodegradable'},
    'metal':         {'color': (245, 166, 35),  'hex': '#F5A623', 'icon': '🔩',  'bin': 'Blue Bin',    'description': 'Metal cans and items',           'nature': 'Non-Biodegradable'},
    'glass':         {'color': (100, 200, 180), 'hex': '#64C8B4', 'icon': '🍾',  'bin': 'Green Bin',   'description': 'Glass bottles and containers',   'nature': 'Non-Biodegradable'},
    'can':           {'color': (200, 140, 80),  'hex': '#C88C50', 'icon': '🥫',  'bin': 'Blue Bin',    'description': 'Aluminum and tin cans',          'nature': 'Non-Biodegradable'},
    'cable':         {'color': (160, 100, 220), 'hex': '#A064DC', 'icon': '🔌',  'bin': 'E-Waste Bin', 'description': 'Electrical cables',             'nature': 'Non-Biodegradable'},
    'e_waste':       {'color': (220, 100, 80),  'hex': '#DC6450', 'icon': '💻',  'bin': 'E-Waste Bin', 'description': 'Electronic waste',              'nature': 'Non-Biodegradable'},
    'medical_waste': {'color': (220, 60, 80),   'hex': '#DC3C50', 'icon': '💉',  'bin': 'Red Bin',     'description': 'Medical and hazardous waste',   'nature': 'Hazardous'},
    'paper':         {'color': (80, 160, 220),  'hex': '#50A0DC', 'icon': '📄',  'bin': 'Blue Bin',    'description': 'Paper and documents',           'nature': 'Biodegradable'},
    'cardboard':     {'color': (160, 120, 80),  'hex': '#A07850', 'icon': '📦',  'bin': 'Blue Bin',    'description': 'Cardboard boxes',               'nature': 'Biodegradable'},
    'organic_waste': {'color': (80, 200, 120),  'hex': '#50C878', 'icon': '🍃',  'bin': 'Green Bin',   'description': 'Biodegradable waste',           'nature': 'Biodegradable'},
}

DEFAULT_CATEGORY = {
    'color': (120, 130, 140), 'hex': '#78828C',
    'icon': '🗑️', 'bin': 'General Bin', 'description': 'Unknown waste type', 'nature': 'Unknown'
}

# Nature badge colours
NATURE_COLORS = {
    'Biodegradable':     {'bg': '#0D3320', 'text': '#50C878', 'border': '#50C878'},
    'Non-Biodegradable': {'bg': '#2A1A08', 'text': '#F5A623', 'border': '#F5A623'},
    'Hazardous':         {'bg': '#2A0A0A', 'text': '#DC3C50', 'border': '#DC3C50'},
    'Unknown':           {'bg': '#1A1E24', 'text': '#78828C', 'border': '#78828C'},
}

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
# Global CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #070D1A !important;
    color: #D0D8E8;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.2rem !important; padding-bottom: 2rem !important; }

/* ── Government header banner ── */
.gov-banner {
    background: linear-gradient(90deg, #040912 0%, #0A1628 40%, #0D2040 70%, #040912 100%);
    border-bottom: 2px solid #00B4A6;
    padding: 0.5rem 2rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 0;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.72rem;
    color: #6E7E9A;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}
.gov-banner span { color: #00B4A6; font-weight: 600; }

/* ── Main hero ── */
.hero-block {
    background: linear-gradient(135deg, #080F1E 0%, #0A1C30 50%, #060E1C 100%);
    border: 1px solid #0F2545;
    border-radius: 16px;
    padding: 2.2rem 2.8rem 1.8rem;
    margin-bottom: 1.6rem;
    position: relative;
    overflow: hidden;
}
.hero-block::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 220px; height: 220px;
    background: radial-gradient(circle, rgba(0,180,166,0.12) 0%, transparent 70%);
    pointer-events: none;
}
.hero-block::after {
    content: '';
    position: absolute;
    bottom: -40px; left: -40px;
    width: 160px; height: 160px;
    background: radial-gradient(circle, rgba(245,166,35,0.07) 0%, transparent 70%);
    pointer-events: none;
}
.hero-eyebrow {
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    color: #00B4A6;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}
.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.4rem;
    font-weight: 700;
    color: #EDF2FA;
    line-height: 1.15;
    margin-bottom: 0.5rem;
}
.hero-title span { color: #00B4A6; }
.hero-sub {
    font-size: 0.92rem;
    color: #6E7E9A;
    max-width: 560px;
    line-height: 1.6;
}
.hero-badge-row {
    display: flex;
    gap: 0.6rem;
    margin-top: 1.2rem;
    flex-wrap: wrap;
}
.hero-badge {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 4px 12px;
    border-radius: 4px;
    border: 1px solid;
}
.hb-teal  { border-color: #00B4A6; color: #00B4A6; background: rgba(0,180,166,0.08); }
.hb-amber { border-color: #F5A623; color: #F5A623; background: rgba(245,166,35,0.08); }
.hb-blue  { border-color: #4080C0; color: #80B0E0; background: rgba(64,128,192,0.08); }

/* ── Status bar (live detection counter) ── */
.status-bar {
    background: #040912;
    border: 1px solid #0F2545;
    border-radius: 10px;
    padding: 0.75rem 1.4rem;
    display: flex;
    align-items: center;
    gap: 2rem;
    margin-bottom: 1.4rem;
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    color: #4A5A6E;
    letter-spacing: 0.05em;
}
.status-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    display: inline-block;
    margin-right: 6px;
}
.dot-green  { background: #00C878; box-shadow: 0 0 6px #00C878; }
.dot-amber  { background: #F5A623; box-shadow: 0 0 6px #F5A623; }
.dot-grey   { background: #3A4A5A; }
.status-val { color: #00B4A6; font-weight: 600; }

/* ── Section header ── */
.section-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    color: #00B4A6;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    margin-bottom: 0.8rem;
    padding-left: 2px;
}

/* ── Detection card ── */
.det-card {
    background: #080F1E;
    border: 1px solid #0F2545;
    border-left: 4px solid;
    border-radius: 8px;
    padding: 0.9rem 1rem;
    margin: 0.5rem 0;
    position: relative;
    transition: border-color 0.2s;
}
.det-card-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.95rem;
    font-weight: 600;
    color: #D0D8E8;
    margin-bottom: 4px;
}
.det-card-meta {
    font-size: 0.78rem;
    color: #5A6A7E;
    display: flex;
    align-items: center;
    gap: 0.8rem;
    flex-wrap: wrap;
}
.conf-pill {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    padding: 2px 8px;
    border-radius: 4px;
    font-weight: 600;
}
.nature-tag {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.08em;
    padding: 2px 8px;
    border-radius: 4px;
    border: 1px solid;
    text-transform: uppercase;
}

/* ── Bin label ── */
.bin-tag {
    background: #0A1628;
    border: 1px solid #1A3050;
    border-radius: 4px;
    padding: 2px 8px;
    font-size: 0.72rem;
    color: #80A0C0;
    font-family: 'DM Mono', monospace;
}

/* ── Info panel ── */
.info-panel {
    background: #040912;
    border: 1px solid #0F2545;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin-bottom: 1rem;
}
.info-panel-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.85rem;
    font-weight: 600;
    color: #8090A8;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.6rem;
    border-bottom: 1px solid #0F2545;
    padding-bottom: 0.4rem;
}

/* ── Category list in sidebar ── */
.cat-row {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    padding: 0.4rem 0;
    border-bottom: 1px solid #0A1628;
    font-size: 0.78rem;
}
.cat-dot { width:10px; height:10px; border-radius:2px; flex-shrink:0; }
.cat-name { color: #B0C0D8; font-weight: 500; flex:1; }
.cat-bin  { font-family: 'DM Mono', monospace; font-size:0.66rem; color:#4A6080; }

/* ── Upload zone ── */
.upload-hint {
    border: 2px dashed #1A3050;
    border-radius: 12px;
    padding: 2.5rem;
    text-align: center;
    color: #3A5070;
    font-size: 0.85rem;
    margin-bottom: 1rem;
}

/* ── Tab overrides ── */
.stTabs [data-baseweb="tab-list"] {
    background: #040912 !important;
    border-radius: 8px !important;
    border: 1px solid #0F2545 !important;
    padding: 4px !important;
    gap: 4px !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    border-radius: 6px !important;
    color: #4A6080 !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.85rem !important;
    padding: 0.5rem 1.2rem !important;
    border: none !important;
    transition: all 0.2s !important;
}
.stTabs [aria-selected="true"] {
    background: #0A1C30 !important;
    color: #00B4A6 !important;
    border: 1px solid #00B4A6 !important;
}

/* ── Buttons ── */
.stButton > button {
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    border-radius: 8px !important;
    transition: all 0.2s !important;
    letter-spacing: 0.04em !important;
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #00B4A6, #008A80) !important;
    color: #FFFFFF !important;
    border: none !important;
    box-shadow: 0 4px 16px rgba(0,180,166,0.3) !important;
}
.stButton > button[kind="primary"]:hover {
    box-shadow: 0 6px 20px rgba(0,180,166,0.45) !important;
    transform: translateY(-1px) !important;
}
.stButton > button[kind="secondary"],
.stButton > button:not([kind="primary"]) {
    background: #080F1E !important;
    color: #80A0C0 !important;
    border: 1px solid #1A3050 !important;
}
.stButton > button:not([kind="primary"]):hover {
    border-color: #00B4A6 !important;
    color: #00B4A6 !important;
}

/* ── Slider ── */
.stSlider > div > div > div > div { background: #00B4A6 !important; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #040912 !important;
    border-right: 1px solid #0A1C2E !important;
}
section[data-testid="stSidebar"] .stMarkdown h3,
section[data-testid="stSidebar"] .stMarkdown h2 {
    font-family: 'Space Grotesk', sans-serif !important;
    color: #6E7E9A !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    font-weight: 600 !important;
}

/* ── Selectbox / number input ── */
.stNumberInput > div > div > input {
    background: #080F1E !important;
    border: 1px solid #1A3050 !important;
    color: #D0D8E8 !important;
    border-radius: 8px !important;
    font-family: 'DM Mono', monospace !important;
}

/* ── Alerts/info ── */
.stAlert { border-radius: 8px !important; }

/* ── Footer ── */
.gov-footer {
    text-align: center;
    font-family: 'DM Mono', monospace;
    font-size: 0.66rem;
    color: #2A3A4A;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 1.5rem 0 0.5rem;
    border-top: 1px solid #0A1628;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Hero
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero-block">
    <div class="hero-eyebrow">⬡ Smart Waste Management System &nbsp;/&nbsp; Real-Time Detection Module</div>
    <div class="hero-title">Classify. Sort. <span>Recover.</span></div>
    <div class="hero-sub">
        AI-powered waste classification for municipal solid waste streams.
        Identify, categorise, and route waste to the correct disposal facility — in real time.
    </div>
    <div class="hero-badge-row">
        <span class="hero-badge hb-teal">YOLOv8 Detection</span>
        <span class="hero-badge hb-amber">10 Waste Classes</span>
        <span class="hero-badge hb-blue">Live Webcam + Image Upload</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Detection Settings")
    conf_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05,
                               help="Lower = more detections but noisier results")
    camera_index   = st.number_input("Camera Index", min_value=0, max_value=5, value=0, step=1,
                                     help="0 = default webcam. Try 1 or 2 if camera fails.")
    st.markdown("---")
    st.page_link("pages/dashboard.py", label="📊 Analytics Dashboard", icon="📊")
    st.markdown("---")

    st.markdown("### 🗂 Waste Class Index")
    for category, info in WASTE_CATEGORIES.items():
        nc = NATURE_COLORS.get(info['nature'], NATURE_COLORS['Unknown'])
        st.markdown(f"""
        <div class="cat-row">
            <div class="cat-dot" style="background:{info['hex']};"></div>
            <span class="cat-name">{info['icon']} {category.replace('_',' ').title()}</span>
            <span class="cat-bin">{info['bin']}</span>
        </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────
tab1, tab2 = st.tabs(["🎥  Live Webcam Detection", "📁  Image Upload & Analysis"])

# ══════════════════════════════════════════════
# TAB 1 — LIVE WEBCAM
# ══════════════════════════════════════════════
with tab1:
    col_vid, col_info = st.columns([3, 1], gap="medium")

    with col_vid:
        st.markdown('<div class="section-label">▸ Camera Feed</div>', unsafe_allow_html=True)
        btn_row = st.columns(2)
        with btn_row[0]:
            start_btn = st.button("▶  Start Detection", type="primary", use_container_width=True)
        with btn_row[1]:
            stop_btn  = st.button("⏹  Stop", use_container_width=True)

        frame_placeholder  = st.empty()
        status_placeholder = st.empty()

    with col_info:
        st.markdown('<div class="section-label">▸ Live Detections</div>', unsafe_allow_html=True)
        det_placeholder = st.empty()

        st.markdown("""
        <div class="info-panel">
            <div class="info-panel-title">Disposal Guide</div>
            <div style="font-size:0.75rem; color:#4A6080; line-height:1.8;">
                🟢 <b style="color:#50C878;">Green Bin</b> — Organic / Glass<br>
                🔵 <b style="color:#50A0DC;">Blue Bin</b> — Paper / Metal<br>
                🟡 <b style="color:#F5A623;">Yellow Bin</b> — Plastic<br>
                🔴 <b style="color:#DC3C50;">Red Bin</b> — Medical / Hazardous<br>
                ⚫ <b style="color:#A064DC;">E-Waste Bin</b> — Electronics
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
        cap = cv2.VideoCapture(int(camera_index), cv2.CAP_DSHOW)

        if not cap.isOpened():
            st.error(f"❌ Could not open camera {int(camera_index)}. Try changing the Camera Index in the sidebar.")
            st.session_state.webcam_running = False
        else:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)
            status_placeholder.markdown("""
            <div class="status-bar">
                <span><span class="status-dot dot-green"></span>CAMERA ACTIVE</span>
                <span><span class="status-dot dot-amber"></span>AI MODEL RUNNING</span>
                <span>STREAM: <span class="status-val">LIVE</span></span>
            </div>""", unsafe_allow_html=True)

            SAVE_EVERY_N_FRAMES = 30
            frame_count = 0

            while st.session_state.webcam_running:
                ret, frame = cap.read()
                if not ret:
                    status_placeholder.warning("⚠️ Frame capture failed — retrying…")
                    time.sleep(0.1)
                    continue

                if model is not None:
                    annotated, detections = run_inference_on_frame(frame, conf_threshold)
                else:
                    annotated, detections = frame, []

                frame_count += 1
                if detections and frame_count % SAVE_EVERY_N_FRAMES == 0:
                    save_detections(detections, source="webcam")

                display = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(display, channels="RGB", use_container_width=True)

                if detections:
                    det_html = ""
                    for det in detections:
                        c    = det['category_info']['hex']
                        nat  = det['category_info'].get('nature', 'Unknown')
                        nc   = NATURE_COLORS.get(nat, NATURE_COLORS['Unknown'])
                        det_html += f"""
                        <div class="det-card" style="border-left-color:{c};">
                            <div class="det-card-title">{det['category_info']['icon']} {det['class'].replace('_',' ').title()}</div>
                            <div class="det-card-meta">
                                <span class="conf-pill" style="background:{c}20;color:{c};">{det['confidence']*100:.0f}%</span>
                                <span class="bin-tag">🗑 {det['category_info']['bin']}</span>
                                <span class="nature-tag" style="background:{nc['bg']};color:{nc['text']};border-color:{nc['border']};">{nat}</span>
                            </div>
                        </div>"""
                    det_placeholder.markdown(det_html, unsafe_allow_html=True)
                else:
                    det_placeholder.markdown("""
                    <div style="color:#2A3A4A; font-size:0.8rem; font-family:'DM Mono',monospace;
                                padding:1rem; text-align:center; letter-spacing:0.08em;">
                        NO OBJECTS DETECTED<br><span style="font-size:0.65rem;">SCANNING...</span>
                    </div>""", unsafe_allow_html=True)

                time.sleep(0.05)

            cap.release()
            frame_placeholder.empty()
            status_placeholder.markdown("""
            <div class="status-bar">
                <span><span class="status-dot dot-grey"></span>CAMERA OFFLINE</span>
                <span>STREAM: <span style="color:#4A5A6E;">STOPPED</span></span>
            </div>""", unsafe_allow_html=True)
    else:
        frame_placeholder.markdown("""
        <div style="background:#040912; border:2px dashed #0F2545; border-radius:12px;
                    padding:4rem 2rem; text-align:center; color:#2A3A4A;">
            <div style="font-size:2.5rem; margin-bottom:1rem;">📷</div>
            <div style="font-family:'Space Grotesk',sans-serif; font-size:1rem; color:#3A5070; font-weight:600;">
                Camera feed inactive
            </div>
            <div style="font-size:0.78rem; margin-top:0.4rem; font-family:'DM Mono',monospace;
                        letter-spacing:0.08em; color:#1E2E3E;">
                CLICK START DETECTION TO BEGIN
            </div>
        </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════
# TAB 2 — UPLOAD IMAGE
# ══════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-label">▸ Image Analysis</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload an image to analyse",
        type=['png', 'jpg', 'jpeg', 'bmp', 'gif'],
        help="Supports JPG, PNG, BMP, GIF"
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        col1, col2 = st.columns([1, 1], gap="large")

        with col1:
            st.markdown('<div class="section-label">▸ Input Image</div>', unsafe_allow_html=True)
            st.image(image, use_container_width=True)

        with col2:
            st.markdown('<div class="section-label">▸ Classification Report</div>', unsafe_allow_html=True)
            if model is not None:
                with st.spinner("Running AI analysis…"):
                    results, detections = classify_image(image, conf_threshold)

                if detections:
                    save_detections(detections, source="upload")

                    # Summary strip
                    bio_count    = sum(1 for d in detections if d['category_info'].get('nature') == 'Biodegradable')
                    nonbio_count = sum(1 for d in detections if d['category_info'].get('nature') == 'Non-Biodegradable')
                    haz_count    = sum(1 for d in detections if d['category_info'].get('nature') == 'Hazardous')

                    st.markdown(f"""
                    <div class="status-bar" style="margin-bottom:1rem;">
                        <span>DETECTED: <span class="status-val">{len(detections)}</span></span>
                        <span style="color:#50C878;">BIO: {bio_count}</span>
                        <span style="color:#F5A623;">NON-BIO: {nonbio_count}</span>
                        <span style="color:#DC3C50;">HAZ: {haz_count}</span>
                    </div>""", unsafe_allow_html=True)

                    for det in detections:
                        c   = det['category_info']['hex']
                        nat = det['category_info'].get('nature', 'Unknown')
                        nc  = NATURE_COLORS.get(nat, NATURE_COLORS['Unknown'])
                        st.markdown(f"""
                        <div class="det-card" style="border-left-color:{c};">
                            <div class="det-card-title">{det['category_info']['icon']} {det['class'].replace('_',' ').title()}</div>
                            <div class="det-card-meta">
                                <span class="conf-pill" style="background:{c}20;color:{c};">{det['confidence']*100:.1f}%</span>
                                <span class="bin-tag">🗑 {det['category_info']['bin']}</span>
                                <span class="nature-tag" style="background:{nc['bg']};color:{nc['text']};border-color:{nc['border']};">{nat}</span>
                            </div>
                            <div style="font-size:0.72rem;color:#3A4A5A;margin-top:5px;">
                                {det['category_info']['description']}
                            </div>
                        </div>""", unsafe_allow_html=True)

                    st.markdown('<div class="section-label" style="margin-top:1.2rem;">▸ Annotated Output</div>', unsafe_allow_html=True)
                    st.image(results[0].plot(), use_container_width=True)
                else:
                    st.markdown("""
                    <div style="background:#040912; border:1px solid #0F2545; border-radius:10px;
                                padding:2rem; text-align:center; color:#3A5070;">
                        <div style="font-size:1.8rem;">🔍</div>
                        <div style="font-family:'Space Grotesk',sans-serif; font-weight:600; margin:0.5rem 0;">
                            No waste items detected
                        </div>
                        <div style="font-size:0.78rem; font-family:'DM Mono',monospace;">
                            TRY LOWERING THE CONFIDENCE THRESHOLD IN THE SIDEBAR
                        </div>
                    </div>""", unsafe_allow_html=True)
            else:
                st.error("❌ Model not loaded. Please verify `models/best.pt` exists.")
    else:
        st.markdown("""
        <div class="upload-hint">
            <div style="font-size:2rem; margin-bottom:0.8rem;">📂</div>
            <div style="font-family:'Space Grotesk',sans-serif; font-size:0.9rem; color:#2A4060; font-weight:600;">
                Drop an image file above to begin analysis
            </div>
            <div style="font-size:0.72rem; margin-top:0.4rem; font-family:'DM Mono',monospace; letter-spacing:0.08em; color:#1A2A3A;">
                SUPPORTED FORMATS: JPG · PNG · BMP · GIF
            </div>
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────
st.markdown("""
<div class="gov-footer">
    Smart Waste Management System &nbsp;·&nbsp; Urban Local Body Intelligence Platform
    &nbsp;·&nbsp; Ministry of Environment, Forest &amp; Climate Change &nbsp;·&nbsp; Government of India
</div>
""", unsafe_allow_html=True)
