import os
import time
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
from database import save_detections   # ← NEW

# ─────────────────────────────────────────────
# Page configuration
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Waste Classification System",
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
    'plastic':       {'color': (107, 107, 255), 'hex': '#FF6B6B', 'icon': '♻️',  'bin': 'Yellow Bin',  'description': 'Recyclable plastic items'},
    'metal':         {'color': (196, 205, 78),  'hex': '#4ECDC4', 'icon': '🔩',  'bin': 'Blue Bin',    'description': 'Metal cans and items'},
    'glass':         {'color': (209, 183, 69),  'hex': '#45B7D1', 'icon': '🍾',  'bin': 'Green Bin',   'description': 'Glass bottles and containers'},
    'can':           {'color': (180, 206, 150), 'hex': '#96CEB4', 'icon': '🥫',  'bin': 'Blue Bin',    'description': 'Aluminum and tin cans'},
    'cable':         {'color': (167, 234, 255), 'hex': '#FFEAA7', 'icon': '🔌',  'bin': 'E-Waste Bin', 'description': 'Electrical cables'},
    'e_waste':       {'color': (233, 230, 223), 'hex': '#DFE6E9', 'icon': '💻',  'bin': 'E-Waste Bin', 'description': 'Electronic waste'},
    'medical_waste': {'color': (117, 118, 255), 'hex': '#FF7675', 'icon': '💉',  'bin': 'Red Bin',     'description': 'Medical and hazardous waste'},
    'paper':         {'color': (255, 185, 116), 'hex': '#74B9FF', 'icon': '📄',  'bin': 'Blue Bin',    'description': 'Paper and documents'},
    'cardboard':     {'color': (254, 155, 162), 'hex': '#A29BFE', 'icon': '📦',  'bin': 'Blue Bin',    'description': 'Cardboard boxes'},
    'organic_waste': {'color': (196, 239, 85),  'hex': '#55EFC4', 'icon': '🍃',  'bin': 'Green Bin',   'description': 'Biodegradable waste'},
}

DEFAULT_CATEGORY = {
    'color': (166, 165, 149), 'hex': '#95A5A6',
    'icon': '🗑️', 'bin': 'General Bin', 'description': 'Unknown waste type'
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
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame_bgr, (x1, y1 - th - 12), (x1 + tw + 8, y1), cat['color'], -1)
                cv2.putText(frame_bgr, label, (x1 + 4, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

                detections.append({
                    'class': class_name, 'confidence': confidence, 'category_info': cat,
                })

    return frame_bgr, detections

# ─────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=Inter:wght@400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.main-header {
    font-family: 'Syne', sans-serif;
    font-size: 2.8rem; font-weight: 800; text-align: center;
    background: linear-gradient(135deg, #2E7D32, #66BB6A);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 0.3rem;
}
.sub-header { font-size:1.05rem; text-align:center; color:#888; margin-bottom:1.8rem; letter-spacing:0.04em; }
.detection-card { padding:0.8rem 1rem; border-radius:10px; margin:0.4rem 0; border-left:5px solid; background:rgba(255,255,255,0.03); }
.badge { display:inline-block; padding:1px 9px; border-radius:99px; font-size:0.72rem; font-weight:600; color:#fff; margin-left:6px; vertical-align:middle; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">♻️ Waste Classification System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI-Powered Waste Detection & Sorting — Real-Time Webcam + Image Upload</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    conf_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05)
    camera_index   = st.number_input("Camera Index (0 = default webcam)", min_value=0, max_value=5, value=0, step=1)
    st.markdown("---")

    # ← NEW: link to dashboard page
    st.page_link("pages/dashboard.py", label="📊 Open Dashboard", icon="📊")
    st.markdown("---")

    st.header("📊 Waste Categories")
    for category, info in WASTE_CATEGORIES.items():
        st.markdown(f"{info['icon']} **{category.replace('_', ' ').title()}**")
        st.caption(f"{info['bin']} — {info['description']}")

# ─────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────
tab1, tab2 = st.tabs(["🎥 Live Webcam", "📁 Upload Image"])

# ══════════════════════════════════════════════
# TAB 1 — LIVE WEBCAM
# ══════════════════════════════════════════════
with tab1:
    st.header("Live Webcam Classification")

    col_vid, col_info = st.columns([3, 1])

    with col_vid:
        start_btn = st.button("▶ Start Webcam", type="primary", use_container_width=True)
        stop_btn  = st.button("⏹ Stop Webcam",  use_container_width=True)
        frame_placeholder  = st.empty()
        status_placeholder = st.empty()

    with col_info:
        st.markdown("### 📋 Detections")
        det_placeholder = st.empty()

    if "webcam_running" not in st.session_state:
        st.session_state.webcam_running = False

    if start_btn:
        st.session_state.webcam_running = True
    if stop_btn:
        st.session_state.webcam_running = False

    if st.session_state.webcam_running:
        cap = cv2.VideoCapture(int(camera_index), cv2.CAP_DSHOW)

        if not cap.isOpened():
            st.error(
                f"❌ Could not open camera {int(camera_index)}. "
                "Try changing the Camera Index in the sidebar (try 1 or 2)."
            )
            st.session_state.webcam_running = False
        else:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)
            status_placeholder.success("🟢 Webcam active — click **Stop Webcam** to end")

            # ── Save every N frames to avoid DB spam ──────────────────────
            SAVE_EVERY_N_FRAMES = 30          # save ~1×/sec at 30 fps
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

                # ← NEW: periodic DB save
                frame_count += 1
                if detections and frame_count % SAVE_EVERY_N_FRAMES == 0:
                    save_detections(detections, source="webcam")

                display = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(display, channels="RGB", use_container_width=True)

                if detections:
                    det_html = ""
                    for det in detections:
                        c = det['category_info']['hex']
                        det_html += f"""
                        <div class="detection-card" style="border-left-color:{c};">
                            <strong>{det['category_info']['icon']} {det['class'].replace('_',' ').title()}</strong>
                            <span class="badge" style="background:{c};">{det['confidence']*100:.0f}%</span><br>
                            <small>🗑 {det['category_info']['bin']}</small>
                        </div>"""
                    det_placeholder.markdown(det_html, unsafe_allow_html=True)
                else:
                    det_placeholder.caption("No waste detected.")

                time.sleep(0.05)

            cap.release()
            frame_placeholder.empty()
            status_placeholder.info("⏹ Webcam stopped.")
    else:
        frame_placeholder.info("👆 Click **Start Webcam** to begin live detection.")

# ══════════════════════════════════════════════
# TAB 2 — UPLOAD IMAGE
# ══════════════════════════════════════════════
with tab2:
    st.header("Upload Image Classification")
    st.info("📤 Upload a JPG / PNG / BMP image to classify waste items.")

    uploaded_file = st.file_uploader("Choose an image…", type=['png', 'jpg', 'jpeg', 'bmp', 'gif'])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Uploaded Image")
            st.image(image, use_container_width=True)

        with col2:
            st.subheader("Classification Results")
            if model is not None:
                with st.spinner("Analysing image…"):
                    results, detections = classify_image(image, conf_threshold)

                if detections:
                    # ← NEW: save to database
                    save_detections(detections, source="upload")
                    st.success(f"✅ Detected **{len(detections)}** item(s) — saved to database")

                    for det in detections:
                        color = det['category_info']['hex']
                        st.markdown(f"""
                            <div class="detection-card" style="border-left-color:{color};">
                                <h4 style="margin:0;">{det['category_info']['icon']}
                                {det['class'].replace('_',' ').title()}
                                <span class="badge" style="background:{color};">{det['confidence']*100:.1f}%</span></h4>
                                <p style="margin:4px 0 0;">🗑 <strong>{det['category_info']['bin']}</strong><br>
                                <small>{det['category_info']['description']}</small></p>
                            </div>
                        """, unsafe_allow_html=True)
                    st.subheader("Annotated Image")
                    st.image(results[0].plot(), use_container_width=True)
                else:
                    st.warning("⚠️ No waste items detected. Try lowering the confidence threshold.")
            else:
                st.error("❌ Model not loaded. Please check `models/best.pt`.")

st.markdown("---")
st.markdown('<div style="text-align:center;color:#888;font-size:0.9rem;">🌍 Help save the planet by sorting waste correctly!</div>', unsafe_allow_html=True)
