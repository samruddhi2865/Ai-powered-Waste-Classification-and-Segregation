import os
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
import tempfile

# Page configuration
st.set_page_config(
    page_title="Waste Classification System",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load model
@st.cache_resource
def load_model():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(BASE_DIR, "models", "best.pt")
    try:
        model = YOLO(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# Waste categories with colors and descriptions
WASTE_CATEGORIES = {
    'plastic': {'color': '#FF6B6B', 'icon': '♻️', 'bin': 'Yellow Bin', 'description': 'Recyclable plastic items'},
    'metal': {'color': '#4ECDC4', 'icon': '🔩', 'bin': 'Blue Bin', 'description': 'Metal cans and items'},
    'glass': {'color': '#45B7D1', 'icon': '🍾', 'bin': 'Green Bin', 'description': 'Glass bottles and containers'},
    'can': {'color': '#96CEB4', 'icon': '🥫', 'bin': 'Blue Bin', 'description': 'Aluminum and tin cans'},
    'cable': {'color': '#FFEAA7', 'icon': '🔌', 'bin': 'E-Waste Bin', 'description': 'Electrical cables'},
    'e_waste': {'color': '#DFE6E9', 'icon': '💻', 'bin': 'E-Waste Bin', 'description': 'Electronic waste'},
    'medical_waste': {'color': '#FF7675', 'icon': '💉', 'bin': 'Red Bin', 'description': 'Medical and hazardous waste'},
    'paper': {'color': '#74B9FF', 'icon': '📄', 'bin': 'Blue Bin', 'description': 'Paper and documents'},
    'cardboard': {'color': '#A29BFE', 'icon': '📦', 'bin': 'Blue Bin', 'description': 'Cardboard boxes'},
    'organic_waste': {'color': '#55EFC4', 'icon': '🍃', 'bin': 'Green Bin', 'description': 'Biodegradable waste'}
}

def classify_image(image, conf_threshold=0.25):
    """Classify waste in an image"""
    results = model(image, conf=conf_threshold)
    detections = []
    
    for result in results:
        if result.boxes is not None and len(result.boxes) > 0:
            names = result.names
            for box in result.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                class_name = names.get(class_id, str(class_id))
                
                category_info = WASTE_CATEGORIES.get(class_name, {
                    'color': '#95A5A6',
                    'icon': '🗑️',
                    'bin': 'General Bin',
                    'description': 'Unknown waste type'
                })
                
                detections.append({
                    'class': class_name,
                    'confidence': confidence,
                    'category_info': category_info
                })
    
    return results, detections

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #2E7D32;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .detection-card {
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 5px solid;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">♻️ Waste Classification System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI-Powered Waste Detection and Sorting</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    conf_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05)
    
    st.markdown("---")
    st.header("📊 Waste Categories")
    for category, info in WASTE_CATEGORIES.items():
        st.markdown(f"{info['icon']} **{category.replace('_', ' ').title()}**")
        st.caption(f"{info['bin']} - {info['description']}")

# Main content - Tabs
tab1, tab2 = st.tabs(["📷 Live Camera", "📁 Upload Image"])

# Tab 1: Live Camera
with tab1:
    st.header("Live Camera Classification")
    st.info("📸 Capture an image from your camera to classify waste in real-time")
    
    camera_image = st.camera_input("Take a picture")
    
    if camera_image is not None:
        image = Image.open(camera_image)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Captured Image")
            st.image(image, use_column_width=True)
        
        with col2:
            st.subheader("Classification Results")
            
            if model is not None:
                with st.spinner("Analyzing image..."):
                    results, detections = classify_image(image, conf_threshold)
                    
                    if detections:
                        st.success(f"✅ Detected {len(detections)} item(s)")
                        
                        for i, det in enumerate(detections, 1):
                            color = det['category_info']['color']
                            st.markdown(f"""
                                <div class="detection-card" style="border-left-color: {color};">
                                    <h4>{det['category_info']['icon']} {det['class'].replace('_', ' ').title()}</h4>
                                    <p><strong>Confidence:</strong> {det['confidence']*100:.2f}%</p>
                                    <p><strong>Bin:</strong> {det['category_info']['bin']}</p>
                                    <p>{det['category_info']['description']}</p>
                                </div>
                            """, unsafe_allow_html=True)
                        
                        # Display annotated image
                        st.subheader("Annotated Image")
                        annotated_img = results[0].plot()
                        st.image(annotated_img, use_column_width=True)
                    else:
                        st.warning("⚠️ No waste items detected. Try adjusting the confidence threshold.")
            else:
                st.error("❌ Model not loaded. Please check the model file.")

# Tab 2: Upload Image
with tab2:
    st.header("Upload Image Classification")
    st.info("📤 Upload an image to classify waste items")
    
    uploaded_file = st.file_uploader("Choose an image...", type=['png', 'jpg', 'jpeg', 'bmp', 'gif'])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Uploaded Image")
            st.image(image, use_column_width=True)
        
        with col2:
            st.subheader("Classification Results")
            
            if model is not None:
                with st.spinner("Analyzing image..."):
                    results, detections = classify_image(image, conf_threshold)
                    
                    if detections:
                        st.success(f"✅ Detected {len(detections)} item(s)")
                        
                        for i, det in enumerate(detections, 1):
                            color = det['category_info']['color']
                            st.markdown(f"""
                                <div class="detection-card" style="border-left-color: {color};">
                                    <h4>{det['category_info']['icon']} {det['class'].replace('_', ' ').title()}</h4>
                                    <p><strong>Confidence:</strong> {det['confidence']*100:.2f}%</p>
                                    <p><strong>Bin:</strong> {det['category_info']['bin']}</p>
                                    <p>{det['category_info']['description']}</p>
                                </div>
                            """, unsafe_allow_html=True)
                        
                        # Display annotated image
                        st.subheader("Annotated Image")
                        annotated_img = results[0].plot()
                        st.image(annotated_img, use_column_width=True)
                    else:
                        st.warning("⚠️ No waste items detected. Try adjusting the confidence threshold.")
            else:
                st.error("❌ Model not loaded. Please check the model file.")

# Footer
st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: #666;">
        <p>🌍 Help save the planet by sorting waste correctly!</p>
    </div>
""", unsafe_allow_html=True)
