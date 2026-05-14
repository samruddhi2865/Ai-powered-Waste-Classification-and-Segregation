# 🗑️ Waste Classification System

An AI-powered Streamlit application that uses YOLOv8 deep learning model to automatically classify and segregate waste items into proper disposal categories.

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32-red)
![YOLO](https://img.shields.io/badge/YOLO-v8-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 🌟 Features

- **🎥 Live Camera Classification**: Capture images from your camera in real-time
- **📁 Image Upload Classification**: Upload images for waste detection
- **AI-Powered Detection**: Uses YOLOv8 for accurate waste detection and classification
- **10 Waste Categories**: Plastic, Metal, Glass, Can, Cable, E-waste, Medical waste, Paper, Cardboard, Organic waste
- **Real-time Processing**: Fast inference with adjustable confidence threshold
- **Beautiful UI**: Modern, responsive Streamlit interface with color-coded results
- **Disposal Guidance**: Provides proper bin recommendations for each waste type
- **Annotated Images**: Visual bounding boxes showing detected items

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Webcam (for live camera mode)
- Modern web browser

### Installation

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd Waste-Classification-Segregation-Web-Application-
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Verify model file**
Ensure `models/best.pt` exists in the project directory

### Running the Application

**Option 1: Using the batch file (Windows)**
```bash
run.bat
```

**Option 2: Using Python directly**
```bash
python -m streamlit run app.py
```

**Option 3: If streamlit is in PATH**
```bash
streamlit run app.py
```

The app will automatically open in your browser at `http://localhost:8501`
streamlit run app.py
```

The app will automatically open in your browser at `http://localhost:8501`

---

## 🎯 How to Use

### 📷 Live Camera Mode
1. Click on the **"Live Camera"** tab
2. Allow camera permissions when prompted by your browser
3. Click **"Take a picture"** to capture an image
4. View instant classification results with:
   - Waste category and icon
   - Confidence score
   - Proper disposal bin
   - Category description
   - Annotated image with bounding boxes

### 📁 Upload Image Mode
1. Click on the **"Upload Image"** tab
2. Click **"Browse files"** or drag and drop an image
3. Supported formats: PNG, JPG, JPEG, BMP, GIF
4. View classification results with annotated bounding boxes

### ⚙️ Adjust Settings
- Use the **confidence threshold slider** in the sidebar (default: 0.25)
- Lower values: More detections, potentially more false positives
- Higher values: Fewer detections, higher confidence required

---

## 🗑️ Waste Categories

The system can detect and classify 10 different types of waste:

| Category | Icon | Bin Type | Description |
|----------|------|----------|-------------|
| **Plastic** | ♻️ | Yellow Bin | Recyclable plastic items |
| **Metal** | 🔩 | Blue Bin | Metal cans and items |
| **Glass** | 🍾 | Green Bin | Glass bottles and containers |
| **Can** | 🥫 | Blue Bin | Aluminum and tin cans |
| **Cable** | 🔌 | E-Waste Bin | Electrical cables |
| **E-Waste** | 💻 | E-Waste Bin | Electronic waste |
| **Medical Waste** | 💉 | Red Bin | Medical and hazardous waste |
| **Paper** | 📄 | Blue Bin | Paper and documents |
| **Cardboard** | 📦 | Blue Bin | Cardboard boxes |
| **Organic Waste** | 🍃 | Green Bin | Biodegradable waste |

---

## 🧠 Model Information

- **Architecture**: YOLOv8 (Ultralytics)
- **Model File**: `models/best.pt`
- **Classes**: 10 waste categories
- **Input**: RGB images (any size, auto-resized)
- **Output**: Bounding boxes with class labels and confidence scores
- **Default Confidence Threshold**: 0.25 (adjustable)

---

## 📁 Project Structure

```
Waste-Classification-System/
│
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── runtime.txt                 # Python version
├── .gitignore                  # Git ignore rules
│
├── models/
│   └── best.pt                 # YOLOv8 trained model
│
├── config/
│   └── data.yaml               # Dataset configuration
│
├── .streamlit/
│   └── config.toml             # Streamlit theme settings
│
└── scripts/
    ├── convert_coco_to_yolo.py
    ├── convert_trashnet_to_yolo.py
    └── datasets/                # Dataset files
```

---

## 🛠️ Technologies Used

- **Streamlit**: Web application framework
- **Ultralytics YOLO**: Object detection model
- **PyTorch**: Deep learning framework
- **Pillow**: Image processing
- **OpenCV**: Computer vision operations
- **NumPy**: Numerical computing

---

## 🐛 Troubleshooting

### Camera Not Working
- Ensure your browser has camera permissions enabled
- Try using HTTPS or localhost
- Check if another application is using the camera
- Use Chrome, Firefox, or Edge for best compatibility

### Model Not Loading
- Verify `models/best.pt` exists in the correct location
- Check file permissions
- Ensure sufficient disk space
- Verify all dependencies are installed

### Slow Performance
- Reduce image resolution before uploading
- Increase confidence threshold to reduce detections
- Close other resource-intensive applications
- Consider using GPU if available (PyTorch with CUDA)

### No Detections Found
- Lower the confidence threshold in the sidebar
- Ensure good lighting in the image
- Try images with clearer waste items
- Check that the waste type is one of the 10 supported categories

---

## ⚙️ Configuration

### Confidence Threshold
Adjust in the sidebar (range: 0.0 - 1.0, default: 0.25)

### Streamlit Theme
Edit `.streamlit/config.toml` to customize colors and appearance

### Model Path
The model is loaded from `models/best.pt`. To use a different model:
1. Replace `models/best.pt` with your trained model
2. Ensure the model has the same 10 classes defined in `config/data.yaml`

## 🔮 Future Enhancements

- [ ] Real-time video stream classification
- [ ] Batch image processing
- [ ] Export results to CSV/PDF
- [ ] Multi-language support
- [ ] Statistics and analytics dashboard
- [ ] Mobile app version
- [ ] API integration for third-party apps

## 🌟 Features

- **AI-Powered Classification**: Uses YOLOv8 for accurate waste detection and classification
- **10 Waste Categories**: Plastic, Metal, Glass, Can, Cable, E-waste, Medical waste, Paper, Cardboard, Organic waste
- **Real-time Processing**: Fast inference with confidence scores
- **Beautiful UI**: Modern, responsive design with smooth animations
- **Drag & Drop**: Easy image upload interface
- **Disposal Guidance**: Provides proper bin recommendations for each waste type
- **Mobile Responsive**: Works seamlessly on all devices

## 📁 Project Structure

```
Waste-Classification-Segregation-Web-Application-/
│
├── app.py                          # Main Flask application
├── requirements.txt                # Python dependencies
├── Procfile                        # Deployment configuration
├── runtime.txt                     # Python version
├── .gitignore                      # Git ignore rules
│
├── models/                         # Model files
│   ├── best.pt                     # YOLO trained model
│   └── model.pkl                   # Pickle model (optional)
│
├── config/                         # Configuration files
│   └── data.yaml                   # Dataset configuration
│
├── static/                         # Static assets
│   ├── css/
│   │   └── style.css              # Stylesheet
│   ├── js/
│   │   └── script.js              # JavaScript
│   └── images/                     # Static images
│
├── templates/                      # HTML templates
│   └── index.html                 # Main page
│
├── uploads/                        # Temporary uploads (auto-created)
├── results/                        # Results storage (auto-created)
│
└── scripts/                        # Utility scripts
    ├── datasets/                   # Dataset processing
    └── *.py                        # Various utility scripts
```

## 🚀 Quick Start

### Prerequisites

- Python 3.11 or higher
- pip (Python package manager)
- Git

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/waste-classifier.git
cd Waste-Classification-Segregation-Web-Application-
```

2. **Create virtual environment**
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Run the application**
```bash
python app.py
```

5. **Open in browser**
```
http://localhost:5000
```

## 🎯 Usage

1. **Upload Image**: Click or drag & drop an image of waste items
2. **Wait for Analysis**: AI processes the image (typically < 2 seconds)
3. **View Results**: See detected items with:
   - Waste category
   - Confidence score
   - Proper disposal bin
   - Item description

## 🧠 Model Information

- **Architecture**: YOLOv8 (Ultralytics)
- **Training Dataset**: Custom waste classification dataset from roboflow (merged COCO + TrashNet)
- **Classes**: 10 waste categories
- **Input**: RGB images (any size, auto-resized)
- **Output**: Bounding boxes with class labels and confidence scores
- **Confidence Threshold**: 0.25 (25%)

### Waste Categories

| Category | Bin Type | Icon |
|----------|----------|------|
| Plastic | Yellow Bin | ♻️ |
| Metal | Blue Bin | 🔩 |
| Glass | Green Bin | 🍾 |
| Can | Blue Bin | 🥫 |
| Cable | E-Waste Bin | 🔌 |
| E-waste | E-Waste Bin | 💻 |
| Medical Waste | Red Bin | 💉 |
| Paper | Blue Bin | 📄 |
| Cardboard | Blue Bin | 📦 |
| Organic Waste | Green Bin | 🍃 |

## 📊 Model Performance & Accuracy

### Overall Model Performance

| Metric | Value | Description |
|--------|-------|-------------|
| **Precision** | **0.948 (94.8%)** | Correct predictions out of total predictions |
| **Recall** | **0.887 (88.7%)** | Detected objects out of actual objects |
| **mAP@0.5** | **0.935 (93.5%)** | Detection accuracy at IoU = 0.5 |
| **mAP@0.5:0.95** | **0.860 (86.0%)** | Strict accuracy across multiple IoU thresholds |

### Per-Class Performance Metrics

| Class | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|-------|-----------|--------|---------|--------------|
| **Paper** | 1.00 | 0.944 | 0.994 | 0.991 |
| **Cardboard** | 0.94 | 1.00 | 0.994 | 0.994 |
| **Organic Waste** | 0.903 | 0.718 | 0.817 | 0.595 |
| **Plastic** | 0.963 | 0.889 | 0.949 | 0.881 |
| **Metal** | 0.952 | 0.857 | 0.929 | 0.857 |
| **Glass** | 0.944 | 0.889 | 0.944 | 0.889 |
| **Can** | 0.952 | 0.952 | 0.976 | 0.905 |
| **Cable** | 0.929 | 0.857 | 0.905 | 0.810 |
| **E-waste** | 0.937 | 0.882 | 0.929 | 0.857 |
| **Medical Waste** | 0.905 | 0.882 | 0.913 | 0.821 |

### Key Performance Insights

- **Best Performing Classes**: Paper (99.4% mAP@0.5) and Cardboard (99.4% mAP@0.5)
- **Most Challenging Class**: Organic Waste (81.7% mAP@0.5) - due to high variability in appearance
- **High Precision**: 94.8% ensures minimal false positives
- **Good Recall**: 88.7% captures most waste items in images
- **Production Ready**: mAP@0.5 of 93.5% indicates excellent real-world performance

## �️ System Architecture & Design

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Client Layer                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │   Browser    │  │    Mobile    │  │   Desktop    │          │
│  │  (HTML/CSS)  │  │   Browser    │  │   Browser    │          │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘          │
│         │                  │                  │                  │
│         └──────────────────┴──────────────────┘                  │
│                            │                                     │
│                    HTTP/HTTPS Requests                           │
└────────────────────────────┼────────────────────────────────────┘
                             │
┌────────────────────────────┼────────────────────────────────────┐
│                    Presentation Layer                            │
│                            │                                     │
│         ┌──────────────────▼──────────────────┐                 │
│         │      Flask Web Server                │                 │
│         │  ┌────────────────────────────┐     │                 │
│         │  │   Route Handlers           │     │                 │
│         │  │  - / (index)               │     │                 │
│         │  │  - /classify (POST)        │     │                 │
│         │  │  - /about                  │     │                 │
│         │  └────────────────────────────┘     │                 │
│         └──────────────┬─────────────────────┘                  │
└────────────────────────┼────────────────────────────────────────┘
                         │
┌────────────────────────┼────────────────────────────────────────┐
│                  Business Logic Layer                            │
│                        │                                         │
│    ┌───────────────────▼───────────────────┐                    │
│    │   Image Processing Module             │                    │
│    │  - File validation                    │                    │
│    │  - Image preprocessing                │                    │
│    │  - Format conversion                  │                    │
│    └───────────────────┬───────────────────┘                    │
│                        │                                         │
│    ┌───────────────────▼───────────────────┐                    │
│    │   Classification Engine               │                    │
│    │  - YOLO model inference               │                    │
│    │  - Confidence filtering (>25%)        │                    │
│    │  - Bounding box extraction            │                    │
│    └───────────────────┬───────────────────┘                    │
│                        │                                         │
│    ┌───────────────────▼───────────────────┐                    │
│    │   Result Processing Module            │                    │
│    │  - Category mapping                   │                    │
│    │  - Metadata enrichment                │                    │
│    │  - Response formatting                │                    │
│    └───────────────────────────────────────┘                    │
└─────────────────────────────────────────────────────────────────┘
                         │
┌────────────────────────┼────────────────────────────────────────┐
│                    Data/Model Layer                              │
│                        │                                         │
│    ┌───────────────────▼───────────────────┐                    │
│    │   YOLOv8 Model (best.pt)              │                    │
│    │  - 10 waste classes                   │                    │
│    │  - PyTorch backend                    │                    │
│    │  - 93.5% mAP@0.5                      │                    │
│    └────────────────────────────────────────┘                   │
│                                                                  │
│    ┌────────────────────────────────────────┐                   │
│    │   Configuration (data.yaml)            │                   │
│    │  - Class definitions                   │                   │
│    │  - Dataset paths                       │                   │
│    └────────────────────────────────────────┘                   │
│                                                                  │
│    ┌────────────────────────────────────────┐                   │
│    │   Waste Categories Metadata            │                   │
│    │  - Bin assignments                     │                   │
│    │  - Color codes                         │                   │
│    │  - Icons & descriptions                │                   │
│    └────────────────────────────────────────┘                   │
└──────────────────────────────────────────────────────────────────┘
```

### System Components

#### 1. Web Server (Flask Application)
- **Purpose**: Handle HTTP requests and serve web interface
- **Responsibilities**:
  - Route management
  - Request validation
  - File upload handling
  - Response serialization
  - Static file serving

#### 2. Image Processing Pipeline
- **Input Validation**: Check file type, size, and format
- **Preprocessing**: Resize, normalize, and prepare for model
- **Post-processing**: Convert results to base64 for web display

#### 3. YOLO Detection Engine
- **Model**: YOLOv8 (Ultralytics implementation)
- **Inference**: Real-time object detection
- **Output**: Bounding boxes, class IDs, confidence scores

#### 4. Classification Service
- **Category Mapping**: Convert class IDs to waste types
- **Metadata Enrichment**: Add bin type, color, icon, description
- **Confidence Filtering**: Only return detections > 25% confidence

### Data Flow Diagram

```
User Upload Image
      │
      ▼
┌─────────────────┐
│ File Validation │ ──── Reject invalid files
└────────┬────────┘
         │ Valid
         ▼
┌─────────────────┐
│  Save to Disk   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ YOLO Inference  │ ──── Model: best.pt
└────────┬────────┘      Confidence: >0.25
         │
         ▼
┌─────────────────┐
│ Parse Results   │ ──── Extract boxes, classes, scores
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Enrich Metadata │ ──── Add bin, color, icon, description
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Format Response │ ──── JSON with base64 image
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Return to UI   │
└─────────────────┘
         │
         ▼
┌─────────────────┐
│ Display Results │ ──── Show detections with guidance
└─────────────────┘
```

## 📐 Class Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         Flask Application                        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ uses
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      <<Module>> app.py                           │
├─────────────────────────────────────────────────────────────────┤
│ + app: Flask                                                     │
│ + yolo_model: YOLO                                               │
│ + pickle_model: object                                           │
│ + WASTE_CATEGORIES: dict                                         │
│ + ALLOWED_EXTENSIONS: set                                        │
├─────────────────────────────────────────────────────────────────┤
│ + index() -> render_template                                     │
│ + classify() -> jsonify                                          │
│ + about() -> render_template                                     │
│ + allowed_file(filename: str) -> bool                            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ depends on
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    <<Class>> YOLO (Ultralytics)                  │
├─────────────────────────────────────────────────────────────────┤
│ - model_path: str                                                │
│ - device: str                                                    │
│ - conf_threshold: float                                          │
├─────────────────────────────────────────────────────────────────┤
│ + __init__(model_path: str)                                      │
│ + __call__(image_path: str, save: bool, conf: float) -> Results │
│ + predict(source: str) -> Results                                │
│ + train(data: str, epochs: int) -> None                          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ returns
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      <<Class>> Results                           │
├─────────────────────────────────────────────────────────────────┤
│ + boxes: Boxes                                                   │
│ + names: dict                                                    │
│ + orig_img: ndarray                                              │
│ + path: str                                                      │
├─────────────────────────────────────────────────────────────────┤
│ + plot() -> ndarray                                              │
│ + save(filename: str) -> None                                    │
│ + to_json() -> str                                               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ contains
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                       <<Class>> Boxes                            │
├─────────────────────────────────────────────────────────────────┤
│ + xyxy: Tensor                                                   │
│ + conf: Tensor                                                   │
│ + cls: Tensor                                                    │
│ + data: Tensor                                                   │
├─────────────────────────────────────────────────────────────────┤
│ + __len__() -> int                                               │
│ + __getitem__(idx: int) -> Box                                   │
└─────────────────────────────────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────┐
│              <<Data Structure>> WasteCategory                    │
├─────────────────────────────────────────────────────────────────┤
│ + color: str                                                     │
│ + icon: str                                                      │
│ + bin: str                                                       │
│ + description: str                                               │
└─────────────────────────────────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────┐
│              <<Data Structure>> Detection                        │
├─────────────────────────────────────────────────────────────────┤
│ + class: str                                                     │
│ + confidence: float                                              │
│ + color: str                                                     │
│ + icon: str                                                      │
│ + bin: str                                                       │
│ + description: str                                               │
└─────────────────────────────────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────┐
│              <<Data Structure>> ClassificationResponse           │
├─────────────────────────────────────────────────────────────────┤
│ + success: bool                                                  │
│ + detections: List[Detection]                                    │
│ + image: str (base64)                                            │
│ + total_items: int                                               │
└─────────────────────────────────────────────────────────────────┘
```

### Component Relationships

```
┌──────────────┐         ┌──────────────┐         ┌──────────────┐
│    Flask     │────────▶│    YOLO      │────────▶│   PyTorch    │
│   Server     │  loads  │    Model     │  uses   │   Backend    │
└──────────────┘         └──────────────┘         └──────────────┘
       │                        │
       │                        │
       │ uses                   │ produces
       │                        │
       ▼                        ▼
┌──────────────┐         ┌──────────────┐
│   Werkzeug   │         │   Results    │
│  (Security)  │         │   Object     │
└──────────────┘         └──────────────┘
       │                        │
       │                        │
       │ validates              │ enriched by
       │                        │
       ▼                        ▼
┌──────────────┐         ┌──────────────┐
│  File Upload │         │   Waste      │
│              │         │  Categories  │
└──────────────┘         └──────────────┘
```

### Sequence Diagram: Image Classification Flow

```
User          Flask App       File System      YOLO Model      Response Builder
 │                │                │                │                │
 │──Upload Image─▶│                │                │                │
 │                │                │                │                │
 │                │──Validate──────│                │                │
 │                │                │                │                │
 │                │──Save File────▶│                │                │
 │                │                │                │                │
 │                │──Load Image────│                │                │
 │                │                │                │                │
 │                │──Predict──────────────────────▶│                │
 │                │                │                │                │
 │                │                │                │──Process────▶  │
 │                │                │                │                │
 │                │◀──────────────────────Results──│                │
 │                │                │                │                │
 │                │──Parse & Enrich────────────────────────────────▶│
 │                │                │                │                │
 │                │◀──────────────────────────────────JSON Response─│
 │                │                │                │                │
 │                │──Delete File──▶│                │                │
 │                │                │                │                │
 │◀──JSON Result──│                │                │                │
 │                │                │                │                │
```

## 🛠️ Technologies Used

### Backend
- **Flask**: Web framework
- **Ultralytics YOLO**: Object detection
- **PyTorch**: Deep learning framework
- **Pillow**: Image processing
- **OpenCV**: Computer vision

### Frontend
- **Steamlit**: Structure and styling

## 📊 API Endpoints

### `GET /`
Returns the main web interface

### `POST /classify`
Classifies uploaded waste image

**Request:**
- Method: POST
- Content-Type: multipart/form-data
- Body: file (image)

**Response:**
```json
{
  "success": true,
  "detections": [
    {
      "class": "plastic",
      "confidence": 95.5,
      "color": "#FF6B6B",
      "icon": "♻️",
      "bin": "Yellow Bin",
      "description": "Recyclable plastic items"
    }
  ],
  "image": "base64_encoded_image",
  "total_items": 1
}
```

## 🔧 Configuration

### Model Configuration
Edit `config/data.yaml` to modify dataset paths and classes:
```yaml
path: datasets/merged
train: train/images
val: valid/images
test: test/images

nc: 10
names:
  0: plastic
  1: metal
  # ... etc
```

