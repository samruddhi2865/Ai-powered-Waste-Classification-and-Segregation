from ultralytics import YOLO

# Load your trained model
model = YOLO('models/best.pt')

# ✅ Option A: Single image
results = model.predict(
    source  = 'datasets/merged/test/images/plastic_waste_24_X180_C509_0407_0_jpg.rf.128cca7cba1e47b52caae83f9bbe411f.jpg',
    save    = True,
    project = 'C:/Users/Admin/Desktop/Waste_Classification_App/Waste_Classification_App/runs/detect',
    name    = 'single_prediction',
    conf    = 0.25
)

# ✅ Option B: Entire test folder
results = model.predict(
    source  = 'datasets/merged/test/images',
    save    = True,
    project = 'C:/Users/Admin/Desktop/Waste_Classification_App/Waste_Classification_App/runs/detect',
    name    = 'my_predictions',
    conf    = 0.25
)

print("Done! Check runs/detect/my_predictions/ for output images")