import os

def clean_dataset(path):
    img_dir = os.path.join(path, "images")
    lbl_dir = os.path.join(path, "labels")

    if not os.path.exists(img_dir):
        return

    for file in os.listdir(img_dir):
        img_path = os.path.join(img_dir, file)
        lbl_path = os.path.join(lbl_dir, os.path.splitext(file)[0] + ".txt")

        if not os.path.exists(img_path):
            print("❌ Removing broken reference:", file)
            continue

        if not os.path.exists(lbl_path):
            print("⚠ Removing image without label:", file)
            os.remove(img_path)

# Run for all datasets
sources = [
    "datasets/source/cable-damage/train",
    "datasets/source/cable-damage/valid",
    "datasets/source/cable-damage/test",
    "datasets/source/can_waste/train",
    "datasets/source/can_waste/valid",
    "datasets/source/can_waste/test",
    "datasets/source/cardboard.v1i.yolov8/train",
    "datasets/source/cardboard.v1i.yolov8/valid",
    "datasets/source/cardboard.v1i.yolov8/test",
    "datasets/source/Ewaste/train",
    "datasets/source/Ewaste/valid",
    "datasets/source/Ewaste/test",
    "datasets/source/Glass_waste/train",
    "datasets/source/Glass_waste/valid",
    "datasets/source/Glass_waste/test",
    "datasets/source/Medical_waste/train",
    "datasets/source/Medical_waste/valid",
    "datasets/source/Medical_waste/test",
    "datasets/source/Metal.v1i.yolov8/train",
    "datasets/source/Metal.v1i.yolov8/valid",
    "datasets/source/Metal.v1i.yolov8/test",
    "datasets/source/organic_waste/train",
    "datasets/source/organic_waste/valid",
    "datasets/source/organic_waste/test",
    "datasets/source/paper_waste/train",
    "datasets/source/paper_waste/valid",
    "datasets/source/paper_waste/test",
    "datasets/source/plastic_waste/train",
    "datasets/source/plastic_waste/valid",
    "datasets/source/plastic_waste/test"
]


for src in sources:
    clean_dataset(src)

print("✅ Dataset cleaned")