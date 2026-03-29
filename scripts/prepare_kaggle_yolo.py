import os
import shutil
from sklearn.model_selection import train_test_split

SRC_BASE = r"datasets/source/kaggle_garbage/archive/Garbage classification/Garbage classification"
OUT_BASE = "datasets/source/kaggle_yolo"

CLASS_MAP = {
    "paper": 7,
    "cardboard": 8
}

print(f"Looking for images in: {SRC_BASE}")

for split in ["train", "valid", "test"]:
    os.makedirs(os.path.join(OUT_BASE, split, "images"), exist_ok=True)
    os.makedirs(os.path.join(OUT_BASE, split, "labels"), exist_ok=True)

for class_name, class_id in CLASS_MAP.items():
    folder = os.path.join(SRC_BASE, class_name)
    
    print(f"Checking folder: {folder}")
    
    if not os.path.exists(folder):
        print(f"❌ Folder not found: {folder}")
        continue

    files = [
        f for f in os.listdir(folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    print(f"✅ Found {len(files)} images in {class_name}")

    if len(files) == 0:
        print(f"No images found in: {folder}")
        continue

    train_files, temp_files = train_test_split(files, test_size=0.2, random_state=42)
    valid_files, test_files = train_test_split(temp_files, test_size=0.5, random_state=42)

    split_map = {
        "train": train_files,
        "valid": valid_files,
        "test": test_files
    }

    for split, split_files in split_map.items():
        print(f"  {split}: {len(split_files)} images")
        for file in split_files:
            src = os.path.join(folder, file)
            new_name = f"{class_name}_{file}"

            dst_img = os.path.join(OUT_BASE, split, "images", new_name)
            dst_lbl = os.path.join(OUT_BASE, split, "labels", os.path.splitext(new_name)[0] + ".txt")

            shutil.copy2(src, dst_img)
            
            with open(dst_lbl, "w", encoding="utf-8") as f:
                f.write(f"{class_id} 0.5 0.5 0.9 0.9\n")

print("\n🎉 Kaggle paper/cardboard YOLO dataset prepared successfully!")
print(f"Check: datasets/source/kaggle_yolo/train/images")