import os
import shutil
import random

DATASET_PATH = "datasets/merged/train"
OUTPUT_PATH = "datasets/balanced/train"

MAX_PER_CLASS = 1500

img_dir = os.path.join(DATASET_PATH, "images")
lbl_dir = os.path.join(DATASET_PATH, "labels")

out_img_dir = os.path.join(OUTPUT_PATH, "images")
out_lbl_dir = os.path.join(OUTPUT_PATH, "labels")

os.makedirs(out_img_dir, exist_ok=True)
os.makedirs(out_lbl_dir, exist_ok=True)

# Supported image formats
IMG_EXTS = [".jpg", ".jpeg", ".png"]

# Step 1: group images by class
class_files = {}

for file in os.listdir(lbl_dir):
    label_path = os.path.join(lbl_dir, file)

    try:
        with open(label_path, "r") as f:
            line = f.readline().strip()
            if not line:
                continue

            class_id = int(line.split()[0])

    except:
        print("⚠ Skipping invalid label:", file)
        continue

    class_files.setdefault(class_id, []).append(file)

# Step 2: balance dataset
for class_id, files in class_files.items():
    print(f"\nClass {class_id}: {len(files)} images")

    # Limit large classes
    if len(files) > MAX_PER_CLASS:
        files = random.sample(files, MAX_PER_CLASS)
        print(f"→ Reduced to {MAX_PER_CLASS}")

    for lbl_file in files:
        base_name = os.path.splitext(lbl_file)[0]

        # 🔍 find correct image file
        src_img = None
        for ext in IMG_EXTS:
            temp_path = os.path.join(img_dir, base_name + ext)
            if os.path.exists(temp_path):
                src_img = temp_path
                break

        if src_img is None:
            print("❌ Missing image:", base_name)
            continue

        src_lbl = os.path.join(lbl_dir, lbl_file)

        dst_img = os.path.join(out_img_dir, os.path.basename(src_img))
        dst_lbl = os.path.join(out_lbl_dir, lbl_file)

        shutil.copy2(src_img, dst_img)
        shutil.copy2(src_lbl, dst_lbl)

print("\n✅ Balanced dataset created successfully!")