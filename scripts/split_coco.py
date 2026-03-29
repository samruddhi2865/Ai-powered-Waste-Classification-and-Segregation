import json
import os
import shutil
from sklearn.model_selection import train_test_split

BASE = "datasets/source/biodegradable_coco/biodegradable.coco"
IMG_DIR = os.path.join(BASE, "train")
ANN_FILE = os.path.join(BASE, "train", "_annotations.coco.json")
OUT = "datasets/source/biodegradable_split"

if not os.path.exists(ANN_FILE):
    print(f"Annotation file not found: {ANN_FILE}")
    raise SystemExit

os.makedirs(OUT, exist_ok=True)

with open(ANN_FILE, "r", encoding="utf-8") as f:
    coco = json.load(f)

images = coco["images"]
annotations = coco["annotations"]
categories = coco["categories"]

image_ids = [img["id"] for img in images]
train_ids, temp_ids = train_test_split(image_ids, test_size=0.2, random_state=42)
valid_ids, test_ids = train_test_split(temp_ids, test_size=0.5, random_state=42)

splits = {
    "train": set(train_ids),
    "valid": set(valid_ids),
    "test": set(test_ids),
}

for split in splits:
    os.makedirs(os.path.join(OUT, split, "images"), exist_ok=True)

for img in images:
    for split, ids in splits.items():
        if img["id"] in ids:
            src = os.path.join(IMG_DIR, img["file_name"])
            dst = os.path.join(OUT, split, "images", img["file_name"])
            if os.path.exists(src):
                shutil.copy2(src, dst)

for split, ids in splits.items():
    split_images = [img for img in images if img["id"] in ids]
    split_annotations = [ann for ann in annotations if ann["image_id"] in ids]
    split_coco = {
        "images": split_images,
        "annotations": split_annotations,
        "categories": categories
    }
    with open(os.path.join(OUT, split, "_annotations.coco.json"), "w", encoding="utf-8") as f:
        json.dump(split_coco, f, indent=2)

print("COCO dataset split completed successfully.")