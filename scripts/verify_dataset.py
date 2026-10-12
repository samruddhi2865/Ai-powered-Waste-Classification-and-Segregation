import os

BASE = "datasets/merged"

for split in ["train", "valid", "test"]:
    img_dir = os.path.join(BASE, split, "images")
    lbl_dir = os.path.join(BASE, split, "labels")

    images = [f for f in os.listdir(img_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    labels = [f for f in os.listdir(lbl_dir) if f.endswith(".txt")]

    print(f"{split}: images={len(images)}, labels={len(labels)}")