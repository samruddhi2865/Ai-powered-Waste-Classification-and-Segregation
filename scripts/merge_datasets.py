import os
import shutil

OUT = "datasets/merged"

SOURCES = [
    "datasets/source/trashnetpp",
    "datasets/source/biodegradable_split",
    "datasets/source/kaggle_yolo"
]

for split in ["train", "valid", "test"]:
    os.makedirs(os.path.join(OUT, split, "images"), exist_ok=True)
    os.makedirs(os.path.join(OUT, split, "labels"), exist_ok=True)

def copy_from_source(src_root, split):
    candidates = [
        (os.path.join(src_root, split, "images"), os.path.join(src_root, split, "labels")),
        (os.path.join(src_root, "images", split), os.path.join(src_root, "labels", split)),
    ]

    for img_dir, lbl_dir in candidates:
        if os.path.exists(img_dir) and os.path.exists(lbl_dir):
            prefix = os.path.basename(src_root)
            for file in os.listdir(img_dir):
                if file.lower().endswith((".jpg", ".jpeg", ".png")):
                    src_img = os.path.join(img_dir, file)
                    src_lbl = os.path.join(lbl_dir, os.path.splitext(file)[0] + ".txt")

                    new_name = f"{prefix}_{file}"
                    dst_img = os.path.join(OUT, split, "images", new_name)
                    dst_lbl = os.path.join(OUT, split, "labels", os.path.splitext(new_name)[0] + ".txt")

                    shutil.copy2(src_img, dst_img)
                    if os.path.exists(src_lbl):
                        shutil.copy2(src_lbl, dst_lbl)
            return

for split in ["train", "valid", "test"]:
    for src in SOURCES:
        copy_from_source(src, split)

print("Merged dataset created successfully.")