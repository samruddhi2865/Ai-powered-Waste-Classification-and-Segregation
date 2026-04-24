import os
import shutil

OUT = "datasets/merged"

SOURCES = [
    "datasets/source/cable-damage",
    "datasets/source/can_waste",
    "datasets/source/cardboard.v1i.yolov8",
    "datasets/source/Ewaste",
    "datasets/source/Glass_waste",
    "datasets/source/Medical_waste",
    "datasets/source/Metal.v1i.yolov8",
    "datasets/source/organic_waste",
    "datasets/source/paper_waste",
    "datasets/source/plastic_waste"
]

# Create output folders
for split in ["train", "valid", "test"]:
    os.makedirs(os.path.join(OUT, split, "images"), exist_ok=True)
    os.makedirs(os.path.join(OUT, split, "labels"), exist_ok=True)

def copy_from_source(src_root, split):
    img_dir = os.path.join(src_root, split, "images")
    lbl_dir = os.path.join(src_root, split, "labels")

    # Handle 'val' vs 'valid'
    if not os.path.exists(img_dir):
        alt_split = "val" if split == "valid" else None
        if alt_split:
            img_dir = os.path.join(src_root, alt_split, "images")
            lbl_dir = os.path.join(src_root, alt_split, "labels")

    if not os.path.exists(img_dir):
        print(f"⚠ Skipping {src_root} ({split} not found)")
        return

    prefix = os.path.basename(src_root)

    for file in os.listdir(img_dir):
        if not file.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        src_img = os.path.join(img_dir, file)
        src_lbl = os.path.join(lbl_dir, os.path.splitext(file)[0] + ".txt")

        # ✅ Check if image exists before copying
        if not os.path.exists(src_img):
            print(f"❌ Missing image: {src_img}")
            continue

        new_name = f"{prefix}_{file}"

        dst_img = os.path.join(OUT, split, "images", new_name)
        dst_lbl = os.path.join(OUT, split, "labels", os.path.splitext(new_name)[0] + ".txt")

        try:
            shutil.copy2(src_img, dst_img)
        except Exception as e:
            print(f"❌ Error copying image: {src_img} → {e}")
            continue

        if os.path.exists(src_lbl):
            shutil.copy2(src_lbl, dst_lbl)
        else:
            print(f"⚠ Missing label: {src_lbl}")

# Run merging
for split in ["train", "valid", "test"]:
    for src in SOURCES:
        copy_from_source(src, split)

print("\n✅ FINAL MERGED DATASET CREATED SUCCESSFULLY")