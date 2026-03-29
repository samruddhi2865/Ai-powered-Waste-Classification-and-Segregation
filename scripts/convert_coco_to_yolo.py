import json
import os

BASE = "datasets/source/biodegradable_split"
FINAL_CLASS_ID = 9

for split in ["train", "valid", "test"]:
    ann_path = os.path.join(BASE, split, "_annotations.coco.json")
    img_dir = os.path.join(BASE, split, "images")
    label_dir = os.path.join(BASE, split, "labels")

    if not os.path.exists(ann_path):
        print(f"Annotation file not found: {ann_path}")
        continue

    if not os.path.exists(img_dir):
        print(f"Image folder not found: {img_dir}")
        continue

    os.makedirs(label_dir, exist_ok=True)

    with open(ann_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    images = {img["id"]: img for img in coco["images"]}
    anns_by_image = {}

    for ann in coco["annotations"]:
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    for image_id, img in images.items():
        file_name = img["file_name"]
        width = float(img["width"])
        height = float(img["height"])

        label_file = os.path.splitext(file_name)[0] + ".txt"
        label_path = os.path.join(label_dir, label_file)

        with open(label_path, "w", encoding="utf-8") as out:
            for ann in anns_by_image.get(image_id, []):
                bbox = ann["bbox"]

                x = float(bbox[0])
                y = float(bbox[1])
                w = float(bbox[2])
                h = float(bbox[3])

                if width <= 0 or height <= 0 or w <= 0 or h <= 0:
                    continue

                x_center = (x + w / 2.0) / width
                y_center = (y + h / 2.0) / height
                w_norm = w / width
                h_norm = h / height

                x_center = max(0.0, min(1.0, x_center))
                y_center = max(0.0, min(1.0, y_center))
                w_norm = max(0.0, min(1.0, w_norm))
                h_norm = max(0.0, min(1.0, h_norm))

                out.write(
                    f"{FINAL_CLASS_ID} "
                    f"{x_center:.6f} {y_center:.6f} "
                    f"{w_norm:.6f} {h_norm:.6f}\n"
                )

    print(f"{split} split converted successfully.")

print("COCO to YOLO conversion completed.")