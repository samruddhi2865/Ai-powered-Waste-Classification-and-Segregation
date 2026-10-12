import os

base_path = "Updated_Datset/clean_dataset"

image_root = os.path.join(base_path, "images")
label_root = os.path.join(base_path, "labels")

class_map = {
    "plastic": 0,
    "metal": 1,
    "glass": 2,
    "can": 3,
    "cable": 4,
    "e_waste": 5,
    "medical_waste": 6,
    "paper": 7,
    "cardboard": 8,
    "organic_waste": 9
}

# loop over train/val/test
for split in ["train", "val", "test"]:
    split_img_path = os.path.join(image_root, split)
    split_label_path = os.path.join(label_root, split)

    os.makedirs(split_label_path, exist_ok=True)

    # loop over classes
    for class_name in os.listdir(split_img_path):
        class_path = os.path.join(split_img_path, class_name)

        if not os.path.isdir(class_path):
            continue

        class_id = class_map[class_name]

        for img_file in os.listdir(class_path):
            img_path = os.path.join(class_path, img_file)

            label_name = os.path.splitext(img_file)[0] + ".txt"
            label_path = os.path.join(split_label_path, label_name)

            # create label folder if needed
            os.makedirs(os.path.dirname(label_path), exist_ok=True)

            with open(label_path, "w") as f:
                f.write(f"{class_id} 0.5 0.5 1.0 1.0")

print("✅ Labels created successfully!")