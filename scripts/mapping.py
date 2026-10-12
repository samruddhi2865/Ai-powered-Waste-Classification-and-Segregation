import os

LABEL_PATH = "datasets/merged"

DATASET_CLASS_MAP = {
    "cable-damage": 4,
    "can_waste": 3,
    "cardboard.v1i.yolov8": 8,
    "Ewaste": 5,
    "Glass_waste": 2,
    "Medical_waste": 6,
    "Metal.v1i.yolov8": 1,
    "organic_waste": 9,
    "paper_waste": 7,
    "plastic_waste": 0
}

for split in ["train", "valid", "test"]:
    label_dir = os.path.join(LABEL_PATH, split, "labels")

    for file in os.listdir(label_dir):
        file_path = os.path.join(label_dir, file)

        prefix = None
        for key in DATASET_CLASS_MAP:
            if file.startswith(key):
                prefix = key
                break

        if prefix is None:
            print("⚠ Unknown prefix:", file)
            continue

        new_class = DATASET_CLASS_MAP[prefix]

        with open(file_path, "r") as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            parts = line.strip().split()
            parts[0] = str(new_class)
            new_lines.append(" ".join(parts))

        with open(file_path, "w") as f:
            f.write("\n".join(new_lines))

print("✅ Class IDs fixed correctly!")