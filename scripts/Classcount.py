import os

label_path = "datasets/merged/train/labels"

class_count = {}

for file in os.listdir(label_path):
    if file.endswith(".txt"):
        with open(os.path.join(label_path, file), "r") as f:
            lines = f.readlines()
            for line in lines:
                class_id = int(line.split()[0])
                class_count[class_id] = class_count.get(class_id, 0) + 1

print("Class Distribution:\n")
for k in sorted(class_count.keys()):
    print(f"Class {k}: {class_count[k]}")