from collections import Counter
import os

label_dir = "datasets/merged/train/labels"
counter = Counter()

for file in os.listdir(label_dir):
    with open(os.path.join(label_dir, file)) as f:
        for line in f:
            class_id = int(line.split()[0])
            counter[class_id] += 1

print(counter)