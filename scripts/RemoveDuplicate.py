import os
from PIL import Image
import imagehash

path = "Updated_Datset/clean_dataset/images"

hashes = {}
duplicates = []

for class_name in os.listdir(path):
    class_path = os.path.join(path, class_name)

    for file in os.listdir(class_path):
        img_path = os.path.join(class_path, file)
        try:
            img = Image.open(img_path)
            h = imagehash.phash(img)

            if h in hashes:
                duplicates.append(img_path)
            else:
                hashes[h] = img_path
        except:
            continue

# delete duplicates
for dup in duplicates:
    os.remove(dup)

print("Removed duplicates:", len(duplicates))