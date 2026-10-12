import os
import time
from PIL import Image

path = "Updated_Datset/clean_dataset/images"

for class_name in os.listdir(path):
    class_path = os.path.join(path, class_name)

    for file in os.listdir(class_path):
        img_path = os.path.join(class_path, file)

        try:
            with Image.open(img_path) as img:
                width, height = img.size

            # 🔴 Now image is closed here

            if width < 200 or height < 200:
                time.sleep(0.05)  # small delay

                try:
                    os.remove(img_path)
                    print("Deleted:", img_path)

                except PermissionError:
                    # Retry after delay
                    time.sleep(0.2)
                    try:
                        os.remove(img_path)
                        print("Deleted (retry):", img_path)
                    except Exception as e:
                        print("Still locked:", img_path, e)

        except Exception as e:
            print("Error reading:", img_path, e)