import sys, platform
from importlib import metadata

print("Python  :", sys.version.split()[0], f"({platform.system()} {platform.release()})")
print("Location:", sys.executable)
print()

packages = ["streamlit", "ultralytics", "torch", "torchvision",
            "opencv-python", "numpy", "pillow", "pandas", "plotly"]
for name in packages:
    try:
        print(f"{name:<15} {metadata.version(name)}")
    except metadata.PackageNotFoundError:
        print(f"{name:<15} NOT INSTALLED")

try:
    import torch
    print("\nCUDA available:", torch.cuda.is_available())
except Exception as e:
    print("\ntorch import failed:", e)