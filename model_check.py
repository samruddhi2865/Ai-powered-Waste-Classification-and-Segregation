"""
model_check.py
--------------
Checks that a trained YOLO model has the 8 waste classes this app expects.

Use 1 (before starting the app):
    python model_check.py                      -> checks models/best.pt
    python model_check.py path/to/other.pt     -> checks another file

Use 2 (inside app.py):
    from model_check import validate_model
    ok, message = validate_model(model)
"""

import os
import sys

EXPECTED_CLASSES = [
    "plastic", "metal", "glass", "e_waste",
    "medical_waste", "paper", "cardboard", "organic_waste",
]


def validate_model(model):
    """
    Returns (ok, message).

    ok is True when the model has exactly the 8 expected class names.
    The app looks classes up by NAME (not by number), so a different
    order is harmless and is only mentioned in the message.
    """
    names = [model.names[i] for i in range(len(model.names))]
    missing = [c for c in EXPECTED_CLASSES if c not in names]
    extra = [c for c in names if c not in EXPECTED_CLASSES]

    if missing or extra or len(names) != len(EXPECTED_CLASSES):
        msg = f"This model has {len(names)} classes, but the app expects {len(EXPECTED_CLASSES)}. "
        if missing:
            msg += f"Missing: {', '.join(missing)}. "
        if extra:
            msg += f"Not in the app's list: {', '.join(extra)}. "
        msg += "Check that models/best.pt is the new YOLOv8m 8-class file."
        return False, msg

    if names != EXPECTED_CLASSES:
        return True, "Model OK: all 8 classes found (order differs, which is harmless)."
    return True, "Model OK: all 8 classes found in the expected order."


if __name__ == "__main__":
    import numpy as np
    from ultralytics import YOLO

    default_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "best.pt")
    path = sys.argv[1] if len(sys.argv) > 1 else default_path

    if not os.path.exists(path):
        print(f"NOT FOUND: {path}")
        print("Copy the new best.pt into the models/ folder and run this again.")
        sys.exit(1)

    print(f"File : {path}")
    print(f"Size : {os.path.getsize(path) / 1e6:.1f} MB")

    model = YOLO(path)
    print("\nClasses inside the model:")
    for i in range(len(model.names)):
        print(f"  {i}  {model.names[i]}")

    ok, msg = validate_model(model)
    print(f"\n{'PASS' if ok else 'FAIL'}: {msg}")

    # Run one blank image through the model to prove inference works
    model(np.zeros((640, 640, 3), dtype=np.uint8), verbose=False)
    print("Test inference: OK (the model loads and runs)")

    sys.exit(0 if ok else 1)
