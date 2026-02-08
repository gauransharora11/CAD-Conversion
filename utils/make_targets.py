import cv2
import os
import numpy as np

input_dir = "Photos"       # ORIGINAL JEWELRY PHOTOS
output_dir = "cad_targets" # TARGET MASKS

os.makedirs(output_dir, exist_ok=True)

files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg','.jpeg','.png'))]

print("Found", len(files), "photos")

for file in files:
    path = os.path.join(input_dir, file)
    img = cv2.imread(path)

    if img is None:
        print("❌ Skipping:", file)
        continue

    img = cv2.resize(img, (512,512))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)

    # Binary mask
    _, mask = cv2.threshold(blur, 120, 255, cv2.THRESH_BINARY)

    # Clean noise
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    cv2.imwrite(os.path.join(output_dir, file), mask)

print("✅ CAD masks created!")
