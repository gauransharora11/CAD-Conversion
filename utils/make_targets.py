import cv2
import os
import numpy as np

input_folder = "data"
target_folder = "cad_targets"

os.makedirs(target_folder, exist_ok=True)

for file in os.listdir(input_folder):
    path = os.path.join(input_folder, file)
    img = cv2.imread(path)

    if img is None:
        print(f"❌ Cannot read {file}")
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    # Correct replacement for cv2.zeros_like
    mask = np.zeros_like(edges)

    mask[edges > 0] = 255

    cv2.imwrite(os.path.join(target_folder, file), mask)

print("✅ CAD targets created!")
