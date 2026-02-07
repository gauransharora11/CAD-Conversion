import cv2
import os

input_folder = "data"
edge_folder = "edges"

os.makedirs(edge_folder, exist_ok=True)

for file in os.listdir(input_folder):
    path = os.path.join(input_folder, file)
    img = cv2.imread(path)

    if img is None:
        print(f"❌ Cannot read {file}")
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 150)

    cv2.imwrite(os.path.join(edge_folder, file), edges)

print("✅ Edges created!")
