import cv2
import torch
import os
import numpy as np
from models.unet import UNet

MODEL_PATH = "models/jewellery_unet.pth"
INPUT_DIR = "edges"
OUTPUT_DIR = "outputs"
IMG_SIZE = 512

os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device("cpu")

model = UNet()
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

for file in os.listdir(INPUT_DIR):
    path = os.path.join(INPUT_DIR, file)

    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Skipping:", file)
        continue

    # 🔥 RESIZE HERE
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = torch.tensor(img).unsqueeze(0).unsqueeze(0).float()

    with torch.no_grad():
        pred = model(img)

    pred = pred.squeeze().numpy()
    pred = (pred * 255).astype("uint8")

    out_path = os.path.join(OUTPUT_DIR, file)
    cv2.imwrite(out_path, pred)
    print("Saved:", out_path)

print("✅ Inference Complete")
