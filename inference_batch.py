import os
import cv2
import numpy as np
import subprocess
import torch
from models.unet import UNet

# ==============================
# PATHS
# ==============================
EDGE_DIR = "edges"
OUT_DIR = "outputs"
SVG_DIR = "outputs/svg"
MODEL_PATH = "models/jewellery_unet.pth"

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(SVG_DIR, exist_ok=True)

# ==============================
# LOAD MODEL
# ==============================
device = "cuda" if torch.cuda.is_available() else "cpu"
model = UNet(in_channels=1, out_channels=1).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# ==============================
# UTILS
# ==============================
def skeletonize(img):
    skel = np.zeros(img.shape, np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))

    while True:
        eroded = cv2.erode(img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()
        if cv2.countNonZero(img) == 0:
            break
    return skel


def perfect_circles(img):
    circles = cv2.HoughCircles(
        img,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=40,
        param1=100,
        param2=30,
        minRadius=20,
        maxRadius=300
    )

    if circles is not None:
        for c in circles[0]:
            x, y, r = map(int, c)
            cv2.circle(img, (x, y), r, 255, 1)
    return img


def smooth_lines(img):
    contours, _ = cv2.findContours(
        img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )

    smooth = np.zeros_like(img)
    for cnt in contours:
        epsilon = 0.002 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        cv2.drawContours(smooth, [approx], -1, 255, 1)

    return smooth


def export_svg(png_path, svg_path):
    subprocess.run([
        "potrace",
        png_path,
        "-s",
        "-o",
        svg_path
    ])


# ==============================
# INFERENCE LOOP
# ==============================
for fname in os.listdir(EDGE_DIR):
    if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
        continue

    print("Processing:", fname)

    img = cv2.imread(os.path.join(EDGE_DIR, fname), 0)
    img = cv2.resize(img, (512, 512))
    img = img / 255.0

    x = torch.tensor(img).unsqueeze(0).unsqueeze(0).float().to(device)

    with torch.no_grad():
        pred = model(x).squeeze().cpu().numpy()

    # Binary
    _, mask = cv2.threshold(pred, 0.5, 1, cv2.THRESH_BINARY)
    mask = (mask * 255).astype(np.uint8)

    # ==============================
    # CAD GEOMETRY PIPELINE
    # ==============================
    mask = skeletonize(mask)          # 1️⃣
    mask = perfect_circles(mask)      # 3️⃣
    mask = smooth_lines(mask)          # 4️⃣
    mask = cv2.dilate(mask, None, 1)

    # Save PNG
    out_png = os.path.join(OUT_DIR, fname)
    cv2.imwrite(out_png, mask)

    # Save SVG
    out_svg = os.path.join(
        SVG_DIR, fname.rsplit(".",1)[0] + ".svg"
    )
    export_svg(out_png, out_svg)

print("DONE ✅")
