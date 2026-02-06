import torch
import cv2
import os
from models.unet import UNet

device = "cuda" if torch.cuda.is_available() else "cpu"
model = UNet().to(device)
model.load_state_dict(torch.load("models/jewellery_unet.pth"))
model.eval()

input_dir = "data/edges"
output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)

for file in os.listdir(input_dir):
    img = cv2.imread(os.path.join(input_dir, file), 0) / 255.0
    img = torch.tensor(img).unsqueeze(0).unsqueeze(0).float().to(device)

    with torch.no_grad():
        pred = model(img)[0][0].cpu().numpy() * 255

    cv2.imwrite(os.path.join(output_dir, file), pred)
