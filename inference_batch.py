import torch
import cv2
import os
from models.unet import UNet

model = UNet()
model.load_state_dict(torch.load("jewelry_net.pth"))
model.eval()

os.makedirs("outputs", exist_ok=True)

for file in os.listdir("edges"):
    img = cv2.imread(f"edges/{file}", 0)
    img = img / 255.0
    img = torch.tensor(img).unsqueeze(0).unsqueeze(0).float()

    with torch.no_grad():
        pred = model(img)

    mask = pred.squeeze().cpu().numpy()
    mask = (mask * 255).astype("uint8")  # convert to 8-bit
    cv2.imwrite(f"outputs/{file}", mask)


print("✅ Batch inference done!")
