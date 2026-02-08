import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from models.unet import UNet
   # your model file

# =========================
# SETTINGS
# =========================
EDGE_DIR = "edges"
CAD_DIR = "cad_targets"
IMG_SIZE = 512
EPOCHS = 20
BATCH_SIZE = 2
LR = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# DATASET
# =========================
class JewelryDataset(Dataset):
    def __init__(self, edge_dir, cad_dir):
        self.edge_dir = edge_dir
        self.cad_dir = cad_dir

        # Only keep files that exist in BOTH folders
        edge_files = set(os.listdir(edge_dir))
        cad_files  = set(os.listdir(cad_dir))

        self.files = sorted(list(edge_files & cad_files))  # intersection

        print(f"Found {len(self.files)} matching image pairs")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]

        edge_path = os.path.join(self.edge_dir, file)
        cad_path  = os.path.join(self.cad_dir, file)

        edge = cv2.imread(edge_path, cv2.IMREAD_GRAYSCALE)
        cad  = cv2.imread(cad_path,  cv2.IMREAD_GRAYSCALE)

        edge = cv2.resize(edge, (512, 512))
        cad  = cv2.resize(cad,  (512, 512))

        edge = torch.tensor(edge/255.0).unsqueeze(0).float()
        cad  = torch.tensor(cad/255.0).unsqueeze(0).float()

        return edge, cad


# =========================
# LOAD DATA
# =========================
dataset = JewelryDataset(EDGE_DIR, CAD_DIR)

loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# =========================
# MODEL
# =========================
model = UNet().to(DEVICE)
loss_fn = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# =========================
# TRAIN LOOP
# =========================
print("🚀 Training Started...")

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for edge, cad in tqdm(loader):
        edge, cad = edge.to(DEVICE), cad.to(DEVICE)

        pred = model(edge)

        # 🔥 ensure output same size (extra safety)
        if pred.shape != cad.shape:
            pred = torch.nn.functional.interpolate(pred, size=cad.shape[2:])

        loss = loss_fn(pred, cad)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS} Loss: {total_loss:.4f}")

# =========================
# SAVE MODEL
# =========================
torch.save(model.state_dict(), "jewelry_net.pth")
print("✅ Training Complete — Model Saved as jewelry_net.pth")
