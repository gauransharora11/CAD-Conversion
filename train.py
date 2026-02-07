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
class CADDataset(Dataset):
    def __init__(self, edge_dir, cad_dir):
        self.edge_dir = edge_dir
        self.cad_dir = cad_dir
        self.files = os.listdir(edge_dir)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        name = self.files[idx]

        edge_path = os.path.join(self.edge_dir, name)
        cad_path = os.path.join(self.cad_dir, name)

        edge = cv2.imread(edge_path, cv2.IMREAD_GRAYSCALE)
        cad = cv2.imread(cad_path, cv2.IMREAD_GRAYSCALE)

        # Skip broken images
        if edge is None or cad is None:
            edge = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
            cad = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)

        # 🔥 FIX: resize to SAME SIZE as model output
        edge = cv2.resize(edge, (IMG_SIZE, IMG_SIZE))
        cad = cv2.resize(cad, (IMG_SIZE, IMG_SIZE))

        edge = edge / 255.0
        cad = cad / 255.0

        edge = torch.tensor(edge, dtype=torch.float32).unsqueeze(0)
        cad = torch.tensor(cad, dtype=torch.float32).unsqueeze(0)

        return edge, cad


# =========================
# LOAD DATA
# =========================
dataset = CADDataset(EDGE_DIR, CAD_DIR)
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
