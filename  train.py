import torch
from torch.utils.data import Dataset, DataLoader
import os
import cv2
import numpy as np
from models.unet import UNet
from tqdm import tqdm

class JewelleryDataset(Dataset):
    def __init__(self, edge_dir, cad_dir):
        self.edge_files = os.listdir(edge_dir)
        self.edge_dir = edge_dir
        self.cad_dir = cad_dir

    def __len__(self):
        return len(self.edge_files)

    def __getitem__(self, idx):
        name = self.edge_files[idx]
        edge = cv2.imread(os.path.join(self.edge_dir, name), 0)
        cad = cv2.imread(os.path.join(self.cad_dir, name), 0)

        edge = edge / 255.0
        cad = cad / 255.0

        edge = torch.tensor(edge).unsqueeze(0).float()
        cad = torch.tensor(cad).unsqueeze(0).float()
        return edge, cad

device = "cuda" if torch.cuda.is_available() else "cpu"

model = UNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_fn = torch.nn.BCELoss()

dataset = JewelleryDataset("data/edges", "data/cad_targets")
loader = DataLoader(dataset, batch_size=4, shuffle=True)

for epoch in range(20):
    total_loss = 0
    for edge, cad in tqdm(loader):
        edge, cad = edge.to(device), cad.to(device)
        pred = model(edge)
        loss = loss_fn(pred, cad)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch} Loss {total_loss/len(loader)}")

torch.save(model.state_dict(), "models/jewellery_unet.pth")
