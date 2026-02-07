import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.enc1 = self.block(1, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = self.block(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.bottleneck = self.block(128, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.dec2 = self.block(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.dec1 = self.block(128, 64)

        self.out = nn.Conv2d(64, 1, 1)

    def block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))

        b = self.bottleneck(self.pool2(e2))

        d2 = self.up2(b)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        return torch.sigmoid(self.out(d1))
