import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        def block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(out_c, out_c, 3, padding=1),
                nn.ReLU(),
            )

        # DOWN
        self.d1 = block(1, 64)
        self.d2 = block(64, 128)
        self.d3 = block(128, 256)
        self.d4 = block(256, 512)

        # UP (NO ConvTranspose — matches your trained model)
        self.u1 = block(512 + 256, 256)
        self.u2 = block(256 + 128, 128)
        self.u3 = block(128 + 64, 64)

        self.pool = nn.MaxPool2d(2)
        self.final = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        d1 = self.d1(x)
        d2 = self.d2(self.pool(d1))
        d3 = self.d3(self.pool(d2))
        d4 = self.d4(self.pool(d3))

        # 🔼 Upsample with interpolation (not transpose conv)
        u1 = F.interpolate(d4, scale_factor=2, mode='bilinear', align_corners=False)
        u1 = torch.cat([u1, d3], dim=1)
        u1 = self.u1(u1)

        u2 = F.interpolate(u1, scale_factor=2, mode='bilinear', align_corners=False)
        u2 = torch.cat([u2, d2], dim=1)
        u2 = self.u2(u2)

        u3 = F.interpolate(u2, scale_factor=2, mode='bilinear', align_corners=False)
        u3 = torch.cat([u3, d1], dim=1)
        u3 = self.u3(u3)

        return torch.sigmoid(self.final(u3))
