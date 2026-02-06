import torch
import torch.nn as nn

def double_conv(in_c, out_c):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_c, out_c, 3, padding=1),
        nn.ReLU(inplace=True),
    )

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.d1 = double_conv(1, 64)
        self.d2 = double_conv(64, 128)
        self.d3 = double_conv(128, 256)
        self.d4 = double_conv(256, 512)

        self.pool = nn.MaxPool2d(2)

        self.u1 = double_conv(512+256, 256)
        self.u2 = double_conv(256+128, 128)
        self.u3 = double_conv(128+64, 64)

        self.up = nn.Upsample(scale_factor=2, mode='bilinear')

        self.final = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        d1 = self.d1(x)
        d2 = self.d2(self.pool(d1))
        d3 = self.d3(self.pool(d2))
        d4 = self.d4(self.pool(d3))

        u1 = self.up(d4)
        u1 = self.u1(torch.cat([u1, d3], dim=1))

        u2 = self.up(u1)
        u2 = self.u2(torch.cat([u2, d2], dim=1))

        u3 = self.up(u2)
        u3 = self.u3(torch.cat([u3, d1], dim=1))

        return torch.sigmoid(self.final(u3))
