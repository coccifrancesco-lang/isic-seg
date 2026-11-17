"""U-Net standard (Ronneberger et al., 2015), versione con BatchNorm e padding.

Input : (B, 3, H, W)  immagine RGB
Output: (B, 1, H, W)  LOGITS (nessuna sigmoid: la applichiamo nella loss/metrica)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(Conv 3x3 -> BatchNorm -> ReLU) x 2"""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    """Encoder: MaxPool 2x2 (dimezza H e W) -> DoubleConv"""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool_conv = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_ch, out_ch))

    def forward(self, x):
        return self.pool_conv(x)


class Up(nn.Module):
    """Decoder: ConvTranspose (raddoppia H e W) -> concatena la skip connection -> DoubleConv"""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)  # in_ch = (in_ch//2 dall'up) + (in_ch//2 dalla skip)

    def forward(self, x, skip):
        x = self.up(x)
        # se H/W non sono divisibili per 16, le dimensioni possono differire di 1 px
        dy = skip.size(2) - x.size(2)
        dx = skip.size(3) - x.size(3)
        x = F.pad(x, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])
        x = torch.cat([skip, x], dim=1)  # concatenazione lungo i canali
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, base=64):
        super().__init__()
        b = base
        self.inc = DoubleConv(in_channels, b)
        self.down1 = Down(b, 2 * b)
        self.down2 = Down(2 * b, 4 * b)
        self.down3 = Down(4 * b, 8 * b)
        self.down4 = Down(8 * b, 16 * b)  # bottleneck
        self.up1 = Up(16 * b, 8 * b)
        self.up2 = Up(8 * b, 4 * b)
        self.up3 = Up(4 * b, 2 * b)
        self.up4 = Up(2 * b, b)
        self.outc = nn.Conv2d(b, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)


if __name__ == "__main__":
    # Test rapido delle forme (funziona anche su CPU, con dati finti)
    model = UNet(base=64)
    x = torch.randn(2, 3, 256, 256)
    y = model(x)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"input {tuple(x.shape)} -> output {tuple(y.shape)} | parametri: {n_params / 1e6:.1f} M")
    assert y.shape == (2, 1, 256, 256)
