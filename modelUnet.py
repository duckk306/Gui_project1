
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """
    (Conv => BN => ReLU) * 2
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class DownBlock(nn.Module):
    """
    Down block: DoubleConv + MaxPool
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        skip = self.conv(x)
        down = self.pool(skip)
        return skip, down


class UpBlock(nn.Module):
    """
    Up block: TransposedConv + concat skip + DoubleConv
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upconv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=2, stride=2
        )
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x, skip):
        x = self.upconv(x)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """
    UNet for semantic segmentation
    Input : (B, 3, H, W)
    Output: (B, num_classes, H, W)  (LOGITS)
    """
    def __init__(self, in_channels=3, num_classes=3):
        super().__init__()

        # Encoder
        self.down1 = DownBlock(in_channels, 64)
        self.down2 = DownBlock(64, 128)
        self.down3 = DownBlock(128, 256)
        self.down4 = DownBlock(256, 512)

        # Bottleneck
        self.bottleneck = DoubleConv(512, 1024)

        # Decoder
        self.up1 = UpBlock(1024, 512)
        self.up2 = UpBlock(512, 256)
        self.up3 = UpBlock(256, 128)
        self.up4 = UpBlock(128, 64)

        # Classifier
        self.head = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        f1, d1 = self.down1(x)
        f2, d2 = self.down2(d1)
        f3, d3 = self.down3(d2)
        f4, d4 = self.down4(d3)

        b = self.bottleneck(d4)

        u1 = self.up1(b, f4)
        u2 = self.up2(u1, f3)
        u3 = self.up3(u2, f2)
        u4 = self.up4(u3, f1)

        return self.head(u4)  # logits (NO softmax)
