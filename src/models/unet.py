import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """
    Two consecutive (Conv2d -> BatchNorm -> ReLU) layers.
    This is the basic repeating building block used at every level
    of both the encoder and decoder in the U-Net.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    """
    Standard 2D U-Net for multi-class brain tumor segmentation.

    Input:  (batch, in_channels, H, W)  - e.g. 4 stacked MRI modalities
    Output: (batch, num_classes, H, W)  - per-pixel class scores

    Architecture: symmetric encoder-decoder with skip connections.
    The encoder progressively downsamples while extracting increasingly
    abstract features. The decoder progressively upsamples back to full
    resolution. Skip connections pass fine-grained spatial detail from
    each encoder level directly to the matching decoder level, which is
    what allows the model to produce sharp, precise segmentation
    boundaries instead of blurry ones.
    """

    def __init__(self, in_channels=4, num_classes=4):
        super().__init__()

        # ---- Encoder (contracting path) ----
        self.enc1 = DoubleConv(in_channels, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # ---- Bottleneck ----
        self.bottleneck = DoubleConv(512, 1024)

        # ---- Decoder (expanding path) ----
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(1024, 512)  # 1024 = 512 (upsampled) + 512 (skip)

        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(128, 64)

        # ---- Final output layer ----
        self.out_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # Bottleneck
        b = self.bottleneck(self.pool(e4))

        # Decoder with skip connections (concatenation)
        d4 = self.up4(b)
        d4 = torch.cat([d4, e4], dim=1)  # skip connection
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        return self.out_conv(d1)
