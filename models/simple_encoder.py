import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleEncoderDecoder(nn.Module):
    """
    A minimal encoder-decoder CNN for segmentation.
    Used as a baseline before UNet.
    """
    def __init__(self, in_channels=3, out_channels=1, base_channels=32):
        super(SimpleEncoderDecoder, self).__init__()

        # --- Encoder ---
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool2d(2, 2)

        self.enc2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 2, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool2d(2, 2)

        # --- Bottleneck ---
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 4, 3, padding=1),
            nn.ReLU(inplace=True)
        )

        # --- Decoder ---
        self.up1 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 2, 3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.up2 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )

        # --- Output layer ---
        self.final = nn.Conv2d(base_channels, out_channels, 1)

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.pool1(x1)
        x3 = self.enc2(x2)
        x4 = self.pool2(x3)
        x5 = self.bottleneck(x4)
        x6 = self.up1(x5)
        x7 = self.dec1(x6)
        x8 = self.up2(x7)
        x9 = self.dec2(x8)
        out = self.final(x9)
        return torch.sigmoid(out)
