import torch
import torch.nn as nn
import numpy as np
from einops import rearrange
import torch.nn.functional as F

from braindecode.models.modules import (
    FilterBankLayer,
    LogVarLayer,
)

from braindecode.models.fblightconvnet import _LightweightConv1d


# Data shape: batch * filterBand * chan * time
class LightConvNet(nn.Module):
    def __init__(
        self,
        num_classes=4,
        num_samples=1000,
        num_channels=22,
        num_bands=9,
        embed_dim=32,
        win_len=250,
        num_heads=4,
        weight_softmax=True,
        bias=False,
    ):
        super().__init__()

        self.win_len = win_len

        self.spacial_block = nn.Sequential(
            nn.Conv2d(num_bands, embed_dim, (num_channels, 1)),
            nn.BatchNorm2d(embed_dim),
            nn.ELU(),
        )

        self.temporal_block = LogVarLayer(dim=3)

        self.conv = _LightweightConv1d(
            embed_dim,
            (num_samples // win_len),
            heads=num_heads,
            weight_softmax=weight_softmax,
            bias=bias,
        )

        self.classify = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        # torch.Size([1, 9, 22, 1000])
        out = self.spacial_block(x)
        out = out.reshape([*out.shape[0:2], -1, self.win_len])
        out = self.temporal_block(out)

        out = self.conv(out)

        out = out.view(out.size(0), -1)
        out = self.classify(out)

        return out


if __name__ == "__main__":
    x = torch.zeros(1, 9, 22, 1000)

    model = LightConvNet()
    print(model)
    with torch.no_grad():
        out = model(x)
