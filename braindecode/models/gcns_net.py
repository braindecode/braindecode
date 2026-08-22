from __future__ import annotations

import torch.nn as nn

from braindecode.models.base import EEGModuleMixin
from braindecode.models.dgcnn import _ChebyshevGraphConvolution


class GCNsNet(EEGModuleMixin, nn.Module):
    def __init__(self):
        pass

    def forward(self):
        pass