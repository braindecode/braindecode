# Authors: Robin Schirrmeister <robintibor@gmail.com>
#
# License: BSD (3-clause)

from torch import nn


class FakeConvModel(nn.Module):
    def __init__(self, in_chans, n_classes,):
        super().__init__()
        self.conv = nn.Conv1d(in_chans, n_classes, 9,)

    def forward(self, x):
        return self.conv(x)
