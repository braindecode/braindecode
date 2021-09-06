# Authors: Hubert Banville <hubert.jbanville@gmail.com>
#
# License: BSD (3-clause)

import torch
from torch import nn

from braindecode.models import TimeDistributed


def test_time_distributed():
    n_channels = 4
    n_times = 100
    feat_size = 5
    n_windows = 4
    batch_size = 2
    X = torch.rand((batch_size, n_windows, n_channels, n_times))

    feat_extractor = nn.Sequential(
        nn.Flatten(start_dim=1),
        nn.Linear(n_channels * n_times, feat_size)
    )
    model = TimeDistributed(feat_extractor)

    out = model(X)
    out2 = [model(X[:, [i]]) for i in range(X.shape[1])]
    out2 = torch.stack(out2, dim=1).flatten(start_dim=2)

    assert out.shape == (batch_size, n_windows, feat_size)
    assert torch.allclose(out, out2)
