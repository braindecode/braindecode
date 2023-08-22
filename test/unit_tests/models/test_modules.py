# Authors: Hubert Banville <hubert.jbanville@gmail.com>
#
# License: BSD (3-clause)

import torch
from torch import nn

from braindecode.models import TimeDistributed
from braindecode.models.tidnet import _BatchNormZG, _DenseSpatialFilter


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


def test_reset_parameters():
    num_channels = 3

    bn = _BatchNormZG(num_channels)
    bn.reset_parameters()

    # Check running stats
    assert bn.running_mean.size(0) == num_channels
    assert torch.allclose(bn.running_mean, torch.zeros(num_channels))

    assert bn.running_var.size(0) == num_channels
    assert torch.allclose(bn.running_var, torch.ones(num_channels))

    # Check weight and bias
    assert bn.weight.size(0) == num_channels
    assert torch.allclose(bn.weight, torch.zeros(num_channels))

    assert bn.bias.size(0) == num_channels
    assert torch.allclose(bn.bias, torch.zeros(num_channels))


def test_dense_spatial_filter_forward_collapse_true():
    in_chans = 3
    growth = 8
    depth = 4
    in_ch = 1
    bottleneck = 4
    drop_prob = 0.0
    activation = torch.nn.LeakyReLU
    collapse = True

    dense_spatial_filter = _DenseSpatialFilter(
        in_chans, growth, depth, in_ch, bottleneck, drop_prob, activation,
        collapse
    )

    x = torch.rand(5, 3, 10)  # 3-dimensional input
    output = dense_spatial_filter(x)
    assert output.shape[:2] == torch.Size([5, 33])


def test_dense_spatial_filter_forward_collapse_false():
    in_chans = 3
    growth = 8
    depth = 4
    in_ch = 1
    bottleneck = 4
    drop_prob = 0.0
    activation = torch.nn.LeakyReLU
    collapse = False

    dense_spatial_filter = _DenseSpatialFilter(
        in_chans, growth, depth, in_ch, bottleneck, drop_prob, activation,
        collapse
    )

    x = torch.rand(5, 3, 10)  # 3-dimensional input
    output = dense_spatial_filter(x)
    assert output.shape[:2] == torch.Size([5, 33])
