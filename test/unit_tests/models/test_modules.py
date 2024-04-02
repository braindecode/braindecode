# Authors: Hubert Banville <hubert.jbanville@gmail.com>
#
# License: BSD (3-clause)

import pytest

import torch
from torch import nn

from braindecode.models.tidnet import _BatchNormZG, _DenseSpatialFilter
from braindecode.models.modules import CombinedConv, MLP, TimeDistributed, DropPath
from braindecode.models.labram import _SegmentPatch

from braindecode.models.functions import drop_path


def test_time_distributed():
    n_channels = 4
    n_times = 100
    feat_size = 5
    n_windows = 4
    batch_size = 2
    X = torch.rand((batch_size, n_windows, n_channels, n_times))

    feat_extractor = nn.Sequential(
        nn.Flatten(start_dim=1), nn.Linear(n_channels * n_times, feat_size)
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
        in_chans, growth, depth, in_ch, bottleneck, drop_prob, activation, collapse
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
        in_chans, growth, depth, in_ch, bottleneck, drop_prob, activation, collapse
    )

    x = torch.rand(5, 3, 10)  # 3-dimensional input
    output = dense_spatial_filter(x)
    assert output.shape[:2] == torch.Size([5, 33])


@pytest.mark.parametrize(
    "bias_time,bias_spat", [(False, False), (False, True), (True, False), (True, True)]
)
def test_combined_conv(bias_time, bias_spat):
    batch_size = 64
    in_chans = 44
    timepoints = 1000

    data = torch.rand([batch_size, 1, timepoints, in_chans])
    conv = CombinedConv(in_chans=in_chans, bias_spat=bias_spat, bias_time=bias_time)

    combined_out = conv(data)
    sequential_out = conv.conv_spat(conv.conv_time(data))

    assert torch.isclose(combined_out, sequential_out, atol=1e-6).all()

    diff = combined_out - sequential_out
    assert ((diff**2).mean().sqrt() / sequential_out.std()) < 1e-5
    assert (diff.abs().median() / sequential_out.abs().median()) < 1e-5


@pytest.mark.parametrize("hidden_features", [None, (10, 10), (50, 50, 50), [10, 10, 10]])
def test_mlp_increase(hidden_features):
    model = MLP(in_features=40, hidden_features=hidden_features)
    if hidden_features is None:
        assert len(model) == 6
    else:
        # For each layer that we add, the model
        # increase with 2 layers + 2 initial layers (input, output layer)
        assert len(model) == 2 * (len(hidden_features)) + 2


def test_segm_patch_not_learning():
    n_chans = 64
    patch_size = 200
    embed_dim = 200
    n_segments = 5
    X = []
    for i in range(n_segments):
        if i % 2 == 0:
            X.append(torch.zeros((n_chans, patch_size)))
        else:
            X.append(torch.ones((n_chans, patch_size)))

    n_times = patch_size * n_segments
    X = torch.concat(X, dim=1)

    assert n_times == X.shape[-1]

    module = _SegmentPatch(
        n_times=n_times,
        n_chans=n_chans,
        patch_size=patch_size,
        emb_dim=embed_dim,
        learned_patcher=False,
    )

    with torch.no_grad():
        # Adding batch dimension
        X_split = module(X.unsqueeze(0))

        assert X_split.shape[1] == n_chans
        assert X_split.shape[2] == n_times // patch_size
        assert X_split.shape[3] == embed_dim

        assert torch.allclose(X_split[0, 0, 0].sum(), torch.zeros(1))
        assert torch.equal(X_split[0, 0, 1].unique(), torch.ones(1))


@pytest.fixture(scope="module")
def x_metainfo():
    return {
        "batch_size": 2,
        "n_chans": 64,
        "patch_size": 200,
        "n_segments": 5,
        "n_times": 1000,
        "X": torch.zeros((2, 64, 1000)),
    }


def test_segm_patch(x_metainfo):

    module = _SegmentPatch(
        n_times=x_metainfo["n_times"],
        n_chans=x_metainfo["n_chans"],
        patch_size=x_metainfo["patch_size"],
        emb_dim=x_metainfo["patch_size"],
    )

    with torch.no_grad():
        # Adding batch dimension
        X_split = module(x_metainfo["X"])
        assert X_split.shape[0] == x_metainfo["batch_size"]
        assert X_split.shape[1] == x_metainfo["n_chans"]
        assert X_split.shape[2] == x_metainfo["n_times"] // x_metainfo["patch_size"]
        assert X_split.shape[3] == x_metainfo["patch_size"]


def test_drop_path_prob_1(x_metainfo):
    """
    Test that the DropPath module sets the input tensor to zero.
    """

    module = DropPath(drop_prob=1)

    with torch.no_grad():
        # Adding batch dimension
        X_zeros = module(x_metainfo["X"])
        assert torch.allclose(X_zeros, torch.zeros_like(x_metainfo["X"]))


def test_drop_path_prob_0(x_metainfo):
    """
    Test that the DropPath module using prob equal 0
    """

    module = DropPath(drop_prob=0)

    with torch.no_grad():
        # Adding batch dimension
        X_original = module(x_metainfo["X"])
        assert torch.allclose(X_original, x_metainfo["X"])


def test_drop_path_representation():
    drop_prob = 0.5
    module = DropPath(drop_prob=0.5)
    expected_repr = f"p={drop_prob}"

    # Get the actual repr string
    actual_repr = repr(module)

    assert expected_repr in actual_repr


def test_drop_path_no_drop():
    x = torch.rand(3, 3)  # Example input tensor
    output = drop_path(x, drop_prob=0.0, training=False)
    assert torch.equal(
        output, x
    ), "Output should be equal to input when drop_prob is 0.0 or training is False."


def test_drop_path_with_dropout_shape():
    x = torch.rand(5, 4)  # Example input tensor
    output = drop_path(x, drop_prob=0.5, training=True)
    assert (
        output.shape == x.shape
    ), "Output tensor must have the same shape as the input tensor."


def test_drop_path_scale_by_keep():
    torch.manual_seed(0)
    x = torch.rand(1, 10)  # Single-dimension tensor for simplicity
    drop_prob = 0.2
    scaled_output = drop_path(x, drop_prob=drop_prob, training=True, scale_by_keep=True)
    unscaled_output = drop_path(
        x, drop_prob=drop_prob, training=True, scale_by_keep=False
    )
    # This test relies on statistical expectation and may need multiple runs or adjustments
    scale_factor = 1 / (1 - drop_prob)
    assert torch.allclose(
        scaled_output.mean(), unscaled_output.mean() * scale_factor, atol=0.1
    ), "Scaled output does not match expected scaling."


def test_drop_path_different_dimensions():
    x_2d = torch.rand(2, 2)  # 2D tensor
    x_3d = torch.rand(2, 2, 2)  # 3D tensor
    output_2d = drop_path(x_2d, drop_prob=0.5, training=True)
    output_3d = drop_path(x_3d, drop_prob=0.5, training=True)
    assert (
        output_2d.shape == x_2d.shape and output_3d.shape == x_3d.shape
    ), "Output tensor must maintain input shape across different dimensions."
