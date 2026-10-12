import pytest
import torch
from torch import nn

from braindecode.models import EEGSym
from braindecode.models.eegsym import _BatchNorm3d, _SpatialConv3d, _TemporalConv3d

_EEGSYM_8CH_CHS = [
    {"ch_name": ch} for ch in ["F3", "C3", "P3", "Cz", "Pz", "F4", "C4", "P4"]
]
_EEGSYM_8CH_LEFT_RIGHT_CHS = [("F3", "F4"), ("C3", "C4"), ("P3", "P4")]
_EEGSYM_8CH_MIDDLE_CHS = ["Cz", "Pz"]


def _make_reference_8ch_eegsym() -> EEGSym:
    return EEGSym(
        n_outputs=2,
        n_times=384,
        sfreq=128,
        chs_info=_EEGSYM_8CH_CHS,
        filters_per_branch=24,
        drop_prob=0.4,
        spatial_resnet_repetitions=1,
        left_right_chs=_EEGSYM_8CH_LEFT_RIGHT_CHS,
        middle_chs=_EEGSYM_8CH_MIDDLE_CHS,
    )


def _reference_state_numel(model: EEGSym) -> int:
    braindecode_runtime_buffers = {"left_idx", "right_idx", "middle_idx"}
    return sum(
        tensor.numel()
        for name, tensor in model.state_dict().items()
        if not name.endswith("num_batches_tracked")
        and name not in braindecode_runtime_buffers
    )


def test_eegsym_matches_reference_8ch_architecture():
    model = _make_reference_8ch_eegsym()

    assert model.scales_samples == [16, 32, 64]
    assert [
        conv[0].kernel_size[1] for conv in model.inception_block1.temporal_convs
    ] == [16, 32, 64]
    assert [
        conv[0].kernel_size[1] for conv in model.inception_block2.temporal_convs
    ] == [4, 8, 16]

    assert [block.temporal_conv[0].out_channels for block in model.residual_blocks] == [
        36,
        36,
        18,
    ]
    assert [block.projection[0].out_channels for block in model.residual_blocks] == [
        36,
        36,
        18,
    ]
    assert [block.spatial_convs[0][0].groups for block in model.residual_blocks] == [
        1,
        1,
        1,
    ]

    channel_merge_conv = model.channel_merging.grouped_conv[0]
    assert channel_merge_conv.groups == 9
    assert channel_merge_conv.in_channels // channel_merge_conv.groups == 2

    batch_norms = [
        module for module in model.modules() if isinstance(module, nn.BatchNorm3d)
    ]
    assert {batch_norm.eps for batch_norm in batch_norms} == {1e-3}
    assert {batch_norm.momentum for batch_norm in batch_norms} == {0.01}
    assert _reference_state_numel(model) == 144440


def test_eegsym_reference_8ch_forward_is_finite():
    model = _make_reference_8ch_eegsym()
    model.eval()

    with torch.no_grad():
        out = model(torch.randn(2, 8, 384))

    assert out.shape == (2, 2)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("filters_per_branch", [0, -8, 12])
def test_eegsym_filters_per_branch_must_be_positive_multiple_of_8(filters_per_branch):
    with pytest.raises(
        ValueError, match="filters_per_branch must be a positive multiple of 8"
    ):
        EEGSym(
            n_outputs=2,
            n_times=384,
            sfreq=128,
            chs_info=_EEGSYM_8CH_CHS,
            filters_per_branch=filters_per_branch,
            left_right_chs=_EEGSYM_8CH_LEFT_RIGHT_CHS,
            middle_chs=_EEGSYM_8CH_MIDDLE_CHS,
        )


def test_spatial_conv3d_matches_conv3d():
    """The (1, 1, k) spatial conv runs as conv2d on a view (Gaudi segfaults in the
    depthwise conv3d backward); same values as the Conv3d with the same weights."""
    conv = nn.Conv3d(24, 24, (1, 1, 5), groups=24, bias=False)
    spatial = _SpatialConv3d(24, 24, (1, 1, 5), groups=24, bias=False)
    spatial.load_state_dict(conv.state_dict())
    x = torch.randn(2, 24, 2, 50, 5)
    torch.testing.assert_close(spatial(x), conv(x))


@pytest.mark.parametrize(
    "kernel, padding, groups", [(5, "same", 1), (4, "same", 2), (7, (0, 0, 0), 2)]
)
def test_temporal_conv3d_time_last_matches_conv3d(kernel, padding, groups):
    # the conv1d form used on HPU against the Conv3d it replaces (run on CPU)
    torch.manual_seed(0)
    conv = _TemporalConv3d(4, 6, (1, kernel, 1), padding=padding, groups=groups)
    x = torch.randn(2, 4, 2, 37, 3)
    torch.testing.assert_close(conv._time_last(x), conv(x))


def test_batchnorm3d_matches_torch():
    # train-mode output, running statistics and counter, then eval output
    ref = nn.BatchNorm3d(4, eps=1e-3, momentum=0.01)
    bn = _BatchNorm3d(4, eps=1e-3, momentum=0.01)
    x = torch.randn(3, 4, 2, 7, 5)
    torch.testing.assert_close(bn(x), ref(x))
    for a, b in zip(bn.buffers(), ref.buffers()):
        torch.testing.assert_close(a, b)
    bn.eval()
    ref.eval()
    torch.testing.assert_close(bn(x), ref(x))
