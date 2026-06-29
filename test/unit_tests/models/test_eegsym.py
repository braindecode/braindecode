import pytest
import torch
from torch import nn

from braindecode.models import EEGSym

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
