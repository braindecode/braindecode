# Authors: Bruno Aristimunha <b.aristimunha@gmail.com>
#
# License: BSD (3-clause)
"""Focused behavior tests for the emg2pose baseline models.

The generic integration suite covers the braindecode model contract. These
tests pin pose-specific behavior and the released-checkpoint compatibility
surface while using reduced model widths and windows where possible.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
import torch
from torch import nn

from braindecode.models.neuropose import NeuroPose
from braindecode.models.sensingdynamics import SensingDynamics
from braindecode.models.vemg2pose import VEMG2Pose


def _small_neuropose(**overrides: Any) -> NeuroPose:
    kwargs: dict[str, Any] = dict(
        n_chans=4,
        n_outputs=3,
        n_times=64,
        sfreq=64.0,
        encoder_channels=(4,),
        encoder_pool_sizes=((2, 2),),
        decoder_channels=(1,),
        decoder_upsample_sizes=((2, 2),),
        n_res_blocks=1,
        n_convs_per_block=1,
    )
    kwargs.update(overrides)
    return NeuroPose(**kwargs)


def _small_vemg2pose(**overrides: Any) -> VEMG2Pose:
    kwargs: dict[str, Any] = dict(
        n_chans=4,
        n_outputs=3,
        n_times=64,
        sfreq=64.0,
        encoder_channels=8,
        feature_dim=4,
        hidden_size=4,
        lstm_layers=1,
        stem_kernel_sizes=(3,),
        stem_strides=(2,),
        tds_subsample_kernels=(3,),
        tds_subsample_strides=(2,),
        tds_kernel_widths=(3,),
        tds_blocks=1,
        tds_channels=2,
        rollout_rate=8.0,
        num_position_steps=4,
    )
    kwargs.update(overrides)
    return VEMG2Pose(**kwargs)


def _small_sensingdynamics(**overrides: Any) -> SensingDynamics:
    kwargs: dict[str, Any] = dict(
        n_chans=16,
        n_outputs=3,
        n_times=192,
        sfreq=2000.0,
        temporal_channels=4,
        mid_channels=2,
        spatial_channels=4,
        mlp_hidden=8,
    )
    kwargs.update(overrides)
    return SensingDynamics(**kwargs)


@pytest.mark.parametrize(
    ("factory", "input_shape"),
    [
        (_small_neuropose, (2, 4, 64)),
        (_small_vemg2pose, (2, 4, 64)),
        (_small_sensingdynamics, (2, 16, 192)),
    ],
    ids=("neuropose", "vemg2pose", "sensingdynamics"),
)
def test_pose_models_return_a_sequence(
    factory: Callable[[], nn.Module], input_shape: tuple[int, int, int]
):
    model = factory().eval()
    output = model(torch.randn(input_shape))

    n_times = input_shape[2] - getattr(model, "left_context", 0)
    assert output.shape == (input_shape[0], n_times, 3)


@pytest.mark.parametrize(
    "factory",
    [_small_neuropose, _small_vemg2pose, _small_sensingdynamics],
    ids=("neuropose", "vemg2pose", "sensingdynamics"),
)
def test_pose_models_apply_centralized_initialization(factory):
    model = factory()

    for module in model.modules():
        if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            if module.bias is not None:
                torch.testing.assert_close(module.bias, torch.zeros_like(module.bias))
        elif isinstance(
            module,
            (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm),
        ):
            if module.weight is not None:
                torch.testing.assert_close(
                    module.weight, torch.ones_like(module.weight)
                )
            if module.bias is not None:
                torch.testing.assert_close(module.bias, torch.zeros_like(module.bias))
        elif isinstance(module, nn.LSTM):
            for name, parameter in module.named_parameters(recurse=False):
                if "bias" in name:
                    torch.testing.assert_close(parameter, torch.zeros_like(parameter))


def test_vemg2pose_tracking_anchor_changes_trajectory():
    model = _small_vemg2pose(parameterization="velocity").eval()
    x = torch.randn(2, 4, 64)
    zero_anchor = torch.zeros(2, 3)
    shifted_anchor = torch.full((2, 3), 3.0)

    output_zero = model.tracking_forward(x, zero_anchor)
    output_shifted = model.tracking_forward(x, shifted_anchor)

    assert not torch.allclose(output_zero, output_shifted)
    torch.testing.assert_close(output_zero, model(x, y0=zero_anchor))


def test_vemg2pose_tracking_forward_preserves_gradients():
    model = _small_vemg2pose(parameterization="velocity")
    output = model.tracking_forward(torch.randn(2, 4, 64), torch.zeros(2, 3))

    output.sum().backward()

    assert model.final_layer.weight.grad is not None


def test_vemg2pose_rejects_windows_without_valid_samples():
    model = _small_vemg2pose()

    with pytest.raises(ValueError, match="left context"):
        model(torch.randn(1, 4, model.left_context))


def test_vemg2pose_supports_a_single_rollout_step():
    model = _small_vemg2pose(rollout_rate=1.0)

    assert model(torch.randn(1, 4, 64)).shape == (1, 50, 3)


def test_vemg2pose_position_and_velocity_parameterizations_diverge():
    torch.manual_seed(7)
    position = _small_vemg2pose(parameterization="position").eval()
    velocity = _small_vemg2pose(parameterization="velocity").eval()
    velocity.load_state_dict(position.state_dict())
    x = torch.randn(1, 4, 64)
    y0 = torch.ones(1, 3)

    with torch.no_grad():
        position_output = position(x, y0=y0)
        velocity_output = velocity(x, y0=y0)

    assert not torch.allclose(position_output, velocity_output)
    assert position.final_layer.out_features == 3
    assert velocity.final_layer.out_features == 3
    assert _small_vemg2pose(parameterization="hybrid").final_layer.out_features == 6


@pytest.mark.parametrize(
    "kwargs, message",
    [
        (
            {
                "tds_subsample_kernels": (),
                "tds_subsample_strides": (),
                "tds_kernel_widths": (),
            },
            "TDS stage",
        ),
        ({"tds_channels": 0}, "tds_channels"),
        (
            {"decoder": "mlp", "decoder_hidden_sizes": ()},
            "decoder_hidden_sizes",
        ),
    ],
    ids=("empty-tds-schedule", "zero-tds-channels", "empty-mlp-schedule"),
)
def test_vemg2pose_rejects_malformed_schedules(kwargs, message):
    with pytest.raises(ValueError, match=message):
        _small_vemg2pose(**kwargs)


def test_sensingdynamics_derives_custom_electrode_stride_geometry():
    model = _small_sensingdynamics(
        temporal_stride=(2, 8),
        electrode_wrap=8,
    ).eval()

    # Conv2d output geometry: 16 --stride 2--> 8, +16 circular padding,
    # -14 from the dilated mid kernel, -2 from the spatial kernel = 8.
    assert model.mlp[1].in_features == 4 * 8
    assert model(torch.randn(1, 16, 192)).shape == (1, 192, 3)


def test_sensingdynamics_batch_one_training():
    model = _small_sensingdynamics().train()

    assert model(torch.randn(1, 16, 192)).shape == (1, 192, 3)


@pytest.mark.parametrize(
    ("model_cls", "kwargs", "expected_count"),
    [
        (
            NeuroPose,
            dict(n_chans=16, n_outputs=20, n_times=4000, sfreq=2000.0),
            149,
        ),
        (
            VEMG2Pose,
            dict(n_chans=16, n_outputs=20, n_times=2000, sfreq=2000.0),
            68,
        ),
    ],
    ids=("neuropose", "vemg2pose"),
)
def test_published_checkpoint_mapping_is_complete(model_cls, kwargs, expected_count):
    model = model_cls(**kwargs)
    mapping = model.mapping
    state_keys = set(model.state_dict())

    assert len(mapping) == expected_count
    assert len(set(mapping)) == expected_count
    assert len(set(mapping.values())) == expected_count
    assert set(mapping.values()) <= state_keys


@pytest.mark.parametrize(
    ("factory", "backbone_key", "input_shape"),
    [
        (_small_neuropose, "encoder.0.network.0.weight", (1, 4, 64)),
        (_small_vemg2pose, "encoder.0.conv.0.weight", (1, 4, 64)),
        (_small_sensingdynamics, "conv1.conv.weight", (1, 16, 192)),
    ],
    ids=("neuropose", "vemg2pose", "sensingdynamics"),
)
def test_reset_head_preserves_backbone_and_updates_config(
    factory, backbone_key, input_shape
):
    model = factory().double()
    backbone_before = model.state_dict()[backbone_key].clone()

    model.reset_head(5)

    torch.testing.assert_close(model.state_dict()[backbone_key], backbone_before)
    assert {parameter.dtype for parameter in model.parameters()} == {torch.float64}
    assert model.n_outputs == 5
    assert model.get_config()["n_outputs"] == 5
    restored = type(model).from_config(model.get_config())
    assert restored.n_outputs == 5
    n_times = input_shape[2] - getattr(model, "left_context", 0)
    assert model(torch.randn(input_shape, dtype=torch.float64)).shape == (
        input_shape[0],
        n_times,
        5,
    )
