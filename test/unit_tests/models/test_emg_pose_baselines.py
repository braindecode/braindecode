# Authors: Bruno Aristimunha <b.aristimunha@gmail.com>
#
# License: BSD (3-clause)
"""Behavior tests for the emg2pose baseline trio.

The parametrized integration suite (test_integration.py) already covers
the braindecode contract; these tests pin the pose-specific semantics:
sequence output shapes, position-vs-velocity parameterization, the
Tracking state anchor, and grid-geometry handling for
SensingDynamicsNet.
"""

import pytest
import torch
from torch import nn

from braindecode.models.base import HAS_HF_HUB
from braindecode.models.neuropose import NeuroPoseNet
from braindecode.models.sensingdynamics import SensingDynamicsNet
from braindecode.models.vemg2pose import VEMG2PoseNet


@pytest.fixture()
def emg_window():
    return torch.randn(2, 16, 10000)


@pytest.fixture()
def vemg2pose():
    return VEMG2PoseNet(
        n_chans=16,
        n_outputs=20,
        n_times=10000,
        sfreq=2000.0,
        input_window_seconds=5.0,
        hidden_size=64,  # shrink for test speed
    )


def test_vemg2pose_sequence_output(vemg2pose, emg_window):
    out = vemg2pose(emg_window)
    assert out.shape == (2, 10000, 20)


def test_vemg2pose_position_vs_velocity_diverge(emg_window):
    pos = VEMG2PoseNet(
        n_chans=16,
        n_outputs=20,
        n_times=10000,
        sfreq=2000.0,
        input_window_seconds=5.0,
        hidden_size=32,
        parameterization="position",
    )
    vel = VEMG2PoseNet(
        n_chans=16,
        n_outputs=20,
        n_times=10000,
        sfreq=2000.0,
        input_window_seconds=5.0,
        hidden_size=32,
        parameterization="velocity",
    )
    vel.load_state_dict(pos.state_dict())
    assert not torch.allclose(pos(emg_window), vel(emg_window))


def test_vemg2pose_encoder_is_causal():
    model = VEMG2PoseNet(
        n_chans=16,
        n_outputs=20,
        n_times=10000,
        sfreq=2000.0,
        hidden_size=16,
        encoder_channels=16,
        feature_dim=16,
        tds_blocks=1,
    ).eval()
    prefix = torch.randn(1, 16, 5000)
    full = torch.cat([prefix, torch.randn(1, 16, 5000)], dim=-1)

    prefix_features = model.tds_stages(model.stem(prefix))
    full_features = model.tds_stages(model.stem(full))

    torch.testing.assert_close(
        prefix_features, full_features[..., : prefix_features.shape[-1]]
    )
    torch.testing.assert_close(model(prefix), model(full)[:, : prefix.shape[-1]])


def test_vemg2pose_tracking_anchor_changes_trajectory(vemg2pose, emg_window):
    y_a = torch.zeros(2, 20)
    y_b = torch.full((2, 20), 3.0)
    out_a = vemg2pose(emg_window, y0=y_a)
    out_b = vemg2pose(emg_window, y0=y_b)
    # Different anchors must yield different rollouts at least early on.
    assert not torch.allclose(out_a[:, :500], out_b[:, :500])


def test_vemg2pose_tracking_forward_preserves_gradients(vemg2pose, emg_window):
    output = vemg2pose.tracking_forward(emg_window, torch.zeros(2, 20))
    output.sum().backward()

    assert vemg2pose.final_layer.weight.grad is not None


def test_vemg2pose_rolls_out_at_decoder_rate():
    model = VEMG2PoseNet(
        n_chans=16,
        n_outputs=20,
        n_times=10000,
        sfreq=2000.0,
        input_window_seconds=5.0,
        hidden_size=16,
        encoder_channels=16,
        feature_dim=16,
        tds_blocks=1,
    )
    rollout_steps = 0

    def count_step(_module, _args, _output):
        nonlocal rollout_steps
        rollout_steps += 1

    handle = model.lstm.register_forward_hook(count_step)
    try:
        model(torch.randn(1, 16, 10000))
    finally:
        handle.remove()

    assert rollout_steps == 250


def test_vemg2pose_invalid_parameterization():
    with pytest.raises(ValueError, match="parameterization"):
        VEMG2PoseNet(parameterization="quaternion")


def test_vemg2pose_rejects_nonpositive_decoder_rate():
    with pytest.raises(ValueError, match="decoder_rate must be positive"):
        VEMG2PoseNet(decoder_rate=0)


def test_vemg2pose_rejects_nonpositive_sampling_rate():
    with pytest.raises(ValueError, match="sfreq must be positive"):
        VEMG2PoseNet(sfreq=0)


@pytest.mark.parametrize(
    "model",
    [
        VEMG2PoseNet(n_chans=16, n_outputs=20, n_times=11790, sfreq=2000),
        NeuroPoseNet(n_chans=16, n_outputs=20, n_times=10000, sfreq=2000),
    ],
    ids=["vemg2pose", "neuropose"],
)
def test_published_checkpoint_mapping_targets(model):
    model_state = model.state_dict()
    mapped_state = {
        old_key: torch.full_like(model_state[new_key], 0.125)
        for old_key, new_key in model.mapping.items()
    }

    incompatible = model.load_state_dict(mapped_state, strict=False)

    assert not incompatible.unexpected_keys
    loaded_state = model.state_dict()
    for new_key in model.mapping.values():
        torch.testing.assert_close(
            loaded_state[new_key], torch.full_like(loaded_state[new_key], 0.125)
        )


def test_neuropose_sequence_and_reset_head(emg_window):
    model = NeuroPoseNet(
        n_chans=16,
        n_outputs=20,
        n_times=10000,
        sfreq=2000.0,
        input_window_seconds=5.0,
        base_channels=8,
        encoder_dim=32,
        n_res_blocks=1,
    )
    assert model(emg_window).shape == (2, 10000, 20)
    model.reset_head(22)
    assert model(emg_window[:1]).shape == (1, 10000, 22)


def test_neuropose_rejects_input_too_short_for_encoder():
    model = NeuroPoseNet(
        n_chans=16,
        n_outputs=20,
        n_times=10000,
        sfreq=2000.0,
        base_channels=8,
        encoder_dim=32,
        n_res_blocks=1,
    )
    with pytest.raises(ValueError, match="at least 400 time samples"):
        model(torch.randn(1, 16, 399))


def test_neuropose_rejects_too_few_bands():
    with pytest.raises(ValueError, match="n_bands must be at least 7"):
        NeuroPoseNet(
            n_chans=16,
            n_outputs=20,
            n_times=10000,
            sfreq=2000.0,
            n_bands=6,
        )


def test_neuropose_rejects_short_noninteger_resampling():
    model = NeuroPoseNet(
        n_chans=8,
        n_outputs=20,
        n_times=98,
        sfreq=500.0,
        internal_sfreq=200.0,
    )
    with pytest.raises(ValueError, match="requires at least 40"):
        model(torch.randn(1, 8, 98))


def test_neuropose_resamples_noninteger_sampling_ratio():
    model = NeuroPoseNet(
        n_chans=8,
        n_outputs=20,
        n_times=500,
        sfreq=500.0,
        internal_sfreq=200.0,
        base_channels=8,
        encoder_dim=32,
        n_res_blocks=1,
    )
    internal_times = None

    def capture_shape(_module, _args, output):
        nonlocal internal_times
        internal_times = output.shape[-2]

    handle = model.input_to_encoder.register_forward_hook(capture_shape)
    try:
        output = model(torch.randn(1, 8, 500))
    finally:
        handle.remove()

    assert internal_times == 200
    assert output.shape == (1, 500, 20)


def test_neuropose_predicts_a_time_varying_sequence(emg_window):
    model = NeuroPoseNet(
        n_chans=16,
        n_outputs=20,
        n_times=10000,
        sfreq=2000.0,
        input_window_seconds=5.0,
        base_channels=8,
        encoder_dim=32,
        n_res_blocks=1,
    ).eval()

    with torch.no_grad():
        output = model(emg_window)

    temporal_range = (output.amax(dim=1) - output.amin(dim=1)).abs().max()
    assert temporal_range > output.abs().max() * 1e-3


def test_neuropose_decoder_restores_paper_output_grid():
    model = NeuroPoseNet(
        n_chans=16,
        n_outputs=20,
        n_times=10000,
        sfreq=2000.0,
        input_window_seconds=5.0,
        base_channels=8,
        encoder_dim=32,
        n_res_blocks=1,
    )
    decoder_shape = None

    def capture_shape(_module, _args, output):
        nonlocal decoder_shape
        decoder_shape = output.shape

    handle = model.decoder.register_forward_hook(capture_shape)
    try:
        model(torch.randn(1, 16, 10000))
    finally:
        handle.remove()

    assert decoder_shape == (1, 8, 1000, 16)


def test_neuropose_supports_a_wider_internal_band_adapter():
    model = NeuroPoseNet(
        n_chans=16,
        n_outputs=20,
        n_times=10000,
        sfreq=2000.0,
        input_window_seconds=5.0,
        n_bands=16,
        base_channels=8,
        encoder_dim=32,
        n_res_blocks=1,
    )

    assert model(torch.randn(1, 16, 10000)).shape == (1, 10000, 20)


def test_sensingdynamics_sequence_shape():
    model = SensingDynamicsNet(
        n_chans=16,
        n_outputs=20,
        n_times=480,
        sfreq=2000.0,
        input_window_seconds=480 / 2000.0,
        temporal_channels=16,
        mid_channels=8,
        spatial_channels=16,
        mlp_hidden=64,
    )
    x = torch.randn(2, 16, 480)
    assert model(x).shape == (2, 480, 20)


def test_sensingdynamics_matches_derived_adaptation_geometry():
    model = SensingDynamicsNet(n_chans=16, n_outputs=20, n_times=480, sfreq=2000.0)

    assert model.conv1.conv.kernel_size == (1, 31)
    assert model.conv1.conv.stride == (1, 8)
    assert model.circular_pad.padding == 4
    assert model.conv2.conv.kernel_size == (8, 18)
    assert model.conv2.conv.dilation == (2, 1)
    assert model.conv3.conv.kernel_size == (3, 1)
    assert model.mlp[1].in_features == 64 * 8
    assert model.receptive_field_samples == 167
    assert model.benchmark_window_samples == 10_167


def test_sensingdynamics_rejects_non_emg2pose_channel_geometry():
    with pytest.raises(ValueError, match="16-channel emg2pose adaptation"):
        SensingDynamicsNet(n_chans=20, n_outputs=20, sfreq=2000.0)


def test_sensingdynamics_rejects_short_input():
    model = SensingDynamicsNet(n_chans=16, n_outputs=20, sfreq=2000.0)
    with pytest.raises(ValueError, match="at least 167 input samples"):
        model(torch.randn(1, 16, 166))


def test_sensingdynamics_uses_circular_grid_padding(monkeypatch):
    model = SensingDynamicsNet(
        n_chans=16,
        n_outputs=20,
        n_times=480,
        sfreq=2000.0,
        input_window_seconds=480 / 2000.0,
        temporal_channels=16,
        mid_channels=8,
        spatial_channels=16,
        mlp_hidden=64,
    )
    modes = []
    original_pad = torch.nn.functional.pad

    def record_pad(input, pad, mode="constant", value=None):
        modes.append(mode)
        if value is None:
            return original_pad(input, pad, mode=mode)
        return original_pad(input, pad, mode=mode, value=value)

    monkeypatch.setattr("braindecode.models.sensingdynamics.F.pad", record_pad)
    model(torch.randn(1, 16, 480))

    assert modes == ["circular"]


def test_sensingdynamics_stacks_raw_and_lowpass_planes():
    model = SensingDynamicsNet(
        n_chans=16,
        n_outputs=20,
        n_times=480,
        sfreq=2000.0,
        temporal_channels=16,
        mid_channels=8,
        spatial_channels=16,
        mlp_hidden=64,
    ).eval()
    captured = {}

    def capture_input(_module, args):
        captured["x"] = args[0].detach()

    handle = model.conv1.register_forward_pre_hook(capture_input)
    x = torch.randn(1, 16, 480)
    model(x)
    handle.remove()

    torch.testing.assert_close(captured["x"][:, 0], x)
    assert not torch.equal(captured["x"][:, 1], x)


def test_sensingdynamics_accepts_precomputed_lowpass():
    model = SensingDynamicsNet(
        n_chans=16,
        n_outputs=20,
        n_times=480,
        sfreq=2000.0,
        temporal_channels=16,
        mid_channels=8,
        spatial_channels=16,
        mlp_hidden=64,
    ).eval()
    x = torch.randn(1, 16, 480)
    x_lowpass = model.lowpass(x)

    with torch.no_grad():
        inferred = model(x)
        precomputed = model(x, x_lowpass=x_lowpass)

    torch.testing.assert_close(inferred, precomputed)


def test_sensingdynamics_rejects_bad_precomputed_lowpass_shape():
    model = SensingDynamicsNet(n_chans=16, n_outputs=20, sfreq=2000.0)
    with pytest.raises(ValueError, match="same shape"):
        model(torch.randn(1, 16, 480), x_lowpass=torch.randn(1, 16, 479))


@pytest.mark.parametrize(
    ("model_cls", "kwargs", "backbone_key"),
    [
        (
            VEMG2PoseNet,
            dict(
                n_chans=16,
                n_outputs=20,
                n_times=1000,
                sfreq=2000.0,
                input_window_seconds=0.5,
                hidden_size=16,
                encoder_channels=16,
                feature_dim=16,
                tds_blocks=1,
            ),
            "stem.1.weight",
        ),
        (
            NeuroPoseNet,
            dict(
                n_chans=16,
                n_outputs=20,
                n_times=10000,
                sfreq=2000.0,
                input_window_seconds=5.0,
                base_channels=8,
                encoder_dim=32,
                n_res_blocks=1,
            ),
            "encoder.0.conv.weight",
        ),
        (
            SensingDynamicsNet,
            dict(
                n_chans=16,
                n_outputs=20,
                n_times=480,
                sfreq=2000.0,
                input_window_seconds=480 / 2000.0,
                temporal_channels=16,
                mid_channels=8,
                spatial_channels=16,
                mlp_hidden=64,
            ),
            "conv1.conv.weight",
        ),
    ],
)
def test_reset_head_preserves_backbone_dtype_and_config(
    model_cls, kwargs, backbone_key
):
    model = model_cls(**kwargs).double()
    backbone_before = model.state_dict()[backbone_key].clone()

    model.reset_head(7)

    torch.testing.assert_close(model.state_dict()[backbone_key], backbone_before)
    assert {parameter.dtype for parameter in model.parameters()} == {torch.float64}
    assert model.n_outputs == 7
    assert model.get_config()["n_outputs"] == 7
    assert model_cls.from_config(model.get_config()).n_outputs == 7


def test_stateful_activations_are_not_shared_between_layers():
    vemg2pose = VEMG2PoseNet(
        n_chans=16,
        n_outputs=20,
        n_times=1000,
        sfreq=2000.0,
        input_window_seconds=0.5,
        hidden_size=16,
        encoder_channels=16,
        feature_dim=16,
        tds_blocks=1,
        activation=nn.PReLU,
    )
    sensingdynamics = SensingDynamicsNet(
        n_chans=16,
        n_outputs=20,
        n_times=480,
        sfreq=2000.0,
        input_window_seconds=480 / 2000.0,
        temporal_channels=16,
        mid_channels=8,
        spatial_channels=16,
        mlp_hidden=64,
    )

    stem_activations = [
        module for module in vemg2pose.stem.modules() if isinstance(module, nn.PReLU)
    ]
    assert len(stem_activations) == 2
    assert stem_activations[0] is not stem_activations[1]
    smu_layers = [
        module
        for module in sensingdynamics.modules()
        if module.__class__.__name__ == "_SMU"
    ]
    assert len(smu_layers) == 5
    assert len({id(module.mu) for module in smu_layers}) == 5


def test_all_three_are_non_classification_models():
    from braindecode.models.util import non_classification_models

    for name in ("VEMG2PoseNet", "NeuroPoseNet", "SensingDynamicsNet"):
        assert name in non_classification_models


@pytest.mark.skipif(not HAS_HF_HUB, reason="huggingface_hub is not installed")
def test_vemg2pose_hub_license_matches_source_license():
    assert VEMG2PoseNet._hub_mixin_info.model_card_data["license"] == "cc-by-nc-sa-4.0"
