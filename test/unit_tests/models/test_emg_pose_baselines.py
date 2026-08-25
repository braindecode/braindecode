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


def test_neuropose_rejects_input_shorter_than_decimation():
    model = NeuroPoseNet(
        n_chans=16,
        n_outputs=20,
        n_times=10000,
        sfreq=2000.0,
        base_channels=8,
        encoder_dim=32,
        n_res_blocks=1,
    )
    with pytest.raises(ValueError, match="at least 10 time samples"):
        model(torch.randn(1, 16, 9))


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


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(n_chans=16, n_grids=1),
        dict(n_chans=20, n_grids=5, elec_per_grid=4, temporal_channels=16),
    ],
)
def test_sensingdynamics_grid_geometries(kwargs):
    model = SensingDynamicsNet(
        n_outputs=20,
        n_times=480,
        sfreq=2048.0,
        input_window_seconds=480 / 2048.0,
        mlp_hidden=64,
        **kwargs,
    )
    x = torch.randn(2, kwargs["n_chans"], 480)
    assert model(x).shape == (2, 480, 20)


def test_sensingdynamics_rejects_bad_grid():
    with pytest.raises(ValueError, match="n_chans must equal"):
        SensingDynamicsNet(n_chans=17, n_grids=5, n_outputs=20)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"n_grids": 0}, "n_grids must be positive"),
        ({"n_grids": 1, "elec_per_grid": 0}, "elec_per_grid must be positive"),
    ],
)
def test_sensingdynamics_rejects_nonpositive_grid_geometry(kwargs, message):
    with pytest.raises(ValueError, match=message):
        SensingDynamicsNet(n_chans=16, n_outputs=20, **kwargs)


def test_sensingdynamics_rejects_nonpositive_frame_count():
    with pytest.raises(ValueError, match="n_frames must be positive"):
        SensingDynamicsNet(n_chans=16, n_outputs=20, n_frames=0)


def test_sensingdynamics_uses_circular_grid_padding(monkeypatch):
    model = SensingDynamicsNet(
        n_chans=16,
        n_outputs=20,
        n_times=480,
        sfreq=2048.0,
        input_window_seconds=480 / 2048.0,
        temporal_channels=16,
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
                sfreq=2048.0,
                input_window_seconds=480 / 2048.0,
                temporal_channels=16,
                mlp_hidden=64,
            ),
            "stem.0.weight",
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
        sfreq=2048.0,
        input_window_seconds=480 / 2048.0,
        temporal_channels=16,
        mlp_hidden=64,
        activation=nn.PReLU,
    )

    stem_activations = [
        module for module in vemg2pose.stem.modules() if isinstance(module, nn.PReLU)
    ]
    assert len(stem_activations) == 2
    assert stem_activations[0] is not stem_activations[1]
    assert len([m for m in sensingdynamics.modules() if isinstance(m, nn.PReLU)]) == 4


def test_all_three_are_non_classification_models():
    from braindecode.models.util import non_classification_models

    for name in ("VEMG2PoseNet", "NeuroPoseNet", "SensingDynamicsNet"):
        assert name in non_classification_models
