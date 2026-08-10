# Authors: Bruno Aristimunha <b.aristimunha@gmail.com>
#
# License: BSD-3

from __future__ import annotations

import numpy as np
import pytest
import torch

from braindecode.models import ZUNA

# Three channels with distinct scalp coordinates (metres).
_POSITIONS = torch.tensor([[0.0, 0.0, 0.0], [0.01, 0.02, 0.03], [-0.01, 0.02, 0.03]])

# Small architecture override so the tests stay fast; the defaults build the
# full 1024-dim, 16-layer pretrained encoder.
_SMALL_KWARGS = dict(
    dim=16,
    n_layers=1,
    n_heads=2,
    head_dim=8,
    fine_time_pts=32,
    latent_dim=12,
    max_seqlen=256,
    rope_theta=10000.0,
    rope_dim=4,
)


@pytest.fixture
def small_zuna():
    return ZUNA(n_chans=3, n_outputs=2, n_times=1280, **_SMALL_KWARGS)


def _chs_info(locs):
    return [
        {"ch_name": name, "loc": np.asarray(loc)}
        for name, loc in zip("ABC", locs, strict=True)
    ]


def test_forward_returns_logits(small_zuna):
    out = small_zuna(torch.randn(2, 3, 1280), channel_positions=_POSITIONS)
    assert out.shape == (2, 2)
    assert small_zuna.get_output_shape() == (1, 2)
    assert small_zuna.sfreq == 256.0
    assert small_zuna.input_window_seconds == 5.0


def test_forward_return_features_dict(small_zuna):
    out = small_zuna(
        torch.randn(2, 3, 1280),
        channel_positions=_POSITIONS,
        return_features=True,
    )
    assert set(out) == {"features", "cls_token", "token_latents", "structured_latents"}
    assert out["cls_token"] is None
    assert out["features"].shape == (2, 3, 12)
    assert out["token_latents"].shape == (2, 120, 12)
    assert out["structured_latents"].shape == (2, 3, 40, 12)
    # features are the per-channel mean over coarse time.
    torch.testing.assert_close(out["features"], out["structured_latents"].mean(dim=2))


def test_uses_configured_activation():
    model = ZUNA(
        n_chans=3,
        n_outputs=2,
        n_times=1280,
        activation=torch.nn.GELU,
        **_SMALL_KWARGS,
    )
    assert isinstance(
        model.encoder.layers[0].feed_forward.activation, torch.nn.GELU
    )


@pytest.mark.parametrize("n_times, coarse_time", [(128, 4), (7680, 240)])
def test_forward_accepts_variable_zuna11_lengths(n_times, coarse_time):
    model = ZUNA(n_chans=3, n_outputs=2, **_SMALL_KWARGS)
    out = model(
        torch.randn(1, 3, n_times),
        channel_positions=_POSITIONS,
        return_features=True,
    )
    assert out["token_latents"].shape == (1, 3 * coarse_time, 12)
    assert out["structured_latents"].shape == (1, 3, coarse_time, 12)


def test_batched_forward_matches_per_sample(small_zuna):
    model = small_zuna.eval()
    x = torch.randn(3, 3, 1280)
    with torch.no_grad():
        batched = model(x, channel_positions=_POSITIONS)
        per_sample = torch.cat(
            [model(x[i : i + 1], channel_positions=_POSITIONS) for i in range(3)],
            dim=0,
        )
    torch.testing.assert_close(batched, per_sample)


def test_resolves_standard_montage_channel_names(small_zuna):
    out = small_zuna(
        torch.randn(1, 3, 1280),
        channel_names=["Fz", "Cz", "Pz"],
        return_features=True,
    )
    assert out["structured_latents"].shape == (1, 3, 40, 12)


def test_uses_chs_info_positions():
    model = ZUNA(
        n_outputs=2,
        n_times=1280,
        chs_info=_chs_info([[0.0, 0.0, 0.0], [0.01, 0.02, 0.03], [-0.01, 0.02, 0.03]]),
        **_SMALL_KWARGS,
    )
    out = model(torch.randn(1, 3, 1280), return_features=True)
    assert out["structured_latents"].shape == (1, 3, 40, 12)


def test_non_finite_chs_info_falls_back_to_channel_names():
    """NaN coordinates must not silently bucket into an arbitrary position."""
    model = ZUNA(
        n_chans=3,
        n_outputs=2,
        n_times=1280,
        chs_info=_chs_info([[np.nan] * 3, [0.01, 0.02, 0.03], [-0.01, 0.02, 0.03]]),
        **_SMALL_KWARGS,
    )
    assert model._cached_positions is None
    with pytest.raises(ValueError, match="requires channel coordinates or names"):
        model(torch.randn(1, 3, 1280))
    out = model(torch.randn(1, 3, 1280), channel_names=["Fz", "Cz", "Pz"])
    assert out.shape == (1, 2)


def test_variable_window_config_round_trip():
    model = ZUNA(
        n_chans=3,
        n_outputs=2,
        input_window_seconds=10.0,
        sfreq=256.0,
        **_SMALL_KWARGS,
    )
    assert model.n_times == 2560
    assert model.input_window_seconds == 10.0

    restored = ZUNA.from_config(model.get_config())
    assert restored.n_times == model.n_times
    assert restored.input_window_seconds == model.input_window_seconds


def test_requires_channel_positions_or_names(small_zuna):
    with pytest.raises(ValueError, match="ZUNA requires channel coordinates or names"):
        small_zuna(torch.randn(1, 3, 1280))


def test_requires_montage_when_names_only(small_zuna):
    with pytest.raises(ValueError, match="ZUNA requires a montage"):
        small_zuna(
            torch.randn(1, 3, 1280),
            channel_names=["Fz", "Cz", "Pz"],
            montage=None,
        )


@pytest.mark.parametrize(
    "bad_n_times, match",
    [
        (96, "0.375 seconds"),
        (7712, "30.125 seconds"),
        (1279, "divisible by fine_time_pts"),
    ],
)
def test_rejects_invalid_constructor_window(bad_n_times, match):
    with pytest.raises(ValueError, match=match):
        ZUNA(n_chans=3, n_outputs=2, n_times=bad_n_times)


def test_rejects_non_256_hz_inputs():
    with pytest.raises(ValueError, match="256 Hz"):
        ZUNA(n_chans=3, n_outputs=2, sfreq=250.0)


def test_rejects_unsupported_rope_dim():
    with pytest.raises(ValueError, match="rope_dim must be 4"):
        ZUNA(n_chans=3, n_outputs=2, n_times=1280, rope_dim=2)


def test_rejects_non_zuna_forward_length(small_zuna):
    with pytest.raises(ValueError, match="divisible by fine_time_pts"):
        small_zuna(torch.randn(1, 3, 1279), channel_positions=_POSITIONS)


def test_reset_head_replaces_final_layer(small_zuna):
    small_zuna.reset_head(5)
    out = small_zuna(torch.randn(2, 3, 1280), channel_positions=_POSITIONS)
    assert small_zuna.n_outputs == 5
    assert out.shape == (2, 5)


def _randomized_encoder_state(model):
    return {k: torch.randn_like(v) for k, v in model.encoder.state_dict().items()}


def test_load_state_dict_strips_upstream_prefix(small_zuna):
    randomized = _randomized_encoder_state(small_zuna)
    upstream = {f"model.encoder.{k}": v for k, v in randomized.items()}
    upstream["model.decoder.dummy"] = torch.zeros(1)

    small_zuna.load_state_dict(upstream)

    loaded = small_zuna.encoder.state_dict()
    for key, expected in randomized.items():
        torch.testing.assert_close(loaded[key], expected)


def test_load_state_dict_maps_zuna11_norm_keys(small_zuna):
    randomized = _randomized_encoder_state(small_zuna)
    norm_names = (
        "q_norm.weight",
        "k_norm.weight",
        "attention_norm.weight",
        "attention_norm_post.weight",
        "ffn_norm.weight",
        "ffn_norm_post.weight",
    )
    upstream = {}
    for key, value in randomized.items():
        upstream_key = f"model.encoder.{key}"
        if key == "norm.weight" or key.endswith(norm_names):
            upstream_key = f"{upstream_key.removesuffix('.weight')}.norm.weight"
        upstream[upstream_key] = value

    small_zuna.load_state_dict(upstream)

    loaded = small_zuna.encoder.state_dict()
    for key, expected in randomized.items():
        torch.testing.assert_close(loaded[key], expected)


def test_load_state_dict_rejects_unmatched_upstream_checkpoint(small_zuna):
    """A drifted upstream layout must fail loudly, not load random weights."""
    upstream = {"model.encoder.does.not.exist": torch.zeros(1)}
    with pytest.raises(ValueError, match="No upstream ZUNA keys matched"):
        small_zuna.load_state_dict(upstream)
