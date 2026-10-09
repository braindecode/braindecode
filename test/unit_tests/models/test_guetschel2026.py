# Authors: Pierre Guetschel <pierre.guetschel@gmail.com>
#
# License: BSD-3
"""Tests of :class:`braindecode.models.Guetschel2026`.

The golden values below were computed with the original implementation
(https://github.com/PierreGtch/eeg-fm-masking, commit 38a089e, MIT) on the
inputs and weights built by the helpers of this file, and the braindecode model
was checked to give *exactly* the same tensors (``torch.equal``) when they were
generated. The tolerance only absorbs the BLAS differences of the CI machines.
"""

import inspect
import json
import math
import warnings
from urllib.error import URLError

import numpy as np
import pytest
import torch
from torch import nn

import braindecode.models.guetschel2026 as guetschel2026_module
from braindecode.models import Guetschel2026
from braindecode.models.guetschel2026 import _GaussianRandomProjection
from braindecode.models.signal_jepa import _pos_encode_time
from braindecode.modules.channels import _standard_positions

try:
    from safetensors.torch import load_file, save_file

    HAS_SAFETENSORS = True
except ImportError:  # pragma: no cover
    HAS_SAFETENSORS = False

try:
    import huggingface_hub  # noqa: F401

    HAS_HF_HUB = True
except ImportError:  # pragma: no cover
    HAS_HF_HUB = False

needs_safetensors = pytest.mark.skipif(
    not (HAS_SAFETENSORS and HAS_HF_HUB),
    reason="safetensors and huggingface_hub are required",
)

# ----------------------------------------------------------------------------
# Inputs shared with the golden-value generator
# ----------------------------------------------------------------------------

# standard_1020 positions in the MNE head frame (MNE 1.13.2), inline so that no
# montage lookup (renamed in MNE >= 1.13) is needed.
_POS = {
    "C3": (-0.0671, 0.0234, 0.1045),
    "Cz": (-0.0014, 0.0276, 0.1402),
    "C4": (0.0653, 0.0236, 0.1037),
    "Oz": (-0.0021, -0.0828, 0.0607),
}
_NAMES = ("C3", "Cz", "C4", "Oz")

_N_BACKBONE_PARAMS = 12_687_872
_EMBED_DIM = 512


def _chs(names=_NAMES):
    return [
        {"ch_name": n, "kind": "eeg", "loc": np.r_[_POS[n], np.zeros(9)]}
        for n in names
    ]


def _released_keys():
    """The 27 (key, shape) pairs of the released checkpoints, no head."""
    keys = {
        "feature_encoder.linear.weight": (512, 200),
        "feature_encoder.linear.bias": (512,),
        "model.pos_encoder.encoding_time": (33, 128),
    }
    for i in range(4):
        p = f"model.transformer.layers.{i}."
        keys[p + "self_attn.packed_proj.weight"] = (1536, 512)
        keys[p + "self_attn.out_proj.weight"] = (512, 512)
        keys[p + "linear1.weight"] = (2730, 512)
        keys[p + "linear2.weight"] = (512, 1365)
        keys[p + "norm1.weight"] = (512,)
        keys[p + "norm2.weight"] = (512,)
    return keys


_RELEASED_KEYS = _released_keys()


def _reference_state_dict(seed=0):
    """Deterministic random backbone weights, with the keys of the checkpoints."""
    g = torch.Generator().manual_seed(seed)
    sd = {}
    for key, shape in sorted(_RELEASED_KEYS.items()):
        noise = torch.randn(shape, generator=g)
        if key.endswith(("norm1.weight", "norm2.weight")):
            sd[key] = 1 + 0.1 * noise
        elif key.endswith("encoding_time"):
            # the formula + noise, so that a test can tell the buffer from the formula
            sd[key] = _pos_encode_time(33, 128, 666) + 1e-3 * noise
        else:
            sd[key] = 0.02 * noise
    return sd


def _reference_head(n_features, n_outputs, seed=1):
    g = torch.Generator().manual_seed(seed)
    return {
        "weight": 0.02 * torch.randn(n_outputs, n_features, generator=g),
        "bias": 0.02 * torch.randn(n_outputs, generator=g),
    }


def _windows(n_times, n_batch=2, dtype=torch.float32):
    """Closed-form windows in volts, ``(2, 4, n_times)``.

    Sample 0 of channel 0 is multiplied by 50, so that the clipping is exercised.
    """
    t = torch.arange(n_times, dtype=torch.float64)
    freqs = torch.tensor([7.0, 11.0, 13.0, 17.0], dtype=torch.float64)
    phases = torch.tensor([0.0, 1.0], dtype=torch.float64)[:n_batch]
    arg = (
        2 * math.pi * freqs[None, :, None] * t[None, None, :] / 200.0
        + phases[:, None, None]
    )
    x = 2e-5 * torch.sin(arg)
    x[:, 0, 0] *= 50
    return x.to(dtype)


def _synthetic_chs(n_chans, seed=0):
    """``n_chans`` electrodes on a 9 cm sphere, as inline positions (no montage lookup)."""
    g = np.random.default_rng(seed)
    xyz = g.normal(size=(n_chans, 3))
    xyz[:, 2] = np.abs(xyz[:, 2])
    xyz = 0.09 * xyz / np.linalg.norm(xyz, axis=1, keepdims=True)
    return [
        {"ch_name": f"E{i}", "kind": "eeg", "loc": np.r_[xyz[i], np.zeros(9)]}
        for i in range(n_chans)
    ]


def _model(names=_NAMES, n_times=1000, n_outputs=3, **kwargs):
    kwargs.setdefault("random_projection", 16)
    return Guetschel2026(
        chs_info=_chs(names), n_times=n_times, n_outputs=n_outputs, sfreq=200.0, **kwargs
    ).eval()


def _backbone_state(model):
    return {k: v for k, v in model.state_dict().items() if not k.startswith("final_layer.")}


# ----------------------------------------------------------------------------
# Golden values, computed with the original implementation (see module docstring)
# ----------------------------------------------------------------------------

# Patches kept in the golden values. At 6200 samples there are 34 patches: patch
# 32 is the last one of the stored time table, patch 33 comes from the formula.
_GOLDEN_PATCHES = {400: [0, 1], 6200: [0, 32, 33]}

# GOLDEN-BEGIN
_GOLDEN = {'features': {400: {'head': [[[-0.483565, -1.153964, 1.436575, -1.211162],
                              [-0.46402, -0.493362, 0.819583, -1.471726]],
                             [[-0.270506, -0.591355, 0.859832, -1.303077],
                              [-0.731, -0.545635, 0.813553, -0.997456]]],
                    'norm': [54.1079, 54.5672]},
              6200: {'head': [[[-0.723383, -1.100275, 1.283384, -1.023651],
                               [-1.168171, -0.868809, 0.717148, -0.457322],
                               [-0.875782, -1.218004, 1.219099, -0.770452]],
                              [[-0.608419, -0.599261, 0.700834, -1.163213],
                               [-1.138656, -1.229702, 1.078599, -0.425404],
                               [-0.531339, -0.879694, 1.078857, -1.191724]]],
                     'norm': [218.7685, 218.8866]}},
 'logits_400': [[-0.212637, 0.63963], [0.078282, 0.964873]],
 'released': {'jepa_r9cm_L2': {'head': [[[0.331767, 0.421402, 2.706304, -0.608073],
                                         [-1.531529, -1.367116, -3.903232, 0.907897],
                                         [0.63347, -1.214064, -3.842947, -0.523668],
                                         [1.64588, -0.554232, -4.738009, -0.080801],
                                         [0.869505, -0.374627, -3.907137, 0.773716]],
                                        [[-1.454329, -3.198446, -1.486266, 2.266468],
                                         [0.185472, 0.077642, 3.223316, -0.74854],
                                         [0.396006, -0.348669, -3.704621, -0.149019],
                                         [1.156456, -0.737943, -4.302522, 0.912485],
                                         [0.915067, -0.753952, -3.638466, 0.636177]]],
                               'norm': [221.5102, 223.7583]},
              'mae_r9cm_L2': {'head': [[[1.9346, -1.583328, 1.568627, -1.04385],
                                        [3.543942, -1.205949, 0.17277, -0.960314],
                                        [1.049286, -1.462192, 1.420502, -0.778552],
                                        [-0.879295, -0.657542, 3.417289, -1.904511],
                                        [-1.295799, 0.708528, 0.353715, -2.424924]],
                                       [[1.444591, -2.802545, 0.347266, -4.363357],
                                        [2.571631, -1.060053, 0.030681, -0.870344],
                                        [1.045072, -1.794099, 2.240225, -1.311754],
                                        [-0.086807, -0.121534, 2.597287, -2.762115],
                                        [-2.186162, 1.78706, -0.027488, -1.668947]]],
                              'norm': [240.0849, 241.4536]}}}
# GOLDEN-END


def _summary(features, patches):
    """What the golden values hold: features[:, 0, patches, :4] and the norms."""
    return features[:, 0, patches, :4], features.flatten(1).norm(dim=1)


# ----------------------------------------------------------------------------
# Architecture
# ----------------------------------------------------------------------------


def test_state_dict_matches_released_layout():
    model = _model(random_projection=16)
    state = model.state_dict()
    layout = {k: tuple(v.shape) for k, v in _backbone_state(model).items()}
    assert layout == _RELEASED_KEYS
    assert len(layout) == 27
    n_features = 4 * 5 * _EMBED_DIM
    assert tuple(state["final_layer.1.projection"].shape) == (16, n_features)
    assert tuple(state["final_layer.2.weight"].shape) == (3, 16)
    assert tuple(state["final_layer.2.bias"].shape) == (3,)
    assert {k for k in state if k.startswith("final_layer.")} == {
        "final_layer.1.projection",
        "final_layer.2.weight",
        "final_layer.2.bias",
    }
    n_backbone = sum(
        p.numel() for n, p in model.named_parameters() if not n.startswith("final_layer.")
    )
    assert n_backbone == _N_BACKBONE_PARAMS


def test_state_dict_without_projection():
    model = _model(random_projection=None)
    state = model.state_dict()
    assert {k: tuple(v.shape) for k, v in _backbone_state(model).items()} == _RELEASED_KEYS
    assert {k for k in state if k.startswith("final_layer.")} == {
        "final_layer.1.weight",
        "final_layer.1.bias",
    }
    assert tuple(state["final_layer.1.weight"].shape) == (3, 4 * 5 * _EMBED_DIM)


def test_architecture_invariants():
    model = _model()
    # Drifts that a loose tolerance would hide: RMSNorm eps 1e-6 gives 2.4e-6,
    # the tanh GELU gives 4.7e-4.
    norms = [m for m in model.modules() if isinstance(m, nn.RMSNorm)]
    assert len(norms) == 8
    assert all(m.eps is None for m in norms)
    gelus = [m for m in model.modules() if isinstance(m, nn.GELU)]
    assert len(gelus) == 4
    assert all(m.approximate == "none" for m in gelus)
    for m in model.model.transformer.modules():
        if isinstance(m, nn.Linear):
            assert m.bias is None
    enc = model.model.pos_encoder
    assert dict(enc.named_buffers())["encoding_time"].shape == (33, 128)
    assert "model.pos_encoder.encoding_time" in model.state_dict()
    assert enc.ch_pos.dtype == torch.float32
    assert "model.pos_encoder.ch_pos" not in model.state_dict()
    # the channel positions are the ones of chs_info
    np.testing.assert_allclose(
        enc.ch_pos.numpy(), np.array([_POS[n] for n in _NAMES], dtype=np.float32)
    )


def test_time_table_is_the_buffer_and_never_grows():
    model = _model()
    model.load_state_dict(_reference_state_dict(), strict=False)  # head-less, as the checkpoints
    enc = model.model.pos_encoder
    table = enc.encoding_time.clone()
    for p in (1, 2, 33):
        got = enc(p, 2)[..., 384:]  # (batch, n_chans, n_patches, 128)
        assert torch.equal(got, table[:p][None, None].expand(2, 4, -1, -1))
    formula = _pos_encode_time(34, 128, 666)
    beyond = enc(34, 2)[..., 384:]
    assert torch.equal(beyond[1, 0], formula)  # recomputed from the formula
    assert not torch.equal(beyond[1, 0, :33], table)  # not the (perturbed) buffer rows

    x400, x6200 = _windows(400), _windows(6200)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    with torch.no_grad():
        f400 = model(x400, return_features=True)["features"]
        model(x6200, return_features=True)
        f400_after = model(x400, return_features=True)["features"]
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())
    assert torch.equal(f400, f400_after)


def test_spatial_encoding_runs_on_the_batch_expanded_positions(monkeypatch):
    """Guard of the bit-for-bit parity with the reference implementation.

    ``torch.sin`` and ``torch.cos`` on the CPU can differ in the last bit for
    the same value depending on its place in the flattened tensor and on the
    number of threads; one ulp per element accumulates to about 2e-5 on 22
    channels at a batch size of 3. The reference computes the encoding on the
    ``(B, C, 3)`` positions, so this layout must be kept (not ``(C, 3)`` then
    broadcast).
    """
    shapes = []
    real = guetschel2026_module._pos_encode_xyz

    def spy(ch_pos, *args, **kwargs):
        shapes.append(tuple(ch_pos.shape))
        return real(ch_pos, *args, **kwargs)

    monkeypatch.setattr(guetschel2026_module, "_pos_encode_xyz", spy)
    model = Guetschel2026(
        chs_info=_synthetic_chs(22),
        n_times=400,
        n_outputs=2,
        sfreq=200.0,
        random_projection=None,
    ).eval()
    enc = model.model.pos_encoder
    assert enc(5, 3).shape == (3, 22, 5, _EMBED_DIM)
    assert shapes == [(3, 22, 3)]
    shapes.clear()
    with torch.no_grad():
        model(torch.randn(3, 22, 400) * 1e-5, return_features=True)
    assert shapes == [(3, 22, 3)]


def _reference_pos_encode(x, x_min, x_max, n_dim):
    """``pos_encode_continuous_batched`` of the original (MIT, commit 38a089e)."""
    out = torch.empty(x.shape + (n_dim,), dtype=torch.float32)
    div_term = torch.exp(
        (1 - torch.arange(0, n_dim, 2, device=out.device) / n_dim) * 2 * math.pi
    )
    xx = torch.as_tensor((x - x_min) / (x_max - x_min)).unsqueeze(-1)
    out[..., 0::2] = torch.sin(xx * div_term)
    out[..., 1::2] = torch.cos(xx * div_term)
    return out


def test_spatial_encoding_equals_the_reference_bitwise():
    # Weak on its own: whether a layout change is visible depends on the CPU and
    # on the thread count (SIMD tails, thread chunks). The structural test above
    # is the guard that holds everywhere; this one catches it on some machines.
    model = Guetschel2026(
        chs_info=_synthetic_chs(22),
        n_times=400,
        n_outputs=2,
        sfreq=200.0,
        random_projection=None,
    )
    enc = model.model.pos_encoder
    h = enc.pos_half_range
    n_threads = torch.get_num_threads()
    try:
        for threads in (1, 8):
            torch.set_num_threads(threads)
            for batch in (2, 3, 7, 13):
                ref = _reference_pos_encode(
                    enc.ch_pos[None].expand(batch, -1, -1), -h, h, 128
                ).flatten(-2)
                ref = ref[:, :, None].expand(-1, -1, 4, -1)
                assert torch.equal(enc(4, batch)[..., :384], ref)
    finally:
        torch.set_num_threads(n_threads)


@pytest.mark.parametrize("random_projection", [None, 16])
@pytest.mark.parametrize(
    "n_times, n_patches",
    [
        (200, 1),
        (379, 1),
        (380, 2),
        (1000, 5),
        (2000, 11),
        (5960, 33),
        (6000, 33),
        (6139, 33),
        (6140, 34),
        (6200, 34),
    ],
)
def test_patch_count_and_head(n_times, n_patches, random_projection):
    model = _model(n_times=n_times, random_projection=random_projection)
    n_features = 4 * n_patches * _EMBED_DIM
    assert model.n_patches == n_patches
    if random_projection is None:
        assert model.final_layer[1].in_features == n_features
    else:
        assert model.final_layer[1].projection.shape == (random_projection, n_features)
        assert model.final_layer[2].in_features == random_projection
    with torch.no_grad():
        out = model(_windows(n_times))
        feats = model(_windows(n_times), return_features=True)["features"]
    assert out.shape == (2, 3)
    assert feats.shape == (2, 4, n_patches, _EMBED_DIM)
    assert torch.isfinite(out).all()


def test_rejects_short_windows():
    with pytest.raises(ValueError, match="n_times"):
        _model(n_times=199)
    model = _model()
    with pytest.raises(ValueError, match="patch_size"):
        model(_windows(150))


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(lambda chs: [dict(c, loc=np.zeros(12)) for c in chs], id="all-zero"),
        pytest.param(
            lambda chs: [dict(chs[0], loc=np.r_[np.nan, np.zeros(11)])] + chs[1:],
            id="nan",
        ),
        pytest.param(
            lambda chs: [{k: v for k, v in c.items() if k != "loc"} for c in chs],
            id="no-loc-key",
        ),
    ],
)
def test_requires_channel_locations(mutate):
    with pytest.raises(ValueError, match="locations"):
        Guetschel2026(
            chs_info=mutate(_chs()), n_times=1000, n_outputs=2, sfreq=200.0,
            random_projection=None,
        )


def _build(chs, **kwargs):
    kwargs.setdefault("random_projection", None)
    return Guetschel2026(
        chs_info=chs, n_times=1000, n_outputs=2, sfreq=200.0, **kwargs
    )


def _scaled(factor, names=_NAMES):
    return [dict(c, loc=c["loc"] * factor) for c in _chs(names)]


def _on_axis(distance, name="E0"):
    """One channel at ``distance`` metres from the origin."""
    loc = np.r_[distance, 0.0, 0.0, np.zeros(9)]
    return {"ch_name": name, "kind": "eeg", "loc": loc}


def test_one_zero_location_among_valid_ones_is_rejected():
    # An all-zero loc is a placeholder for an unknown position (MNE marks one
    # with NaN or zeros); the original accepted it (it checks only for NaN),
    # this class does not.
    chs = _chs()
    chs[1] = dict(chs[1], loc=np.zeros(12))
    with pytest.raises(ValueError, match=r"METRES.*'Cz' \(0 m\)"):
        _build(chs)


@pytest.mark.parametrize("factor", [100, 1000], ids=["cm", "mm"])
def test_rejects_positions_in_centimetres_or_millimetres(factor):
    with pytest.raises(ValueError, match="METRES") as err:
        _build(_scaled(factor))
    msg = str(err.value)
    assert "centimetres or millimetres" in msg
    assert "4 of 4" in msg
    assert "'C3'" in msg and "'Oz'" in msg
    # channel_strategy does not replace non-zero positions: the message must say so
    assert "does not replace given positions" in msg


@pytest.mark.filterwarnings("ignore:Montage name .* is deprecated")
def test_the_advice_of_the_error_works():
    """Names only + channel_strategy='exact' builds from the rejected names."""
    names = [c["ch_name"] for c in _scaled(100)]
    with pytest.raises(ValueError, match="METRES"):
        _build(_scaled(100), channel_strategy="exact")  # alone: same error
    names_only = [{"ch_name": n, "kind": "eeg"} for n in names]
    model = _build(names_only, channel_strategy="exact")
    dist = model.model.pos_encoder.ch_pos.norm(dim=-1)
    assert dist.shape == (len(names),)
    assert ((dist >= 0.05) & (dist <= 0.20)).all()


@pytest.mark.filterwarnings("ignore:Montage name .* is deprecated")
def test_exact_fills_only_the_zero_row_and_keeps_the_given_positions():
    """The advice of the error for a channel without position: 'exact' alone."""
    chs = _scaled(1.05)  # given positions that differ from the standard ones
    zero_at = 1
    chs[zero_at] = dict(chs[zero_at], loc=np.zeros(12))
    with pytest.raises(ValueError, match=r"METRES.*'Cz' \(0 m\)") as err:
        _build(chs)
    assert "pass channel_strategy='exact'" in str(err.value)
    model = _build(chs, channel_strategy="exact")
    ch_pos = model.model.pos_encoder.ch_pos.numpy()
    expected = np.array([c["loc"][:3] for c in chs])
    expected[zero_at] = _standard_positions()[chs[zero_at]["ch_name"].lower()]
    np.testing.assert_allclose(ch_pos, expected, atol=1e-6)
    # the other rows are the given ones, not the standard_1005 template
    given = [i for i in range(len(chs)) if i != zero_at]
    template = np.array([_standard_positions()[chs[i]["ch_name"].lower()] for i in given])
    assert not np.allclose(ch_pos[given], template, atol=1e-4)


def test_exact_cannot_look_up_non_standard_names():
    """Names outside standard_1005 (EGI 'E1'...) need positions in metres."""
    egi = [{"ch_name": f"E{i}", "kind": "eeg"} for i in range(1, 4)]
    with pytest.raises(ValueError, match="channel locations") as err:
        _build(egi, channel_strategy="exact")
    assert "other names" in str(err.value)
    assert "positions in metres" in str(err.value)


def test_exact_cannot_look_up_non_standard_names_with_an_all_zero_loc():
    """The all-zero error says that 'exact' leaves an unknown name at the origin."""
    chs = _chs() + [{"ch_name": "E1", "kind": "eeg", "loc": np.zeros(12)}]
    with pytest.raises(ValueError, match="METRES") as err:
        _build(chs, channel_strategy="exact")
    msg = str(err.value)
    assert "'E1' (0 m)" in msg
    assert "other names need positions in metres" in msg


def test_rejects_positions_scaled_by_1e_3():
    with pytest.raises(ValueError, match="METRES") as err:
        _build(_scaled(1e-3))
    msg = str(err.value)
    assert "'C3' (0.000126 m)" in msg  # 0.126 m x 1e-3


def test_error_names_the_channel_at_the_origin_among_valid_ones():
    chs = _chs() + [{"ch_name": "Fp1", "kind": "eeg", "loc": np.zeros(12)}]
    with pytest.raises(ValueError, match="METRES") as err:
        _build(chs)
    msg = str(err.value)
    assert "1 of 5" in msg
    assert "'Fp1' (0 m)" in msg
    for name in _NAMES:
        assert repr(name) not in msg


def test_error_lists_only_the_first_offending_channels():
    chs = _chs() * 3  # 12 channels, all too far
    chs = [dict(c, ch_name=f"E{i}", loc=c["loc"] * 100) for i, c in enumerate(chs)]
    with pytest.raises(ValueError, match="METRES") as err:
        _build(chs)
    msg = str(err.value)
    assert "12 of 12" in msg
    assert "'E4'" in msg and "'E5'" not in msg
    assert "'E4'" in msg and msg.count(" m)") == 5 and ", ..." in msg


def test_a_single_valid_channel_builds():
    model = _build(_chs(("Cz",)))
    assert model.model.pos_encoder.ch_pos.shape == (1, 3)


@pytest.mark.parametrize("distance", [0.001, 0.01, 0.1, 1.0, 10.0])
def test_a_single_channel_is_checked(distance):
    if 0.05 <= distance <= 0.2:
        _build([_on_axis(distance)])
    else:
        with pytest.raises(ValueError, match="METRES"):
            _build([_on_axis(distance)])


def test_a_single_channel_at_the_origin_is_rejected():
    # caught by the all-zero check, which runs first
    with pytest.raises(ValueError, match="locations|positions"):
        _build([_on_axis(0.0)])


@pytest.mark.parametrize(
    "distance, ok",
    [
        (0.0499, False),
        (0.0499999, False),
        # both are accepted once rounded to float32: the distance is taken on
        # the float64 positions as given
        (0.049999999, False),
        (0.05, True),  # the bounds are inclusive, whatever the float32 rounding
        (0.0501, True),
        (0.1999, True),
        (0.2, True),
        (0.20000001, False),  # accepted once rounded to float32, see above
        (0.2000001, False),
        (0.2001, False),
    ],
)
def test_distance_boundaries(distance, ok):
    chs = _chs() + [_on_axis(distance, name="B")]
    if ok:
        _build(chs)
    else:
        with pytest.raises(ValueError, match="'B'"):
            _build(chs)


def test_the_distance_is_the_euclidean_norm_not_one_coordinate():
    # each coordinate is below 0.2 m but the norm is 0.208 m
    loc = np.r_[0.12, 0.12, 0.12, np.zeros(9)]
    with pytest.raises(ValueError, match="'D' \\(0.208 m\\)"):
        _build([{"ch_name": "D", "kind": "eeg", "loc": loc}])


def test_the_registry_fixture_builds():
    from braindecode.models.util import _get_signal_params, models_mandatory_parameters

    _, required, signal_params = next(
        p for p in models_mandatory_parameters if p[0] == "Guetschel2026"
    )
    sp = _get_signal_params(signal_params, required)
    model = Guetschel2026(**sp, random_projection=None)
    assert model.model.pos_encoder.ch_pos.shape[0] == len(sp["chs_info"])


def test_inline_metre_positions_build():
    model = _build(_synthetic_chs(32))
    assert model.model.pos_encoder.ch_pos.shape == (32, 3)


def test_positions_of_a_real_montage_build_and_their_scalings_do_not():
    mne = pytest.importorskip("mne")
    from braindecode.util import resolve_montage_name

    montage = mne.channels.make_standard_montage(resolve_montage_name("standard_1020"))
    info = mne.create_info(montage.ch_names, 200.0, "eeg")
    info.set_montage(montage)
    chs = [
        {"ch_name": c["ch_name"], "kind": "eeg", "loc": c["loc"].copy()}
        for c in info["chs"]
    ]
    assert len(chs) > 80
    _build(chs)
    for factor in (100, 1000, 1e-3):
        with pytest.raises(ValueError, match="METRES"):
            _build([dict(c, loc=c["loc"] * factor) for c in chs])


@pytest.mark.filterwarnings("ignore:Montage name .* is deprecated")
def test_exact_channel_strategy_looks_positions_up_by_name():
    chs = [dict(c, loc=np.zeros(12)) for c in _chs()]
    model = Guetschel2026(
        chs_info=chs,
        n_times=1000,
        n_outputs=2,
        sfreq=200.0,
        channel_strategy="exact",
        random_projection=None,
    )
    ch_pos = model.model.pos_encoder.ch_pos
    assert torch.isfinite(ch_pos).all()
    assert (ch_pos.abs().sum(-1) > 0).all()


def test_warns_when_sfreq_is_not_the_pretraining_one():
    with pytest.warns(UserWarning, match="200"):
        Guetschel2026(
            chs_info=_chs(), n_times=1000, n_outputs=2, sfreq=250.0, random_projection=None
        )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _model(random_projection=None)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(normalization="chan_std"), "normalization"),
        (dict(embed_dim=100), "embed_dim"),
        (dict(embed_dim=512, num_heads=7), "num_heads"),
        (dict(num_heads=0), "num_heads"),
        (dict(num_heads=-8), "num_heads"),
        (dict(num_heads=8.0), "num_heads"),
        (dict(pos_half_range=0.0), "pos_half_range"),
        (dict(pos_half_range=-0.15), "pos_half_range"),
        (dict(pos_half_range=float("inf")), "pos_half_range"),
        (dict(pos_half_range=float("nan")), "pos_half_range"),
        (dict(patch_overlap=200), "patch_overlap"),
    ],
)
def test_invalid_arguments(kwargs, match):
    with pytest.raises(ValueError, match=match):
        _model(random_projection=None, **kwargs)


# ----------------------------------------------------------------------------
# Input scaling
# ----------------------------------------------------------------------------


def test_median_std_clip():
    model = _model(random_projection=None)
    g = torch.Generator().manual_seed(0)
    z = torch.randn(1, 4, 1000, generator=g)
    z = (z - z.mean(-1, keepdim=True)) / z.std(-1, keepdim=True, correction=0)
    stds = torch.tensor([1.0, 2.0, 3.0, 4.0])  # microvolts
    x = z * stds[None, :, None] * 1e-6

    # (a) the divisor is the *lower* median of the four stds (2, not 2.5)
    scaled = model._scale(x)
    expected = x * 1e6 / (2.0 + 1e-6)
    torch.testing.assert_close(scaled, expected, rtol=1e-5, atol=1e-5)
    assert scaled.abs().max() < 15  # nothing is clipped here

    # (b) values beyond clip_sigma are clipped
    spike = x.clone()
    spike[0, 0, 10] = 1e-3  # 1000 uV
    spike[0, 0, 11] = -1e-3
    clipped = model._scale(spike)
    assert clipped.max().item() == 15.0
    assert clipped.min().item() == -15.0
    assert (clipped.abs() <= 15.0).all()

    # (c) float16 inputs do not overflow
    half = model._scale(torch.randn(2, 4, 1000, generator=g).half())
    assert half.dtype == torch.float16
    assert torch.isfinite(half).all()

    # (d) no normalization
    plain = _model(random_projection=None, normalization="none", input_scale=1.0)
    assert torch.equal(plain._scale(x), x)
    only_scale = _model(random_projection=None, normalization="none")
    torch.testing.assert_close(only_scale._scale(x), x * 1e6)


def test_scale_keeps_the_dtype():
    model = _model(random_projection=None)
    for dtype in (torch.float32, torch.float64):
        assert model._scale(_windows(1000, dtype=dtype)).dtype == dtype


# ----------------------------------------------------------------------------
# Golden values (computed with the original code)
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("n_times", [400, 6200])
def test_reference_features(n_times):
    model = _model(n_times=n_times, random_projection=None)
    model.load_state_dict(_reference_state_dict(), strict=False)
    with torch.no_grad():
        features = model(_windows(n_times), return_features=True)["features"]
    head, norm = _summary(features, _GOLDEN_PATCHES[n_times])
    gold = _GOLDEN["features"][n_times]
    torch.testing.assert_close(head, torch.tensor(gold["head"]), rtol=0, atol=1e-5)
    torch.testing.assert_close(norm, torch.tensor(gold["norm"]), rtol=1e-5, atol=1e-4)


def test_reference_logits_with_the_plain_head():
    # random_projection=None is the head of the original wrapper
    model = _model(n_times=400, n_outputs=2, random_projection=None)
    model.load_state_dict(_reference_state_dict(), strict=False)
    model.final_layer[1].load_state_dict(_reference_head(4 * 2 * _EMBED_DIM, 2))
    with torch.no_grad():
        logits = model(_windows(400))
    torch.testing.assert_close(
        logits, torch.tensor(_GOLDEN["logits_400"]), rtol=0, atol=1e-5
    )


def test_features_do_not_depend_on_the_head():
    kwargs = dict(n_times=400, n_outputs=2)
    plain, projected = _model(random_projection=None, **kwargs), _model(
        random_projection=32, **kwargs
    )
    for m in (plain, projected):
        m.load_state_dict(_reference_state_dict(), strict=False)
    with torch.no_grad():
        a = plain(_windows(400), return_features=True)["features"]
        b = projected(_windows(400), return_features=True)["features"]
    assert torch.equal(a, b)


def test_float64_backbone_is_float64():
    model = _model(n_times=400, random_projection=None)
    model.load_state_dict(_reference_state_dict(), strict=False)
    f32 = model(_windows(400), return_features=True)["features"]
    model.double()
    with torch.no_grad():
        f64 = model(_windows(400, dtype=torch.float64), return_features=True)["features"]
    assert f64.dtype == torch.float64
    # float64 runs end to end and agrees with float32 to float32 accuracy
    torch.testing.assert_close(f32.double(), f64, rtol=0, atol=1e-3)


@pytest.mark.parametrize("scale", [1.0, 1e-6, 1e-12])
def test_scaling_is_exact_in_float64(scale):
    # Exact, so that a hidden float32 step or a missing epsilon cannot pass
    # (a float32 detour changes the float64 result by about 7e-8).
    model = _model(n_times=400, random_projection=None)
    x64 = _windows(400, dtype=torch.float64) * scale
    y = x64 * model.input_scale
    std = y.std(dim=-1, keepdim=True, correction=0)
    y = (y / (std.median(dim=-2, keepdim=True).values + 1e-6)).clamp(
        -model.clip_sigma, model.clip_sigma
    )
    got = model._scale(x64)
    assert got.dtype == torch.float64
    assert torch.equal(got, y)


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------


def test_partial_state_dict_raises():
    model = _model(random_projection=16)
    sd = _reference_state_dict()
    partial = {k: v for k, v in sd.items() if k != "model.transformer.layers.0.linear1.weight"}
    with pytest.raises(RuntimeError, match="misses backbone"):
        model.load_state_dict(partial, strict=False)

    # The released files have no head: its keys (projection and linear) are
    # the only missing ones, and nothing is unexpected.
    result = model.load_state_dict(sd, strict=False)
    assert set(result.missing_keys) == {
        "final_layer.1.projection",
        "final_layer.2.weight",
        "final_layer.2.bias",
    }
    assert result.unexpected_keys == []
    with pytest.raises(RuntimeError):
        model.load_state_dict(sd, strict=True)

    plain = _model(random_projection=None)
    result = plain.load_state_dict(sd, strict=False)
    assert set(result.missing_keys) == {"final_layer.1.weight", "final_layer.1.bias"}
    assert result.unexpected_keys == []


def test_loading_the_checkpoint_keeps_the_projection():
    model = _model(random_projection=16, random_projection_seed=3)
    before = model.final_layer[1].projection.clone()
    model.load_state_dict(_reference_state_dict(), strict=False)
    assert torch.equal(model.final_layer[1].projection, before)


_ORIGINAL_CONFIG = {
    "clip_sigma": 15.0,
    "factor": 1000000.0,
    "feature_encoder": {
        "dim": 512,
        "modelName": "LinearPatchEmbedding",
        "patch_overlap": 20,
        "patch_size": 200,
    },
    "masker": {
        "length_blocks": 2,
        "n_target_blocks": None,
        "pct_unmasked": 0.44999999999999996,
        # json.dump writes the Infinity of the r=all repositories
        "radius_blocks": float("inf"),
        "scalp_surface": 0.0942477796076938,
        "vectorized": True,
    },
    "pos_encoder": {
        "max_seconds": 600.0,
        "max_x": 0.15,
        "modelName": "AdditivePositionalEncoder",
        "sfreq_features": 1.1111111111111112,
        "spat_dim": 384,
        "time_dim": 128,
    },
    "scaler": "median_std_clip",
    "shared_feature_encoder": True,
    "transformer": {
        "activation": "gelu",
        "bias": False,
        "d_model": 512,
        "dim_feedforward": 1365,
        "dropout": 0.0,
        "glu": True,
        "nhead": 8,
        "norm": "rms_norm",
        "num_layers": 4,
    },
}


@needs_safetensors
def test_from_pretrained_original_repo_layout(tmp_path):
    """A local copy of a repository, laid out exactly as on the Hub."""
    repo = tmp_path / "eeg-fm-masking_mae_r9cm_L2"
    (repo / "epoch_01").mkdir(parents=True)
    (repo / "config.json").write_text(json.dumps(_ORIGINAL_CONFIG))
    final, epoch = _reference_state_dict(0), _reference_state_dict(1)
    save_file(final, str(repo / "model.safetensors"))
    save_file(epoch, str(repo / "epoch_01" / "model.safetensors"))

    kwargs = dict(
        chs_info=_chs(), n_times=1000, n_outputs=3, sfreq=200.0, random_projection=16
    )
    model = Guetschel2026.from_pretrained(str(repo), **kwargs)
    assert all(torch.equal(model.state_dict()[k], v) for k, v in final.items())
    assert model.clip_sigma == 15.0
    assert model.final_layer[2].out_features == 3
    config = model.get_config()
    assert not {"masker", "factor", "scaler"} & set(config)
    assert config["random_projection"] == 16

    other = Guetschel2026.from_pretrained(
        str(repo), filename="epoch_01/model.safetensors", **kwargs
    )
    assert all(torch.equal(other.state_dict()[k], v) for k, v in epoch.items())
    assert not torch.equal(
        other.feature_encoder.linear.weight, model.feature_encoder.linear.weight
    )

    saved = tmp_path / "saved"
    model.save_pretrained(saved)
    reloaded = Guetschel2026.from_pretrained(
        str(saved), chs_info=_chs(), n_times=1000, sfreq=200.0
    )
    a, b = model.state_dict(), reloaded.state_dict()
    assert a.keys() == b.keys()
    assert all(torch.equal(a[k], b[k]) for k in a)


@needs_safetensors
def test_from_pretrained_rejects_a_checkpoint_missing_backbone_weights(tmp_path):
    repo = tmp_path / "broken"
    repo.mkdir()
    (repo / "config.json").write_text(json.dumps(_ORIGINAL_CONFIG))
    sd = _reference_state_dict()
    del sd["feature_encoder.linear.bias"]
    save_file(sd, str(repo / "model.safetensors"))
    with pytest.raises(RuntimeError, match="misses backbone"):
        Guetschel2026.from_pretrained(
            str(repo), chs_info=_chs(), n_times=1000, n_outputs=2, sfreq=200.0,
            random_projection=16,
        )


# ----------------------------------------------------------------------------
# hub_repo_id
# ----------------------------------------------------------------------------

# The 58 repositories of the Hub (https://huggingface.co/PierreGtch), as listed
# by the file names of their config.json.
_HUB_SUFFIXES = {
    **{r: (1, 2, 4, 8, 16, 33) for r in ("rone", "r6cm", "r9cm", "r12cm")},
    "rall": (1, 2, 4, 8, 16),
}
_HUB_REPOS = sorted(
    f"PierreGtch/eeg-fm-masking_{pretext}_{radius}_L{length}"
    for pretext in ("mae", "jepa")
    for radius, lengths in _HUB_SUFFIXES.items()
    for length in lengths
)


def test_hub_repo_id_generates_the_58_repositories():
    assert len(_HUB_REPOS) == 58
    generated = []
    for pretext in ("mae", "jepa"):
        for radius in ("one", "6cm", "9cm", "12cm", "all"):
            for length in (1, 2, 4, 8, 16, 33):
                if (radius, length) == ("all", 33):
                    continue
                generated.append(Guetschel2026.hub_repo_id(pretext, radius, length))
    assert sorted(generated) == _HUB_REPOS
    assert len(set(generated)) == 58


@pytest.mark.parametrize(
    "args, expected",
    [
        (("mae", "9cm", 2), "PierreGtch/eeg-fm-masking_mae_r9cm_L2"),
        (("jepa", "9cm", 2), "PierreGtch/eeg-fm-masking_jepa_r9cm_L2"),
        (("jepa", "all", 1), "PierreGtch/eeg-fm-masking_jepa_rall_L1"),
        (("mae", "one", 1), "PierreGtch/eeg-fm-masking_mae_rone_L1"),
        (("mae", "12cm", 33), "PierreGtch/eeg-fm-masking_mae_r12cm_L33"),
    ],
)
def test_hub_repo_id_examples(args, expected):
    assert Guetschel2026.hub_repo_id(*args) == expected


def test_hub_repo_id_is_a_classmethod():
    assert Guetschel2026.hub_repo_id("mae", "9cm", 2) == _model().hub_repo_id("mae", "9cm", 2)


@pytest.mark.parametrize("pretext", ["mae", "jepa"])
def test_hub_repo_id_refuses_the_untrained_cell(pretext):
    with pytest.raises(ValueError, match="trained"):
        Guetschel2026.hub_repo_id(pretext, "all", 33)


@pytest.mark.parametrize(
    "args, bad",
    [
        (("vae", "9cm", 2), "pretext"),
        (("MAE", "9cm", 2), "pretext"),
        (("mae", "5cm", 2), "mask_radius"),
        (("mae", "9", 2), "mask_radius"),
        (("mae", 9, 2), "mask_radius"),
        (("mae", "9cm", 3), "mask_length"),
        (("mae", "9cm", 0), "mask_length"),
        (("mae", "9cm", "2"), "mask_length"),
        (("mae", "9cm", True), "mask_length"),
        (("mae", "9cm", 2.0), "mask_length"),
        (("mae", "9cm", None), "mask_length"),
    ],
)
def test_hub_repo_id_rejects_invalid_values(args, bad):
    with pytest.raises(ValueError, match=bad) as excinfo:
        Guetschel2026.hub_repo_id(*args)
    assert "one of" in str(excinfo.value)  # the message lists the valid values


# ----------------------------------------------------------------------------
# Random projection head
# ----------------------------------------------------------------------------

_N_FEATURES = 4 * 5 * _EMBED_DIM  # 4 channels, 1000 samples


def test_random_projection_defaults():
    params = inspect.signature(Guetschel2026.__init__).parameters
    assert params["random_projection"].default is None
    assert params["random_projection_seed"].default == 0
    # the default head has no projection
    model = Guetschel2026(chs_info=_chs(["Cz"]), n_times=200, n_outputs=2, sfreq=200.0)
    assert [type(m) for m in model.final_layer] == [nn.Flatten, nn.Linear]
    assert model.random_projection is None
    # the paper's 5000-feature projection (1 channel, 1 patch)
    with pytest.warns(UserWarning, match="random_projection"):
        model = Guetschel2026(
            chs_info=_chs(["Cz"]),
            n_times=200,
            n_outputs=2,
            sfreq=200.0,
            random_projection=5000,
        )
    assert model.final_layer[1].projection.shape == (5000, _EMBED_DIM)
    assert model.final_layer[2].in_features == 5000
    assert model.random_projection == 5000
    assert model.random_projection_seed == 0


def test_construction_warnings_point_at_the_calling_line():
    kwargs = dict(chs_info=_chs(["Cz"]), n_times=200, n_outputs=2)
    with pytest.warns(UserWarning) as record:
        Guetschel2026(sfreq=250.0, random_projection=5000, **kwargs)
    assert len(record) == 2, [str(w.message) for w in record]
    # The sampling-rate warning comes from the shared warn_if_sfreq_differs, whose
    # stacklevel is the one of every pretrained model; only ours is checked here.
    projection = [w for w in record if "random_projection" in str(w.message)]
    assert [w.filename for w in projection] == [__file__]


def test_random_projection_warns_when_it_expands_the_features():
    kwargs = dict(chs_info=_chs(["Cz"]), n_times=200, n_outputs=2, sfreq=200.0)
    with pytest.warns(UserWarning, match="expands rather than reduces"):
        Guetschel2026(random_projection=513, **kwargs)  # 512 features
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        Guetschel2026(random_projection=512, **kwargs)
        Guetschel2026(random_projection=None, **kwargs)


def test_default_device_context_puts_every_buffer_and_parameter_together():
    with torch.device("meta"):
        model = _model(random_projection=16)
    tensors = {**dict(model.named_parameters()), **dict(model.named_buffers())}
    assert "final_layer.1.projection" in tensors
    assert {t.device.type for t in tensors.values()} == {"meta"}


def test_random_projection_is_drawn_on_the_cpu_whatever_the_default_device(
    monkeypatch,
):
    draws = []
    normal_ = torch.Tensor.normal_

    def spy(self, *args, **kwargs):
        if kwargs.get("generator") is not None:
            draws.append(self.device.type)
        return normal_(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "normal_", spy)
    with torch.device("meta"):
        model = _model(random_projection=16)
    assert draws and set(draws) == {"cpu"}  # drawn on the CPU ...
    assert model.final_layer[1].projection.device.type == "meta"  # ... then moved


def test_random_projection_is_deterministic():
    expected = _model(random_projection=16)
    model = _model(random_projection=16)
    assert torch.equal(
        model.final_layer[1].projection, expected.final_layer[1].projection
    )
    assert torch.equal(
        model.model.pos_encoder.encoding_time,
        expected.model.pos_encoder.encoding_time,
    )


_ACCELERATORS = [
    d
    for d, ok in (
        ("cuda", torch.cuda.is_available()),
        ("mps", torch.backends.mps.is_available()),
    )
    if ok
]


@pytest.mark.skipif(not _ACCELERATORS, reason="no accelerator available")
@pytest.mark.parametrize("random_projection", [None, 16])
def test_forward_under_an_accelerator_default_device_context(random_projection):
    device = _ACCELERATORS[0]
    x = _windows(1000).to(device)  # float64 intermediates: not under the context
    with torch.device(device):
        model = _model(random_projection=random_projection)
    out = model(x)
    assert out.device.type == device
    if random_projection is not None:
        assert torch.equal(
            model.final_layer[1].projection.cpu(),
            _model(random_projection=random_projection).final_layer[1].projection,
        )


def test_random_projection_follows_the_default_dtype():
    previous = torch.get_default_dtype()
    expected = _model(random_projection=16).final_layer[1].projection
    try:
        torch.set_default_dtype(torch.float64)
        model = _model(random_projection=16)
        assert model.final_layer[1].projection.dtype == torch.float64
        with torch.no_grad():
            out = model(_windows(1000, dtype=torch.float64))
        assert out.dtype == torch.float64
        # drawn in float32, then stored in the default dtype
        assert torch.equal(model.final_layer[1].projection.float(), expected)
    finally:
        torch.set_default_dtype(previous)


def test_random_projection_layout():
    model = _model(random_projection=32, n_outputs=3)
    head = model.final_layer
    assert isinstance(head, nn.Sequential) and len(head) == 3
    assert isinstance(head[0], nn.Flatten)
    assert isinstance(head[1], _GaussianRandomProjection)
    assert isinstance(head[2], nn.Linear)
    assert head[1].projection.shape == (32, _N_FEATURES)
    assert head[1].projection.dtype == torch.float32
    assert (head[2].in_features, head[2].out_features) == (32, 3)
    assert model.final_layer[-1] is head[2]


def test_no_random_projection_layout():
    model = _model(random_projection=None, n_outputs=3)
    head = model.final_layer
    assert isinstance(head, nn.Sequential) and len(head) == 2
    assert isinstance(head[0], nn.Flatten)
    assert isinstance(head[1], nn.Linear)
    assert (head[1].in_features, head[1].out_features) == (_N_FEATURES, 3)
    assert not any(isinstance(m, _GaussianRandomProjection) for m in model.modules())


def test_random_projection_is_a_persistent_buffer_not_a_parameter():
    model = _model()
    proj = model.final_layer[1]
    name = "final_layer.1.projection"
    assert name in dict(model.named_buffers())
    assert name in model.state_dict()
    assert name not in dict(model.named_parameters())
    assert "projection" not in proj._non_persistent_buffers_set
    assert not isinstance(proj.projection, nn.Parameter)
    assert not proj.projection.requires_grad
    assert all(p is not proj.projection for p in model.parameters())
    assert list(proj.parameters()) == []


def test_random_projection_is_never_trained():
    model = _model(n_outputs=2)
    model.train()
    before = model.final_layer[1].projection.clone()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    out = model(_windows(1000))
    out.sum().backward()
    optimizer.step()
    assert model.final_layer[1].projection.grad is None
    assert model.final_layer[2].weight.grad is not None
    assert torch.equal(model.final_layer[1].projection, before)


def test_random_projection_is_deterministic_per_seed():
    a = _GaussianRandomProjection(2048, 64, seed=0).projection
    b = _GaussianRandomProjection(2048, 64, seed=0).projection
    c = _GaussianRandomProjection(2048, 64, seed=1).projection
    assert torch.equal(a, b)
    assert not torch.equal(a, c)
    m0 = _model(random_projection=16, random_projection_seed=0)
    m0b = _model(random_projection=16, random_projection_seed=0)
    m1 = _model(random_projection=16, random_projection_seed=1)
    assert torch.equal(m0.final_layer[1].projection, m0b.final_layer[1].projection)
    assert not torch.equal(m0.final_layer[1].projection, m1.final_layer[1].projection)
    assert m1.random_projection_seed == 1


def test_random_projection_does_not_depend_on_the_global_seed():
    torch.manual_seed(1)
    a = _model(random_projection=16)
    torch.manual_seed(2)
    b = _model(random_projection=16)
    assert torch.equal(a.final_layer[1].projection, b.final_layer[1].projection)
    # while the layers initialised by the global RNG do differ
    assert not torch.equal(a.final_layer[2].weight, b.final_layer[2].weight)


def test_random_projection_generation_does_not_touch_the_global_rng():
    torch.manual_seed(123)
    state = torch.get_rng_state()
    _GaussianRandomProjection(4096, 128, seed=5)
    assert torch.equal(torch.get_rng_state(), state)
    # and a different global state does not change the matrix either
    reference = _GaussianRandomProjection(4096, 128, seed=5).projection
    torch.manual_seed(999)
    assert torch.equal(_GaussianRandomProjection(4096, 128, seed=5).projection, reference)


def test_random_projection_distribution():
    n_components, n_features = 256, 10240
    proj = _GaussianRandomProjection(n_features, n_components, seed=0).projection
    assert proj.shape == (n_components, n_features)
    sigma = 1 / math.sqrt(n_components)  # N(0, 1/n_components), as sklearn
    # 2.6 M entries: the standard error of the mean is 4e-5
    assert abs(proj.double().mean().item()) < 5e-4
    assert proj.double().std().item() == pytest.approx(sigma, rel=5e-3)
    z = proj.double() / sigma
    assert (z.abs() < 1).double().mean().item() == pytest.approx(0.6827, abs=3e-3)
    assert (z.abs() < 2).double().mean().item() == pytest.approx(0.9545, abs=3e-3)
    assert (z**4).mean().item() == pytest.approx(3.0, abs=0.05)  # Gaussian kurtosis
    # the rows and the columns are not systematically different
    assert proj.double().mean(dim=1).abs().max().item() < 5 * sigma / math.sqrt(n_features) * 1.5


@pytest.mark.parametrize("n_components", [1, 7, 5000])
def test_random_projection_scale_follows_n_components(n_components):
    proj = _GaussianRandomProjection(2000, n_components, seed=0).projection
    assert proj.std().item() == pytest.approx(1 / math.sqrt(n_components), rel=0.05)


def test_head_is_flatten_projection_linear():
    model = _model(n_times=400, random_projection=32, n_outputs=2)
    model.load_state_dict(_reference_state_dict(), strict=False)
    with torch.no_grad():
        features = model(_windows(400), return_features=True)["features"]
        logits = model(_windows(400))
        manual = model.final_layer[2](
            features.flatten(1) @ model.final_layer[1].projection.T
        )
    assert logits.shape == (2, 2)
    torch.testing.assert_close(logits, manual, rtol=1e-5, atol=1e-6)
    # shape of the output of the projection alone
    assert model.final_layer[:2](features).shape == (2, 32)


def test_reset_head_keeps_the_projection_bitwise():
    model = _model(random_projection=16, n_outputs=3)
    proj_module = model.final_layer[1]
    proj_before = proj_module.projection.clone()
    old_linear = model.final_layer[2]
    model.reset_head(5)
    assert model.final_layer[1] is proj_module
    assert torch.equal(model.final_layer[1].projection, proj_before)
    assert model.final_layer[2] is not old_linear
    assert (model.final_layer[2].in_features, model.final_layer[2].out_features) == (16, 5)
    assert model.n_outputs == 5
    with torch.no_grad():
        assert model(_windows(1000)).shape == (2, 5)


def test_reset_head_without_projection():
    model = _model(random_projection=None, n_outputs=3)
    model.reset_head(7)
    assert len(model.final_layer) == 2
    assert (model.final_layer[1].in_features, model.final_layer[1].out_features) == (
        _N_FEATURES,
        7,
    )
    assert model.n_outputs == 7


def test_reset_head_keeps_the_dtype():
    model = _model(random_projection=16).double()
    model.reset_head(2)
    assert model.final_layer[2].weight.dtype == torch.float64
    assert model.final_layer[1].projection.dtype == torch.float64
    with torch.no_grad():
        assert model(_windows(1000, dtype=torch.float64)).dtype == torch.float64


def test_state_dict_round_trip_keeps_the_projection():
    source = _model(random_projection=16, random_projection_seed=7)
    target = _model(random_projection=16, random_projection_seed=0)
    assert not torch.equal(
        source.final_layer[1].projection, target.final_layer[1].projection
    )
    target.load_state_dict(source.state_dict())
    assert torch.equal(source.final_layer[1].projection, target.final_layer[1].projection)
    x = _windows(1000)
    with torch.no_grad():
        assert torch.equal(source(x), target(x))


@needs_safetensors
def test_save_and_load_pretrained_keeps_the_projection(tmp_path):
    model = _model(random_projection=16, random_projection_seed=4)
    model.save_pretrained(tmp_path)
    # the projection is persisted in the weights ...
    saved = load_file(str(tmp_path / "model.safetensors"))
    assert torch.equal(saved["final_layer.1.projection"], model.final_layer[1].projection)
    # ... so that it survives, even when another seed is requested at load time
    reloaded = Guetschel2026.from_pretrained(
        str(tmp_path), chs_info=_chs(), n_times=1000, sfreq=200.0, random_projection_seed=9
    ).eval()
    assert torch.equal(reloaded.final_layer[1].projection, model.final_layer[1].projection)
    x = _windows(1000)
    with torch.no_grad():
        assert torch.equal(model(x), reloaded(x))
    config = model.get_config()
    assert config["random_projection"] == 16
    assert config["random_projection_seed"] == 4


def test_random_projection_float64_cast():
    model = _model(random_projection=16)
    float32_matrix = model.final_layer[1].projection.clone()
    model.double()
    assert model.final_layer[1].projection.dtype == torch.float64
    assert torch.equal(model.final_layer[1].projection, float32_matrix.double())
    with torch.no_grad():
        out = model(_windows(1000, dtype=torch.float64))
    assert out.dtype == torch.float64
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("bad", [0, -3, 1.5, True, "5000"])
def test_invalid_random_projection(bad):
    with pytest.raises(ValueError, match="random_projection"):
        Guetschel2026(
            chs_info=_chs(), n_times=1000, n_outputs=2, sfreq=200.0, random_projection=bad
        )


@pytest.mark.parametrize("bad", [1.5, True, "0", None])
def test_invalid_random_projection_seed(bad):
    with pytest.raises(ValueError, match="random_projection_seed"):
        _model(random_projection_seed=bad)


def test_numpy_integers_are_accepted():
    model = _model(random_projection=np.int64(8), random_projection_seed=np.int32(2))
    assert model.final_layer[1].projection.shape == (8, _N_FEATURES)
    assert type(model.random_projection) is int
    assert type(model.random_projection_seed) is int


# ----------------------------------------------------------------------------
# Released checkpoints (network)
# ----------------------------------------------------------------------------

_PINNED = {
    "mae_r9cm_L2": "c861ade23dddcb0fc0be718944334a2316d635a5",
    "jepa_r9cm_L2": "88877d4cf2304cd3419181a7b1b416ce1f5d508f",
}


def _from_hub(pretext_geometry, **kwargs):
    pretext, radius, length = pretext_geometry
    repo = Guetschel2026.hub_repo_id(pretext, radius, int(length))
    try:
        return Guetschel2026.from_pretrained(
            repo,
            chs_info=_chs(),
            n_times=1000,
            n_outputs=2,
            sfreq=200.0,
            random_projection=None,
            **kwargs,
        ).eval()
    except (URLError, OSError) as err:  # offline CI
        pytest.skip(f"Hugging Face Hub not reachable: {err}")


@pytest.mark.network
@pytest.mark.huggingface
@needs_safetensors
@pytest.mark.parametrize(
    "name, geometry",
    [
        ("mae_r9cm_L2", ("mae", "9cm", 2)),
        ("jepa_r9cm_L2", ("jepa", "9cm", 2)),
    ],
)
def test_released_checkpoints_reproduce_reference_features(name, geometry):
    model = _from_hub(geometry, revision=_PINNED[name])
    with torch.no_grad():
        features = model(_windows(1000), return_features=True)["features"]
    head, norm = _summary(features, list(range(5)))
    gold = _GOLDEN["released"][name]
    torch.testing.assert_close(head, torch.tensor(gold["head"]), rtol=0, atol=1e-5)
    torch.testing.assert_close(norm, torch.tensor(gold["norm"]), rtol=1e-5, atol=1e-4)
    assert model.clip_sigma == 15.0


@pytest.mark.network
@pytest.mark.huggingface
@needs_safetensors
def test_released_epoch_file_loads():
    final = _from_hub(("mae", "9cm", 2), revision=_PINNED["mae_r9cm_L2"])
    epoch = _from_hub(
        ("mae", "9cm", 2),
        revision=_PINNED["mae_r9cm_L2"],
        filename="epoch_05/model.safetensors",
    )
    assert not torch.equal(
        epoch.feature_encoder.linear.weight, final.feature_encoder.linear.weight
    )
    assert {k: v.shape for k, v in _backbone_state(epoch).items()} == {
        k: torch.Size(v) for k, v in _RELEASED_KEYS.items()
    }


def test_attention_dropout_is_off_in_eval_mode():
    model = _model(random_projection=None, drop_prob=0.5).eval()
    x = _windows(1000)
    with torch.no_grad():
        assert torch.equal(model(x), model(x))
        # only the attention is checked: the other dropouts are modules in eval
        attention = model.model.transformer.layers[0].self_attn
        z = torch.randn(2, 6, attention.packed_proj.in_features)
        assert torch.equal(attention(z), attention(z))
        attention.train()
        assert not torch.equal(attention(z), attention(z))


@pytest.mark.filterwarnings("ignore:Mismatch dtype")
def test_autocast_keeps_the_residual_stream_in_the_autocast_dtype():
    model = _model(random_projection=None).eval()
    x = _windows(1000)
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        features = model(x, return_features=True)["features"]
    assert features.dtype == torch.bfloat16


def test_random_projection_repr_shows_its_shape():
    head = _model(random_projection=16).final_layer[1]
    assert "n_components=16" in repr(head)
    assert f"n_features={head.n_features}" in repr(head)


def test_numpy_integer_head_arguments_round_trip_through_the_config():
    model = _model(random_projection=np.int64(16), random_projection_seed=np.int32(3))
    config = model.get_config()
    assert config["random_projection"] == 16 and type(config["random_projection"]) is int
    assert config["random_projection_seed"] == 3
    assert type(config["random_projection_seed"]) is int
    rebuilt = Guetschel2026.from_config(config)
    assert torch.equal(
        rebuilt.final_layer[1].projection, model.final_layer[1].projection
    )


def test_logits_require_the_configured_window_length():
    model = _model(random_projection=None).eval()
    n_times = model.n_times
    x = torch.randn(2, model.n_chans, n_times + 200) * 1e-5
    with pytest.raises(ValueError, match="return_features=True"):
        model(x)
    with torch.no_grad():
        features = model(x, return_features=True)["features"]
    assert features.shape[2] == (n_times + 200 - 200) // 180 + 1
    # trailing samples that do not fill a patch keep the patch count
    with torch.no_grad():
        model(torch.randn(2, model.n_chans, n_times + 10) * 1e-5)


@pytest.mark.parametrize("return_features", [False, True])
def test_rejects_inputs_with_another_number_of_channels(return_features):
    model = _model(random_projection=None)
    for n_chans in (1, model.n_chans + 1):
        x = torch.randn(2, n_chans, model.n_times) * 1e-5
        with pytest.raises(ValueError, match="channels"):
            model(x, return_features=return_features)
