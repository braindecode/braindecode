"""``on_non_divisible`` of the patch-based models: pad (default), crop, error.

For every model, a window that patches tile exactly gives the same parameters
and the same output in all three modes (the default ``"pad"`` changes nothing
for divisible windows). A window they do not tile is

- ``"pad"``: the output of the same weights on the right zero-padded window,
- ``"crop"``: the output of the same weights on the cropped window (what the
  reference implementations of STEEGFormer, Brant, BaRISTA, SleepFM and REVE
  did unconditionally),
- ``"error"``: a ``ValueError`` (what NeuroRVQ, NeuroRVQTokenizer and DuIN
  did unconditionally).

The parametrization uses string keys, not the classes: ``test/conftest.py``
marks a test parametrized with the REVE class as network-only.
"""

import json
import warnings

import pytest
import torch
import torch.nn.functional as F

from braindecode.models import (
    REVE,
    BaRISTA,
    Brant,
    CSBrain,
    DuIN,
    NeuroRVQ,
    NeuroRVQTokenizer,
    SleepFM,
    SleepFMStager,
    STEEGFormer,
)
from braindecode.models.neurorvq_tokenizer import _EMAEmbedding

_NEURORVQ_CHANNELS = ("f3", "f4", "cz")

# name -> (class, kwargs, patch_size, step, divisible, non-divisible, short)
# ``step`` is the hop between patch starts (``patch_size`` unless overlapping).
_MODELS = {
    "STEEGFormer": (
        STEEGFormer,
        dict(n_chans=3, n_outputs=2, embed_dim=32, depth=1, num_heads=2),
        16,
        16,
        64,
        70,
        10,
    ),
    "Brant": (
        Brant,
        dict(
            n_chans=2,
            n_outputs=2,
            sfreq=250.0,
            patch_size=250,
            embed_dim=32,
            ffn_dim=64,
            temporal_n_layers=1,
            spatial_n_layers=1,
            n_heads=2,
        ),
        250,
        250,
        500,
        600,
        100,
    ),
    "BaRISTA": (
        BaRISTA,
        # learned pooling: its weight is sized by the patch count
        dict(
            n_chans=4,
            n_outputs=2,
            spatial_scale="none",
            patch_size=32,
            d_model=16,
            n_layers=1,
            num_heads=2,
            cnn_depth=1,
        ),
        32,
        32,
        128,
        140,
        20,
    ),
    "SleepFM": (
        SleepFM,
        dict(
            n_chans=3,
            n_outputs=5,
            sfreq=128.0,
            patch_size=64,
            embed_dim=16,
            drop_prob=0.0,
            max_seq_length=32,
            num_heads=4,
            num_layers=1,
            pooling_heads=4,
        ),
        64,
        64,
        256,
        300,
        40,
    ),
    "SleepFMStager": (
        SleepFMStager,
        # chunks of 2 patches: the padded patch starts a chunk of its own
        dict(
            n_chans=3,
            n_outputs=5,
            sfreq=128.0,
            patch_size=64,
            embed_dim=16,
            drop_prob=0.0,
            max_seq_length=32,
            channel_modalities=["A", "A", "B"],
            encoder_chunk_patches=2,
            encoder_num_heads=4,
            encoder_num_layers=1,
            encoder_pooling_heads=4,
            staging_pooling_heads=4,
        ),
        64,
        64,
        256,
        300,
        40,
    ),
    "NeuroRVQ": (
        NeuroRVQ,
        dict(
            n_chans=3,
            n_outputs=2,
            sfreq=200,
            channel_names=_NEURORVQ_CHANNELS,
            depth=1,
            num_heads=4,
            out_chans=4,
            max_patches=8,
        ),
        200,
        200,
        400,
        450,
        100,
    ),
    "NeuroRVQTokenizer": (
        NeuroRVQTokenizer,
        dict(
            n_chans=3,
            sfreq=200,
            channel_names=_NEURORVQ_CHANNELS,
            max_patches=4,
            out_chans=4,
            num_heads=4,
            encoder_depth=1,
            decoder_depth=1,
            n_code=16,
            code_dim=16,
            num_quantizers=2,
        ),
        200,
        200,
        400,
        450,
        100,
    ),
    "DuIN": (
        DuIN,
        dict(
            n_chans=3,
            n_outputs=2,
            sfreq=1000.0,
            patch_size=20,
            n_filters=(8, 4),
            kernel_sizes=(5, 3),
            strides=(5, 1),
            n_layers=1,
            n_heads=2,
            head_dim=8,
            ffn_dim=16,
            head_hidden_dim=8,
        ),
        20,
        20,
        60,
        70,
        15,
    ),
    "CSBrain": (
        CSBrain,
        dict(n_chans=3, n_outputs=2, n_layer=1, nhead=2),
        200,
        200,
        400,
        450,
        100,
    ),
    # Overlapping patches: 20 samples every 16 (patch_overlap=4).
    "REVE-overlap": (
        REVE,
        dict(
            chs_info=[{"ch_name": "Cz"}, {"ch_name": "Pz"}],
            n_outputs=2,
            embed_dim=32,
            depth=1,
            heads=2,
            head_dim=8,
            patch_size=20,
            patch_overlap=4,
        ),
        20,
        16,
        68,
        75,
        10,
    ),
}
_NAMES = list(_MODELS)


@pytest.fixture(autouse=True)
def _reve_position_bank(tmp_path, monkeypatch):
    """Serve REVE's position bank from a local file instead of the Hub."""
    bank = {"Cz": [0.0, 0.0, 0.09], "Pz": [0.0, -0.07, 0.06]}
    (tmp_path / "reve_positions.json").write_text(json.dumps(bank))
    monkeypatch.setenv("REVE_POSITIONS_PATH", str(tmp_path))


def _build(name, n_times, seed=0, **kwargs):
    cls, base, *_ = _MODELS[name]
    torch.manual_seed(seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = cls(**base, n_times=n_times, **kwargs).eval()
    if cls is NeuroRVQTokenizer:
        # Warm codebooks, so no data-dependent k-means runs in forward.
        gen = torch.Generator().manual_seed(seed)
        for module in model.modules():
            if isinstance(module, _EMAEmbedding):
                weight = torch.randn(module.weight.shape, generator=gen)
                module.weight.data.copy_(F.normalize(weight, dim=-1))
                module.initted.fill_(1.0)
    return model


def _n_chans(name):
    base = _MODELS[name][1]
    return base.get("n_chans") or len(base["chs_info"])


def _run(model, x):
    with torch.no_grad():
        return model(x)


def _tiled_length(name, n_times, pad):
    """Length the window is padded (``pad``) or cropped to."""
    _, _, patch_size, step, *_ = _MODELS[name]
    if n_times < patch_size:
        return patch_size
    n_steps, rest = divmod(n_times - patch_size, step)
    return patch_size + (n_steps + (pad and rest > 0)) * step


@pytest.mark.parametrize("name", _NAMES)
def test_divisible_window_is_identical_in_every_mode(name):
    """Parity: a tiled window gives the same state dict and the same output
    bit for bit whatever ``on_non_divisible`` is, so the ``"pad"`` default
    reproduces the former ``"crop"``/``"error"`` behaviour exactly."""
    n_times = _MODELS[name][4]
    x = torch.randn(2, _n_chans(name), n_times)
    ref = _build(name, n_times, on_non_divisible="crop")
    ref_state = ref.state_dict()
    ref_out = _run(ref, x)
    for mode in ("pad", "error", None):
        kwargs = {} if mode is None else {"on_non_divisible": mode}
        model = _build(name, n_times, **kwargs)
        state = model.state_dict()
        assert list(state) == list(ref_state)
        for key, value in state.items():
            assert torch.equal(value, ref_state[key]), key
        torch.testing.assert_close(_run(model, x), ref_out, rtol=0, atol=0)


@pytest.mark.parametrize("which", ["non_divisible", "short"])
@pytest.mark.parametrize("name", _NAMES)
def test_pad_equals_zero_padded_window(name, which):
    """``"pad"`` (the default) is the same weights on the right zero-padded
    window: the positional tables, heads and patch guards count the padded
    patch, so the state dict loads strictly into a model built at that length."""
    n_times = _MODELS[name][5 if which == "non_divisible" else 6]
    padded = _tiled_length(name, n_times, pad=True)
    assert padded > n_times
    model = _build(name, n_times)
    reference = _build(name, padded, seed=1)
    reference.load_state_dict(model.state_dict())
    x = torch.randn(2, _n_chans(name), n_times)
    out = _run(model, x)
    torch.testing.assert_close(
        out, _run(reference, F.pad(x, (0, padded - n_times))), rtol=0, atol=0
    )
    for tensor in out if isinstance(out, tuple) else (out,):
        assert torch.isfinite(tensor).all()


@pytest.mark.parametrize("name", _NAMES)
def test_crop_equals_cropped_window(name):
    """``"crop"`` is the same weights on the window cropped to whole patches."""
    n_times = _MODELS[name][5]
    cropped = _tiled_length(name, n_times, pad=False)
    assert cropped < n_times
    model = _build(name, n_times, on_non_divisible="crop")
    reference = _build(name, cropped, seed=1)
    reference.load_state_dict(model.state_dict())
    x = torch.randn(2, _n_chans(name), n_times)
    torch.testing.assert_close(
        _run(model, x), _run(reference, x[..., :cropped]), rtol=0, atol=0
    )


@pytest.mark.parametrize("name", _NAMES)
def test_crop_rejects_a_window_shorter_than_one_patch(name):
    n_times = _MODELS[name][6]
    with pytest.raises(ValueError, match="full patch"):
        model = _build(name, n_times, on_non_divisible="crop")
        _run(model, torch.randn(1, _n_chans(name), n_times))


@pytest.mark.parametrize("name", _NAMES)
def test_error_rejects_non_divisible_window(name):
    _, base, *_, non_divisible, short = _MODELS[name]
    for n_times in (non_divisible, short):
        with pytest.raises(ValueError, match="divisible"):
            _MODELS[name][0](**base, n_times=n_times, on_non_divisible="error")


@pytest.mark.parametrize("name", _NAMES)
def test_pad_warns_and_bad_mode_raises(name):
    cls, base, *_ = _MODELS[name]
    with pytest.warns(UserWarning, match="padded"):
        cls(**base, n_times=_MODELS[name][5])
    with pytest.raises(ValueError, match="on_non_divisible"):
        cls(**base, n_times=_MODELS[name][4], on_non_divisible="bogus")


@pytest.mark.parametrize("name", _NAMES)
def test_on_non_divisible_is_in_the_config(name):
    """The mode is saved with the model, so ``from_config`` rebuilds the same
    patch count (and parameter shapes) for a non-divisible window."""
    n_times = _MODELS[name][5]
    model = _build(name, n_times, on_non_divisible="crop")
    config = model.get_config()
    assert config["on_non_divisible"] == "crop"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rebuilt = type(model).from_config(json.loads(json.dumps(config)))
    rebuilt.load_state_dict(model.state_dict())


_LEGACY_CROP = [
    n for n in _NAMES if getattr(_MODELS[n][0], "_legacy_on_non_divisible", None)
]


def test_legacy_crop_models_are_the_ones_that_cropped():
    assert sorted({_MODELS[n][0].__name__ for n in _LEGACY_CROP}) == sorted(
        ["STEEGFormer", "Brant", "BaRISTA", "SleepFM", "SleepFMStager", "REVE"]
    )


@pytest.mark.parametrize("name", _NAMES)
def test_config_without_the_key_rebuilds_the_old_model(name):
    """A config saved before ``on_non_divisible`` existed has no such key: models
    that used to crop rebuild with ``"crop"`` (same shapes as the saved weights),
    the others (which used to raise) keep the new default."""
    n_times = _MODELS[name][5]
    legacy = "crop" if name in _LEGACY_CROP else "pad"
    model = _build(name, n_times, on_non_divisible=legacy)
    config = model.get_config()
    del config["on_non_divisible"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rebuilt = type(model).from_config(json.loads(json.dumps(config)))
    assert rebuilt.get_config()["on_non_divisible"] == legacy
    rebuilt.load_state_dict(model.state_dict(), strict=True)


@pytest.mark.parametrize("name", _LEGACY_CROP)
def test_checkpoint_without_the_key_loads_as_crop(name, tmp_path):
    """``from_pretrained`` of a pre-``on_non_divisible`` save loads strictly."""
    pytest.importorskip("huggingface_hub")
    n_times = _MODELS[name][5]
    model = _build(name, n_times, on_non_divisible="crop")
    ckpt = tmp_path / "ckpt"
    model.save_pretrained(ckpt)
    config = json.loads((ckpt / "config.json").read_text())
    del config["on_non_divisible"]
    (ckpt / "config.json").write_text(json.dumps(config))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        loaded = type(model).from_pretrained(str(ckpt), strict=True)
    assert loaded.get_config()["on_non_divisible"] == "crop"
    x = torch.randn(2, _n_chans(name), n_times)
    torch.testing.assert_close(_run(loaded.eval(), x), _run(model, x))
