"""``on_non_divisible`` of the patch-based models: pad (default), crop, error."""

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

_NRVQ = dict(
    n_chans=3, sfreq=200, channel_names=("f3", "f4", "cz"), out_chans=4, num_heads=4
)
# name -> (class, kwargs, patch_size, step, divisible n_times, non-divisible n_times)
# fmt: off
_MODELS = {
    "STEEGFormer": (STEEGFormer, dict(n_chans=3, n_outputs=2, embed_dim=32, depth=1, num_heads=2), 16, 16, 64, 70),
    "Brant": (Brant, dict(n_chans=2, n_outputs=2, sfreq=250.0, patch_size=250, embed_dim=32, ffn_dim=64,
                          temporal_n_layers=1, spatial_n_layers=1, n_heads=2), 250, 250, 500, 600),
    "BaRISTA": (BaRISTA, dict(n_chans=4, n_outputs=2, spatial_scale="none", patch_size=32, d_model=16,
                              n_layers=1, num_heads=2, cnn_depth=1), 32, 32, 128, 140),
    "SleepFM": (SleepFM, dict(n_chans=3, n_outputs=5, sfreq=128.0, patch_size=64, embed_dim=16, drop_prob=0.0,
                              max_seq_length=32, num_heads=4, num_layers=1, pooling_heads=4), 64, 64, 256, 300),
    "SleepFMStager": (SleepFMStager, dict(n_chans=3, n_outputs=5, sfreq=128.0, patch_size=64, embed_dim=16,
                                          drop_prob=0.0, max_seq_length=32, channel_modalities=["A", "A", "B"],
                                          encoder_chunk_patches=2, encoder_num_heads=4, encoder_num_layers=1,
                                          encoder_pooling_heads=4, staging_pooling_heads=4), 64, 64, 256, 300),
    "NeuroRVQ": (NeuroRVQ, dict(_NRVQ, n_outputs=2, depth=1, max_patches=8), 200, 200, 400, 450),
    "NeuroRVQTokenizer": (NeuroRVQTokenizer, dict(_NRVQ, max_patches=4, encoder_depth=1, decoder_depth=1,
                                                  n_code=16, code_dim=16, num_quantizers=2), 200, 200, 400, 450),
    "DuIN": (DuIN, dict(n_chans=3, n_outputs=2, sfreq=1000.0, patch_size=20, n_filters=(8, 4), kernel_sizes=(5, 3),
                        strides=(5, 1), n_layers=1, n_heads=2, head_dim=8, ffn_dim=16, head_hidden_dim=8),
             20, 20, 60, 70),
    "CSBrain": (CSBrain, dict(n_chans=3, n_outputs=2, n_layer=1, nhead=2), 200, 200, 400, 450),
    # string key: test/conftest.py marks tests parametrized with the REVE class as network-only
    "REVE-overlap": (REVE, dict(chs_info=[{"ch_name": "Cz"}, {"ch_name": "Pz"}], n_outputs=2, embed_dim=32,
                                depth=1, heads=2, head_dim=8, patch_size=20, patch_overlap=4), 20, 16, 68, 75),
}
# fmt: on
_LEGACY_CROP = [
    "STEEGFormer",
    "Brant",
    "BaRISTA",
    "SleepFM",
    "SleepFMStager",
    "REVE-overlap",
]


@pytest.fixture(autouse=True)
def _reve_position_bank(tmp_path, monkeypatch):
    bank = {"Cz": [0.0, 0.0, 0.09], "Pz": [0.0, -0.07, 0.06]}
    (tmp_path / "reve_positions.json").write_text(json.dumps(bank))
    monkeypatch.setenv("REVE_POSITIONS_PATH", str(tmp_path))


def _build(name, n_times, seed=0, **kwargs):
    cls, base, *_ = _MODELS[name]
    torch.manual_seed(seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = cls(**base, n_times=n_times, **kwargs).eval()
    for module in model.modules():  # warm codebooks: no k-means in forward
        if isinstance(module, _EMAEmbedding):
            module.weight.data.copy_(
                F.normalize(torch.randn(module.weight.shape), dim=-1)
            )
            module.initted.fill_(1.0)
    return model


def _run(model, n_times, n_cut=None):
    torch.manual_seed(1)
    x = torch.randn(2, model.n_chans, n_times)
    if n_cut is not None:
        x = F.pad(x, (0, n_cut - n_times)) if n_cut > n_times else x[..., :n_cut]
    with torch.no_grad():
        return model(x)


@pytest.mark.parametrize("name", list(_MODELS))
def test_on_non_divisible(name):
    cls, base, patch, step, divisible, odd = _MODELS[name]
    # Tiled window: identical model and output in every mode (parity of the default).
    ref = _build(name, divisible, on_non_divisible="crop")
    for mode in ("pad", "error"):
        model = _build(name, divisible, on_non_divisible=mode)
        assert all(
            torch.equal(v, ref.state_dict()[k]) for k, v in model.state_dict().items()
        )
        torch.testing.assert_close(
            _run(model, divisible), _run(ref, divisible), rtol=0, atol=0
        )
    # Non-divisible: pad / crop are the same weights on the zero-padded / cropped window.
    n_steps, rest = divmod(odd - patch, step)
    for mode, cut in (
        ("pad", patch + (n_steps + 1) * step),
        ("crop", patch + n_steps * step),
    ):
        model = _build(name, odd, on_non_divisible=mode)
        same = _build(name, cut, seed=1)
        same.load_state_dict(model.state_dict())
        torch.testing.assert_close(
            _run(model, odd), _run(same, odd, cut), rtol=0, atol=0
        )
    with pytest.raises(ValueError, match="divisible"):
        cls(**base, n_times=odd, on_non_divisible="error")
    with pytest.raises(ValueError, match="on_non_divisible"):
        cls(**base, n_times=divisible, on_non_divisible="bogus")


@pytest.mark.parametrize("name", list(_MODELS))
def test_config_without_the_key_rebuilds_the_old_model(name):
    """Configs saved before the option existed load as the model they were."""
    legacy = "crop" if name in _LEGACY_CROP else "pad"
    model = _build(name, _MODELS[name][5], on_non_divisible=legacy)
    config = model.get_config()
    del config["on_non_divisible"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rebuilt = type(model).from_config(json.loads(json.dumps(config)))
    rebuilt.load_state_dict(model.state_dict(), strict=True)


def test_checkpoint_without_the_key_loads_as_crop(tmp_path):
    pytest.importorskip("huggingface_hub")
    model = _build("Brant", 600, on_non_divisible="crop")
    model.save_pretrained(tmp_path / "ckpt")
    config_file = tmp_path / "ckpt" / "config.json"
    config = json.loads(config_file.read_text())
    del config["on_non_divisible"]
    config_file.write_text(json.dumps(config))
    loaded = Brant.from_pretrained(str(tmp_path / "ckpt"), strict=True)
    assert loaded.get_config()["on_non_divisible"] == "crop"
