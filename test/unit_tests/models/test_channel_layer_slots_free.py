"""Channel layer in the slots model (EEGDINO) and the free models (CBraMod,
Brant, BrainBERT): strategy x geometry grid, native state, masks, TorchScript.
"""

from __future__ import annotations

import re
import warnings

import numpy as np
import pytest
import torch

from braindecode.models import EEGDINO, BrainBERT, Brant, CBraMod
from braindecode.models.eegdino import EEGDINO_SLOT_CHANNELS
from braindecode.modules.channels import ChannelEncoding

from .test_channel_layer_montage import (
    EXACT_MISSING,
    GEOMETRIES,
    NO_POSITIONS,
    STRATEGIES,
    WIENER_FAR,
    build,
    check_grid_cell,
    make_dense_set,
)
from .test_pretrained_compat import COMPAT, chs_from_montage, geometries

SMALL = {
    "EEGDINO": dict(n_layer=1, nhead=2, dim_feedforward=32),
    "CBraMod": dict(n_layer=1, nhead=2, dim_feedforward=32),
    "Brant": dict(
        embed_dim=32, ffn_dim=32, temporal_n_layers=1, spatial_n_layers=1, n_heads=2
    ),
    "BrainBERT": dict(hidden_dim=32, ffn_dim=32, n_layers=1, n_heads=2),
}

DECLARED = {
    # Coordinates-only and partial name sets miss slot electrodes.
    ("EEGDINO", "G3", "exact"): EXACT_MISSING,
    ("EEGDINO", "G3b", "exact"): EXACT_MISSING,
    ("EEGDINO", "G3", "wiener"): WIENER_FAR,
    # BrainBERT's single channel "E1" has no position.
    ("BrainBERT", "G1", "source"): NO_POSITIONS,
    ("BrainBERT", "G1", "latent"): NO_POSITIONS,
    ("BrainBERT", "G4", "source"): NO_POSITIONS,
    ("BrainBERT", "G4", "latent"): NO_POSITIONS,
}


@pytest.fixture(scope="module")
def dense_set():
    return make_dense_set()


def _canonical_chs(name):
    if name == "BrainBERT":
        return [{"ch_name": "E1", "kind": "eeg", "loc": np.zeros(12)}]
    return chs_from_montage(EEGDINO_SLOT_CHANNELS)


@pytest.mark.parametrize("name", list(SMALL))
@pytest.mark.parametrize("gname", GEOMETRIES)
@pytest.mark.parametrize("strategy", STRATEGIES)
def test_slots_free_strategy_geometry_grid(name, gname, strategy, dense_set):
    check_grid_cell(
        name,
        gname,
        strategy,
        SMALL[name],
        DECLARED.get((name, gname, strategy)),
        dense_set,
    )


@pytest.mark.parametrize("name", list(SMALL))
def test_native_keeps_state_and_does_not_warn(name):
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        model = build(name, _canonical_chs(name), SMALL[name])
    assert not model._channel_layer
    assert not any(k.startswith("channel_tokenizer") for k in model.state_dict())
    assert model.get_config()["channel_strategy"] == "native"


def test_eegdino_exact_on_permuted_montage_reproduces_native():
    canon = _canonical_chs("EEGDINO")
    torch.manual_seed(0)
    native = build("EEGDINO", canon, SMALL["EEGDINO"])
    exact = build("EEGDINO", canon[::-1], SMALL["EEGDINO"], channel_strategy="exact")
    exact.load_state_dict(native.state_dict(), strict=True)
    x = torch.randn(2, 19, 800)
    with torch.no_grad():
        torch.testing.assert_close(exact(x.flip(1)), native(x), rtol=0, atol=0)


def test_eegdino_native_more_than_19_channels_is_declared():
    gkw = geometries(COMPAT["EEGDINO"])["G2"]
    with pytest.raises(ValueError, match=re.escape("channel_strategy=...")):
        build("EEGDINO", gkw["chs_info"], SMALL["EEGDINO"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = build(
            "EEGDINO", gkw["chs_info"], SMALL["EEGDINO"], channel_strategy="spline"
        )
    assert model.channel_tokenizer.target.n_slots == 19


@pytest.mark.parametrize("name", ["CBraMod", "Brant"])
def test_free_source_feeds_parcels(name):
    model = build(
        name,
        _canonical_chs(name),
        SMALL[name],
        channel_strategy="source",
        channel_strategy_kwargs={"n_parcels": 32},
    )
    seen = {}
    hooked = model.patch_embedding if name == "CBraMod" else model.patch_tokenizer
    hooked.register_forward_pre_hook(lambda m, a: seen.update(x=a[0]))
    with torch.no_grad():
        model(torch.randn(1, 19, COMPAT[name]["n_times"]))
    assert seen["x"].shape[1] == 32


def test_cbramod_unobserved_channels_reach_the_patch_mask(monkeypatch):
    model = build(
        "CBraMod", _canonical_chs("CBraMod"), SMALL["CBraMod"], channel_strategy="zero"
    )
    observed = torch.ones(19, dtype=torch.bool)
    observed[[0, 5]] = False

    def fake_encode(x, chs_info=None):
        return ChannelEncoding(x, None, None, observed, observed.float(), None)

    monkeypatch.setattr(model, "_encode_channels", fake_encode)
    seen = {}
    model.patch_embedding.register_forward_pre_hook(
        lambda m, a: seen.update(mask=a[1])
    )
    with torch.no_grad():
        model(torch.randn(2, 19, 800))
    mask = seen["mask"]
    assert mask.shape == (2, 19, 4)
    assert mask[:, [0, 5]].eq(1).all() and mask.sum() == 2 * 2 * 4


def test_cbramod_all_observed_leaves_mask_untouched():
    model = build(
        "CBraMod",
        _canonical_chs("CBraMod"),
        SMALL["CBraMod"],
        channel_strategy="spline",
    )
    seen = {}
    model.patch_embedding.register_forward_pre_hook(
        lambda m, a: seen.update(mask=a[1])
    )
    with torch.no_grad():
        model(torch.randn(1, 19, 800))
    assert seen["mask"] is None


@pytest.mark.parametrize("cls", [Brant, BrainBERT])
@pytest.mark.parametrize("strategy", ["native", "spline"])
def test_free_scriptable_models_still_script(cls, strategy):
    name = cls.__name__
    chs = _canonical_chs(name)
    # float sfreq: the mixin's scripted ``sfreq`` property is typed float.
    model = build(
        name,
        chs,
        SMALL[name],
        sfreq=float(COMPAT[name]["sfreq"]),
        channel_strategy=strategy,
    )
    scripted = torch.jit.script(model)
    x = torch.randn(1, len(chs), COMPAT[name]["n_times"])
    if strategy == "native":
        with torch.no_grad():
            torch.testing.assert_close(scripted(x), model(x), rtol=0, atol=0)
    else:  # the channel layer is eager-only: a declared error, not a silent skip
        with pytest.raises(Exception, match="eager mode only"):
            scripted(x)


def test_slots_and_free_models_declare_their_interface():
    assert EEGDINO._channel_target.interface == "slots"
    assert EEGDINO._channel_target.n_slots == 19
    for cls in (CBraMod, Brant, BrainBERT):
        assert cls._channel_target.interface == "free"
