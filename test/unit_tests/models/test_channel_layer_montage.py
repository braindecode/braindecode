"""Channel layer in the montage models: BIOT, CodeBrain, MIRepNet.

Every registered strategy is built and run on the compatibility geometries
G1-G4 of ``test_pretrained_compat``; a cell either forwards a finite output or
raises one of the declared ``ValueError`` listed in ``DECLARED``.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import torch

from braindecode.models import BIOT, CodeBrain, MIRepNet
from braindecode.models.biot import BIOT_CHANNEL_ORDER
from braindecode.models.codebrain import CODEBRAIN_CHANNEL_ORDER
from braindecode.models.mirepnet import MIREPNET_CHANNEL_ORDER

from .test_pretrained_compat import (
    COMPAT,
    _montage,
    chs_from_montage,
    geometries,
)

STRATEGIES = [
    "exact",
    "zero",
    "nearest",
    "idw",
    "spline",
    "field",
    "source",
    "wiener",
    "region",
    "latent",
]
GEOMETRIES = ["G1", "G2", "G3", "G3b", "G4"]

EXACT_MISSING = r"Strategy 'exact': target channels .* are not in the input"
NO_POSITIONS = r"needs at least \d+ channels with a position; got 0"
WIENER_FAR = r"'wiener': channels .* no electrode of the fitted dense montage"

# Small backbones: the grid checks the channel layer, not the model size.
SMALL = {
    "BIOT": dict(embed_dim=32, num_heads=2, num_layers=1),
    "CodeBrain": dict(
        res_channels=16, skip_channels=16, out_channels=16, num_res_layers=1
    ),
    "MIRepNet": dict(embed_dim=32, num_heads=2, num_layers=1),
}

# Declared errors: (model, geometry, strategy) -> message pattern.
DECLARED: dict[tuple[str, str, str], str] = {}
# BIOT under a channel strategy takes monopolar EEG; the bipolar canonical
# names (G1, G4) carry no electrode and no position.
for _g in ("G1", "G4"):
    DECLARED[("BIOT", _g, "exact")] = EXACT_MISSING
    for _s in STRATEGIES:
        if _s not in ("exact", "zero"):
            DECLARED[("BIOT", _g, _s)] = NO_POSITIONS
# biosemi64 lacks A1/A2; coordinates-only and partial name sets miss targets.
for _m, _gs in {
    "BIOT": ("G2", "G3", "G3b"),
    "CodeBrain": ("G3", "G3b"),
    "MIRepNet": ("G1", "G3", "G3b"),
}.items():
    for _g in _gs:
        DECLARED[(_m, _g, "exact")] = EXACT_MISSING
# Electrodes farther than 15 mm from the fitted dense (standard_1005) montage.
for _m, _g in [
    ("BIOT", "G2"),
    ("BIOT", "G3"),
    ("CodeBrain", "G3"),
    ("MIRepNet", "G1"),
    ("MIRepNet", "G3"),
]:
    DECLARED[(_m, _g, "wiener")] = WIENER_FAR


def make_dense_set():
    """Tiny synthetic dense recording to fit ``wiener`` (standard_1005)."""
    chs = chs_from_montage(_montage("standard_1005").ch_names)
    pos = np.array([ch["loc"][:3] for ch in chs])
    dist = np.linalg.norm(pos[:, None] - pos[None], axis=-1)
    chol = np.linalg.cholesky(np.exp(-dist / 0.05) + 1e-6 * np.eye(len(pos)))
    X = np.random.default_rng(0).standard_normal((300, len(pos))) @ chol.T
    return X, chs


@pytest.fixture(scope="module")
def dense_set():
    return make_dense_set()


def build(name, chs_info, small, n_times=None, **kw):
    """Model ``name`` of ``COMPAT`` with a small backbone on ``chs_info``."""
    spec = COMPAT[name]
    args = dict(
        n_outputs=2,
        chs_info=chs_info,
        sfreq=spec["sfreq"],
        n_times=n_times or spec["n_times"],
        **spec.get("kwargs", {}),
        **small,
    )
    args.update(kw)
    return spec["cls"](**args).eval()


def check_grid_cell(name, gname, strategy, small, declared, dense):
    """Build + forward one cell; finite output or the declared error."""
    geos = geometries(COMPAT[name])
    if gname not in geos:
        pytest.skip(f"{name} has no {gname} geometry")
    gkw = geos[gname]
    x = torch.randn(2, len(gkw["chs_info"]), gkw["n_times"])

    def run():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = build(name, gkw["chs_info"], small, channel_strategy=strategy)
            if strategy == "wiener":
                model.channel_tokenizer.fit(*dense)
            with torch.no_grad():
                return model(x)

    if declared is not None:
        with pytest.raises(ValueError, match=declared):
            run()
        return
    y = run()
    assert y.shape[0] == 2
    assert torch.isfinite(y).all()


@pytest.mark.parametrize("name", list(SMALL))
@pytest.mark.parametrize("gname", GEOMETRIES)
@pytest.mark.parametrize("strategy", STRATEGIES)
def test_montage_strategy_geometry_grid(name, gname, strategy, dense_set):
    check_grid_cell(
        name,
        gname,
        strategy,
        SMALL[name],
        DECLARED.get((name, gname, strategy)),
        dense_set,
    )


CANONICAL = {
    "BIOT": BIOT_CHANNEL_ORDER,
    "CodeBrain": CODEBRAIN_CHANNEL_ORDER,
    "MIRepNet": MIREPNET_CHANNEL_ORDER,
}


@pytest.mark.parametrize("name", list(SMALL))
def test_native_canonical_keeps_state_and_does_not_warn(name):
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        model = build(name, chs_from_montage(CANONICAL[name]), SMALL[name])
    assert not model._channel_layer
    assert not any(k.startswith("channel_tokenizer") for k in model.state_dict())
    assert model.get_config()["channel_strategy"] == "native"


@pytest.mark.parametrize("name", list(SMALL))
def test_native_non_canonical_montage_warns_and_runs(name):
    gkw = geometries(COMPAT[name])["G2"]
    with pytest.warns(FutureWarning, match=r"pass channel_strategy="):
        model = build(name, gkw["chs_info"], SMALL[name])
    with torch.no_grad():
        y = model(torch.randn(1, 64, gkw["n_times"]))
    assert torch.isfinite(y).all()


@pytest.mark.parametrize("n_chans,warns", [(18, False), (16, False), (22, True)])
def test_biot_native_without_chs_info_checks_n_chans(n_chans, warns):
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        BIOT(n_outputs=2, n_chans=n_chans, n_times=800, sfreq=200, **SMALL["BIOT"])
    assert any(w.category is FutureWarning for w in rec) is warns


@pytest.mark.parametrize("name", ["CodeBrain", "MIRepNet"])
def test_exact_on_permuted_montage_reproduces_native(name):
    """``exact`` turns the reversed montage back into the canonical order."""
    canon = chs_from_montage(CANONICAL[name])
    torch.manual_seed(0)
    native = build(name, canon, SMALL[name])
    exact = build(name, canon[::-1], SMALL[name], channel_strategy="exact")
    exact.load_state_dict(native.state_dict(), strict=True)
    x = torch.randn(2, len(canon), COMPAT[name]["n_times"])
    with torch.no_grad():
        torch.testing.assert_close(exact(x.flip(1)), native(x), rtol=0, atol=0)


def test_biot_strategy_forms_bipolar_derivations_from_monopolar_input():
    electrodes = list(
        dict.fromkeys(e for ch in BIOT_CHANNEL_ORDER for e in ch.split("-"))
    )
    mono = torch.randn(2, len(electrodes), 800)
    bipolar = torch.stack(
        [
            mono[:, electrodes.index(a)] - mono[:, electrodes.index(b)]
            for a, b in (ch.split("-") for ch in BIOT_CHANNEL_ORDER)
        ],
        dim=1,
    )
    torch.manual_seed(0)
    native = build("BIOT", chs_from_montage(BIOT_CHANNEL_ORDER), SMALL["BIOT"])
    layered = build(
        "BIOT",
        chs_from_montage(electrodes[::-1]),
        SMALL["BIOT"],
        channel_strategy="exact",
    )
    layered.load_state_dict(native.state_dict(), strict=True)
    with torch.no_grad():
        torch.testing.assert_close(layered(mono.flip(1)), native(bipolar))


def test_forward_takes_a_per_call_montage():
    model = build(
        "CodeBrain",
        chs_from_montage(CODEBRAIN_CHANNEL_ORDER),
        SMALL["CodeBrain"],
        channel_strategy="spline",
    )
    bio = _montage("biosemi64").ch_names[:32]
    with torch.no_grad():
        y = model(torch.randn(1, 32, 800), chs_info=chs_from_montage(bio, "biosemi64"))
    assert y.shape == (1, 2) and torch.isfinite(y).all()
    assert len(model.channel_tokenizer._cache) == 2


def test_montage_models_declare_their_target():
    assert len(MIREPNET_CHANNEL_ORDER) == 45
    for cls, n in ((BIOT, 18), (CodeBrain, 19), (MIRepNet, 45)):
        assert cls._channel_target.interface == "montage"
        assert len(cls._channel_target.sensors().names) == n


def test_source_lead_field_is_finite_on_the_z_axis():
    """biosemi64 Cz sits exactly above the sphere centre (BIOT G2 + source)."""
    from braindecode.modules.channels.head import get_sphere_head

    head = get_sphere_head()
    on_axis = np.array([[0.0, 0.0, 0.095], [0.05, 0.0, 0.08], [0.0, 0.05, 0.08]])
    assert np.isfinite(head.leadfield(on_axis)).all()
