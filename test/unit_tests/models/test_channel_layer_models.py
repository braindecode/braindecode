# Authors: Bruno Aristimunha <b.aristimunha@gmail.com>
#
# License: BSD (3-clause)
"""Pretrained models on the channel layer (:mod:`braindecode.modules.channels`).

BENDR is the first pretrained model on the layer.
"""

import warnings

import mne
import numpy as np
import pytest
import torch

from braindecode.models import BENDR
from braindecode.models.bendr import (
    _BENDR_NATIVE_TARGET,
    _BENDR_TARGET_CHS_INFO,
    BENDR_CHANNEL_ORDER,
)
from braindecode.modules.channels import ChannelTarget, ChannelTokenizer

from .test_pretrained_compat import COMPAT, geometries

STRATEGIES = (
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
)
GEOMETRIES = ("G1", "G2", "G3", "G3b", "G4")
# Cells where a strategy cannot serve the montage: a declared ValueError.
DECLARED_ERRORS = {
    ("exact", "G3"): "not in the input montage",  # coords-only E1..E8
    ("exact", "G3b"): "not in the input montage",  # 8 of the 19 targets
    ("wiener", "G3"): "fitted dense montage",  # E6..E8 far from any 10-05 site
}
N_TIMES = 1024

_POS = mne.channels.make_standard_montage("standard_1005").get_positions()["ch_pos"]


def _chs(names):
    return [
        {"ch_name": n, "kind": "eeg", "loc": np.r_[_POS[n], np.zeros(9)]}
        for n in names
    ]


def _dense_fit_set(n_samples=400, seed=0):
    """Smooth synthetic fields on all standard_1005 sites (for ``wiener``)."""
    dense = [
        {"ch_name": n, "kind": "eeg", "loc": np.r_[p, np.zeros(9)]}
        for n, p in _POS.items()
    ]
    P = np.array([ch["loc"][:3] for ch in dense])
    basis = np.c_[np.ones(len(P)), P, P**2, P[:, [0]] * P[:, [1]]]
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_samples, basis.shape[1])) @ basis.T * 1e2
    X += 0.1 * rng.standard_normal(X.shape)
    return X, dense


FOUR = _chs(["Fz", "Cz", "Pz", "Oz"])


def test_bendr_declares_its_channel_contract():
    target = BENDR._channel_target
    assert isinstance(target, ChannelTarget)
    assert target.interface == "montage"
    assert [ch["ch_name"] for ch in target.chs_info] == BENDR_CHANNEL_ORDER
    assert target.non_electrode == ("SCALE",)
    assert len(BENDR_CHANNEL_ORDER) == 20 and BENDR_CHANNEL_ORDER[-1] == "SCALE"


@pytest.mark.parametrize(
    "kwargs",
    [dict(n_chans=20), dict(chs_info=_BENDR_TARGET_CHS_INFO)],
    ids=["n_chans", "canonical_chs_info"],
)
def test_native_canonical_has_no_layer_and_matches_exact(kwargs):
    # Native + canonical: no tokenizer, so released checkpoints load strictly
    # and run bit-identically. ``exact`` on the canonical montage is the
    # identity map, so it gives the same output with the same weights.
    torch.manual_seed(0)
    native = BENDR(n_outputs=2, n_times=N_TIMES, **kwargs).eval()
    assert native.channel_tokenizer is None
    assert not any(k.startswith("channel_tokenizer") for k in native.state_dict())
    exact = BENDR(
        n_outputs=2,
        n_times=N_TIMES,
        chs_info=_BENDR_TARGET_CHS_INFO,
        channel_strategy="exact",
    ).eval()
    exact.load_state_dict(native.state_dict(), strict=True)
    x = torch.randn(2, 20, N_TIMES)
    with torch.no_grad():
        assert torch.equal(native(x), exact(x))


def test_native_non_canonical_keeps_the_former_projection():
    model = BENDR(chs_info=FOUR + _chs(["C3"]), n_outputs=2, n_times=N_TIMES)
    tok = model.channel_tokenizer
    assert tok is not None and tok.strategy_name == "spline"
    legacy = ChannelTokenizer(
        _BENDR_NATIVE_TARGET, "spline", src_chs_info=FOUR + _chs(["C3"]), reg=0.0
    )
    x = torch.randn(1, 5, N_TIMES)
    torch.testing.assert_close(tok(x).x, legacy(x).x, rtol=0, atol=0)
    # The native projection still spline-fills SCALE at its placeholder.
    assert tok(x).weights[-1].abs().sum() > 0


def test_scale_is_a_non_electrode_under_a_strategy():
    model = BENDR(chs_info=FOUR, n_outputs=2, n_times=N_TIMES, channel_strategy="spline")
    enc = model.channel_tokenizer(torch.randn(1, 4, N_TIMES))
    assert enc.x.shape == (1, 20, N_TIMES)
    assert torch.equal(enc.x[0, -1], torch.zeros(N_TIMES))
    assert not enc.observed[-1]
    assert enc.weights[:-1].abs().sum(1).gt(0).all()  # electrodes reconstructed


def test_input_scale_is_copied_never_used_as_a_spatial_source():
    # 19 EEG sites minus Cz, plus SCALE with its placeholder position.
    keep = [ch for ch in _BENDR_TARGET_CHS_INFO if ch["ch_name"] != "CZ"]
    model = BENDR(chs_info=keep, n_outputs=2, n_times=N_TIMES, channel_strategy="spline")
    W = model.channel_tokenizer(torch.randn(1, 19, N_TIMES)).weights
    scale_col = W[:, -1]
    assert scale_col[-1] == 1.0  # SCALE copied by name
    assert scale_col[:-1].abs().sum() == 0  # never feeds an electrode


def test_source_on_four_channels_keeps_input_scale():
    torch.manual_seed(0)
    model = BENDR(
        chs_info=FOUR, n_outputs=2, n_times=N_TIMES, channel_strategy="source"
    ).eval()
    x = torch.randn(3, 4, N_TIMES)
    enc = model.channel_tokenizer(x)
    electrodes = enc.x[:, :-1]
    assert torch.isfinite(enc.x).all()
    ratio = (electrodes.std() / x.std()).item()
    assert 1 / 3 < ratio < 3, ratio
    with torch.no_grad():
        assert torch.isfinite(model(x)).all()


def test_strategy_model_loads_the_native_checkpoint_strictly():
    native = BENDR(n_chans=20, n_outputs=2, n_times=N_TIMES)
    model = BENDR(chs_info=FOUR, n_outputs=2, n_times=N_TIMES, channel_strategy="idw")
    assert list(model.state_dict()) == list(native.state_dict())
    model.load_state_dict(native.state_dict(), strict=True)


def test_channel_strategy_round_trips_through_config():
    model = BENDR(
        chs_info=FOUR,
        n_outputs=2,
        n_times=N_TIMES,
        channel_strategy="spline",
        channel_strategy_kwargs={"reg": 1e-2},
    )
    config = model.get_config()
    assert config["channel_strategy"] == "spline"
    assert config["channel_strategy_kwargs"] == {"reg": 1e-2}
    rebuilt = BENDR.from_config(config)
    assert rebuilt.channel_tokenizer.strategy_name == "spline"
    assert rebuilt.channel_tokenizer.strategy.reg == 1e-2


def test_per_call_montage():
    model = BENDR(n_outputs=2, n_times=N_TIMES, n_chans=20, channel_strategy="spline")
    model.eval()
    with torch.no_grad():
        y4 = model(torch.randn(1, 4, N_TIMES), chs_info=FOUR)
        y5 = model(torch.randn(1, 5, N_TIMES), chs_info=FOUR + _chs(["C3"]))
    assert y4.shape == y5.shape == (1, 2)


def test_declared_errors():
    with pytest.raises(ValueError, match="channel_strategy_kwargs"):
        BENDR(n_chans=20, n_outputs=2, n_times=N_TIMES, channel_strategy_kwargs={"a": 1})
    with pytest.raises(ValueError, match="per call"):
        BENDR(n_chans=20, n_outputs=2, n_times=N_TIMES)(
            torch.randn(1, 4, N_TIMES), chs_info=FOUR
        )
    with pytest.raises(ValueError, match="Did you mean"):
        BENDR(chs_info=FOUR, n_outputs=2, n_times=N_TIMES, channel_strategy="splin")


def test_bendr_adapts_non_canonical_chs():
    # Moved from test_interpolated.py: BENDR adapts any montage with
    # coordinates, without a separate wrapper class.
    model = BENDR(chs_info=FOUR + _chs(["C3"]), n_outputs=2, n_times=1000, sfreq=256)
    with torch.no_grad():
        y = model.eval()(torch.randn(1, 5, 1000))
    assert y.shape == (1, 2) and torch.isfinite(y).all()


def _cells():
    geos = geometries(COMPAT["BENDR"])
    for s in STRATEGIES:
        for g in GEOMETRIES:
            yield pytest.param(s, g, geos[g], id=f"{s}-{g}")


@pytest.mark.parametrize("strategy,gname,gkw", list(_cells()))
def test_bendr_strategy_by_geometry(strategy, gname, gkw):
    n_in = len(gkw["chs_info"])

    def build_and_forward():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = BENDR(n_outputs=2, channel_strategy=strategy, **gkw).eval()
            if strategy == "wiener":
                model.channel_tokenizer.fit(*_dense_fit_set())
            with torch.no_grad():
                return model(torch.randn(1, n_in, gkw["n_times"]))

    if (strategy, gname) in DECLARED_ERRORS:
        with pytest.raises(ValueError, match=DECLARED_ERRORS[(strategy, gname)]):
            build_and_forward()
    else:
        y = build_and_forward()
        assert y.shape == (1, 2) and torch.isfinite(y).all()


def test_labram_warns_on_non_canonical_chs():
    # Moved from test_interpolated.py; the message now points at the
    # channel layer.
    from braindecode.models import Labram

    with pytest.warns(UserWarning, match="ChannelTokenizer"):
        Labram(chs_info=FOUR, n_outputs=2, n_times=200)


def test_from_pretrained_with_a_strategy(tmp_path):
    pytest.importorskip("huggingface_hub")
    small = dict(encoder_h=64, contextualizer_hidden=128, transformer_layers=2)
    native = BENDR(n_chans=20, n_outputs=2, n_times=N_TIMES, **small).eval()
    native.save_pretrained(tmp_path)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = BENDR.from_pretrained(
            tmp_path, chs_info=FOUR, channel_strategy="source"
        ).eval()
    # Map-quality warnings (4 inputs -> 20 targets) are expected; the
    # "freshly initialised" warning is not, since source adds no weights.
    assert not [w for w in caught if "freshly initialised" in str(w.message)]
    assert model.channel_tokenizer.strategy_name == "source"
    for k, v in native.state_dict().items():
        assert torch.equal(model.state_dict()[k], v), k
    with torch.no_grad():
        assert torch.isfinite(model(torch.randn(1, 4, N_TIMES))).all()
