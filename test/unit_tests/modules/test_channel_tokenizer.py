# Authors: Bruno Aristimunha <b.aristimunha@gmail.com>
#
# License: BSD (3-clause)
import numpy as np
import pytest
import torch

from braindecode.models.bendr import _BENDR_TARGET_CHS_INFO
from braindecode.modules import ChannelTarget, ChannelTokenizer

BENDR19 = [c for c in _BENDR_TARGET_CHS_INFO if c["ch_name"] != "SCALE"]
TARGET = ChannelTarget("montage", chs_info=BENDR19)


def _named(names, kind="eeg"):
    return [{"ch_name": n, "kind": kind} for n in names]


def test_native_is_identity():
    tok = ChannelTokenizer(TARGET)
    x = torch.randn(2, 5, 7)
    enc = tok(x, _named(["A", "B", "C", "D", "E"]))
    assert enc.x is x
    assert enc.observed.tolist() == [True] * 5 and (enc.support == 1).all()
    assert enc.channel_ids is None and enc.positions is None and enc.weights is None
    assert tok.state_dict() == {}


def test_non_trainable_strategy_adds_no_state():
    tok = ChannelTokenizer(TARGET, "spline", src_chs_info=BENDR19[:8])
    assert tok.state_dict() == {}
    assert list(tok.parameters()) == []


def test_construction_montage_is_the_default():
    tok = ChannelTokenizer(TARGET, "exact", src_chs_info=BENDR19[::-1])
    x = torch.randn(2, 19, 4)
    torch.testing.assert_close(tok(x).x, x.flip(1), rtol=0, atol=0)
    with pytest.raises(ValueError, match="chs_info"):
        ChannelTokenizer(TARGET, "exact")(x)


def test_channel_count_mismatch_is_declared_error():
    tok = ChannelTokenizer(TARGET, "spline", src_chs_info=BENDR19[:8])
    with pytest.raises(ValueError, match="7 channels.*8"):
        tok(torch.randn(1, 7, 10))
    with pytest.raises(ValueError, match="8 channels.*9"):
        tok(torch.randn(1, 8, 10), BENDR19[:9])


def test_alternating_montages_use_their_own_map():
    tok = ChannelTokenizer(TARGET, "zero")
    a, b = BENDR19[:10], BENDR19[5:15]
    xa, xb = torch.randn(1, 10, 3), torch.randn(1, 10, 3)
    for _ in range(2):
        # Montage a fills rows 0-9, montage b rows 5-14, in input order.
        assert torch.equal(tok(xa, a).x[0, :10], xa[0])
        assert torch.equal(tok(xb, b).x[0, 5:15], xb[0])
        assert tok(xb, b).x[0, :5].abs().sum() == 0


def test_cache_keeps_at_most_8_montages():
    tok = ChannelTokenizer(TARGET, "zero")
    x = torch.randn(1, 10, 3)
    for start in range(10):
        tok(x, BENDR19[start : start + 10])
    assert len(tok._cache) == 8


def test_maps_follow_dtype():
    tok = ChannelTokenizer(TARGET, "spline", src_chs_info=BENDR19[:8])
    x = torch.randn(2, 8, 5, dtype=torch.float64)
    enc = tok.double()(x)
    assert enc.x.dtype == torch.float64 and enc.weights.dtype == torch.float64
    ref = tok(x.float()).x
    torch.testing.assert_close(enc.x.float(), ref)


def test_non_eeg_dropped_from_the_signal_when_asked():
    chs = _named(["Cz"]) + _named(["EOG"], kind="eog") + _named(["Pz"])
    tok = ChannelTokenizer(TARGET, "zero", src_chs_info=chs, drop_non_eeg=True)
    x = torch.tensor([1.0, 2.0, 3.0]).reshape(1, 3, 1)
    out = tok(x).x[0, :, 0]
    assert out[9].item() == 1.0 and out[14].item() == 3.0  # CZ, PZ rows
    assert out.sum().item() == 4.0  # the EOG signal went nowhere
    with pytest.raises(ValueError, match="EOG"):
        ChannelTokenizer(TARGET, "zero", src_chs_info=chs)


def test_unknown_strategy_lists_names():
    with pytest.raises(ValueError, match="native"):
        ChannelTokenizer(TARGET, "splin")


def test_bendr_non_canonical_montage_goes_through_the_layer():
    from braindecode.models import BENDR

    model = BENDR(chs_info=BENDR19[:8], n_outputs=2, n_times=1000).eval()
    assert isinstance(model.channel_tokenizer, ChannelTokenizer)
    W = model.channel_tokenizer(torch.zeros(1, 8, 1)).weights
    assert W.shape == (20, 8)
    np.testing.assert_array_equal(W[:8].numpy(), np.eye(8))


def _dense_fields(rng, n):
    """Spatially smooth random fields on the 19 BENDR sites (wiener training)."""
    pos = np.stack([np.asarray(c["loc"], float)[:3] for c in BENDR19])
    d = np.linalg.norm(pos[:, None] - pos[None], axis=-1)
    cov = np.exp(-d / 0.06)
    return rng.multivariate_normal(np.zeros(19), cov, size=n)


def _accelerator():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return None


@pytest.mark.skipif(_accelerator() is None, reason="needs a cuda or mps device")
@pytest.mark.parametrize(
    "strategy,kwargs",
    [
        ("spline", {}),
        ("source", {"trainable": True}),
        ("latent", {}),
        ("wiener", {}),
    ],
)
def test_new_montage_after_moving_to_a_device(strategy, kwargs):
    device = _accelerator()
    tok = ChannelTokenizer(TARGET, strategy, src_chs_info=BENDR19[:8], **kwargs)
    if strategy == "wiener":
        tok.fit(_dense_fields(np.random.default_rng(0), 500), BENDR19)
    tok = tok.to(device)
    x = torch.randn(2, 9, 16)
    # A montage the tokenizer has not built yet: the map is built on the device.
    enc = tok(x.to(device), BENDR19[4:13])
    assert enc.x.device.type == device.type and enc.x.dtype == torch.float32
    assert torch.isfinite(enc.x).all()
    ref = tok.cpu()(x, BENDR19[4:13]).x
    torch.testing.assert_close(enc.x.cpu(), ref, rtol=1e-4, atol=1e-4)


def test_loading_a_fitted_state_refreshes_the_cached_maps():
    rng = np.random.default_rng(0)
    a = ChannelTokenizer(TARGET, "wiener", src_chs_info=BENDR19[:8])
    a.fit(_dense_fields(rng, 500), BENDR19)
    b = ChannelTokenizer(TARGET, "wiener", src_chs_info=BENDR19[:8])
    b.fit(_dense_fields(rng, 50), BENDR19)  # a different covariance
    x = torch.randn(1, 8, 5)
    assert not torch.allclose(a(x).weights, b(x).weights)  # b's map is cached
    b.load_state_dict(a.state_dict())
    torch.testing.assert_close(b(x).weights, a(x).weights, rtol=0, atol=0)


def test_quality_warnings_once_per_montage():
    import warnings

    tok = ChannelTokenizer(TARGET, "idw")
    x = torch.randn(1, 4, 3)
    chs = _named(["Fz", "Cz", "Pz", "Oz"])
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        for _ in range(3):
            tok(x, chs)
    assert sum("support < 0.5" in str(w.message) for w in record) == 1


@pytest.mark.parametrize("strategy", ["source", "spline", "latent"])
def test_reconstructing_strategy_without_training_montage_warns(strategy):
    with pytest.warns(UserWarning, match="has no effect.*chs_info"):
        ChannelTokenizer(ChannelTarget("positions"), strategy)


def test_exact_without_training_montage_does_not_warn():
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ChannelTokenizer(ChannelTarget("positions"), "exact")


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize(
    "target,strategy,kwargs,expected",
    [
        (TARGET, "spline", {}, 19),  # sensor target: the target size
        (ChannelTarget("ids", vocabulary=("Fz", "Cz", "Pz")), "exact", {}, 8),
        (ChannelTarget("ids", vocabulary=("Fz", "Cz", "Pz", "Oz")), "idw", {}, 4),
        (ChannelTarget("free"), "idw", {}, 8),  # pass-through
        (ChannelTarget("free"), "source", dict(n_parcels=16), 16),
        (ChannelTarget("free"), "latent", dict(n_latents=5), 5),
        (ChannelTarget("positions"), "spline", {}, 8),
    ],
)
def test_n_outputs_matches_forward(target, strategy, kwargs, expected):
    chs = BENDR19[:8]
    if target.interface == "ids" and strategy == "exact":
        chs, expected = _named(["Pz", "Fz", "Cz"]), 3  # one id per input
    tok = ChannelTokenizer(target, strategy, **kwargs)
    with_montage = ChannelTokenizer(target, strategy, src_chs_info=chs, **kwargs)
    x = torch.randn(1, len(chs), 4)
    assert with_montage.n_outputs() == with_montage(x).x.shape[1] == expected
    assert tok.n_outputs(chs) == expected
    # Without a montage, K follows from the target, the strategy and n_chans.
    assert tok.n_outputs(n_chans=len(chs)) == expected


def test_n_outputs_needs_a_size_when_it_depends_on_the_input():
    with pytest.raises(ValueError, match="pass chs_info"):
        ChannelTokenizer(ChannelTarget("free"), "idw").n_outputs()
    # The target size needs no montage.
    assert ChannelTokenizer(TARGET, "spline").n_outputs() == 19
    # native hands the input over unchanged.
    native = ChannelTokenizer(TARGET)
    assert native.n_outputs(BENDR19[:8]) == native.n_outputs(n_chans=8) == 8
