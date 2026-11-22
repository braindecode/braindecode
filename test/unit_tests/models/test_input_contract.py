"""Contract tests for declared input axes, minimum length, and length-grid snapping."""

import inspect
import warnings

import pytest
import torch

from braindecode.datasets import pad_channels_collate
from braindecode.models import LUNA, EEGConformer
from braindecode.models.util import models_dict, models_mandatory_parameters
from test.unit_tests.models._fixtures import (
    _TOKENIZERS,
    PARAMS,
    _build_case,
    _materialize,
    _shapes,
    _tensor_leaves,
    non_classification_models,
)

# channel_tokens is an upper-bound table (1 <= n_chans_in <= n_chans); see #1542.
_UPPER_BOUND_CHANNEL_MODELS = {"BIOT"}


def _matches_free_time(out, ref, *, longer: bool) -> bool:
    if _shapes(out) == _shapes(ref):
        return True
    out_leaves = list(_tensor_leaves(out))
    ref_leaves = list(_tensor_leaves(ref))
    if len(out_leaves) != len(ref_leaves):
        return False
    for lo, lr in zip(out_leaves, ref_leaves):
        if lo.ndim != lr.ndim or lo.ndim < 3 or lo.shape[0] != lr.shape[0]:
            return False
        diffs = [d for d in range(1, lo.ndim) if lo.shape[d] != lr.shape[d]]
        if len(diffs) > 1:
            return False
        if diffs:
            d = diffs[0]
            if longer and lo.shape[d] <= lr.shape[d]:
                return False
    return True


def _matches_free_channels(out, ref, n_chans: int) -> bool:
    if _shapes(out) == _shapes(ref):
        return True
    out_leaves = list(_tensor_leaves(out))
    ref_leaves = list(_tensor_leaves(ref))
    if len(out_leaves) != len(ref_leaves):
        return False
    for lo, lr in zip(out_leaves, ref_leaves):
        if lo.ndim != lr.ndim or lo.ndim < 3:
            return False
        if lo.shape[0] != lr.shape[0] or lo.shape[1] != n_chans or lo.shape[2:] != lr.shape[2:]:
            return False
    return True


def test_every_registered_model_declares_axes_in_own_class_body():
    for name, cls in models_dict.items():
        assert cls.__dict__.get("_time_axis") in {"fixed", "free", "dense"}, name
        assert cls.__dict__.get("_channel_axis") in {"fixed", "free"}, name


@PARAMS
def test_time_axis_matches_forward(model_name, required_params, signal_params):
    model, x = _build_case(model_name, required_params, signal_params)
    _materialize(model, x)
    C, T = x.shape[1:]
    calls = []
    model.register_forward_pre_hook(lambda m, a: calls.append(1))
    with torch.no_grad():
        ref = model(x)
        calls.clear()
        if model._time_axis == "fixed":
            with pytest.raises(
                ValueError, match=rf"built for n_times={T} and got {2 * T} samples"
            ):
                model(torch.randn(2, C, 2 * T))
            assert calls == []
        elif model._time_axis == "free":
            for t in (2 * T, T + max(1, model._n_times_step)):
                out = model(torch.randn(2, C, t))
                if model_name in non_classification_models:
                    assert _matches_free_time(out, ref, longer=(t == 2 * T))
                else:
                    assert _shapes(out) == _shapes(ref)
        else:
            longer = model(torch.randn(2, C, 2 * T))
            assert (
                next(_tensor_leaves(longer)).numel()
                > next(_tensor_leaves(ref)).numel()
            )


@PARAMS
def test_min_n_times_runs_and_one_less_refuses(
    model_name, required_params, signal_params
):
    model, x = _build_case(model_name, required_params, signal_params)
    if model._time_axis == "fixed":
        pytest.skip("fixed length")
    _materialize(model, x)
    m = model._min_n_times
    assert isinstance(m, int) and m >= 1
    with torch.no_grad():
        model(torch.randn(2, x.shape[1], m))
    if m > 1:
        calls = []
        model.register_forward_pre_hook(lambda mod, a: calls.append(1))
        with pytest.raises(ValueError, match=rf"needs at least {m} time samples"):
            model(torch.randn(2, x.shape[1], m - 1))
        assert calls == []


def test_length_grid_snaps_within_cap_and_warns_once():
    model = EEGConformer(n_chans=3, n_outputs=2, n_times=1000).eval()
    with pytest.warns(UserWarning, match="snapped"):
        out = model(torch.randn(2, 3, 1001))
    assert out.shape == (2, 2)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out_pad = model(torch.randn(2, 3, 999))
    assert out_pad.shape == (2, 2)
    assert not [w for w in caught if "snapped" in str(w.message)]
    with pytest.raises(ValueError, match="built for n_times=1000"):
        model(torch.randn(2, 3, 1020))

    strict = EEGConformer(n_chans=3, n_outputs=2, n_times=1000).eval()
    strict.on_non_divisible = "error"
    with pytest.raises(ValueError, match="built for n_times=1000"):
        strict(torch.randn(2, 3, 1001))

    luna = LUNA(n_chans=3, n_outputs=2, n_times=1000).eval()
    with pytest.warns(UserWarning, match=r"LUNA snapped input length 1 to 40"):
        assert luna(torch.randn(2, 3, 1)).shape == (2, 2)


@PARAMS
def test_channel_axis_matches_forward(model_name, required_params, signal_params):
    model, x = _build_case(model_name, required_params, signal_params)
    _materialize(model, x)
    C, T = x.shape[1:]
    calls = []
    model.register_forward_pre_hook(lambda m, a: calls.append(1))
    with torch.no_grad():
        ref = model(x)
        calls.clear()
        candidates = [C + 1] + ([C - 1] if C > 1 else [])
        if model._channel_axis == "fixed":
            for c in candidates:
                if model_name in _UPPER_BOUND_CHANNEL_MODELS and c <= C:
                    assert _shapes(model(torch.randn(2, c, T))) == _shapes(ref)
                    continue
                calls.clear()
                with pytest.raises(
                    ValueError,
                    match=rf"built for n_chans={C} and got {c} channels",
                ):
                    model(torch.randn(2, c, T))
                assert calls == []
        else:
            for c in candidates:
                out = model(torch.randn(2, c, T))
                if model_name in _TOKENIZERS:
                    assert _matches_free_channels(out, ref, c)
                else:
                    assert _shapes(out) == _shapes(ref)


@PARAMS
def test_check_reads_keyword_input(model_name, required_params, signal_params):
    model, x = _build_case(model_name, required_params, signal_params)
    _materialize(model, x)
    C, T = x.shape[1:]
    name = list(inspect.signature(type(model).forward).parameters)[1]
    if model._channel_axis == "fixed":
        calls = []
        model.register_forward_pre_hook(lambda m, a: calls.append(1))
        with pytest.raises(ValueError, match="channels"):
            model(**{name: torch.randn(2, C + 1, T)})
        assert calls == []
    with torch.no_grad():
        assert _shapes(model(**{name: x})) == _shapes(model(x))


def test_padded_batch_passes_channel_check():
    tested = 0
    for model_name, required_params, signal_params in models_mandatory_parameters:
        cls = models_dict[model_name]
        if cls._channel_axis != "free":
            continue
        if "ch_mask" not in inspect.signature(cls.forward).parameters:
            continue
        model, x = _build_case(model_name, required_params, signal_params)
        _materialize(model, x)
        C, T = x.shape[1:]
        c_short = max(1, C - 2)
        windows = [
            (torch.randn(C, T).numpy(), 0, [0, 0, T]),
            (torch.randn(c_short, T).numpy(), 1, [0, 0, T]),
        ]
        X, _, _, mask = pad_channels_collate(windows)
        with torch.no_grad():
            out = model(X, ch_mask=mask)
        leaves = list(_tensor_leaves(out))
        assert leaves, model_name
        for leaf in leaves:
            assert torch.isfinite(leaf).all(), model_name
        tested += 1
    assert tested >= 5


@PARAMS
def test_eval_is_deterministic_and_batch_independent(
    model_name, required_params, signal_params
):
    model, x = _build_case(model_name, required_params, signal_params)
    _materialize(model, x)
    with torch.no_grad():
        a = model(x)
        b = model(x)
        one = model(x[:1])
    a_leaves = list(_tensor_leaves(a))
    b_leaves = list(_tensor_leaves(b))
    one_leaves = list(_tensor_leaves(one))
    assert len(a_leaves) == len(b_leaves) == len(one_leaves)
    for la, lb in zip(a_leaves, b_leaves):
        assert torch.equal(la, lb)
    for la, l1 in zip(a_leaves, one_leaves):
        if la.ndim >= 1 and la.shape[0] == x.shape[0] and l1.shape[0] == 1:
            tol = 1e-4 * max(1.0, l1.abs().max().item())
            torch.testing.assert_close(la[:1], l1, rtol=1e-4, atol=tol)
