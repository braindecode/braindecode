"""Declared input axes, minimum window length, length-grid snapping and eval
determinism across every registered model."""

import inspect
import warnings

import pytest
import torch

from braindecode.datasets import pad_channels_collate
from braindecode.models import EEGConformer
from braindecode.models.util import models_dict, models_mandatory_parameters
from test.unit_tests.models._fixtures import (
    _TOKENIZERS,
    PARAMS,
    _build_case,
    _materialize,
    _shapes,
    _summary_row,  # noqa: F401
    _tensor_leaves,
    non_classification_models,
)


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
            assert calls == []  # refused before forward ran
        elif model._time_axis == "free":
            for t in (2 * T, T + max(1, model._n_times_step)):
                out = model(torch.randn(2, C, t))
                if model_name not in non_classification_models or model_name in {
                    "DANCE",
                    "CareSleepNet",
                }:
                    assert _shapes(out) == _shapes(ref)
                else:
                    assert len(_shapes(out)) == len(_shapes(ref))
                    for got_s, ref_s in zip(_shapes(out), _shapes(ref)):
                        assert len(got_s) == len(ref_s) and got_s[0] == ref_s[0]
                    if t == 2 * T:
                        assert (
                            next(_tensor_leaves(out)).numel()
                            > next(_tensor_leaves(ref)).numel()
                        )
        else:  # dense: a longer input gives more output steps
            longer = model(torch.randn(2, C, 2 * T))
            assert (
                next(_tensor_leaves(longer)).numel() > next(_tensor_leaves(ref)).numel()
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
        out = model(torch.randn(2, x.shape[1], m))
    assert all(
        torch.isfinite(leaf).all()
        for leaf in _tensor_leaves(out)
        if leaf.is_floating_point() or leaf.is_complex()
    )
    if m > 1:
        with pytest.raises(ValueError, match=rf"needs at least {m} time samples"):
            model(torch.randn(2, x.shape[1], m - 1))


def test_length_grid_snaps_within_cap_and_warns_once():
    model = EEGConformer(n_chans=3, n_outputs=2, n_times=1000).eval()
    with pytest.warns(UserWarning, match="snapped"):
        out = model(torch.randn(2, 3, 1001))
    assert out.shape == (2, 2)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model(torch.randn(2, 3, 1001))
    assert not [w for w in caught if "snapped" in str(w.message)]
    with pytest.raises(ValueError, match="built for n_times=1000"):
        model(torch.randn(2, 3, 1020))  # > 1% of 1000: refused


@PARAMS
def test_channel_axis_matches_forward(model_name, required_params, signal_params):
    model, x = _build_case(model_name, required_params, signal_params)
    _materialize(model, x)
    C, T = x.shape[1:]
    candidates = [C + 1, *([C - 1] if C > 1 else [])]
    calls = []
    model.register_forward_pre_hook(lambda m, a: calls.append(1))
    with torch.no_grad():
        ref = model(x)
        for c in candidates:
            calls.clear()
            if model._channel_axis == "fixed":
                with pytest.raises(
                    ValueError,
                    match=rf"built for n_chans={C} and got {c} channels",
                ):
                    model(torch.randn(2, c, T))
                assert calls == []
            elif model_name in _TOKENIZERS:
                expected = [(s[0], c, *s[2:]) for s in _shapes(ref)]
                assert _shapes(model(torch.randn(2, c, T))) == expected
            else:
                assert _shapes(model(torch.randn(2, c, T))) == _shapes(ref)


@PARAMS
def test_check_reads_keyword_input(model_name, required_params, signal_params):
    model, x = _build_case(model_name, required_params, signal_params)
    _materialize(model, x)
    C, T = x.shape[1:]
    name = list(inspect.signature(type(model).forward).parameters)[1]
    calls = []
    model.register_forward_pre_hook(lambda m, a: calls.append(1))
    with torch.no_grad():
        if model._channel_axis == "fixed":
            with pytest.raises(ValueError, match="channels"):
                model(**{name: torch.randn(2, C + 1, T)})
            assert calls == []
        assert _shapes(model(**{name: x})) == _shapes(model(x))


def test_padded_batch_passes_channel_check():
    checked = 0
    for model_name, required_params, signal_params in models_mandatory_parameters:
        cls = models_dict[model_name]
        if cls._channel_axis != "free":
            continue
        if "ch_mask" not in inspect.signature(cls.forward).parameters:
            continue
        model, x = _build_case(model_name, required_params, signal_params)
        _materialize(model, x)
        C, T = x.shape[1:]
        c_full = max(C, 3)
        batch = [
            (torch.randn(c_full, T).numpy(), 0, [0, T, 0]),
            (torch.randn(c_full - 2, T).numpy(), 1, [0, T, 0]),
        ]
        X, _, _, mask = pad_channels_collate(batch)
        with torch.no_grad():
            out = model(X, ch_mask=mask)
        leaves = list(_tensor_leaves(out))
        assert leaves
        assert all(
            torch.isfinite(leaf).all()
            for leaf in leaves
            if leaf.is_floating_point() or leaf.is_complex()
        ), model_name
        checked += 1
    assert checked >= 7


@PARAMS
def test_eval_is_deterministic_and_batch_independent(
    model_name, required_params, signal_params
):
    torch.manual_seed(0)
    model, x = _build_case(model_name, required_params, signal_params)
    _materialize(model, x)
    with torch.no_grad():
        a = model(x)
        b = model(x)
        one = model(x[:1])
    leaves_a = list(_tensor_leaves(a))
    leaves_b = list(_tensor_leaves(b))
    leaves_one = list(_tensor_leaves(one))
    assert len(leaves_a) == len(leaves_b) == len(leaves_one)
    for leaf_a, leaf_b in zip(leaves_a, leaves_b):
        assert torch.equal(leaf_a, leaf_b)
    for leaf_a, leaf_one in zip(leaves_a, leaves_one):
        if leaf_a.ndim >= 1 and leaf_a.shape[0] == x.shape[0]:
            torch.testing.assert_close(leaf_a[:1], leaf_one, atol=5e-5, rtol=1e-5)
