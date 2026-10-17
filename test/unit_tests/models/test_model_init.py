"""Every registered model initializes reproducibly and the same way through its
constructor and :meth:`~braindecode.models.base.EEGModuleMixin.reset_parameters`."""

import pytest
import torch

from braindecode.models.base import EEGModuleMixin
from braindecode.models.util import models_dict, models_mandatory_parameters
from test.unit_tests.models._fixtures import _get_signal_params


def _build(name, required, signal_params, seed):
    torch.manual_seed(seed)
    return models_dict[name](**_get_signal_params(signal_params, required))


def _state(model, params_only=False):
    tensors = model.named_parameters() if params_only else model.state_dict().items()
    return {
        k: v.detach().clone()
        for k, v in tensors
        if not isinstance(v, torch.nn.parameter.UninitializedParameter)
    }


def _assert_same_distribution(name, expected, actual):
    for key, a in expected.items():
        b = actual[key]
        if not a.is_floating_point() or a.numel() == 0:
            continue
        assert torch.isfinite(b).all(), f"{name}.{key}: non-finite after reset"
        if a.numel() > 1 and torch.all(a == a.flatten()[0]):  # zeros, ones, fill values
            assert torch.equal(a, b), f"{name}.{key}: constant {a.flatten()[0]} changed"
        elif a.numel() >= 256:
            ratio = float(b.double().std() / a.double().std())
            assert 0.85 < ratio < 1.18, (
                f"{name}.{key}: std {float(a.std()):.4g} at construction, "
                f"{float(b.std()):.4g} after reset_parameters()"
            )


@pytest.mark.parametrize(
    "name,required,signal_params", models_mandatory_parameters, ids=lambda x: x if isinstance(x, str) else ""
)
def test_seeded_initialization_is_reproducible_and_finite(name, required, signal_params):
    first = _state(_build(name, required, signal_params, seed=0))
    second = _state(_build(name, required, signal_params, seed=0))
    assert first.keys() == second.keys()
    for key, value in first.items():
        assert torch.equal(value, second[key]), f"{name}.{key} depends on more than the torch seed"
        if value.is_floating_point():
            assert torch.isfinite(value).all(), f"{name}.{key} is not finite"


@pytest.mark.parametrize(
    "name,required,signal_params", models_mandatory_parameters, ids=lambda x: x if isinstance(x, str) else ""
)
def test_reset_parameters_initializes_like_the_constructor(name, required, signal_params):
    """``reset_parameters`` draws new weights with the constructor's scheme, and a
    model-specific ``_init_parameters`` really changes weights."""
    constructed = _state(_build(name, required, signal_params, seed=0), True)
    model = _build(name, required, signal_params, seed=0)
    torch.manual_seed(1)
    model.reset_parameters()
    _assert_same_distribution(name, constructed, _state(model, True))

    if type(model)._init_parameters is not EEGModuleMixin._init_parameters:
        # from PyTorch's layer defaults, the model's own initialization must
        # change some weight (it is not a no-op)
        model.apply(_reset_layer)
        before = _state(model)
        model._init_parameters()
        after = _state(model)
        assert any(not torch.equal(before[k], after[k]) for k in before), (
            f"{name}._init_parameters() changes no weight"
        )


def _reset_layer(module):
    if isinstance(module, EEGModuleMixin):
        return
    for name in ("reset_parameters", "_reset_parameters"):
        if callable(getattr(module, name, None)):
            getattr(module, name)()
            break
