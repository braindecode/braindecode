"""Cross-model contract gates for every registered Braindecode model.

These tests turn the repository model conventions into machine-enforced
invariants. A newly registered model is automatically included through
models_mandatory_parameters; no per-model test opt-in is required.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence

import pytest
import torch
from torch import nn

from braindecode.models.util import (
    _get_signal_params,
    interpolated_models_dict,
    models_dict,
    models_mandatory_parameters,
)

all_models_dict = {**models_dict, **interpolated_models_dict}


def _tensor_leaves(value):
    """Yield tensor leaves from nested model outputs."""
    if torch.is_tensor(value):
        yield value
    elif isinstance(value, Mapping):
        for child in value.values():
            yield from _tensor_leaves(child)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for child in value:
            yield from _tensor_leaves(child)


def _build_case(model_name, required_params, signal_params):
    signal = _get_signal_params(signal_params, required_params)
    model = all_models_dict[model_name](**signal).eval()
    x = torch.randn(2, len(signal["chs_info"]), signal["n_times"])
    return model, x


@pytest.mark.parametrize(
    "model_name,required_params,signal_params",
    models_mandatory_parameters,
)
def test_registered_model_runtime_contract(
    model_name, required_params, signal_params
):
    """Registered models preserve inputs and emit finite batched tensors."""
    model, x = _build_case(model_name, required_params, signal_params)
    x_before = x.clone()

    with torch.no_grad():
        output = model(x)

    assert torch.equal(x, x_before), (
        f"{model_name} mutated its input tensor during eval-mode forward"
    )

    leaves = list(_tensor_leaves(output))
    assert leaves, f"{model_name} returned no tensor output"

    batched = [leaf for leaf in leaves if leaf.ndim > 0 and leaf.shape[0] == x.shape[0]]
    assert batched, (
        f"{model_name} returned no tensor leaf preserving batch dimension "
        f"{x.shape[0]}"
    )

    for leaf in leaves:
        if leaf.is_floating_point() or leaf.is_complex():
            assert torch.isfinite(leaf).all(), (
                f"{model_name} emitted non-finite values in eval-mode forward"
            )


@pytest.mark.parametrize(
    "model_name,required_params,signal_params",
    models_mandatory_parameters,
)
def test_registered_model_serialization_contract(
    model_name, required_params, signal_params
):
    """Config + state_dict reconstruction preserves eval-mode outputs."""
    model, x = _build_case(model_name, required_params, signal_params)

    with torch.no_grad():
        expected = list(_tensor_leaves(model(x)))

    config = model.get_config()
    serialized = json.dumps(config)
    rebuilt = type(model).from_config(json.loads(serialized)).eval()

    if any(isinstance(p, nn.UninitializedParameter) for p in rebuilt.parameters()):
        with torch.no_grad():
            rebuilt(x)

    rebuilt.load_state_dict(model.state_dict(), strict=True)

    with torch.no_grad():
        actual = list(_tensor_leaves(rebuilt(x)))

    assert len(actual) == len(expected), (
        f"{model_name} changed tensor-output structure after config/state round-trip"
    )
    for expected_leaf, actual_leaf in zip(expected, actual):
        assert expected_leaf.shape == actual_leaf.shape
        torch.testing.assert_close(actual_leaf, expected_leaf)
