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
    models_dict,
    models_mandatory_parameters,
)

all_models_dict = dict(models_dict)


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
    signal = _get_signal_params(signal_params)
    model_kwargs = _get_signal_params(signal_params, required_params)
    model = all_models_dict[model_name](**model_kwargs).eval()
    x = torch.randn(2, len(signal["chs_info"]), signal["n_times"])
    return model, x


def _clone_state(model):
    """Clone persistent tensor state after any lazy first-forward setup."""
    return {name: value.detach().clone() for name, value in model.state_dict().items()}


def _batched_tensor_leaves(value, batch_size):
    """Return tensor leaves whose leading dimension represents the batch."""
    return [
        leaf
        for leaf in _tensor_leaves(value)
        if leaf.ndim > 0 and leaf.shape[0] == batch_size
    ]


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

    batched = _batched_tensor_leaves(output, x.shape[0])
    assert batched, (
        f"{model_name} returned no tensor leaf preserving batch dimension "
        f"{x.shape[0]}"
    )

    for leaf in leaves:
        if leaf.is_floating_point() or leaf.is_complex():
            assert torch.isfinite(leaf).all(), (
                f"{model_name} emitted non-finite values in eval-mode forward"
            )

    # A first forward may legitimately materialize lazy parameters. Once warm,
    # however, eval-mode inference must not mutate persistent model state.
    # Reuse the permutation probe below as the second forward so the gate stays
    # cheap even for large foundation models.
    state_before = _clone_state(model)

    # Reordering independent samples must only reorder the corresponding
    # outputs. This catches accidental batch-axis mixing that ordinary shape
    # checks and single-sample tests cannot see.
    permutation = torch.tensor([1, 0], device=x.device)
    with torch.no_grad():
        permuted = model(x.index_select(0, permutation))

    state_after = model.state_dict()
    assert state_before.keys() == state_after.keys()
    for name, expected_state in state_before.items():
        torch.testing.assert_close(
            state_after[name],
            expected_state,
            msg=lambda msg: (
                f"{model_name} mutated persistent state {name!r} during "
                f"eval-mode forward: {msg}"
            ),
        )

    permuted_batched = _batched_tensor_leaves(permuted, x.shape[0])
    assert len(permuted_batched) == len(batched)
    for expected_leaf, actual_leaf in zip(batched, permuted_batched):
        torch.testing.assert_close(
            actual_leaf,
            expected_leaf.index_select(0, permutation),
            msg=lambda msg: (
                f"{model_name} is not batch-permutation equivariant in eval "
                f"mode: {msg}"
            ),
        )
    # Permutation equivariance alone cannot detect all cross-sample mixing: a
    # symmetric batch aggregate can influence every output and still permute
    # correctly. Keep sample 0 fixed, change only sample 1, and require sample
    # 0's outputs to remain invariant.
    composed_x = x.clone()
    composed_x[1].mul_(-3.0).add_(1.0)
    with torch.no_grad():
        recomposed = model(composed_x)

    recomposed_batched = _batched_tensor_leaves(recomposed, x.shape[0])
    assert len(recomposed_batched) == len(batched)
    for expected_leaf, actual_leaf in zip(batched, recomposed_batched):
        torch.testing.assert_close(
            actual_leaf[0],
            expected_leaf[0],
            msg=lambda msg: (
                f"{model_name} leaked information across independent batch "
                f"samples in eval mode: {msg}"
            ),
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
