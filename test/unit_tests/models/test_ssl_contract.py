"""Self-supervised contract: one contrastive step through every headless model."""

import pytest
import torch

from braindecode.modules import MLP
from braindecode.training import info_nce
from test.unit_tests.models._fixtures import (
    _TOKENIZERS,
    PARAMS,
    _build_case,
    _materialize,
)


@PARAMS
def test_one_contrastive_step(model_name, required_params, signal_params):
    if model_name in _TOKENIZERS:
        pytest.skip("tokenizer")
    # n_outputs=None: no head parameters for DistributedDataParallel to miss
    model, x = _build_case(model_name, required_params, signal_params, headless=True)
    _materialize(model, x, return_features=True)
    model.train()
    x = torch.randn(4, *x.shape[1:])
    z1, z2 = (
        model(x + 0.1 * torch.randn_like(x), return_features=True)["embedding"]
        for _ in range(2)
    )
    head = MLP(in_features=z1.shape[1], hidden_features=[64], out_features=32)
    loss = info_nce(head(z1), head(z2))
    loss.backward()
    assert torch.isfinite(loss)
    assert list(model.final_layer.parameters()) == []
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters()
    )
