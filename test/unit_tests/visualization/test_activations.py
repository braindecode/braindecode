import pytest
import torch
from torch import nn

from braindecode.visualization import (
    capture_activations,
    run_with_activation_substitution,
)


class _Block(nn.Module):
    def forward(self, x):
        return x * 2, {"mean": x.mean()}


class _Model(nn.Module):
    def __init__(self, explode=False):
        super().__init__()
        self.block = _Block()
        self.explode = explode

    def forward(self, x):
        hidden, _ = self.block(x)
        if self.explode:
            raise RuntimeError("boom")
        return hidden + 1


def test_capture_and_substitute_nested_output():
    model = _Model()
    x = torch.randn(2, 3)
    captured = capture_activations(model, x, model.block)

    torch.testing.assert_close(captured[0], x * 2)
    assert isinstance(captured[1], dict)
    baseline = model(x)
    restored = run_with_activation_substitution(
        model, x, model.block, lambda _output: captured
    )
    changed = run_with_activation_substitution(
        model, x, model.block, lambda output: (output[0] + 1, output[1])
    )
    torch.testing.assert_close(restored, baseline)
    assert not torch.equal(changed, baseline)
    assert not model.block._forward_hooks


@pytest.mark.parametrize(
    "run",
    [
        lambda model, x: capture_activations(model, x, model.block),
        lambda model, x: run_with_activation_substitution(
            model, x, model.block, lambda output: output
        ),
    ],
)
def test_hooks_are_removed_when_forward_raises(run):
    model = _Model(explode=True)
    with pytest.raises(RuntimeError, match="boom"):
        run(model, torch.randn(2, 3))
    assert not model.block._forward_hooks


def test_capture_rejects_an_uncalled_layer():
    model = _Model()
    with pytest.raises(RuntimeError, match="not called"):
        capture_activations(model, torch.randn(2, 3), nn.Identity())


def test_substitution_rejects_none():
    model = _Model()
    with pytest.raises(TypeError, match="must return"):
        run_with_activation_substitution(
            model, torch.randn(2, 3), model.block, lambda _output: None
        )
    assert not model.block._forward_hooks
