import pytest
import torch
from torch import nn

from braindecode.visualization import (
    capture_activations,
    run_with_activation_substitution,
)


class ToyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = nn.Linear(4, 6)
        self.head = nn.Linear(6, 2)

    def forward(self, x):
        x = self.block(x)
        return self.head(x)


class ExplodingNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = nn.Linear(4, 4)

    def forward(self, x):
        x = self.block(x)
        raise RuntimeError("boom")


class TupleBlock(nn.Module):
    """A block returning ``(hidden, aux)``, as transformer blocks often do."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 6)

    def forward(self, x):
        hidden = self.linear(x)
        return hidden, hidden.mean(dim=-1)


class TupleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = TupleBlock()
        self.head = nn.Linear(6, 2)

    def forward(self, x):
        hidden, _aux = self.block(x)
        return self.head(hidden)


def test_capture_activations_returns_one_entry_and_cleans_hooks():
    model = ToyNet().eval()
    captured = capture_activations(model, torch.randn(3, 4), {"block": model.block})
    assert set(captured) == {"block"}
    assert captured["block"].shape == (3, 6)
    assert len(model.block._forward_hooks) == 0


def test_capture_activations_keys_by_position_for_a_sequence():
    model = ToyNet().eval()
    captured = capture_activations(model, torch.randn(3, 4), [model.block, model.head])
    assert set(captured) == {0, 1}
    assert captured[1].shape == (3, 2)


def test_capture_activations_cleans_up_on_exception():
    model = ExplodingNet().eval()
    with pytest.raises(RuntimeError, match="boom"):
        capture_activations(model, torch.randn(2, 4), [model.block])
    assert len(model.block._forward_hooks) == 0


def test_capture_activations_preserves_nested_output_structure():
    model = TupleNet().eval()
    captured = capture_activations(model, torch.randn(3, 4), {"block": model.block})
    hidden, aux = captured["block"]
    assert isinstance(captured["block"], tuple)
    assert hidden.shape == (3, 6)
    assert aux.shape == (3,)
    assert not hidden.requires_grad  # detach walks into the tuple


def test_capture_activations_can_keep_the_graph():
    model = ToyNet().eval()
    captured = capture_activations(
        model, torch.randn(3, 4), [model.block], detach=False
    )
    assert captured[0].requires_grad


def test_activation_substitution_identity_and_change():
    model = ToyNet().eval()
    x = torch.randn(4, 4)

    baseline = model(x)
    identity = run_with_activation_substitution(model, x, model.block, lambda out: out)
    torch.testing.assert_close(identity, baseline)
    assert len(model.block._forward_hooks) == 0

    shifted = run_with_activation_substitution(
        model, x, model.block, lambda out: out + 1.0
    )
    assert not torch.allclose(shifted, baseline)
    assert len(model.block._forward_hooks) == 0


def test_activation_substitution_on_a_tuple_returning_block():
    model = TupleNet().eval()
    x = torch.randn(4, 4)

    baseline = model(x)
    zeroed = run_with_activation_substitution(
        model, x, model.block, lambda out: (torch.zeros_like(out[0]), out[1])
    )
    assert zeroed.shape == baseline.shape
    assert not torch.allclose(zeroed, baseline)
    # With the hidden state zeroed the head sees only its own bias.
    torch.testing.assert_close(zeroed, model.head.bias.expand_as(zeroed))
    assert len(model.block._forward_hooks) == 0


def test_activation_substitution_cleans_up_on_exception():
    model = ExplodingNet().eval()
    with pytest.raises(RuntimeError, match="boom"):
        run_with_activation_substitution(
            model, torch.randn(2, 4), model.block, lambda out: out
        )
    assert len(model.block._forward_hooks) == 0


def test_forward_fn_reaches_a_submodule():
    model = ToyNet().eval()
    x = torch.randn(3, 4)
    captured = capture_activations(
        model, x, [model.block], forward_fn=lambda inp: model.block(inp)
    )
    torch.testing.assert_close(captured[0], model.block(x))
