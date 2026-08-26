import pytest
import torch
from torch import nn

from braindecode.models import ShallowFBCSPNet
from braindecode.visualization import (
    capture_activations,
    run_with_activation_substitution,
)

from .conftest import SEED

N_CHANS = 18
N_TIMES = 600
N_OUTPUTS = 2
BATCH = 4


@pytest.fixture(scope="module")
def model():
    m = ShallowFBCSPNet(
        n_chans=N_CHANS,
        n_outputs=N_OUTPUTS,
        n_times=N_TIMES,
        final_conv_length="auto",
    )
    m.eval()
    return m


@pytest.fixture(scope="module")
def X():
    torch.manual_seed(SEED)
    return torch.randn(BATCH, N_CHANS, N_TIMES)


class _ToyNet(nn.Module):
    """Small enough that the substituted output can be derived by hand."""

    def __init__(self, block=None):
        super().__init__()
        torch.manual_seed(SEED)
        self.block = nn.Linear(4, 6) if block is None else block
        self.head = nn.Linear(6, 2)

    def forward(self, x):
        out = self.block(x)
        hidden = out if torch.is_tensor(out) else _hidden_of(out)
        return self.head(hidden)


class _MultiOutBlock(nn.Module):
    """A block returning a container, as transformer blocks commonly do."""

    def __init__(self, kind):
        super().__init__()
        self.linear = nn.Linear(4, 6)
        self.kind = kind

    def forward(self, x):
        hidden = self.linear(x)
        aux = hidden.mean(dim=-1)
        if self.kind == "tuple":
            return hidden, aux
        return {"hidden": hidden, "aux": aux}


def _hidden_of(out):
    return out[0] if isinstance(out, tuple) else out["hidden"]


class _ExplodingNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = nn.Linear(4, 4)

    def forward(self, x):
        self.block(x)
        raise RuntimeError("boom")


@pytest.mark.parametrize("as_layers", [lambda b: b, lambda b: [b], lambda b: {"b": b}])
def test_captured_activation_substitutes_back_unchanged(model, X, as_layers):
    """Round-trip on a real model.

    Feeding a captured activation straight back in must be a no-op. This
    fails if capture returns anything other than the genuine layer output,
    and it fails if the substitution hook does not actually reach the rest
    of the network -- so it pins both functions against one another without
    needing to know the model's internals.
    """
    layer = model.conv_nonlin_exp
    captured = capture_activations(model, X, as_layers(layer))
    (activation,) = captured.values()

    with torch.no_grad():
        baseline = model(X)
        restored = run_with_activation_substitution(
            model, X, layer, lambda _out: activation
        )
    torch.testing.assert_close(restored, baseline)


def test_substituted_output_matches_hand_computation():
    """Zeroing a block leaves the head with nothing but its own bias."""
    model = _ToyNet().eval()
    with torch.no_grad():
        out = run_with_activation_substitution(
            model, torch.randn(BATCH, 4), model.block, torch.zeros_like
        )
    torch.testing.assert_close(out, model.head.bias.expand_as(out))


@pytest.mark.parametrize("kind", ["tuple", "dict"])
def test_capture_preserves_nested_output_structure(kind):
    block = _MultiOutBlock(kind)
    model = _ToyNet(block=block).eval()
    x = torch.randn(BATCH, 4)

    captured = capture_activations(model, x, {"block": model.block})["block"]

    torch.testing.assert_close(_hidden_of(captured), block.linear(x))
    assert not _hidden_of(captured).requires_grad  # detach walks the container


def test_capture_without_detach_keeps_the_graph():
    model = _ToyNet().eval()
    captured = capture_activations(
        model, torch.randn(BATCH, 4), [model.block], detach=False
    )
    captured[0].sum().backward()
    assert model.block.weight.grad is not None


@pytest.mark.parametrize(
    "run",
    [
        lambda m, x: capture_activations(m, x, [m.block]),
        lambda m, x: run_with_activation_substitution(m, x, m.block, lambda o: o),
    ],
)
def test_hooks_are_removed_when_the_forward_raises(run):
    model = _ExplodingNet().eval()
    with pytest.raises(RuntimeError, match="boom"):
        run(model, torch.randn(2, 4))
    assert not model.block._forward_hooks


def test_hooks_are_removed_when_the_substitute_raises():
    model = _ToyNet().eval()

    def boom(_out):
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        run_with_activation_substitution(model, torch.randn(2, 4), model.block, boom)
    assert not model.block._forward_hooks
