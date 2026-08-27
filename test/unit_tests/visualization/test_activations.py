import pytest
import torch
from torch import nn

from braindecode.models import EEGConformer, ShallowFBCSPNet
from braindecode.visualization import (
    capture_activations,
    run_with_activation_substitution,
)

from .conftest import SEED

N_CHANS = 18
N_TIMES = 600
N_OUTPUTS = 2
BATCH = 4


def _shallow():
    """A convolutional stack, hooked at a top-level child."""
    m = ShallowFBCSPNet(
        n_chans=N_CHANS,
        n_outputs=N_OUTPUTS,
        n_times=N_TIMES,
        final_conv_length="auto",
    )
    return m.eval(), "conv_nonlin_exp"


def _conformer():
    """A transformer, hooked at a residual block three levels down.

    The harder of the two: the hooked module is not a direct child, and a
    second residual block plus the head still run downstream of it, so a
    substitution that fails to propagate has more places to show up.
    """
    m = EEGConformer(n_chans=N_CHANS, n_outputs=N_OUTPUTS, n_times=N_TIMES)
    return m.eval(), "transformer.0.0"


@pytest.fixture(scope="module", params=[_shallow, _conformer], ids=["cnn", "residual"])
def model_and_layer(request):
    model, name = request.param()
    return model, dict(model.named_modules())[name]


def _randn(*shape, seed=SEED):
    """Seeded input drawn from a local generator.

    A bare ``torch.manual_seed`` would reseed the process-global RNG and
    silently change the draws of every test that runs afterwards.
    """
    return torch.randn(*shape, generator=torch.Generator().manual_seed(seed))


@pytest.fixture(scope="module")
def X():
    return _randn(BATCH, N_CHANS, N_TIMES)


class _ToyNet(nn.Module):
    """Small enough that the substituted output can be derived by hand."""

    def __init__(self, block=None):
        super().__init__()
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


@pytest.mark.parametrize("how", ["module", "list", "dict"])
def test_captured_activation_substitutes_back_unchanged(model_and_layer, X, how):
    """Round-trip on a real model.

    Feeding a captured activation straight back in must be a no-op, which
    fails if capture returns anything other than the genuine layer output.
    A hook returning ``None`` also leaves the output untouched, so the
    round-trip alone cannot tell a working substitution from one that never
    fires; perturbing the activation pins that down.
    """
    model, layer = model_and_layer
    layers = {"module": layer, "list": [layer], "dict": {"b": layer}}[how]
    captured = capture_activations(model, X, layers)
    (activation,) = captured.values()

    with torch.no_grad():
        baseline = model(X)
        restored = run_with_activation_substitution(
            model, X, layer, lambda _out: activation
        )
        perturbed = run_with_activation_substitution(
            model, X, layer, lambda _out: activation + 1.0
        )
    torch.testing.assert_close(restored, baseline)
    assert not torch.allclose(perturbed, baseline)


def test_substituted_output_matches_hand_computation():
    """Zeroing a block leaves the head with nothing but its own bias."""
    model = _ToyNet().eval()
    with torch.no_grad():
        out = run_with_activation_substitution(
            model, _randn(BATCH, 4), model.block, torch.zeros_like
        )
    torch.testing.assert_close(out, model.head.bias.expand_as(out))


@pytest.mark.parametrize("kind", ["tuple", "dict"])
def test_capture_preserves_nested_output_structure(kind):
    block = _MultiOutBlock(kind)
    model = _ToyNet(block=block).eval()
    x = _randn(BATCH, 4)

    captured = capture_activations(model, x, {"block": model.block})["block"]

    torch.testing.assert_close(_hidden_of(captured), block.linear(x))
    assert not _hidden_of(captured).requires_grad  # detach walks the container


def test_capture_without_detach_keeps_the_graph():
    model = _ToyNet().eval()
    captured = capture_activations(model, _randn(BATCH, 4), [model.block], detach=False)
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
        run(model, _randn(2, 4))
    assert not model.block._forward_hooks


def test_hooks_are_removed_when_the_substitute_raises():
    model = _ToyNet().eval()

    def boom(_out):
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        run_with_activation_substitution(model, _randn(2, 4), model.block, boom)
    assert not model.block._forward_hooks


def test_capture_raises_when_a_requested_layer_never_runs():
    """A short dict would defer the failure to whoever indexes the key."""
    model = _ToyNet().eval()
    unused = nn.Linear(4, 4)
    with pytest.raises(RuntimeError, match="never called"):
        capture_activations(model, _randn(BATCH, 4), {"unused": unused})


def test_substitution_rejects_a_substitute_fn_that_returns_none():
    """Torch reads a None hook return as 'keep the original output', so a
    missing return would silently disable the substitution."""
    model = _ToyNet().eval()
    with pytest.raises(TypeError, match="returned None"):
        run_with_activation_substitution(
            model, _randn(BATCH, 4), model.block, lambda out: None
        )
