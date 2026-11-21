# Authors: Anton Soloviev <anton@praviel.com>
# License: BSD (3-clause)

import pytest
import torch
from torch.nn import functional as F

from braindecode.models import BENDR
from braindecode.models.bendr import _BENDRContextualizer
from braindecode.util import set_random_seeds


@pytest.mark.parametrize("training", [False, True])
@pytest.mark.parametrize("start_token", [-5, 0.25, None])
def test_contextualizer_unnormalized_output(training, start_token):
    """The release projects transformer outputs directly, including the token.

    Zero residual branches make the transformer an identity, so its output has
    an analytic value independent of the contextualizer's forward implementation.
    Keep a nonidentity final norm in the state dict, as in the released weights.
    """
    model = _BENDRContextualizer(
        in_features=4, hidden_feedforward=8, heads=2, layers=1,
        dropout=0, position_encoder=0, start_token=start_token,
    ).train(training)
    with torch.no_grad():
        layer = model.transformer_layers[0]
        layer.self_attn.out_proj.weight.zero_()
        layer.self_attn.out_proj.bias.zero_()
        layer.linear2.weight.zero_()
        layer.linear2.bias.zero_()
        projection = model.input_conditioning[3]
        projection.weight.zero_()
        projection.weight[:4].copy_(2 * torch.eye(4))
        projection.bias.zero_()
        model.output_layer.weight.zero_()
        model.output_layer.weight[:, :4, 0].copy_(torch.eye(4))
        model.output_layer.bias.zero_()
        model.norm.weight.fill_(0.9235)
        model.norm.bias.fill_(0.5)

    x = torch.arange(24, dtype=torch.float32).reshape(2, 4, 3)
    expected = 2 * F.layer_norm(x.transpose(1, 2), (4,)).transpose(1, 2)
    if start_token is not None:
        expected = torch.cat([torch.full((2, 4, 1), start_token), expected], dim=2)
    torch.testing.assert_close(model(x), expected)
    assert not any(p.requires_grad for p in model.norm.parameters())


# The small configuration: 480 samples give 5 encoded steps (the encoder downsamples by 96).
_SMALL_BENDR = dict(
    n_chans=4,
    n_outputs=3,
    sfreq=256,
    encoder_h=16,
    contextualizer_hidden=32,
    transformer_layers=1,
    transformer_heads=2,
)


@pytest.mark.parametrize(
    "n_chans, n_params", [(1, 157_112_891), (20, 157_142_075), (64, 157_209_659)]
)
def test_bendr_parameter_counts(n_chans, n_params):
    """The model has the paper's ~157M parameters: ~4M encoder, ~153M contextualizer."""
    with torch.device("meta"):  # only the shapes count: do not allocate 157M weights
        model = BENDR(n_chans=n_chans, n_outputs=2, n_times=5120, sfreq=256)

    def count(module):
        return sum(p.numel() for p in module.parameters())

    assert count(model) == n_params
    assert 3_900_000 < count(model.encoder) < 4_100_000
    assert 152_000_000 < count(model.contextualizer) < 154_000_000


@pytest.mark.parametrize("encoder_only", [False, True])
def test_bendr_gradient_flow(encoder_only):
    """The encoder always gets gradients; the contextualizer, unless it is bypassed."""
    set_random_seeds(0, False)
    model = BENDR(n_times=480, encoder_only=encoder_only, **_SMALL_BENDR)
    out = model(torch.randn(2, 4, 480))
    out.sum().backward()

    def has_grad(module):
        return any(
            p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters()
        )

    assert out.shape == (2, 3)
    assert has_grad(model.encoder)
    assert has_grad(model.contextualizer) != encoder_only


def test_bendr_encoder_only_short_input_raises():
    """RuntimeError when input is too short for 4-chunk pooling."""
    model = BENDR(n_times=None, encoder_only=True, **_SMALL_BENDR)
    with pytest.raises(RuntimeError, match="too few"):
        model(torch.randn(2, 4, 96))
