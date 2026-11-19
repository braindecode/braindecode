# Authors: Anton Soloviev <anton@praviel.com>
# License: BSD (3-clause)

import numpy as np
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


def test_bendr_parameter_counts():
    """
    Test BENDR parameter counts match paper specifications.

    Paper reports ~157M parameters total:
    - Encoder: ~4M parameters
    - Contextualizer: ~153M parameters
    """
    set_random_seeds(0, False)

    # Standard 20-channel configuration
    model = BENDR(
        n_chans=20,
        n_outputs=2,
        n_times=5120,
        sfreq=256,
    )

    # Count total parameters
    total_params = sum(p.numel() for p in model.parameters())

    # Should be close to paper: 157,141,049,
    # At braindecode, there are 2 k params difference
    # that might come from implementation details from different
    # torch versions or minor code changes. 157,143,101 in my case.

    # Allow 0.1% tolerance
    expected = 157_141_049
    assert abs(total_params - expected) / expected < 0.001, \
        f"Expected ~{expected:,} params, got {total_params:,}"

    # Count encoder parameters (should be ~4M)
    encoder_params = sum(p.numel() for p in model.encoder.parameters())
    assert 3_900_000 < encoder_params < 4_100_000, \
        f"Encoder should have ~4M params, got {encoder_params:,}"

    # Count contextualizer parameters (should be ~153M)
    contextualizer_params = sum(p.numel() for p in model.contextualizer.parameters())
    assert 152_000_000 < contextualizer_params < 154_000_000, \
        f"Contextualizer should have ~153M params, got {contextualizer_params:,}"


def test_bendr_different_channels():
    """
    Test BENDR with different channel counts.
    Parameter count should scale with number of channels.
    """
    set_random_seeds(0, False)

    configs = [
        (1, 157_112_891),   # Single channel
        (20, 157_142_075),  # Standard
        (64, 157_209_659),  # More channels
    ]

    for n_chans, expected_params in configs:
        model = BENDR(
            n_chans=n_chans,
            n_outputs=2,
            n_times=5120,
            sfreq=256,
        )

        total_params = sum(p.numel() for p in model.parameters())

        # Check exact match
        assert total_params == expected_params, \
            f"For {n_chans} channels: expected {expected_params:,}, got {total_params:,}"


def test_bendr_gradient_flow():
    """
    Test that gradients flow through the entire model.
    """
    set_random_seeds(0, False)

    model = BENDR(n_chans=20, n_outputs=4, n_times=5120, sfreq=256)
    x = torch.randn(2, 20, 5120, requires_grad=True)

    y = model(x)
    loss = y.sum()
    loss.backward()

    # Check gradients exist in encoder
    encoder_has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.encoder.parameters()
    )
    assert encoder_has_grad, "No gradients in encoder"

    # Check gradients exist in contextualizer
    contextualizer_has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.contextualizer.parameters()
    )
    assert contextualizer_has_grad, "No gradients in contextualizer"


@pytest.mark.parametrize("drop_prob", [0.0, 0.1, 0.15])
def test_bendr_dropout_configurations(drop_prob):
    """
    Test BENDR with different dropout rates.
    Paper uses 0.15 for pretraining, 0.0 for fine-tuning.
    """
    set_random_seeds(0, False)

    model = BENDR(
        n_chans=20,
        n_outputs=4,
        n_times=5120,
        sfreq=256,
        drop_prob=drop_prob,
    )

    x = torch.randn(2, 20, 5120)

    # Training mode
    model.train()
    y_train = model(x)
    assert y_train.shape == (2, 4)

    # Eval mode
    model.eval()
    y_eval = model(x)
    assert y_eval.shape == (2, 4)

    # With dropout=0, outputs should be identical
    if drop_prob == 0.0:
        np.testing.assert_allclose(
            y_train.detach().numpy(),
            y_eval.detach().numpy(),
            rtol=1e-5,
            atol=1e-7,
        )


def test_bendr_encoder_only_gradient_flow():
    """Encoder-only mode: encoder has grads, contextualizer does not."""
    set_random_seeds(0, False)

    model = BENDR(
        n_chans=20, n_outputs=4, n_times=5120, sfreq=256, encoder_only=True,
    )
    x = torch.randn(2, 20, 5120, requires_grad=True)
    model(x).sum().backward()

    encoder_has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.encoder.parameters()
    )
    assert encoder_has_grad, "No gradients in encoder"

    ctx_has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.contextualizer.parameters()
    )
    assert not ctx_has_grad, "Contextualizer should have no grads"


def test_bendr_encoder_only_parameter_counts():
    """Encoder/contextualizer params match full model; final_layer is larger."""
    set_random_seeds(0, False)

    model_full = BENDR(
        n_chans=20, n_outputs=4, n_times=5120, sfreq=256, encoder_only=False,
    )
    model_enc = BENDR(
        n_chans=20, n_outputs=4, n_times=5120, sfreq=256, encoder_only=True,
    )

    for name in ("encoder", "contextualizer"):
        p_full = sum(p.numel() for p in getattr(model_full, name).parameters())
        p_enc = sum(p.numel() for p in getattr(model_enc, name).parameters())
        assert p_full == p_enc, f"{name} params differ: {p_full} vs {p_enc}"

    fl_full = sum(p.numel() for p in model_full.final_layer.parameters())
    fl_enc = sum(p.numel() for p in model_enc.final_layer.parameters())
    assert fl_enc > fl_full


def test_bendr_backward_compatibility():
    """encoder_only=False (default) produces identical output to original."""
    set_random_seeds(0, False)

    model_default = BENDR(n_chans=20, n_outputs=4, n_times=5120, sfreq=256)
    model_explicit = BENDR(
        n_chans=20, n_outputs=4, n_times=5120, sfreq=256, encoder_only=False,
    )
    model_explicit.load_state_dict(model_default.state_dict())
    model_default.eval()
    model_explicit.eval()

    x = torch.randn(2, 20, 5120)
    np.testing.assert_allclose(
        model_default(x).detach().numpy(),
        model_explicit(x).detach().numpy(),
        rtol=1e-5, atol=1e-7,
    )


def test_bendr_encoder_only_short_input_raises():
    """RuntimeError when input is too short for 4-chunk pooling."""
    model = BENDR(
        n_chans=20, n_outputs=4, n_times=None, sfreq=256, encoder_only=True,
    )
    with pytest.raises(RuntimeError, match="too few"):
        model(torch.randn(2, 20, 96))
