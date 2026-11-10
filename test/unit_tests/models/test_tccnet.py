# Authors: Arslan Shafique <81501094+arslan-shafique@users.noreply.github.com>
#          OpenAI Codex
#
# License: BSD (3-clause)
"""Tests for :class:`braindecode.models.TCCNet`.

The NumPy oracle below directly transcribes ``tcn.py``, ``custom_layers.py``
and ``models.py`` from the original MIT-licensed repository at commit
``cd28909ccf3696b835446cbec2f15eb60610a2af``. It is deliberately independent
of PyTorch convolution and autograd: output values are computed with explicit
causal-convolution loops, and input gradients with central differences.
"""

import numpy as np
import pytest
import torch

from braindecode.models import TCCNet


def _reference_causal_conv(x, convolution):
    """Apply the source CausalConv1D to an ``(N, T, C)`` NumPy array."""
    kernel = convolution.weight.detach().numpy().transpose(2, 1, 0)
    bias = convolution.bias.detach().numpy()
    dilation = convolution.dilation
    left_padding = (convolution.kernel_size - 1) * dilation
    padded = np.pad(x, ((0, 0), (left_padding, 0), (0, 0)))
    output = np.zeros((x.shape[0], x.shape[1], kernel.shape[-1]), dtype=x.dtype)
    for time_idx in range(x.shape[1]):
        for kernel_idx in range(convolution.kernel_size):
            input_idx = time_idx + kernel_idx * dilation
            output[:, time_idx] += padded[:, input_idx] @ kernel[kernel_idx]
    nonzero = output != 0
    return (output + bias) * nonzero


def _reference_forward(model, x):
    """Evaluate the pinned Keras equations without PyTorch operators."""
    features = x.transpose(0, 2, 1)
    for block in model.feature_extractor:
        residual = _reference_causal_conv(features, block.projection)
        for convolution in block.convolutions:
            features = np.maximum(_reference_causal_conv(features, convolution), 0)
        features = features + residual

    if model.attention is None:
        pooled = features.mean(axis=1)
    else:
        weight = model.attention.weight.detach().numpy()
        context = model.attention.context.detach().numpy()
        scores = np.tanh(features @ weight) @ context
        attention = np.exp(scores)
        attention = attention / (attention.sum(axis=1, keepdims=True) + 1e-7)
        pooled = np.sum(features * attention[..., None], axis=1)

    final_weight = model.final_layer.weight.detach().numpy()
    final_bias = model.final_layer.bias.detach().numpy()
    return pooled @ final_weight.T + final_bias


def _set_deterministic_parameters(model):
    """Use non-symmetric values so layout or dilation errors are visible."""
    with torch.no_grad():
        for index, parameter in enumerate(model.parameters()):
            values = torch.linspace(
                -0.13 + index * 0.01,
                0.17 + index * 0.01,
                parameter.numel(),
                dtype=parameter.dtype,
            )
            parameter.copy_(values.reshape_as(parameter))


def _finite_difference_input_gradient(model, x, coefficients, epsilon=1e-6):
    """Differentiate the independent reference with central differences."""
    gradient = np.empty_like(x)
    for index in np.ndindex(x.shape):
        plus = x.copy()
        minus = x.copy()
        plus[index] += epsilon
        minus[index] -= epsilon
        plus_loss = np.sum(_reference_forward(model, plus) * coefficients)
        minus_loss = np.sum(_reference_forward(model, minus) * coefficients)
        gradient[index] = (plus_loss - minus_loss) / (2 * epsilon)
    return gradient


def _small_model(classification_head="aot", **kwargs):
    return TCCNet(
        n_chans=3,
        n_outputs=4,
        n_times=32,
        residual_blocks=2,
        tcn_layers=(2, 1),
        filters=((4, 5), (6,)),
        kernel_size=((3, 2), (3,)),
        drop_prob=0.0,
        classification_head=classification_head,
        **kwargs,
    )


@pytest.mark.parametrize("classification_head", ["aot", "att"])
def test_tccnet_output_shape_and_gradients(classification_head):
    model = _small_model(classification_head)
    x = torch.randn(2, 3, 32, requires_grad=True)

    output = model(x)
    output.square().mean().backward()

    assert output.shape == (2, 4)
    assert torch.isfinite(output).all()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert x.grad.abs().sum() > 0
    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


@pytest.mark.parametrize("classification_head", ["aot", "att"])
def test_tccnet_matches_pinned_keras_equations(classification_head):
    model = TCCNet(
        n_chans=2,
        n_outputs=2,
        residual_blocks=1,
        tcn_layers=2,
        filters=((2, 3),),
        kernel_size=2,
        drop_prob=0.0,
        classification_head=classification_head,
    ).double()
    _set_deterministic_parameters(model)
    x_numpy = np.linspace(-0.4, 0.6, 10).reshape(1, 2, 5)
    coefficients = np.array([[0.7, -0.3]])
    x = torch.tensor(x_numpy, dtype=torch.float64, requires_grad=True)

    observed = model(x)
    (observed * torch.as_tensor(coefficients)).sum().backward()
    expected = _reference_forward(model, x_numpy)
    expected_gradient = _finite_difference_input_gradient(model, x_numpy, coefficients)

    np.testing.assert_allclose(observed.detach().numpy(), expected, rtol=1e-11)
    np.testing.assert_allclose(x.grad.numpy(), expected_gradient, rtol=2e-5, atol=1e-7)


@pytest.mark.parametrize("classification_head", ["aot", "att"])
def test_tccnet_pre_padding_mask_matches_unpadded_input(classification_head):
    model = _small_model(classification_head, mask_value=-10.0).eval()
    x = torch.randn(2, 3, 24, requires_grad=True)
    padded = torch.cat((torch.full((2, 3, 8), -10.0), x.detach()), dim=-1)
    padded.requires_grad_()

    expected = model(x)
    observed = model(padded)
    observed.sum().backward()

    torch.testing.assert_close(observed, expected, rtol=1e-5, atol=1e-6)
    assert padded.grad is not None
    assert torch.count_nonzero(padded.grad[..., :8]) == 0


def test_tccnet_dilation_continues_across_residual_blocks():
    model = _small_model()

    dilations = [
        convolution.dilation
        for block in model.feature_extractor
        for convolution in block.convolutions
    ]

    assert dilations == [1, 2, 4]
    assert all(block.projection.kernel_size == 1 for block in model.feature_extractor)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"residual_blocks": 0}, "residual_blocks must be positive"),
        (
            {"residual_blocks": 2, "tcn_layers": (1,)},
            "one value per residual block",
        ),
        (
            {
                "residual_blocks": 2,
                "tcn_layers": (1, 1),
                "filters": ((2,),),
            },
            "filters must contain 2 blocks",
        ),
        ({"drop_prob": 1.0}, "drop_prob must be in"),
        ({"classification_head": "mean"}, "must be 'aot' or 'att'"),
    ],
)
def test_tccnet_rejects_invalid_architecture(kwargs, match):
    base = {
        "n_chans": 3,
        "n_outputs": 4,
        "filters": 2,
        "kernel_size": 3,
    }
    base.update(kwargs)

    with pytest.raises(ValueError, match=match):
        TCCNet(**base)


@pytest.mark.parametrize("classification_head", ["aot", "att"])
def test_tccnet_fully_masked_input_is_finite(classification_head):
    model = _small_model(
        classification_head=classification_head, mask_value=-10.0
    ).eval()

    output = model(torch.full((2, 3, 32), -10.0))

    assert torch.isfinite(output).all()
    torch.testing.assert_close(output, model.final_layer.bias.expand_as(output))
