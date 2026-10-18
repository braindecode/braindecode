# Authors: Robin Schirrmeister <robintibor@gmail.com>
#          Anton Soloviev <anton@praviel.com>
#
# License: BSD (3-clause)

import numpy as np
import pytest
import torch
from torch import nn

from braindecode.util import set_random_seeds
from braindecode.visualization.frequency import amplitude_gradients


@pytest.mark.parametrize("requires_grad", [False, True])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_amplitude_gradients_tensor_inputs(device, requires_grad):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    model = nn.Sequential(nn.Flatten(), nn.Linear(34, 2, bias=False)).to(device).eval()
    x = torch.linspace(-1, 1, 68, device=device).reshape(2, 2, 17)
    expected = amplitude_gradients(model, x.cpu().numpy())
    x.requires_grad_(requires_grad)

    actual = amplitude_gradients(model, x)

    np.testing.assert_array_equal(actual, expected)
    assert x.grad is None


def test_amplitude_gradients():
    # If the weights are initialized with a sine function
    # gradient of amplitude should be only in one frequency bin
    set_random_seeds(948, False)
    model = nn.Conv1d(1, 1, 16)
    # torch.linspace(,,n)[:n-1] is same as np.linspace(,,n,endpoint=False)
    model.weight.data[:, :, :] = torch.sin(torch.linspace(0, 2 * np.pi, 17)[:16])
    model.bias.data[:] = 0
    grads = amplitude_gradients(model, torch.randn(1, 1, 16))
    grads = grads.squeeze()
    assert np.abs(grads[1]) / np.sum(np.abs(grads)) > 0.99


def test_amplitude_gradients_two_filters():
    # If the weights are initialized with a sine function
    # gradient of amplitude should be only in one frequency bin
    set_random_seeds(948, False)
    model = nn.Conv1d(1, 2, 16)
    # torch.linspace(,,n)[:n-1] is same as np.linspace(,,n,endpoint=False)
    model.weight.data[0, :, :] = torch.sin(torch.linspace(0, 2 * np.pi, 17)[:16])
    model.weight.data[1, :, :] = torch.sin(torch.linspace(0, 4 * np.pi, 17)[:16])
    model.bias.data[:] = 0
    grads = amplitude_gradients(model, torch.randn(1, 1, 16))
    grads = grads.squeeze()
    assert np.abs(grads[0][1]) / np.sum(np.abs(grads[0])) > 0.99
    assert np.abs(grads[1][2]) / np.sum(np.abs(grads[1])) > 0.99


@pytest.mark.parametrize("pending_grads", [False, True])
def test_amplitude_gradients_preserves_parameter_grads(pending_grads):
    model = nn.Sequential(nn.Conv1d(2, 3, 3), nn.AdaptiveAvgPool1d(1), nn.Flatten(1))
    x = np.random.RandomState(42).randn(4, 2, 16).astype(np.float32)
    if pending_grads:
        model(torch.from_numpy(x)).square().mean().backward()
    parameters = list(model.parameters())
    grad_objects = [p.grad for p in parameters]
    expected = [None if p.grad is None else p.grad.clone() for p in parameters]

    amplitude_gradients(model, x)

    for parameter, grad_object, value in zip(parameters, grad_objects, expected):
        assert parameter.grad is grad_object
        if value is not None:
            torch.testing.assert_close(parameter.grad, value, rtol=0, atol=0)
