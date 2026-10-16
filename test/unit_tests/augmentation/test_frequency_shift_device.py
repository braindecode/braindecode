# Authors: Anton Soloviev <anton@praviel.com>
# License: BSD (3-clause)

import numpy as np
import pytest
import torch

from braindecode.augmentation.functional import frequency_shift


@pytest.mark.parametrize("device", ["cpu", pytest.param(
    "cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
)])
@pytest.mark.parametrize("kind", ["int", "float", "list", "array", "tensor", "device_tensor"])
def test_frequency_shift_places_shift_on_input_device(device, kind):
    X = torch.ones(2, 3, 128, device=device)
    y = torch.tensor([0, 1], device=device)
    shift = {
        "int": 2,
        "float": 2.0,
        "list": [2.0, 2.0],
        "array": np.array([2.0, 2.0]),
        "tensor": torch.tensor([2.0, 2.0]),
        "device_tensor": torch.tensor([2.0, 2.0], device=device),
    }[kind]

    actual, actual_y = frequency_shift(X, y, delta_freq=shift, sfreq=128)

    # The analytic signal of a constant is one, so modulation is a cosine.
    time = torch.arange(128, device=device) / 128
    expected = torch.cos(2 * torch.pi * 2 * time).expand_as(X)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
    assert actual.device == X.device
    assert torch.equal(actual_y, y)
    assert torch.equal(X, torch.ones_like(X))


@pytest.mark.parametrize("device", ["cpu", pytest.param(
    "cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
)])
def test_frequency_shift_keeps_gradient_to_host_shift(device):
    X = torch.ones(2, 3, 128, device=device)
    shift = torch.tensor([1.0, -2.0], dtype=torch.float64, requires_grad=True)
    actual, _ = frequency_shift(X, torch.tensor([0, 1], device=device), shift, 128)
    actual.sum().backward()

    time = torch.arange(128, dtype=shift.dtype) / 128
    phase = 2 * torch.pi * shift.detach()[:, None] * time
    expected = -3 * (2 * torch.pi * time * torch.sin(phase)).sum(dim=-1)
    torch.testing.assert_close(shift.grad, expected, rtol=1e-6, atol=1e-6)
    assert shift.grad.device == shift.device
