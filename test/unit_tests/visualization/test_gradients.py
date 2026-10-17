# Authors: Robin Schirrmeister <robintibor@gmail.com>
#          Anton Soloviev <anton@praviel.com>
#
# License: BSD (3-clause)

import numpy as np
import pytest
import torch
from torch import nn

from braindecode.util import set_random_seeds
from braindecode.visualization.frequency import (
    amplitude_gradients,
    amplitude_gradients_per_trial,
)


@pytest.mark.parametrize("n_times", [16, 17])
@pytest.mark.parametrize("batch_size", [1, 2, 3, 5, 8])
def test_per_trial_amplitude_gradients_match_parseval(n_times, batch_size):
    class EnergyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor([[1.0, 0.5], [-0.25, 2.0]]))

        def forward(self, x):
            return x.square().mean(dim=-1) @ self.weight.T

    model = EnergyModel().eval()
    x = torch.linspace(0.1, 0.9, 5 * 2 * n_times).reshape(5, 2, n_times)
    amplitudes = np.abs(np.fft.rfft(x.numpy(), axis=-1)).astype(np.float32)
    # Parseval: mean(x**2) = sum(multiplicity * amplitude**2) / n_times**2.
    multiplicity = np.full(amplitudes.shape[-1], 2.0)
    multiplicity[0] = 1
    if n_times % 2 == 0:
        multiplicity[-1] = 1
    expected = (
        2 * model.weight.detach().numpy()[:, None, :, None]
        * multiplicity[None, None, None, :] * amplitudes[None] / n_times**2
    )

    actual = amplitude_gradients_per_trial(
        model, torch.utils.data.TensorDataset(x), batch_size=batch_size
    )

    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-7)


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
