# Authors: Robin Schirrmeister <robintibor@gmail.com>
#
# License: BSD (3-clause)

import numpy as np
import torch
from torch import nn
from braindecode.util import set_random_seeds
from braindecode.visualization.gradients import compute_amplitude_gradients_for_X


def test_compute_amplitude_gradients_for_X():
    # If the weights are initalized with a sine function
    # gradient of amplitude should be only in one frequency bin
    set_random_seeds(948, False)
    model = nn.Conv1d(1, 1, 16)
    # torch.linspace(,,n)[:n-1] is same as np.linspace(,,n,endpoint=False)
    model.weight.data[:, :, :] = torch.sin(torch.linspace(0, 2 * np.pi, 17)[:16])
    model.bias.data[:] = 0
    grads = compute_amplitude_gradients_for_X(model, torch.randn(1, 1, 16))
    grads = grads.squeeze()
    assert np.abs(grads[1]) / np.sum(np.abs(grads)) > 0.99


def test_compute_amplitude_gradients_for_X_two_filters():
    # If the weights are initalized with a sine function
    # gradient of amplitude should be only in one frequency bin
    set_random_seeds(948, False)
    model = nn.Conv1d(1, 2, 16)
    # torch.linspace(,,n)[:n-1] is same as np.linspace(,,n,endpoint=False)
    model.weight.data[0, :, :] = torch.sin(torch.linspace(0, 2 * np.pi, 17)[:16])
    model.weight.data[1, :, :] = torch.sin(torch.linspace(0, 4 * np.pi, 17)[:16])
    model.bias.data[:] = 0
    grads = compute_amplitude_gradients_for_X(model, torch.randn(1, 1, 16))
    grads = grads.squeeze()
    assert np.abs(grads[0][1]) / np.sum(np.abs(grads[0])) > 0.99
    assert np.abs(grads[1][2]) / np.sum(np.abs(grads[1])) > 0.99
