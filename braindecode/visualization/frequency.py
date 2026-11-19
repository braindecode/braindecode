# Authors: Robin Schirrmeister <robintibor@gmail.com>
#          Sarthak Tayal <sarthaktayal2@gmail.com>
#          Bruno Aristimunha <b.aristimunha@gmail.com>
#          Anton Soloviev <anton@praviel.com>
#
# License: BSD (3-clause)

"""Frequency-domain attribution.

Computes the gradient of each model output unit with respect to the
amplitude spectrum of the input. Internally splits the input into
amplitude and phase via ``rfft``, treats both as leaf tensors, rebuilds
the time-domain signal, runs the model, and differentiates with respect
to the amplitudes without accumulating model parameter gradients.

This is the one attribution method braindecode implements directly:
captum has no frequency-domain attribution. Time-domain methods live
in :mod:`braindecode.visualization.attribution`.
"""

import numpy as np
import torch
from skorch.utils import to_numpy, to_tensor


def amplitude_gradients(model, x):
    """Per-batch amplitude gradients.

    Parameters
    ----------
    model : torch.nn.Module
        Model in eval mode (or otherwise deterministic for the given
        input). Must accept ``x`` of shape ``(batch, n_chans, n_times)``
        and return outputs of shape ``(batch, n_outputs)``.
    x : numpy.ndarray or torch.Tensor of shape ``(batch, n_chans, n_times)``
        Input batch. Will be moved to ``model``'s device.

    Returns
    -------
    numpy.ndarray of shape ``(n_outputs, batch, n_chans, n_freqs)``
        ``out[i]`` is the gradient of the mean of the i-th output unit
        w.r.t. the input amplitude spectrum, per trial. ``n_freqs`` is
        ``n_times // 2 + 1`` (the size of an ``rfft``).
    """
    device = next(model.parameters()).device
    ffted = np.fft.rfft(to_numpy(x), axis=2)
    amps = np.abs(ffted)
    phases = np.angle(ffted)
    amps_th = to_tensor(amps.astype(np.float32), device=device).requires_grad_(True)
    phases_th = to_tensor(phases.astype(np.float32), device=device).requires_grad_(True)

    # Rebuild fft coefficients from amplitude/phase as real-valued pairs,
    # convert to complex, invert back to time-domain. ``torch.irfft`` was
    # removed in PyTorch 1.8; ``torch.fft.irfft`` expects complex input.
    fft_coefs = amps_th.unsqueeze(-1) * torch.stack(
        (torch.cos(phases_th), torch.sin(phases_th)), dim=-1
    )
    fft_coefs = fft_coefs.squeeze(3)
    iffted = torch.fft.irfft(torch.view_as_complex(fft_coefs), n=x.shape[2], dim=2)

    outs = model(iffted)
    n_outputs = outs.shape[1]
    grads_per_output = np.full((n_outputs,) + ffted.shape, np.nan, dtype=np.float32)
    for i in range(n_outputs):
        grads = torch.autograd.grad(outs[:, i].mean(), amps_th, retain_graph=True)[0]
        grads_per_output[i] = to_numpy(grads)
    assert not np.any(np.isnan(grads_per_output))
    return grads_per_output


def amplitude_gradients_per_trial(model, dataset, batch_size):
    """Concatenated :func:`amplitude_gradients` over every trial in a dataset.

    Parameters
    ----------
    model : torch.nn.Module
    dataset : torch.utils.data.Dataset
        Yields ``(x, ...)`` tuples; only the first element is used.
    batch_size : int

    Returns
    -------
    numpy.ndarray of shape ``(n_outputs, n_trials, n_chans, n_freqs)``
        Gradients of the sum of each minibatch's outputs, in dataset order.
        For a model that processes trials independently in evaluation mode,
        each entry is the gradient of that trial's output.

    Notes
    -----
    This function replaces the dataset-level ``compute_amplitude_gradients``
    from Braindecode 1.4. That function seeded the random generators before
    loading the data. To reproduce the old default, call
    ``set_random_seeds(20240205, cuda=next(model.parameters()).is_cuda)``
    from :mod:`braindecode.util` before computing the gradients.

    The old function returned gradients of the minibatch mean. To recover
    its scaling from the current output, divide each slice by its actual
    batch length, including the last batch::

        gradients = amplitude_gradients_per_trial(model, dataset, batch_size)
        for start in range(0, gradients.shape[1], batch_size):
            batch = gradients[:, start : start + batch_size]
            batch /= batch.shape[1]

    For gradients of a single batch mean, use :func:`amplitude_gradients`.
    """
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, drop_last=False, shuffle=False
    )
    return np.concatenate(
        [len(x) * amplitude_gradients(model, x) for x, *_ in loader], axis=1
    )
