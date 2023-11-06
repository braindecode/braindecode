# Authors: Robin Schirrmeister <robintibor@gmail.com>
#
# License: BSD (3-clause)

import numpy as np
from skorch.utils import to_numpy, to_tensor
import torch


def compute_amplitude_gradients(model, dataset, batch_size):
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         drop_last=False, shuffle=False)
    all_amp_grads = []
    for batch_X, _, _ in loader:
        this_amp_grads = compute_amplitude_gradients_for_X(model, batch_X, )
        all_amp_grads.append(this_amp_grads)
    all_amp_grads = np.concatenate(all_amp_grads, axis=1)
    return all_amp_grads


def compute_amplitude_gradients_for_X(model, X):
    device = next(model.parameters()).device
    ffted = np.fft.rfft(X, axis=2)
    amps = np.abs(ffted)
    phases = np.angle(ffted)
    amps_th = to_tensor(amps.astype(np.float32), device=device).requires_grad_(True)
    phases_th = to_tensor(phases.astype(np.float32), device=device).requires_grad_(True)

    fft_coefs = amps_th.unsqueeze(-1) * torch.stack(
        (torch.cos(phases_th), torch.sin(phases_th)), dim=-1)
    fft_coefs = fft_coefs.squeeze(3)

    try:
        complex_fft_coefs = torch.view_as_complex(fft_coefs)
        iffted = torch.fft.irfft(
            complex_fft_coefs, n=X.shape[2], dim=2)
    except AttributeError:
        iffted = torch.irfft(  # Deprecated since 1.7
            fft_coefs, signal_ndim=1, signal_sizes=(X.shape[2],))

    outs = model(iffted)

    n_filters = outs.shape[1]
    amp_grads_per_filter = np.full((n_filters,) + ffted.shape,
                                   np.nan, dtype=np.float32)
    for i_filter in range(n_filters):
        mean_out = torch.mean(outs[:, i_filter])
        mean_out.backward(retain_graph=True)
        amp_grads = to_numpy(amps_th.grad.clone())
        amp_grads_per_filter[i_filter] = amp_grads
        amps_th.grad.zero_()
    assert not np.any(np.isnan(amp_grads_per_filter))
    return amp_grads_per_filter
