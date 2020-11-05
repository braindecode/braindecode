import numpy as np
from skorch.utils import to_numpy, to_tensor
import torch


def compute_amplitude_gradients(model, dataset, batch_size):
    """
    Compute the gradients of the given model.

    Args:
        model: (todo): write your description
        dataset: (todo): write your description
        batch_size: (int): write your description
    """
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         drop_last=False, shuffle=False)
    all_amp_grads = []
    for batch_X, _, _ in loader:
        this_amp_grads = compute_amplitude_gradients_for_X(model, batch_X, )
        all_amp_grads.append(this_amp_grads)
    all_amp_grads = np.concatenate(all_amp_grads, axis=1)
    return all_amp_grads


def compute_amplitude_gradients_for_X(model, X):
    """
    Compute the gradients of the gradients

    Args:
        model: (todo): write your description
        X: (todo): write your description
    """
    ffted = np.fft.rfft(X, axis=2)
    amps = np.abs(ffted)
    phases = np.angle(ffted)
    amps_th = to_tensor(amps.astype(np.float32), device='cuda').requires_grad_(True)
    phases_th = to_tensor(phases.astype(np.float32), device='cuda').requires_grad_(True)

    fft_coefs = amps_th.unsqueeze(-1) * torch.stack(
        (torch.cos(phases_th), torch.sin(phases_th)), dim=-1)
    fft_coefs = fft_coefs.squeeze(3)

    iffted = torch.irfft(
        fft_coefs, signal_ndim=1, signal_sizes=(X.shape[2],))

    outs = model(iffted)

    n_filters = outs.shape[1]
    amp_grads_per_filter = np.full((n_filters,) + ffted.shape,
                            np.nan, dtype=np.float32)
    for i_filter in range(n_filters):
        mean_out = torch.mean(outs[:,i_filter])
        mean_out.backward(retain_graph=True)
        amp_grads = to_numpy(amps_th.grad)
        amp_grads_per_filter[i_filter] = amp_grads
    assert not np.any(np.isnan(amp_grads_per_filter))
    return amp_grads_per_filter
