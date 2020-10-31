import torch
from .global_variables import fft_args, data_size

def mask_along_axis(X, params):
    r"""
    Apply a mask along ``axis``. Mask will be applied from indices
    ``[v_0, v_0 + v)``, where
    ``v`` is sampled from ``uniform(0, mask_param)``,
    and ``v_0`` from ``uniform(0, max_v - v)``.
    All examples will have the same mask interval.

    Args:
        specgram (Tensor): Real spectrogram (channel, freq, time)
        mask_start (int): First column masked
        mask_end (int): First column unmasked
        mask_value (float): Value to assign to the masked columns
        axis (int): Axis to apply masking on (1 -> frequency, 2 -> time)

    Returns:
        Tensor: Masked spectrogram of dimensions (channel, freq, time)
    """

    mask_start = params["mask_start"]
    mask_end = params["mask_end"]

    if params["axis"] == 1:
        X[:, mask_start:mask_end, :, :] = params["mask_value"]
    elif params["axis"] == 2:
        X[:, :, mask_start:mask_end, :] = params["mask_value"]
    else:
        raise ValueError('Only Frequency and Time masking are supported')
    X = X.reshape(X.shape[:-2] + X.shape[-2:])
    return X


def mask_along_axis_random(X, params):

    specgram = X
    value = torch.rand(1) \
        * params['magnitude'] * specgram.size(params["axis"])

    min_value = torch.rand(1) * (specgram.size(params["axis"]) - value)

    params["mask_start"] = (min_value.long()).squeeze()
    params["mask_end"] = (min_value.long() + value.long()).squeeze()

    X = mask_along_axis(X, params)
    return X


def mask_along_time(datum, magnitude):
    X = datum.X
    params_time = {"magnitude": magnitude, "axis": 2}
    X = signal_to_time_frequency(X)
    X = mask_along_axis_random(X, params_time)
    datum.X = time_frequency_to_signal(X)
    return datum


def mask_along_frequency(datum, magnitude):
    X = datum.X
    params_frequency = {"magnitude": magnitude, "axis": 1}
    X = signal_to_time_frequency(X)
    X = mask_along_axis_random(X, params_frequency)
    datum.X = time_frequency_to_signal(X)
    return datum


def signal_to_time_frequency(X):
    global fft_args
    X = torch.stft(X, n_fft=fft_args["n_fft"],
                   hop_length=fft_args["hop_length"],
                   win_length=fft_args["win_length"],
                   window=torch.hann_window(fft_args["n_fft"]))
    return X
    
def time_frequency_to_signal(X):
    global fft_args, data_size
    X = torch.istft(X, n_fft=fft_args["n_fft"],
                    hop_length=fft_args["hop_length"],
                    win_length=fft_args["win_length"],
                    window=torch.hann_window(fft_args["n_fft"]),
                    length=data_size)
    return X
