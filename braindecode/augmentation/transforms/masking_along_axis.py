import torch
from ..global_variables import fft_args, data_size


def mask_along_axis(X, params):
    r"""
    Apply a mask along ``axis``. Mask will be applied from indices
    ``[v_start, v_end)``, .
    All examples will have the same mask interval.

    Args: 
        X (Tensor): Real spectrogram (channel, freq, time)
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
    """Given a magnitude and an axis, produces a masking interval ``[v_start, v_end)``
    where
    ``v_end - v_start`` is sampled from ``uniform(0, magnitude * v_max)`` and rounded,
    and ``v_start`` from ``uniform(0, v_max - v)`` and rounded.

    Args:
        X (Tensor): data
        params (dict): a dict containing the necessary parameters for applying the transform : the magnitude, the axis and the masking value.

    Returns:
        Tensor: Masked data
    """

    specgram = X
    value = torch.rand(1) \
        * params['magnitude'] * specgram.size(params["axis"])

    min_value = torch.rand(1) * (specgram.size(params["axis"]) - value)

    params["mask_start"] = (min_value.long()).squeeze()
    params["mask_end"] = (min_value.long() + value.long()).squeeze()
    X = mask_along_axis(X, params)
    return X


def mask_along_time(datum, magnitude):
    """Given a magnitude and data, will mask a random band of data along the time axis

    Args:
        datum (Datum): A wrapper containing the data to transform, plus metadata informations useful for certain transforms
        magnitude (float): a ``[0, 1] float, harmonized between transforms, characterizing how much the transform will alter the data 

    Returns:
        Datum: A wrapper containing the transformed data.
    """
    X = datum.X
    params_time = {"magnitude": magnitude, "axis": 2, "mask_value": 0}
    X = signal_to_time_frequency(X)
    X = mask_along_axis_random(X, params_time)
    datum.X = time_frequency_to_signal(X)
    return datum


def mask_along_frequency(datum, magnitude):
    """Given a magnitude and data, will mask a random band of data along the frequency axis

    Args:
        datum (Datum): A wrapper containing the data to transform, plus metadata informations useful for certain transforms
        magnitude (float): a ``[0, 1] float, harmonized between transforms, characterizing how much the transform will alter the data 

    Returns:
        Datum: A wrapper containing the transformed data.
    """
    X = datum.X
    params_frequency = {"magnitude": magnitude, "axis": 1, "mask_value": 0}
    X = signal_to_time_frequency(X)
    X = mask_along_axis_random(X, params_frequency)
    datum.X = time_frequency_to_signal(X)
    return datum


def signal_to_time_frequency(X):
    """Transforms a temporal signal into its time-frequency representation

    Args:
        X (Tensor): Temporal signal (channel, time)

    Returns:
        Tensor: Real spectrogram (channel, freq, time)
    """
    global fft_args
    X = torch.stft(X, n_fft=fft_args["n_fft"],
                   hop_length=fft_args["hop_length"],
                   win_length=fft_args["win_length"],
                   window=torch.hann_window(fft_args["n_fft"]))
    return X


def time_frequency_to_signal(X):
    """Transforms a time-frequency representation back into a signal
    Args:
        X (Tensor): Real spectrogram (channel, freq, time)

    Returns:
        Tensor: Temporal signal (channel, time) 
    """
    global fft_args, data_size
    X = torch.istft(X, n_fft=fft_args["n_fft"],
                    hop_length=fft_args["hop_length"],
                    win_length=fft_args["win_length"],
                    window=torch.hann_window(fft_args["n_fft"]),
                    length=data_size)
    return X
