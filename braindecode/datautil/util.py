def ms_to_samples(ms, fs):
    """
    Compute milliseconds to number of samples.
    
    Parameters
    ----------
    ms: number
        Milliseconds
    fs: number
        Sampling rate

    Returns
    -------
    n_samples: int
        Number of samples

    """
    return ms * fs / 1000.0


def samples_to_ms(n_samples, fs):
    """
    Compute milliseconds to number of samples.
    
    Parameters
    ----------
    n_samples: number
        Number of samples
    fs: number
        Sampling rate

    Returns
    -------
    milliseconds: int
    """
    return n_samples * 1000.0 / fs
