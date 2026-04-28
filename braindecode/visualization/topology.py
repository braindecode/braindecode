# Authors: Vandit Shah <shahvanditt@gmail.com>
#
# License: BSD (3-clause)

import numpy as np


def project_to_topomap(data, chs_info, res=64):
    """Project per-channel attribution values onto a 2-D scalp topomap grid.

    Thin wrapper around :func:`mne.viz.plot_topomap` that returns the
    interpolated ``(res, res)`` image array. MNE handles the channel
    geometry, sphere fitting, and Clough-Tocher interpolation.

    Parameters
    ----------
    data : numpy.ndarray
        Per-channel values of shape ``(n_chans,)``.
    chs_info : list of dict
        MNE-style channel info list. Each entry must have a ``'ch_name'``
        key and a ``'loc'`` array whose first three elements are the 3-D
        Cartesian electrode position in metres.
    res : int, default=64
        Resolution of the output square grid (pixels per side).

    Returns
    -------
    numpy.ndarray
        Interpolated scalp map of shape ``(res, res)``. Pixels outside
        the head outline are NaN.
    """
    import matplotlib.pyplot as plt
    import mne

    info = _info_from_chs_info(chs_info)
    fig, ax = plt.subplots()
    try:
        im, _ = mne.viz.plot_topomap(data, info, axes=ax, show=False, res=res)
        return np.asarray(im.get_array(), dtype=float)
    finally:
        plt.close(fig)


def _info_from_chs_info(chs_info):
    """Build an :class:`mne.Info` object from a braindecode ``chs_info`` list."""
    import mne

    info = mne.create_info(
        ch_names=[ch["ch_name"] for ch in chs_info],
        sfreq=1.0,
        ch_types="eeg",
    )
    with info._unlock():
        for i, ch in enumerate(chs_info):
            if ch.get("loc") is not None:
                info["chs"][i]["loc"] = np.asarray(ch["loc"], dtype=np.float64)
    return info
