# Authors: Vandit Shah <shahvanditt@gmail.com>
#
# License: BSD (3-clause)

import mne
import numpy as np
from mne.channels.layout import _find_topomap_coords
from mne.utils import _check_sphere
from scipy.interpolate import CloughTocher2DInterpolator


def project_to_topomap(data, chs_info, res=64):
    """Project per-channel attribution values onto a 2-D scalp topomap grid.

    Projects channel values onto a ``(res, res)`` grid using Clough-Tocher
    triangulation on the 2-D electrode positions obtained from MNE's sphere
    fitting. Points outside the convex hull of the electrode positions are
    set to ``NaN``.

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
        Interpolated scalp map of shape ``(res, res)``. Pixels outside the
        electrode convex hull are ``NaN``.
    """
    info = _info_from_chs_info(chs_info)
    sphere = _check_sphere(None)
    pos2d = _find_topomap_coords(info, picks=list(range(len(data))), sphere=sphere)

    pad = 0.1
    xmin, xmax = pos2d[:, 0].min(), pos2d[:, 0].max()
    ymin, ymax = pos2d[:, 1].min(), pos2d[:, 1].max()
    xpad = pad * (xmax - xmin)
    ypad = pad * (ymax - ymin)
    xi = np.linspace(xmin - xpad, xmax + xpad, res)
    yi = np.linspace(ymin - ypad, ymax + ypad, res)
    Xi, Yi = np.meshgrid(xi, yi)

    return CloughTocher2DInterpolator(pos2d, data, fill_value=np.nan)(Xi, Yi)


def _info_from_chs_info(chs_info):
    """Build an :class:`mne.Info` object from a braindecode ``chs_info`` list."""
    info = mne.create_info(
        ch_names=[ch["ch_name"] for ch in chs_info],
        sfreq=1.0,  # placeholder, not used in topomap plotting
        ch_types="eeg",
    )
    with info._unlock():
        for i, ch in enumerate(chs_info):
            if ch.get("loc") is not None:
                info["chs"][i]["loc"] = np.asarray(ch["loc"], dtype=np.float64)
    return info
