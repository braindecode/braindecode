# Authors: Pierre Guetschel
#          Bruno Aristimunha <b.aristimunha@gmail.com>
#
# License: BSD (3-clause)
"""MNE-based interpolation strategies: spherical spline and field mapping."""

from __future__ import annotations

import numpy as np

from .base import ChannelStrategy, register_channel_strategy


def _mne_interp_matrix(
    src_pos: np.ndarray, tgt_pos: np.ndarray, method: str = "spline", reg: float = 0.0
) -> np.ndarray:
    """``(M, k)`` matrix of :meth:`mne.io.Raw.interpolate_to` from k to M positions.

    Identity-input trick: interpolating ``eye(k)`` gives the matrix columns.
    MNE errors are re-raised as a declared :class:`ValueError`.
    """
    import mne

    src_names = [f"S{i}" for i in range(len(src_pos))]
    tgt_names = [f"T{i}" for i in range(len(tgt_pos))]
    try:
        info = mne.create_info(src_names, sfreq=100.0, ch_types="eeg")
        info.set_montage(
            mne.channels.make_dig_montage(
                dict(zip(src_names, src_pos)), coord_frame="head"
            )
        )
        raw = mne.io.RawArray(np.eye(len(src_names)), info, verbose="ERROR")
        montage = mne.channels.make_dig_montage(
            dict(zip(tgt_names, tgt_pos)), coord_frame="head"
        )
        with mne.utils.use_log_level("ERROR"):
            W = raw.interpolate_to(montage, method=method, reg=reg).get_data()
    except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
        raise ValueError(
            f"MNE could not build the {method!r} interpolation from "
            f"{len(src_pos)} positions: {exc}"
        ) from exc
    return W


class _MNEStrategy(ChannelStrategy):
    method = "spline"
    min_positions = 4

    def __init__(self, reg: float = 0.0):
        super().__init__()
        self.reg = reg

    def _fill(self, src, use, tgt_pos):
        rows = np.zeros((len(tgt_pos), len(src.names)))
        rows[:, use] = _mne_interp_matrix(
            src.positions[use], tgt_pos, method=self.method, reg=self.reg
        )
        return rows


@register_channel_strategy("spline")
class SplineStrategy(_MNEStrategy):
    """MNE spherical spline, regularised (``reg=1e-3``; ``0`` = unregularised)."""

    def __init__(self, reg: float = 1e-3):
        super().__init__(reg=reg)


@register_channel_strategy("field")
class FieldStrategy(_MNEStrategy):
    """MNE field mapping (``interpolate_to(method="MNE")``)."""

    method = "MNE"
