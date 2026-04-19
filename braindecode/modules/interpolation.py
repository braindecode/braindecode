# Authors: Pierre Guetschel
#
# License: BSD (3-clause)
from __future__ import annotations

from typing import Literal

import numpy as np
import torch
from torch import nn


class ChannelInterpolationLayer(nn.Module):
    """Projects an input from one channel set to another via a fixed (or learnable) matrix.

    .. warning:: Experimental. Public API may change without a deprecation cycle.

    Parameters
    ----------
    src_chs_info : list of dict
        Source (user) channel info; each dict must have ``"ch_name"`` and
        ``"loc"`` keys (MNE-style).
    tgt_chs_info : list of dict
        Target channel info; same structure.
    mode : {"always", "name_match"}
        See design spec. Default ``"always"``.
    method : str
        Forwarded to ``mne.Raw.interpolate_to`` ``method`` argument when an
        MNE-based matrix is needed. Default ``"spline"``.
    trainable : bool
        If ``True`` the matrix is an ``nn.Parameter`` (stored in
        ``state_dict``). If ``False`` it is a non-persistent buffer
        (recomputed from ``chs_info`` at every ``__init__``).
    """

    def __init__(
        self,
        src_chs_info: list[dict],
        tgt_chs_info: list[dict],
        mode: Literal["always", "name_match"] = "always",
        method: str = "spline",
        trainable: bool = False,
    ) -> None:
        super().__init__()
        W = self._build_matrix(src_chs_info, tgt_chs_info, mode=mode, method=method)
        if trainable:
            self.matrix = nn.Parameter(W)
        else:
            self.register_buffer("matrix", W, persistent=False)
        self.src_chs_info = src_chs_info
        self.tgt_chs_info = tgt_chs_info
        self.mode = mode
        self.method = method
        self.trainable = trainable

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C_src, T) → (B, C_tgt, T)
        return self.matrix @ x

    @staticmethod
    def _build_matrix(
        src: list[dict],
        tgt: list[dict],
        mode: str,
        method: str,
    ) -> torch.Tensor:
        name_to_src = {s["ch_name"].lower(): i for i, s in enumerate(src)}
        matches = [
            (i, name_to_src.get(t["ch_name"].lower())) for i, t in enumerate(tgt)
        ]
        if mode == "name_match" and all(j is not None for _, j in matches):
            W = torch.zeros(len(tgt), len(src))
            for i, j in matches:
                W[i, j] = 1.0
            return W
        raise NotImplementedError("only all-match name_match is implemented in Task 1")


def _compute_interpolation_matrix_mne(
    src_chs_info: list[dict],
    tgt_chs_info: list[dict],
    method: str = "spline",
) -> torch.Tensor:
    """Compute an interpolation matrix ``W`` of shape ``(n_tgt, n_src)`` via MNE.

    Uses the identity-input trick: feed an identity matrix through
    ``Raw.interpolate_to`` so each output column corresponds to one
    source channel.
    """
    import mne

    src_names = [s["ch_name"] for s in src_chs_info]
    src_locs = np.stack(
        [np.asarray(s["loc"], dtype=float)[:3] for s in src_chs_info], axis=0
    )
    tgt_names = [t["ch_name"] for t in tgt_chs_info]
    tgt_locs = np.stack(
        [np.asarray(t["loc"], dtype=float)[:3] for t in tgt_chs_info], axis=0
    )

    info_src = mne.create_info(ch_names=src_names, sfreq=100.0, ch_types="eeg")
    montage_src = mne.channels.make_dig_montage(
        ch_pos=dict(zip(src_names, src_locs)), coord_frame="head"
    )
    info_src.set_montage(montage_src)
    identity = np.eye(len(src_chs_info), dtype=np.float64)
    raw = mne.io.RawArray(identity, info_src, verbose="ERROR")

    montage_tgt = mne.channels.make_dig_montage(
        ch_pos=dict(zip(tgt_names, tgt_locs)), coord_frame="head"
    )

    raw_new = raw.interpolate_to(montage_tgt, method=method)
    W = raw_new.get_data()  # (n_tgt, n_src)

    assert W.shape == (len(tgt_chs_info), len(src_chs_info)), (
        f"Unexpected matrix shape {W.shape}; expected "
        f"({len(tgt_chs_info)}, {len(src_chs_info)})."
    )
    return torch.tensor(W, dtype=torch.float32)
