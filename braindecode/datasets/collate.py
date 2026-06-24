"""Collate functions for braindecode datasets.

:func:`pad_channels_collate` makes a :class:`~braindecode.datasets.BaseConcatDataset`
that mixes recordings with *different channel sets* (heterogeneous montages)
batchable: it pads every sample's channels to the batch maximum and emits a
boolean channel mask. Pair it with ``set_return_ch_pos(True)`` so each sample
carries its electrode positions -- channel identity is then encoded by the
positions, not by row index, which is exactly what position-aware models
(e.g. ``REVE``) consume.
"""

# Authors: The braindecode developers
#
# License: BSD (3-clause)

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data._utils.collate import default_collate

__all__ = ["pad_channels_collate"]


def pad_channels_collate(batch, pad_value: float = 0.0, pos_pad_value: float = 0.0):
    """Collate variable-channel windows by padding channels to the batch maximum.

    Each item of ``batch`` is the tuple returned by a windowed dataset's
    ``__getitem__``: either ``(X, y, crop_inds)`` or, when
    ``return_ch_pos=True`` (see
    :meth:`braindecode.datasets.BaseConcatDataset.set_return_ch_pos`),
    ``(X, y, crop_inds, ch_pos)``. Signals with different numbers of channels
    are zero-padded along the channel axis to ``max_ch`` (the largest channel
    count in the batch) so they can be stacked into a single tensor, and a
    boolean ``ch_mask`` marks the real (non-padded) channels.

    The time dimension is assumed uniform (fixed-length windows); a
    :class:`ValueError` is raised otherwise.

    This makes a heterogeneous collection **batchable**. *Consuming* the result
    (so a model ignores padded channels) requires a position/channel-aware
    model; fixed-geometry models (e.g. ``ShallowFBCSPNet``) structurally need a
    constant channel count and cannot use variable batches.

    Parameters
    ----------
    batch : list of tuple
        Samples from a windowed dataset (see above).
    pad_value : float
        Value used to pad the signal ``X`` channel rows (default ``0.0``).
    pos_pad_value : float
        Value used to pad the position ``ch_pos`` rows (default ``0.0``).

    Returns
    -------
    X : torch.Tensor
        Padded signals, shape ``(batch, max_ch, n_times)``.
    y : torch.Tensor
        Targets, collated with ``default_collate``.
    crop_inds : list
        Crop indices, collated with ``default_collate``.
    ch_pos : torch.Tensor
        Only when the samples include positions. Padded positions, shape
        ``(batch, max_ch, 3)``.
    ch_mask : torch.Tensor
        Boolean mask, shape ``(batch, max_ch)``; ``True`` for real channels,
        ``False`` for padding.
    """
    if len(batch) == 0:
        raise ValueError("pad_channels_collate received an empty batch.")

    has_pos = len(batch[0]) >= 4
    Xs = [np.asarray(item[0]) for item in batch]

    n_times = Xs[0].shape[1]
    if any(x.shape[1] != n_times for x in Xs):
        raise ValueError(
            "pad_channels_collate only pads the channel axis; all windows must "
            f"share the same number of time samples, got {sorted({x.shape[1] for x in Xs})}."
        )

    bsz = len(batch)
    max_ch = max(x.shape[0] for x in Xs)

    X_out = torch.full((bsz, max_ch, n_times), pad_value, dtype=torch.float32)
    ch_mask = torch.zeros((bsz, max_ch), dtype=torch.bool)
    pos_out = (
        torch.full((bsz, max_ch, 3), pos_pad_value, dtype=torch.float32)
        if has_pos
        else None
    )
    for i, (item, x) in enumerate(zip(batch, Xs)):
        c = x.shape[0]
        X_out[i, :c] = torch.as_tensor(x, dtype=torch.float32)
        ch_mask[i, :c] = True
        if pos_out is not None:
            pos_out[i, :c] = torch.as_tensor(np.asarray(item[3]), dtype=torch.float32)

    # y and crop_inds carry no channel axis -> standard collation.
    y = default_collate([item[1] for item in batch])
    crop_inds = default_collate([item[2] for item in batch])

    if pos_out is not None:
        return X_out, y, crop_inds, pos_out, ch_mask
    return X_out, y, crop_inds, ch_mask
