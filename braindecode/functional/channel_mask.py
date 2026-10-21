"""Batches of recordings with different numbers of channels.

Recordings are padded along the channel axis to a common count, and a boolean
``ch_mask`` of shape ``(batch, n_chans)`` flags the real channels (``True``) as
:func:`braindecode.datasets.pad_channels_collate` returns it. Every helper
takes ``ch_mask=None`` for a batch without padding, so a model has one code
path for both. They support TorchScript; einops functions do not, and the
merged batch and channel axis is split back with sizes known only at runtime.
"""

# Authors: Julien Gadonneix <juliengado.2001@gmail.com>
#
# License: BSD (3-clause)

from typing import List, Optional

import torch


def check_ch_mask(
    ch_mask: Optional[torch.Tensor], x: torch.Tensor
) -> Optional[torch.Tensor]:
    """Validate the channel mask of a padded batch.

    Parameters
    ----------
    ch_mask : torch.Tensor or None
        Mask of shape ``(batch, n_chans)``, ``True`` for real channels and
        ``False`` for padding.
    x : torch.Tensor
        Input of shape ``(batch, n_chans, ...)`` the mask describes.

    Returns
    -------
    torch.Tensor or None
        ``ch_mask`` as a boolean tensor on the device of ``x``, or ``None``.
        Every sample must have a real channel, which is checked in eager mode
        only.
    """
    if ch_mask is None:
        return None
    if list(ch_mask.shape) != list(x.shape[:2]):
        raise ValueError(
            f"ch_mask must have shape {list(x.shape[:2])}, got {list(ch_mask.shape)}."
        )
    ch_mask = ch_mask.to(device=x.device, dtype=torch.bool)
    # Value check only in eager mode: it branches on tensor data.
    if not torch.jit.is_scripting():
        if not torch.compiler.is_compiling() and not bool(ch_mask.any(dim=1).all()):
            raise ValueError("ch_mask must flag at least one real channel per sample.")
    return ch_mask


def pack_channels(
    x: torch.Tensor, ch_mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Merge the batch and channel axes and drop the padded channels.

    For a stage that encodes each channel on its own, so that no compute is
    spent on padding. :func:`unpack_channels` puts the channels back.

    Parameters
    ----------
    x : torch.Tensor
        Tensor of shape ``(batch, n_chans, ...)``.
    ch_mask : torch.Tensor or None
        Boolean mask of shape ``(batch, n_chans)``, ``True`` for real channels.
        ``None`` keeps every channel.

    Returns
    -------
    torch.Tensor
        Shape ``(n_real, ...)``: the real channels, sample by sample, in the
        order of the input (``(batch * n_chans, ...)`` without ``ch_mask``).
    """
    x = x.flatten(0, 1)
    if ch_mask is None:
        return x
    x = x[ch_mask.flatten()]
    # Tell torch.export the data-dependent count is not zero.
    if not torch.jit.is_scripting():
        torch._check(x.shape[0] > 0)
    return x


def unpack_channels(
    x: torch.Tensor, ch_mask: Optional[torch.Tensor], n_chans: int
) -> torch.Tensor:
    """Split the output of :func:`pack_channels` back into samples and channels.

    Parameters
    ----------
    x : torch.Tensor
        Tensor of shape ``(n_real, ...)``, one row per real channel, in the
        order :func:`pack_channels` returns them.
    ch_mask : torch.Tensor or None
        The boolean mask of shape ``(batch, n_chans)`` given to
        :func:`pack_channels`, or ``None``.
    n_chans : int
        Number of channels of the padded batch.

    Returns
    -------
    torch.Tensor
        Shape ``(batch, n_chans, ...)``, zeros for the padded channels.
    """
    if ch_mask is None:
        return x.unflatten(0, [-1, n_chans])
    shape = list(ch_mask.shape) + list(x.shape[1:])
    mask = ch_mask.reshape(list(ch_mask.shape) + [1] * (x.dim() - 1))
    return x.new_zeros(shape).masked_scatter(mask, x)


def masked_mean(
    x: torch.Tensor, mask: Optional[torch.Tensor], dim: List[int]
) -> torch.Tensor:
    """Mean over ``dim`` of the entries a mask flags.

    Parameters
    ----------
    x : torch.Tensor
        Tensor to average.
    mask : torch.Tensor or None
        Boolean mask over the leading axes of ``x`` (e.g. a ``ch_mask`` of
        shape ``(batch, n_chans)``), ``True`` for the entries that count; it
        is broadcast over the remaining axes. ``None`` is a plain mean.
    dim : list of int
        Axes to average over.

    Returns
    -------
    torch.Tensor
        ``x`` averaged over ``dim``. Masked entries do not count, whatever
        they hold (even NaN).
    """
    if mask is None:
        return x.mean(dim=dim)
    mask = mask.reshape(list(mask.shape) + [1] * (x.dim() - mask.dim()))
    total = x.masked_fill(~mask, 0.0).sum(dim=dim)
    return total / mask.expand(x.shape).sum(dim=dim)
