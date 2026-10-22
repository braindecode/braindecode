"""Cutting a time axis into patches: how many, and padding or cropping to fit."""

# Authors: Bruno Aristimunha <b.aristimunha@gmail.com>
#
# License: BSD (3-clause)

import torch
import torch.nn.functional as F

_ON_NON_DIVISIBLE = ("pad", "crop", "error")


def _check_on_non_divisible(on_non_divisible: str) -> None:
    """Raise a ``ValueError`` for an unknown ``on_non_divisible`` value."""
    if on_non_divisible not in _ON_NON_DIVISIBLE:
        raise ValueError(
            "on_non_divisible must be 'pad', 'crop', or 'error', "
            f"got {on_non_divisible!r}."
        )


def patch_count(
    n_times: int, patch_size: int, on_non_divisible: str = "pad", step: int = 0
) -> int:
    """Number of patches cut from ``n_times`` samples.

    Parameters
    ----------
    n_times : int
        Length of the axis that is patched.
    patch_size : int
        Samples per patch.
    on_non_divisible : {"pad", "crop", "error"}, default="pad"
        A last, incomplete patch is zero-padded (``"pad"``), dropped
        (``"crop"``) or raises a ``ValueError``.
    step : int, default=0
        Distance between patch starts; ``0`` means ``patch_size``
        (non-overlapping).

    Returns
    -------
    int
        Number of patches.
    """
    if step <= 0:
        step = patch_size
    if n_times < 1:
        raise ValueError(f"n_times must be positive, got {n_times}.")
    if n_times < patch_size:
        if on_non_divisible == "pad":
            return 1
        if on_non_divisible == "crop":
            raise ValueError(
                f"Need at least one full patch of {patch_size} samples, got {n_times}."
            )
    elif (n_times - patch_size) % step == 0:
        return (n_times - patch_size) // step + 1
    elif on_non_divisible != "error":
        return (n_times - patch_size) // step + 1 + (on_non_divisible == "pad")
    raise ValueError(
        f"Time dimension ({n_times}) is not divisible by patch_size ({patch_size})"
        + ("." if step == patch_size else f" with patch step {step}.")
    )


def pad_or_crop_to_patches(
    x: torch.Tensor,
    patch_size: int,
    on_non_divisible: str = "pad",
    step: int = 0,
    min_patches: int = 1,
    center: bool = False,
) -> torch.Tensor:
    """Right zero-pad or crop the last axis so the patches tile it.

    A tiled axis is returned as is. An axis short of ``min_patches`` patches is
    zero-padded up to that many, with half the missing ones (whole steps)
    before it if ``center``.

    Parameters
    ----------
    x : torch.Tensor
        Shape ``(..., n_times)``.
    patch_size, on_non_divisible, step
        As in :func:`patch_count`.
    min_patches : int, default=1
        Least number of patches on the returned axis.
    center : bool, default=False
        Pad half the missing patches on the left.

    Returns
    -------
    torch.Tensor
        Shape ``(..., patch_size + (n_patches - 1) * step)``.
    """
    n_patches = patch_count(x.shape[-1], patch_size, on_non_divisible, step)
    step = step if step > 0 else patch_size
    left = 0
    if n_patches < min_patches:
        left = (min_patches - n_patches) // 2 * step if center else 0
        n_patches = min_patches
    target = patch_size + (n_patches - 1) * step
    if target > x.shape[-1]:
        return F.pad(x, (left, target - x.shape[-1] - left))
    return x[..., :target]
