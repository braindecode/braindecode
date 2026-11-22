# Authors: Robin Schirrmeister <robintibor@gmail.com>
#          Hubert Banville <hubert.jbanville@gmail.com>
#
# License: BSD (3-clause)

import numpy as np
import torch
from scipy.special import log_softmax
from torch import nn
from torch.nn import functional as F

from .convolution import AvgPool2dWithConv
from .layers import Chomp1d


def _pad_shift_array(x, stride=1):
    """Zero-pad and shift rows of a 3D array.

    E.g., used to align predictions of corresponding windows in
    sequence-to-sequence models.

    Parameters
    ----------
    x : np.ndarray
        Array of shape (n_rows, n_classes, n_windows).
    stride : int
        Number of non-overlapping elements between two consecutive sequences.

    Returns
    -------
    np.ndarray :
        Array of shape (n_rows, n_classes, (n_rows - 1) * stride + n_windows)
        where each row is obtained by zero-padding the corresponding row in
        ``x`` before and after in the last dimension.
    """
    if x.ndim != 3:
        raise NotImplementedError(
            f"x must be of shape (n_rows, n_classes, n_windows), got {x.shape}"
        )
    x_padded = np.pad(x, ((0, 0), (0, 0), (0, (x.shape[0] - 1) * stride)))
    orig_strides = x_padded.strides
    new_strides = (
        orig_strides[0] - stride * orig_strides[2],
        orig_strides[1],
        orig_strides[2],
    )
    return np.lib.stride_tricks.as_strided(x_padded, strides=new_strides)


def aggregate_probas(logits, n_windows_stride=1):
    """Aggregate predicted probabilities with self-ensembling.

    Aggregate window-wise predicted probabilities obtained on overlapping
    sequences of windows using multiplicative voting as described in
    [Phan2018]_.

    Parameters
    ----------
    logits : np.ndarray
        Array of shape (n_sequences, n_classes, n_windows) containing the
        logits (i.e. the raw unnormalized scores for each class) for each
        window of each sequence.
    n_windows_stride : int
        Number of windows between two consecutive sequences. Default is 1
        (maximally overlapping sequences).

    Returns
    -------
    np.ndarray :
        Array of shape ((n_rows - 1) * stride + n_windows, n_classes)
        containing the aggregated predicted probabilities for each window
        contained in the input sequences.

    References
    ----------
    .. [Phan2018] Phan, H., Andreotti, F., Cooray, N., Chén, O. Y., &
        De Vos, M. (2018). Joint classification and prediction CNN framework
        for automatic sleep stage classification. IEEE Transactions on
        Biomedical Engineering, 66(5), 1285-1296.

    Examples
    --------
    >>> import numpy as np
    >>> from braindecode.modules import aggregate_probas
    >>> logits = np.random.randn(3, 4, 5)  # (n_sequences, n_classes, n_windows)
    >>> probas = aggregate_probas(logits, n_windows_stride=1)
    >>> probas.shape
    (7, 4)
    """
    log_probas = log_softmax(logits, axis=1)
    return _pad_shift_array(log_probas, stride=n_windows_stride).sum(axis=0).T


class _AvgPool1dWithConv(nn.Module):
    """1-D average pooling implemented via depthwise convolution to support dilation."""

    def __init__(
        self,
        kernel_size: int | tuple[int],
        stride: int | tuple[int] = 1,
        dilation: int | tuple[int] = 1,
        padding: int | tuple[int] = 0,
    ) -> None:
        super().__init__()
        self.kernel_size = (
            (kernel_size,) if isinstance(kernel_size, int) else tuple(kernel_size)
        )
        self.stride = (stride,) if isinstance(stride, int) else tuple(stride)
        self.dilation = (dilation,) if isinstance(dilation, int) else tuple(dilation)
        self.padding = (padding,) if isinstance(padding, int) else tuple(padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_channels = x.shape[1]
        k = self.kernel_size[0]
        weights = x.new_full((in_channels, 1, k), 1.0 / float(k))
        return F.conv1d(
            x,
            weights,
            bias=None,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=in_channels,
        )


def _has_strided_layer(model: nn.Module) -> bool:
    """Return whether ``model`` contains any submodule with stride greater than 1."""
    for module in model.modules():
        stride = getattr(module, "stride", None)
        if isinstance(stride, (int, np.integer)) and stride > 1:
            return True
        if isinstance(stride, (tuple, list)) and any(s > 1 for s in stride):
            return True
    return False


def _replace_submodule(root: nn.Module, path: str, new_module: nn.Module) -> None:
    parts = path.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)


def to_dense_prediction_model(
    model: nn.Module, axis: tuple[int, ...] | int = (2, 3)
) -> None:
    """Convert a sequential strided/dilated convolutional stack to dense predictions.

    Replaces strides along ``axis`` with ``1`` and multiplies downstream
    dilations (and numeric paddings / causal chomps) by the cumulative stride
    removed before each layer. Average pooling layers that require ``dilation > 1``
    are replaced with dilated depthwise-convolutional equivalents. Modifies
    ``model`` in place.

    Parameters
    ----------
    model : nn.Module
        Sequential convolutional model to convert in place.
    axis : int or tuple of int, default=(2, 3)
        Intermediate tensor axes to convert; only ``2``, ``3``, or ``(2, 3)``
        are allowed.
    """
    if isinstance(axis, (int, np.integer)):
        axes = (int(axis),)
    elif hasattr(axis, "__iter__"):
        axes = tuple(int(ax) for ax in axis)
    else:
        raise ValueError(f"Only 2 and 3 are allowed for axis; got {axis!r}.")
    if not axes or any(ax not in (2, 3) for ax in axes):
        raise ValueError(f"Only 2 and 3 are allowed for axis; got {axis!r}.")

    ax_2d = [ax - 2 for ax in axes]
    convert_1d = 0 in ax_2d
    stride_1d = 1
    stride_2d = [1, 1]
    replacements: list[tuple[str, nn.Module]] = []

    skip_types = (
        nn.modules.conv._ConvTransposeNd,
        nn.modules.pooling._AdaptiveAvgPoolNd,
        nn.modules.pooling._AdaptiveMaxPoolNd,
        nn.Unfold,
    )
    known_1d = (nn.Conv1d, nn.MaxPool1d, nn.AvgPool1d, _AvgPool1dWithConv)
    known_2d = (nn.Conv2d, nn.MaxPool2d, nn.AvgPool2d, AvgPool2dWithConv)

    for name, module in list(model.named_modules()):
        if module is model or isinstance(module, skip_types):
            continue
        if isinstance(module, Chomp1d):
            if convert_1d and stride_1d > 1:
                module.chomp_size = int(module.chomp_size * stride_1d)
            continue

        is_known = isinstance(module, (*known_1d, *known_2d))
        if not is_known and any(True for _ in module.children()):
            continue

        has_dil = getattr(module, "dilation", None) is not None
        has_str = getattr(module, "stride", None) is not None
        if not has_dil and not has_str:
            continue

        is_1d = isinstance(module, known_1d)
        if not is_1d and not isinstance(module, known_2d):
            sample = module.stride if has_str else module.dilation
            is_1d = isinstance(sample, tuple) and len(sample) == 1

        if is_1d:
            if not convert_1d:
                continue
            raw_d = module.dilation if has_dil else 1
            old_d = int(raw_d[0]) if isinstance(raw_d, (tuple, list)) else int(raw_d)
            new_d = int(old_d * stride_1d)

            raw_s = module.stride if has_str else 1
            old_s = int(raw_s[0]) if isinstance(raw_s, (tuple, list)) else int(raw_s)

            old_p = getattr(module, "padding", 0)
            if isinstance(old_p, tuple):
                new_p_1d: int | tuple[int, ...] | str = (int(old_p[0] * stride_1d),)
            elif isinstance(old_p, (int, np.integer)):
                new_p_1d = int(old_p * stride_1d)
            else:
                new_p_1d = old_p

            stride_1d *= old_s
            if isinstance(module, nn.AvgPool1d) and new_d > 1:
                k_1d = (
                    module.kernel_size[0]
                    if isinstance(module.kernel_size, tuple)
                    else int(module.kernel_size)
                )
                p_1d = (
                    new_p_1d[0]
                    if isinstance(new_p_1d, tuple)
                    else (int(new_p_1d) if isinstance(new_p_1d, int) else 0)
                )
                replacements.append(
                    (
                        name,
                        _AvgPool1dWithConv(
                            kernel_size=k_1d, stride=1, dilation=new_d, padding=p_1d
                        ),
                    )
                )
            else:
                if has_dil:
                    module.dilation = (
                        (new_d,) if isinstance(module.dilation, tuple) else new_d
                    )
                if isinstance(old_p, (int, np.integer, tuple)) and hasattr(
                    module, "padding"
                ):
                    module.padding = new_p_1d  # type: ignore[assignment]
                if has_str:
                    module.stride = (1,) if isinstance(module.stride, tuple) else 1
        else:
            if has_dil:
                raw_d2 = module.dilation
                old_d2 = (
                    [int(v) for v in raw_d2]
                    if isinstance(raw_d2, (tuple, list))
                    else [int(raw_d2), int(raw_d2)]
                )
            else:
                old_d2 = [1, 1]
            new_d2 = list(old_d2)
            for idx in ax_2d:
                if idx < len(new_d2):
                    new_d2[idx] = int(old_d2[idx] * stride_2d[idx])

            if has_str:
                raw_s2 = module.stride
                old_s2 = (
                    [int(v) for v in raw_s2]
                    if isinstance(raw_s2, (tuple, list))
                    else [int(raw_s2), int(raw_s2)]
                )
            else:
                old_s2 = [1, 1]
            new_s2 = list(old_s2)

            old_p2 = getattr(module, "padding", 0)
            if isinstance(old_p2, (int, np.integer)):
                new_p2: list[int] | None = [int(old_p2), int(old_p2)]
            elif isinstance(old_p2, (tuple, list)):
                new_p2 = [int(v) for v in old_p2]
            else:
                new_p2 = None
            if new_p2 is not None:
                for idx in ax_2d:
                    if idx < len(new_p2):
                        new_p2[idx] = int(new_p2[idx] * stride_2d[idx])

            for idx in ax_2d:
                if idx < len(old_s2):
                    stride_2d[idx] *= int(old_s2[idx])
                    new_s2[idx] = 1

            if isinstance(module, nn.AvgPool2d) and any(d > 1 for d in new_d2):
                k_2d = (
                    module.kernel_size
                    if isinstance(module.kernel_size, tuple)
                    else (int(module.kernel_size), int(module.kernel_size))
                )
                p_2d = tuple(new_p2) if new_p2 is not None else (0, 0)
                replacements.append(
                    (
                        name,
                        AvgPool2dWithConv(
                            kernel_size=k_2d,
                            stride=tuple(new_s2),
                            dilation=tuple(new_d2),
                            padding=p_2d,
                        ),
                    )
                )
            else:
                if has_dil:
                    module.dilation = tuple(new_d2)
                if new_p2 is not None and hasattr(module, "padding"):
                    module.padding = tuple(new_p2)  # type: ignore[assignment]
                if has_str:
                    module.stride = tuple(new_s2)

    for name, new_module in replacements:
        _replace_submodule(model, name, new_module)
