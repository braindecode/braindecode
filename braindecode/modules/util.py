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
        kernel_size: tuple[int],
        stride: tuple[int],
        dilation: tuple[int],
        padding: tuple[int],
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.padding = padding

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


def _as_list(value, rank: int) -> list[int]:
    if isinstance(value, (tuple, list)):
        return [int(v) for v in value]
    return [int(value)] * rank


def _like(old, new: list[int]):
    # keep an int attribute an int when it stays one value (MaxPool1d stores ints)
    return tuple(new) if isinstance(old, (tuple, list)) or len(new) > 1 else new[0]


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
        are allowed. 1-D layers convert when ``2`` is included.
    """
    if isinstance(axis, (int, np.integer)):
        axes = (int(axis),)
    elif hasattr(axis, "__iter__"):
        axes = tuple(int(ax) for ax in axis)
    else:
        raise ValueError(f"Only 2 and 3 are allowed for axis; got {axis!r}.")
    if not axes or any(ax not in (2, 3) for ax in axes):
        raise ValueError(f"Only 2 and 3 are allowed for axis; got {axis!r}.")

    # positions in a layer's (H, W) or (T,) stride tuple, and the stride removed so far
    conv_axes = {1: [0] if 2 in axes else [], 2: [ax - 2 for ax in axes]}
    stride_so_far = {1: [1], 2: [1, 1]}
    avg_pools = {
        1: (nn.AvgPool1d, _AvgPool1dWithConv),
        2: (nn.AvgPool2d, AvgPool2dWithConv),
    }
    skip_types = (
        nn.modules.conv._ConvTransposeNd,
        nn.modules.pooling._AdaptiveAvgPoolNd,
        nn.modules.pooling._AdaptiveMaxPoolNd,
        nn.Unfold,
    )
    known_1d = (nn.Conv1d, nn.MaxPool1d, nn.AvgPool1d, _AvgPool1dWithConv)
    known_2d = (nn.Conv2d, nn.MaxPool2d, nn.AvgPool2d, AvgPool2dWithConv)
    replacements: list[tuple[str, nn.Module]] = []

    for name, module in list(model.named_modules()):
        if module is model or isinstance(module, skip_types):
            continue
        if isinstance(module, Chomp1d):
            module.chomp_size = int(module.chomp_size * stride_so_far[1][0])
            continue
        if not isinstance(module, (*known_1d, *known_2d)) and any(
            True for _ in module.children()
        ):
            continue
        dilation = getattr(module, "dilation", None)
        stride = getattr(module, "stride", None)
        if dilation is None and stride is None:
            continue

        sample = stride if stride is not None else dilation
        rank = 1 if isinstance(module, known_1d) else 2
        if not isinstance(module, (*known_1d, *known_2d)):
            rank = 1 if isinstance(sample, tuple) and len(sample) == 1 else 2
        conv, so_far = conv_axes[rank], stride_so_far[rank]
        if not conv:
            continue

        new_d = _as_list(1 if dilation is None else dilation, rank)
        new_s = _as_list(1 if stride is None else stride, rank)
        padding = getattr(module, "padding", 0)
        new_p = None if isinstance(padding, str) else _as_list(padding, rank)
        for i in conv:
            new_d[i] *= so_far[i]
            if new_p is not None:
                new_p[i] *= so_far[i]
            so_far[i] *= new_s[i]
            new_s[i] = 1

        pool_cls, conv_pool_cls = avg_pools[rank]
        if isinstance(module, pool_cls) and any(d > 1 for d in new_d):
            kernel = tuple(_as_list(module.kernel_size, rank))
            pad = tuple(new_p) if new_p is not None else (0,) * rank
            replacements.append(
                (name, conv_pool_cls(kernel, tuple(new_s), tuple(new_d), pad))
            )
            continue
        if dilation is not None:
            module.dilation = _like(dilation, new_d)
        if new_p is not None and hasattr(module, "padding"):
            module.padding = _like(padding, new_p)  # type: ignore[assignment]
        if stride is not None:
            module.stride = _like(stride, new_s)

    for name, new_module in replacements:
        model.set_submodule(name, new_module)
