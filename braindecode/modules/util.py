# Authors: Robin Schirrmeister <robintibor@gmail.com>
#          Hubert Banville <hubert.jbanville@gmail.com>
#
# License: BSD (3-clause)

import numpy as np
import torch.nn.functional as F
from scipy.special import log_softmax
from torch import nn


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


_ONE_D_LAYERS = (nn.Conv1d, nn.MaxPool1d, nn.AvgPool1d)
_DENSE_LAYERS = _ONE_D_LAYERS + (nn.Conv2d, nn.MaxPool2d, nn.AvgPool2d)


def _as_tuple(value, n):
    return tuple(value) if hasattr(value, "__len__") else (value,) * n


def has_strided_layer(module) -> bool:
    """Whether ``module`` holds a convolution or pooling layer with stride > 1."""
    return any(
        isinstance(m, _DENSE_LAYERS) and any(s > 1 for s in _as_tuple(m.stride, 1))
        for m in module.modules()
    )


def dense_convert(module, axis=(2, 3)) -> None:
    """Replace strides by dilations so ``module`` slides over long inputs.

    Walks the 1D and 2D convolution and pooling layers in order, tracks the
    stride accumulated so far on each axis, multiplies each layer's existing
    dilation by it and sets the stride on the converted axes to 1.

    Parameters
    ----------
    module : torch.nn.Module
        Module converted in place.
    axis : int or tuple of int
        Output axes to convert, 2, 3 or (2, 3). 1D layers have one axis, 2.
    """
    axes = [ax - 2 for ax in (axis if hasattr(axis, "__iter__") else (axis,))]
    stride_so_far = [1, 1]
    for parent_name, m in list(module.named_modules()):
        if not isinstance(m, _DENSE_LAYERS):
            continue
        n = 1 if isinstance(m, _ONE_D_LAYERS) else 2
        stride = _as_tuple(m.stride, n)
        conv_axes = [ax for ax in axes if ax < n]
        dilation = list(_as_tuple(getattr(m, "dilation", 1), n))
        for ax in conv_axes:
            dilation[ax] *= stride_so_far[ax]
        for ax in range(n):
            stride_so_far[ax] *= stride[ax]
        m.stride = tuple(1 if ax in conv_axes else s for ax, s in enumerate(stride))
        if hasattr(m, "dilation"):
            m.dilation = tuple(dilation)
        elif any(d > 1 for d in dilation):  # average pooling has no dilation
            parent, _, child = parent_name.rpartition(".")
            setattr(module.get_submodule(parent), child, _DilatedAvgPool(m, dilation))


class _DilatedAvgPool(nn.Module):
    """Average pooling with dilation, written as a depthwise convolution with
    constant weights ``1 / prod(kernel_size)`` (``nn.AvgPool*d`` has no dilation).
    """

    def __init__(self, pool, dilation):
        super().__init__()
        n = len(dilation)
        self.kernel_size = _as_tuple(pool.kernel_size, n)
        self.stride = _as_tuple(pool.stride, n)
        self.padding = _as_tuple(pool.padding, n)
        self.dilation = tuple(dilation)

    def forward(self, x):
        n_maps = x.shape[1]
        weight = x.new_ones(n_maps, 1, *self.kernel_size)
        weight = weight / weight[0].numel()
        conv = F.conv1d if len(self.kernel_size) == 1 else F.conv2d
        return conv(
            x,
            weight,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=n_maps,
        )
