# Authors: Robin Schirrmeister <robintibor@gmail.com>
#          Hubert Banville <hubert.jbanville@gmail.com>
#
# License: BSD (3-clause)

import numpy as np
from scipy.special import log_softmax


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
    """
    log_probas = log_softmax(logits, axis=1)
    return _pad_shift_array(log_probas, stride=n_windows_stride).sum(axis=0).T
