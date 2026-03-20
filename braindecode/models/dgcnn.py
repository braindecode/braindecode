# Authors: Vandit Shah <shahvanditt@gmail.com>
#          Bruno Aristimunha <b.aristimunha@gmail.com>
#
# Architecture based on Song et al. (2018):
#   Official TensorFlow code: http://aip.seu.edu.cn/songtengfei
#   (archived at EEG-TAC-official/lib/models.py)
#
# License: BSD (3-clause)

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from braindecode.models.base import EEGModuleMixin


class _GraphConvolution(nn.Module):
    """Chebyshev spectral graph convolution layer.

    Parameters
    ----------
    in_features : int
        Number of input features per node.
    out_features : int
        Number of output features per node.
    cheb_order : int
        Order of Chebyshev polynomial approximation.
    """

    def __init__(self, in_features, out_features, cheb_order):
        super().__init__()
        self.cheb_order = cheb_order
        self.weight = nn.Linear(in_features * cheb_order, out_features, bias=False)

    def forward(self, x, laplacian):
        """
        Parameters
        ----------
        x : Tensor, shape (B, n_chans, in_features)
        laplacian : Tensor, shape (n_chans, n_chans)

        Returns
        -------
        Tensor, shape (B, n_chans, out_features)
        """
        cheb = [x]  # T_0(L) x = x

        if self.cheb_order > 1:
            cheb.append(torch.matmul(laplacian, x))  # T_1(L) x = L x

        for _ in range(2, self.cheb_order):
            cheb.append(2 * torch.matmul(laplacian, cheb[-1]) - cheb[-2])

        return self.weight(torch.cat(cheb, dim=-1))


def _extract_positions(chs_info, n_chans):
    """Extract 3D electrode positions from MNE channel info.

    MNE stores the electrode position at indices ``0:3`` of the ``loc``
    array.  The utility :func:`extract_channel_locations_from_chs_info`
    reads indices ``3:6`` (the reference/normal vector), which are often
    all-zero.  This helper reads the correct indices and validates that
    the positions are non-degenerate.

    Returns
    -------
    np.ndarray of shape (n_chans, 3) or None
    """
    if chs_info is None:
        return None

    positions = []
    for ch in chs_info[:n_chans]:
        if not isinstance(ch, dict):
            return None
        loc = ch.get("loc")
        if loc is None:
            return None
        loc = np.asarray(loc, dtype=np.float32)
        if loc.size < 3:
            return None
        positions.append(loc[:3])

    positions = np.stack(positions)

    # Check positions are not all zero / degenerate
    if np.allclose(positions, 0):
        return None

    return positions


def _build_initial_adjacency(chs_info, n_chans, n_neighbors=5):
    """Build an initial adjacency matrix from channel spatial positions.

    Connects each channel to its ``n_neighbors`` nearest spatial
    neighbors using a Gaussian kernel weight derived from the 3-D
    electrode coordinates stored in ``chs_info``.

    Parameters
    ----------
    chs_info : list of dict
        MNE-style channel information with ``'loc'`` entries containing
        3-D electrode positions.
    n_chans : int
        Number of channels.
    n_neighbors : int
        How many spatial neighbors to connect per node.

    Returns
    -------
    np.ndarray, shape (n_chans, n_chans)

    Raises
    ------
    ValueError
        If valid 3-D electrode positions cannot be extracted from
        ``chs_info``.
    """
    locs = _extract_positions(chs_info, n_chans)

    if locs is None:
        raise ValueError(
            "DGCNN requires 3-D electrode positions to build the initial "
            "graph adjacency matrix.  Provide ``chs_info`` with valid "
            "``'loc'`` entries (e.g. from ``mne.Info['chs']`` with a "
            "montage set).  See the docstring for details."
        )

    # Pairwise Euclidean distances
    diff = locs[:, None, :] - locs[None, :, :]  # (C, C, 3)
    dist = np.sqrt((diff**2).sum(axis=-1))  # (C, C)

    # Gaussian kernel: w_ij = exp(-d_ij^2 / (2 * sigma^2))
    # sigma = median of kNN distances (robust scale estimate)
    k = min(n_neighbors, n_chans - 1)
    knn_dists = np.sort(dist, axis=1)[:, 1 : k + 1]  # exclude self
    sigma = np.median(knn_dists) + 1e-8

    W = np.exp(-(dist**2) / (2 * sigma**2))

    # Sparsify: keep only k nearest neighbors (symmetric)
    A = np.zeros_like(W)
    for i in range(n_chans):
        neighbors = np.argsort(dist[i])[1 : k + 1]
        A[i, neighbors] = W[i, neighbors]

    # Symmetrize
    A = np.maximum(A, A.T).astype(np.float32)
    return A


class _LearnableAdjacency(nn.Module):
    """Learnable adjacency matrix with ReLU non-negativity.

    The adjacency is initialized from electrode spatial positions
    provided via ``chs_info``.  During training the matrix is updated
    via backpropagation.  The normalized graph Laplacian
    ``L = I - D^{-1/2} A D^{-1/2}`` is recomputed on every forward
    call.

    Parameters
    ----------
    n_chans : int
        Number of graph nodes (EEG channels).
    chs_info : list of dict
        MNE-style channel information with 3-D positions.
    n_neighbors : int
        Neighbors per node used to build the spatial kNN graph.
    """

    def __init__(self, n_chans, chs_info, n_neighbors=5):
        super().__init__()
        self.n_chans = n_chans

        init_adj = _build_initial_adjacency(chs_info, n_chans, n_neighbors)
        self.adjacency = nn.Parameter(
            torch.from_numpy(init_adj) + 0.1 * torch.randn(n_chans, n_chans)
        )
        self.bias = nn.Parameter(torch.tensor(0.0))

    def forward(self):
        """Compute the normalized Laplacian from the current adjacency.

        Returns
        -------
        Tensor, shape (n_chans, n_chans)
        """
        A = F.relu(self.adjacency + self.bias)

        d = A.sum(dim=1)
        d_inv_sqrt = 1.0 / (torch.sqrt(d) + 1e-5)
        D_inv_sqrt = torch.diag(d_inv_sqrt)

        I = torch.eye(self.n_chans, device=A.device)
        return I - D_inv_sqrt @ A @ D_inv_sqrt


class DGCNN(EEGModuleMixin, nn.Module):
    """DGCNN for EEG classification from Song et al. (2018) [dgcnn]_.

    :bdg-success:`Graph Neural Network`

    .. figure:: https://ar5iv.labs.arxiv.org/html/1801.07829/assets/sections/figure/model_architecture.jpg
        :align: center
        :alt: DGCNN Architecture
        :width: 600px

    Dynamic Graph Convolutional Neural Network (DGCNN) treats EEG electrodes
    as nodes in a graph and **learns the adjacency matrix** via
    backpropagation. The graph convolution uses Chebyshev polynomial
    approximation of spectral filters on the learned graph Laplacian.

    .. rubric:: Architectural Overview

    1. **Learnable Adjacency Matrix**: A trainable ``(n_chans, n_chans)``
       matrix with ReLU non-negativity, initialized from electrode spatial
       positions when ``chs_info`` is provided.  The normalized Laplacian
       is derived as ``L = I − D^{−1/2} A D^{−1/2}``.
    2. **Chebyshev Graph Convolution**: K-order polynomial filtering on
       the learned Laplacian maps each node's features to ``graph_dim``
       output features.
    3. **Activation**: ReLU after graph convolution with a per-feature
       bias.
    4. **Fully Connected Head**: Flatten all node features and classify
       via FC layers with dropout.

    Parameters
    ----------
    n_outputs : int
        Number of outputs (classes).
    n_chans : int
        Number of EEG channels (electrodes = graph nodes).
    chs_info : list of dict
        **Required.**  Information about each channel, typically obtained
        from ``mne.Info['chs']``.  Each entry must contain a ``'loc'``
        key with 3-D electrode positions so the initial adjacency
        matrix can be built from spatial proximity.  A montage must be
        set on the ``mne.Info`` object (see
        :meth:`mne.Info.set_montage`).
    n_times : int
        Number of time samples per window. Used as node feature dimension.
    input_window_seconds : float
        Length of input window in seconds.
    sfreq : float
        Sampling frequency of the EEG recording.
    graph_dim : int, default=64
        Output features of the graph convolution layer (``F`` in the paper).
    cheb_order : int, default=2
        Order of Chebyshev polynomial approximation (``K`` in the paper).
    n_neighbors : int, default=5
        Number of spatial neighbors for adjacency initialization.
        Only used when ``chs_info`` carries electrode positions.
    mlp_dims : tuple[int, ...], default=(256,)
        Hidden layer sizes of the classification MLP (``M`` in the paper).
    drop_prob : float, default=0.5
        Dropout probability in the classification head.

    References
    ----------
    .. [dgcnn] Song, T., Zheng, W., Song, P., & Cui, Z. (2018). EEG emotion
        recognition using dynamical graph convolutional neural networks.
        IEEE Transactions on Affective Computing, 11(3), 532-541.
        https://doi.org/10.1109/TAFFC.2018.2817622
    """

    def __init__(
        self,
        n_outputs=None,
        n_chans=None,
        chs_info=None,
        n_times=None,
        input_window_seconds=None,
        sfreq=None,
        graph_dim=64,
        cheb_order=2,
        n_neighbors=5,
        mlp_dims=(256,),
        drop_prob=0.5,
    ):
        super().__init__(
            n_outputs=n_outputs,
            n_chans=n_chans,
            chs_info=chs_info,
            n_times=n_times,
            sfreq=sfreq,
            input_window_seconds=input_window_seconds,
        )

        del n_outputs, n_chans, n_times, input_window_seconds, sfreq

        self.drop_prob = drop_prob

        # Learnable adjacency, spatially-informed when chs_info is provided
        self.learned_adj = _LearnableAdjacency(
            n_chans=self.n_chans,
            chs_info=chs_info,
            n_neighbors=n_neighbors,
        )

        # Chebyshev graph convolution
        self.graph_conv = _GraphConvolution(
            in_features=self.n_times,
            out_features=graph_dim,
            cheb_order=cheb_order,
        )

        # Bias + ReLU after graph conv (official code: b1relu)
        self.graph_bias = nn.Parameter(torch.zeros(1, 1, graph_dim))

        # FC classification head
        fc_in = self.n_chans * graph_dim
        layers = []
        for mlp_out in mlp_dims:
            layers.append(nn.Linear(fc_in, mlp_out))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=drop_prob))
            fc_in = mlp_out

        self.final_layer = nn.Linear(fc_in, self.n_outputs)
        layers.append(self.final_layer)
        self.classifier = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : Tensor, shape (B, n_chans, n_times)

        Returns
        -------
        Tensor, shape (B, n_outputs)
        """
        laplacian = self.learned_adj()
        x = F.relu(self.graph_conv(x, laplacian) + self.graph_bias)
        x = x.reshape(x.size(0), -1)
        return self.classifier(x)
