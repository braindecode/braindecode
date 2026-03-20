# Authors: Vandit Shah <shahvanditt@gmail.com>
#          Bruno Aristimunha <b.aristimunha@gmail.com>
#
# Architecture based on Song et al. (2018):
#   Official TensorFlow code archived at:
#   http://web.archive.org/web/20221122064435/http://aip.seu.edu.cn/wp-content/uploads/2021/08/EEG-TAC.zip
#
# License: BSD (3-clause)

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import kneighbors_graph

from braindecode.models.base import EEGModuleMixin
from braindecode.models.util import extract_channel_locations_from_chs_info


class _GraphConvolution(nn.Module):
    r"""Chebyshev spectral graph convolution with :math:`1 \times 1` mixing.

    Implements Equations 11-13 of Song et al. (2018).  The spectral
    filter :math:`g(\boldsymbol{\Lambda}^*)` is approximated by a
    :math:`K`-order Chebyshev polynomial expansion:

    .. math::

        \mathbf{y}
        = \sum_{k=0}^{K-1} \theta_k\, T_k(\tilde{\mathbf{L}}^*)\,\mathbf{x},

    where :math:`T_k` are Chebyshev polynomials of the first kind
    computed recursively (Eq. 12):

    .. math::

        T_0(x) = 1,\quad T_1(x) = x,\quad
        T_k(x) = 2x\,T_{k-1}(x) - T_{k-2}(x),\quad k \ge 2,

    and :math:`\tilde{\mathbf{L}}^*` is the rescaled normalized Laplacian.
    The learnable coefficients :math:`\theta_k` are absorbed into a single
    linear projection of the concatenated Chebyshev components, which
    simultaneously acts as a :math:`1 \times 1` convolution that mixes
    features across frequency bands (see Fig. 2 in the paper).

    Parameters
    ----------
    in_features : int
        Number of input features per node (e.g. number of time samples
        or frequency bands).
    out_features : int
        Number of output features per node (number of spectral filters).
    cheb_order : int
        Order :math:`K` of the Chebyshev polynomial approximation.
    """

    def __init__(self, in_features, out_features, cheb_order):
        super().__init__()
        self.cheb_order = cheb_order
        self.weight = nn.Linear(in_features * cheb_order, out_features, bias=False)

    def forward(self, x, laplacian):
        r"""Apply Chebyshev graph convolution.

        Computes :math:`[T_0(\mathbf{L})\mathbf{x},\;\ldots,\;
        T_{K-1}(\mathbf{L})\mathbf{x}]` and projects the concatenation
        through a learned weight matrix.

        Parameters
        ----------
        x : Tensor, shape (B, N, F_in)
            Input node features, where *N* is the number of graph nodes
            (EEG channels) and *F_in* the input feature dimension.
        laplacian : Tensor, shape (N, N)
            Normalized graph Laplacian.

        Returns
        -------
        Tensor, shape (B, N, F_out)
            Filtered node features.
        """
        cheb = [x]  # T_0(L) x = x

        if self.cheb_order > 1:
            cheb.append(torch.matmul(laplacian, x))  # T_1(L) x = L x

        for _ in range(2, self.cheb_order):
            cheb.append(2 * torch.matmul(laplacian, cheb[-1]) - cheb[-2])

        return self.weight(torch.cat(cheb, dim=-1))


def _build_initial_adjacency(chs_info, n_chans, n_neighbors=5):
    r"""Build an initial adjacency matrix from electrode spatial positions.

    Implements the Gaussian kernel graph construction from Eq. 1 of
    Song et al. (2018):

    .. math::

        w_{ij} = \begin{cases}
            \exp\!\Big(-\dfrac{\mathrm{dist}(i,j)^{2}}{2\rho^{2}}\Big),
            & \text{if } \mathrm{dist}(i,j) \le \tau \\
            0, & \text{otherwise}
        \end{cases}

    where :math:`\rho` is estimated as the median of :math:`k`-nearest
    neighbor distances and the threshold :math:`\tau` is enforced by
    keeping only the ``n_neighbors`` nearest neighbors per node.
    The resulting matrix is symmetrized so that
    :math:`A = \max(A, A^\top)`.

    Parameters
    ----------
    chs_info : list of dict
        MNE-style channel information with ``'loc'`` entries containing
        3-D electrode positions.
    n_chans : int
        Number of channels (graph nodes :math:`N`).
    n_neighbors : int
        How many spatial neighbors to connect per node (:math:`\tau`
        is implicitly defined by this value).

    Returns
    -------
    np.ndarray, shape (n_chans, n_chans)
        Symmetric adjacency matrix :math:`\mathbf{W}`.

    Raises
    ------
    ValueError
        If valid 3-D electrode positions cannot be extracted from
        ``chs_info``.
    """
    locs = extract_channel_locations_from_chs_info(chs_info, num_channels=n_chans)

    if locs is None:
        raise ValueError(
            "DGCNN requires 3-D electrode positions to build the initial "
            "graph adjacency matrix.  Provide ``chs_info`` with valid "
            "``'loc'`` entries (e.g. from ``mne.Info['chs']`` with a "
            "montage set).  See the docstring for details."
        )

    k = min(n_neighbors, n_chans - 1)

    # Pairwise Euclidean distances
    dist = pairwise_distances(locs, metric="euclidean")

    # Gaussian kernel: w_ij = exp(-d_ij^2 / (2 * sigma^2))
    # sigma = median of kNN distances (robust scale estimate)
    knn_dists = np.sort(dist, axis=1)[:, 1 : k + 1]
    sigma = np.median(knn_dists) + 1e-8
    W = np.exp(-(dist**2) / (2 * sigma**2))

    # Sparsify: keep only k nearest neighbors (symmetric)
    knn = kneighbors_graph(locs, n_neighbors=k, mode="connectivity")
    A = np.array(knn.toarray()) * W

    # Symmetrize
    A = np.maximum(A, A.T).astype(np.float32)
    return A


class _LearnableAdjacency(nn.Module):
    r"""Learnable adjacency matrix with ReLU non-negativity.

    Implements the dynamical adjacency learning described in Section 3
    and Algorithm 1 of Song et al. (2018).  The adjacency matrix
    :math:`\mathbf{W}^*` is initialized from electrode spatial
    positions (Eq. 1) and then optimized jointly with all other model
    parameters via back-propagation (Eqs. 15-16):

    .. math::

        \mathbf{W}^* \leftarrow (1 - \rho)\,\mathbf{W}^*
        + \rho\,\frac{\partial Loss}{\partial \mathbf{W}^*}.

    Following Algorithm 1 (step 3), a ReLU operation is applied after
    every update to keep all entries non-negative.  The normalized
    graph Laplacian is then derived as (Eq. 2, normalized form):

    .. math::

        \mathbf{L} = \mathbf{I}
        - \mathbf{D}^{-1/2}\,\mathbf{W}^*\,\mathbf{D}^{-1/2},

    where :math:`D_{ii} = \sum_j w^*_{ij}`.

    Parameters
    ----------
    n_chans : int
        Number of graph nodes :math:`N` (EEG channels).
    chs_info : list of dict
        MNE-style channel information with 3-D positions used to
        build the initial adjacency via :func:`_build_initial_adjacency`.
    n_neighbors : int
        Neighbors per node for the initial spatial kNN graph.
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
        r"""Compute the normalized Laplacian from the current adjacency.

        Applies ReLU to enforce non-negativity (Algorithm 1, step 3)
        and returns
        :math:`\mathbf{L} = \mathbf{I}
        - \mathbf{D}^{-1/2}\,\mathbf{A}\,\mathbf{D}^{-1/2}`.

        Returns
        -------
        Tensor, shape (N, N)
            Normalized graph Laplacian.
        """
        A = F.relu(self.adjacency + self.bias)

        d = A.sum(dim=1)
        d_inv_sqrt = 1.0 / (torch.sqrt(d) + 1e-5)
        D_inv_sqrt = torch.diag(d_inv_sqrt)

        I = torch.eye(self.n_chans, device=A.device)
        return I - D_inv_sqrt @ A @ D_inv_sqrt


class DGCNN(EEGModuleMixin, nn.Module):
    r"""DGCNN for EEG classification from Song et al. (2018) [dgcnn]_.

    :bdg-light:`Graph Neural Network`

    .. figure:: ../docs/_static/model/DGCNN.gif
        :align: center
        :alt: DGCNN Architecture
        :width: 600px

    Dynamic Graph Convolutional Neural Network (DGCNN) treats EEG
    electrodes as nodes in a graph and **learns the adjacency matrix**
    :math:`\mathbf{W}^*` jointly with all other parameters via
    back-propagation (Algorithm 1).  The graph convolution uses a
    :math:`K`-order Chebyshev polynomial approximation of spectral
    filters on the learned graph Laplacian (Eq. 13):

    .. math::

        \mathbf{y}
        = \sum_{k=0}^{K-1} \theta_k\, T_k(\tilde{\mathbf{L}}^*)\,
          \mathbf{x}.

    .. rubric:: Architectural Overview (Fig. 2)

    1. **Learnable Adjacency Matrix** — A trainable
       :math:`(N \times N)` matrix with ReLU non-negativity
       (Algorithm 1, step 3), initialized from electrode spatial
       positions via the Gaussian kernel of Eq. 1.  The normalized
       Laplacian is derived as
       :math:`\mathbf{L} = \mathbf{I}
       - \mathbf{D}^{-1/2}\,\mathbf{W}^*\,\mathbf{D}^{-1/2}`.
    2. **Chebyshev Graph Convolution** — :math:`K`-order polynomial
       spectral filtering (Eq. 13) combined with a :math:`1 \times 1`
       convolution that maps each node's features to
       ``n_filters`` output features.
    3. **Activation** — ReLU with a learnable per-feature bias.
    4. **Fully Connected Head** — Flatten all node features and
       classify via FC layers with dropout and softmax.

    Parameters
    ----------
    n_outputs : int
        Number of outputs of the model (number of classes :math:`C`).
    n_chans : int
        Number of EEG channels (graph nodes :math:`N`).
    chs_info : list of dict
        **Required.**  Information about each channel, typically obtained
        from ``mne.Info['chs']``.  Each entry must contain a ``'loc'``
        key with 3-D electrode positions so the initial adjacency
        matrix can be built from spatial proximity (Eq. 1).  A montage
        must be set on the ``mne.Info`` object (see
        :meth:`mne.Info.set_montage`).
    n_times : int
        Number of time samples per window.  Used as the input feature
        dimension per node.
    input_window_seconds : float
        Length of input window in seconds.
    sfreq : float
        Sampling frequency of the EEG recording.
    n_filters : int, default=64
        Number of spectral graph-convolutional filters.  This is the
        output feature dimension per node produced by the Chebyshev
        graph convolution followed by the :math:`1 \times 1`
        convolution (see Fig. 2 in the paper).  The original code
        uses 64.
    cheb_order : int, default=2
        Order :math:`K` of the Chebyshev polynomial approximation
        (Eq. 11).
    n_neighbors : int, default=5
        Number of spatial nearest neighbors per node used to build the
        initial adjacency matrix (Eq. 1).
    mlp_dims : tuple[int, ...], default=(256,)
        Hidden-layer sizes of the fully connected classification head.
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
        n_filters=64,
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

        # Learnable adjacency W* (Section 3.1, Algorithm 1)
        self.learned_adj = _LearnableAdjacency(
            n_chans=self.n_chans,
            chs_info=chs_info,
            n_neighbors=n_neighbors,
        )

        # Chebyshev graph convolution + 1x1 conv (Eq. 13 + Fig. 2)
        self.graph_conv = _GraphConvolution(
            in_features=self.n_times,
            out_features=n_filters,
            cheb_order=cheb_order,
        )

        # Per-feature bias before ReLU (b1relu in official code)
        self.graph_bias = nn.Parameter(torch.zeros(1, 1, n_filters))

        # FC classification head (Fig. 2: "Full connection" + softmax)
        fc_in = self.n_chans * n_filters
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
        r"""Forward pass through the DGCNN pipeline (Fig. 2).

        1. Compute normalized Laplacian from the learned adjacency.
        2. Apply Chebyshev graph convolution (Eq. 13).
        3. Add per-feature bias and apply ReLU.
        4. Flatten and classify through the FC head.

        Parameters
        ----------
        x : Tensor, shape (batch, n_chans, n_times)
            Input EEG tensor where each channel corresponds to a
            graph node and the time samples are the input features
            per node.

        Returns
        -------
        Tensor, shape (batch, n_outputs)
            Class logits.
        """
        laplacian = self.learned_adj()
        x = F.relu(self.graph_conv(x, laplacian) + self.graph_bias)
        x = x.reshape(x.size(0), -1)
        return self.classifier(x)
