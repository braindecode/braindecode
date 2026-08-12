# Authors: Vandit Shah <shahvanditt@gmail.com>
#          Bruno Aristimunha <b.aristimunha@gmail.com>
#
# Architecture based on Song et al. (2018):
#   Official TensorFlow code archived at:
#   http://web.archive.org/web/20221122064435/http://aip.seu.edu.cn/wp-content/uploads/2021/08/EEG-TAC.zip
#
# License: BSD (3-clause)

from __future__ import annotations

import mne
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
        self.projection = nn.Linear(in_features * cheb_order, out_features, bias=False)

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
        cheb_components = [x]  # T_0(L) x = x

        if self.cheb_order > 1:
            cheb_components.append(torch.matmul(laplacian, x))  # T_1(L) x = L x

        for _ in range(2, self.cheb_order):
            cheb_components.append(
                2 * torch.matmul(laplacian, cheb_components[-1]) - cheb_components[-2]
            )

        return self.projection(torch.cat(cheb_components, dim=-1))


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

    Notes
    -----
    If valid 3-D positions cannot be extracted from ``chs_info``,
    channel names are looked up in the ``standard_1005`` MNE montage.
    If that also fails (or ``chs_info`` is ``None``)
    """
    electrode_positions = extract_channel_locations_from_chs_info(
        chs_info, num_channels=n_chans
    )

    if electrode_positions is None and chs_info is not None:
        # Try to infer positions from channel names via a standard montage
        try:
            ch_names = [ch["ch_name"] for ch in chs_info[:n_chans]]
            montage = mne.channels.make_standard_montage("standard_1005")
            info = mne.create_info(ch_names=ch_names, sfreq=256, ch_types="eeg")
            info.set_montage(montage, on_missing="raise")
            electrode_positions = extract_channel_locations_from_chs_info(
                info["chs"], num_channels=n_chans
            )
        except Exception:
            raise ValueError(
                "Could not extract valid 3-D channel locations from chs_info. "
                "Please ensure that each entry in chs_info contains a 'loc' key "
                "with 3-D positions or that channel names match a standard montage."
            )

    n_neighbors_capped = min(n_neighbors, n_chans - 1)

    # Pairwise Euclidean distances
    distance_matrix = pairwise_distances(electrode_positions, metric="euclidean")

    # Gaussian kernel: w_ij = exp(-d_ij^2 / (2 * sigma^2))
    # sigma = median of kNN distances (robust scale estimate)
    knn_distances = np.sort(distance_matrix, axis=1)[:, 1 : n_neighbors_capped + 1]
    sigma = np.median(knn_distances) + 1e-8
    gaussian_weights = np.exp(-(distance_matrix**2) / (2 * sigma**2))

    # Sparsify: keep only k nearest neighbors (symmetric)
    knn_connectivity = kneighbors_graph(
        electrode_positions, n_neighbors=n_neighbors_capped, mode="connectivity"
    )
    adjacency = np.array(knn_connectivity.toarray()) * gaussian_weights

    # Symmetrize
    adjacency = np.maximum(adjacency, adjacency.T).astype(np.float32)
    return adjacency


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
        adjacency = F.relu(self.adjacency + self.bias)

        degree = adjacency.sum(dim=1)
        degree_inv_sqrt = 1.0 / (torch.sqrt(degree) + 1e-5)
        degree_inv_sqrt_diag = torch.diag(degree_inv_sqrt)

        identity = torch.eye(self.n_chans, device=adjacency.device)
        return identity - degree_inv_sqrt_diag @ adjacency @ degree_inv_sqrt_diag


class DGCNN(EEGModuleMixin, nn.Module):
    r"""DGCNN for EEG classification from Song et al. (2018) [dgcnn]_.

    :bdg-light:`Graph Neural Network` :bdg-dark-line:`Channel`

    .. figure:: ../_static/model/DGCNN.gif
        :align: center
        :alt: DGCNN Architecture
        :width: 600px

    .. rubric:: Architectural Overview

    DGCNN is a *graph-based* architecture that models EEG channels as nodes
    in a graph and **dynamically learns the adjacency matrix**
    :math:`\mathbf{W}^*` jointly with all other parameters via
    back-propagation (Algorithm 1 in [dgcnn]_). The end-to-end flow is:

    - (i) learn inter-channel relationships by dynamically updating a
      trainable adjacency matrix,
    - (ii) apply spectral graph convolution via Chebyshev polynomial
      approximation to extract graph-structured features, and
    - (iii) classify with a fully connected head.

    Different from traditional GCNN methods that predetermine the connections
    of the graph nodes according to their spatial positions, "the proposed
    DGCNN method learns the adjacency matrix in a dynamic way, i.e., the
    entries of the adjacency matrix are adaptively updated with the changes
    of graph model parameters during the model training" [dgcnn]_.

    .. rubric:: Macro Components

    - :class:`_LearnableAdjacency` **(Dynamical adjacency → graph Laplacian)**

        - *Operations.*
        - A trainable :math:`(N \times N)` matrix :math:`\mathbf{W}^*`
          initialized from electrode spatial positions via a Gaussian kernel
          (Eq. 1): :math:`w_{ij} = \exp(-\mathrm{dist}(i,j)^2 / 2\rho^2)`
          for the :math:`k`-nearest neighbors, zero otherwise.
        - **ReLU** applied after every gradient update to keep all entries
          non-negative (Algorithm 1, step 3).
        - The normalized graph Laplacian is derived as (Eq. 2):
          :math:`\mathbf{L} = \mathbf{I}
          - \mathbf{D}^{-1/2}\,\mathbf{W}^*\,\mathbf{D}^{-1/2}`.

        The adjacency matrix captures intrinsic functional relationships
        between EEG channels that pure spatial proximity may not reflect.

    - :class:`_GraphConvolution` **(Chebyshev spectral graph convolution +
      1x1 mixing)**

        - *Operations.*
        - :math:`K`-order Chebyshev polynomial expansion of spectral graph
          filters on the learned Laplacian (Eqs. 11-13):

          .. math::

              \mathbf{y}
              = \sum_{k=0}^{K-1} \theta_k\, T_k(\tilde{\mathbf{L}}^*)\,
                \mathbf{x},

          where :math:`T_k` are Chebyshev polynomials computed recursively
          (Eq. 12) and :math:`\theta_k` are learnable coefficients.
        - A :math:`1 \times 1` convolution (linear projection) that mixes
          the concatenated Chebyshev components, mapping each node's input
          features to ``n_filters`` output features.

        "Following the graph filtering operation is a :math:`1 \times 1`
        convolution layer, which aims to learn the discriminative features
        among the various frequency domains" [dgcnn]_.

    - **Activation layer.** ReLU with a learnable per-feature bias ensures
      non-negative outputs of the graph filtering layer [dgcnn]_.

    - **Classifier Head.**
      Flatten all node features and classify via a multi-layer fully
      connected network with dropout and softmax.

    .. rubric:: Graph Convolution Details

    - **Spatial (graph structure).** The adjacency matrix encodes pairwise
      relationships between EEG channels. It is initialized from 3-D
      electrode positions using a Gaussian kernel with kNN sparsification
      (Eq. 1), then *jointly optimized* with all other parameters. This
      allows the model to discover functional connectivity patterns that
      differ from the initial spatial layout. The spectral graph
      convolution then propagates information across neighboring nodes
      according to this learned graph topology.

    - **Spectral (graph spectral domain).** The Chebyshev polynomial
      approximation (Eq. 11) operates in the *graph spectral domain*
      defined by the eigenvalues of the graph Laplacian. The :math:`K`-order
      approximation acts as a localized graph filter: each node aggregates
      information from its :math:`K`-hop neighborhood. This is analogous
      to a band-pass filter in the graph frequency domain.

    - **Temporal / Frequency.** No explicit temporal convolution or
      frequency decomposition is performed within the network. In the
      original paper, the input features per node are pre-extracted
      frequency-band features (e.g., differential entropy from
      :math:`\delta`, :math:`\theta`, :math:`\alpha`, :math:`\beta`,
      :math:`\gamma` bands). When used with raw time series, the time
      samples serve directly as node features.

    .. rubric:: Additional Comments

    - **Dynamic vs. static graph.** Traditional GCNN methods fix the
      adjacency matrix before training based on spatial positions.
      DGCNN learns it end-to-end, allowing the graph to capture
      task-relevant functional connectivity rather than mere spatial
      proximity.
    - **Chebyshev order.** The order :math:`K` controls the receptive
      field on the graph: :math:`K=1` uses only direct neighbors,
      :math:`K=2` (default) reaches 2-hop neighborhoods. Higher orders
      increase expressivity but also parameter count.
    - **Regularization.** Dropout in the classification head and the
      ReLU constraint on the adjacency matrix provide implicit
      regularization. The loss function in the original paper also
      includes an explicit :math:`\ell_2` penalty on all parameters
      (Eq. 14).

    Parameters
    ----------
    chs_info : list of dict, optional
        Information about each channel, typically obtained from
        ``mne.Info['chs']``.  Each entry must contain a ``'loc'``
        key with 3-D electrode positions so the initial adjacency
        matrix can be built from spatial proximity (Eq. 1).  A montage
        must be set on the ``mne.Info`` object (see
        :meth:`mne.Info.set_montage`).  If ``None`` or positions
        cannot be extracted, raised ValueError (see Notes).
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
    activation : type[nn.Module], default=nn.ReLU
        Activation function class used after the graph convolution and
        in the classification head.
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
        n_outputs: int | None = None,
        n_chans: int | None = None,
        chs_info: list[dict] | None = None,
        n_times: int | None = None,
        input_window_seconds: float | None = None,
        sfreq: float | None = None,
        n_filters: int = 64,
        cheb_order: int = 2,
        n_neighbors: int = 5,
        mlp_dims: tuple[int, ...] = (256,),
        activation: type[nn.Module] = nn.ReLU,
        drop_prob: float = 0.5,
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

        self.activation = activation()
        self.drop_prob = drop_prob

        # Learnable adjacency W* (Section 3.1, Algorithm 1)
        # Use self.chs_info (populated by EEGModuleMixin with defaults
        # when chs_info=None) so that the adjacency can always be built.
        self.learnable_adjacency = _LearnableAdjacency(
            n_chans=self.n_chans,
            chs_info=self.chs_info,
            n_neighbors=n_neighbors,
        )

        # Chebyshev graph convolution + 1x1 conv (Eq. 13 + Fig. 2)
        self.graph_conv = _GraphConvolution(
            in_features=self.n_times,
            out_features=n_filters,
            cheb_order=cheb_order,
        )

        # Per-feature bias before ReLU (b1relu in official code)
        self.graph_conv_bias = nn.Parameter(torch.zeros(1, 1, n_filters))

        # Fully connected classification head (Fig. 2: "Full connection" + softmax)
        classifier_input_dim = self.n_chans * n_filters
        layers = []
        for hidden_dim in mlp_dims:
            layers.append(nn.Linear(classifier_input_dim, hidden_dim))
            layers.append(activation())
            layers.append(nn.Dropout(p=drop_prob))
            classifier_input_dim = hidden_dim

        self.classifier = nn.Sequential(*layers)
        self.final_layer = nn.Linear(classifier_input_dim, self.n_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""Forward pass through the DGCNN pipeline (Fig. 2).

        .. figure:: ../_static/model/DGCNN.gif
            :align: center
            :alt: DGCNN Architecture
            :width: 600px

        1. Compute normalized Laplacian from the learned adjacency.
        2. Apply Chebyshev graph convolution (Eq. 13).
        3. Add per-feature bias and apply ReLU.
        4. Flatten and classify through the fully connected head.

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
        laplacian = self.learnable_adjacency()
        x = self.activation(self.graph_conv(x, laplacian) + self.graph_conv_bias)
        x = x.reshape(x.size(0), -1)
        x = self.classifier(x)
        return self.final_layer(x)
