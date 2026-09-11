from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from scipy.sparse.csgraph import laplacian
from torch import nn
from torch_geometric.nn.pool import graclus

from braindecode.models.base import EEGModuleMixin
from braindecode.models.dgcnn import _ChebyshevGraphConvolution


class _GraphCoarsening:
    """
    To carry out pooling to reduce dimensionality, the Graclus multilevel
    clustering algorithm is performed.
    """

    def __call__(self, adjacency, levels):
        adjacencies = [adjacency]
        clusters = []

        for _ in range(levels):
            cluster = self._cluster(adjacencies[-1])
            clusters.append(cluster)

            coarsened_adjacency = self._coarsen_adjacency(
                adjacencies[-1],
                cluster,
            )
            adjacencies.append(coarsened_adjacency)

        permutations = self._compute_perm(clusters)

        for i in range(levels):
            adjacencies[i] = self._perm_adjacency(
                adjacencies[i],
                permutations[i],
            )

        return adjacencies, permutations[0] if levels > 0 else None

    def _cluster(self, adjacency):
        row, col = adjacency.nonzero(as_tuple=True)
        edge_index = torch.stack([row, col])
        edge_weight = adjacency[row, col]

        clusters = graclus(
            edge_index,
            weight=edge_weight,
            num_nodes=adjacency.shape[0],
        )

        _, clusters = torch.unique(
            clusters,
            sorted=True,
            return_inverse=True,
        )

        return clusters

    def _coarsen_adjacency(self, adjacency, clusters):
        n_clusters = int(clusters.max().item()) + 1

        membership = nn.functional.one_hot(
            clusters,
            num_classes=n_clusters,
        ).to(adjacency.dtype)

        coarsened_adjacency = membership.T @ adjacency @ membership
        coarsened_adjacency.fill_diagonal_(0)

        return coarsened_adjacency

    def _compute_perm(self, parents):
        """
        Return a list of indices to reorder the adjacency and data matrices so
        that the union of two neighbors from layer to layer forms a binary tree.
        """

        # Order of last layer is arbitrary (chosen by the clustering algorithm).
        indices = []
        if len(parents) > 0:
            M_last = int(parents[-1].max().item()) + 1
            indices.append(list(range(M_last)))

        for parent in parents[::-1]:
            # Fake nodes go after real ones.
            pool_singletons = len(parent)

            indices_layer = []
            for i in indices[-1]:
                indices_node = torch.where(parent == i)[0].tolist()

                if len(indices_node) == 1:
                    # Add a node to go with a singleton.
                    indices_node.append(pool_singletons)
                    pool_singletons += 1

                elif len(indices_node) == 0:
                    # Add two nodes as children of a singleton in the parent.
                    indices_node.append(pool_singletons + 0)
                    indices_node.append(pool_singletons + 1)
                    pool_singletons += 2

                indices_layer.extend(indices_node)
            indices.append(indices_layer)

        return indices[::-1]

    def _perm_adjacency(self, adjacency, indices):
        """
        Permute adjacency matrix, i.e. exchange node ids,
        so that binary unions form the clustering tree.
        """
        n_nodes = adjacency.shape[0]
        n_nodes_new = len(indices)

        # Add isolated fake nodes for padding.
        if n_nodes_new > n_nodes:
            padding = n_nodes_new - n_nodes
            adjacency = F.pad(
                adjacency,
                (0, padding, 0, padding),
            )

        indices = torch.tensor(
            indices,
            dtype=torch.long,
            device=adjacency.device,
        )

        return adjacency[indices][:, indices]


class GCNsNet(EEGModuleMixin, nn.Module):
    r"""GCNs-Net: A Graph Convolutional Neural Network
    Approach for Decoding Time-resolved EEG Motor
    Imagery Signals from Hou et al. (2022) [hou2022gcns]_.

    :bdg-light:`Graph Neural Network` :bdg-dark-line:`Channel`

    .. figure:: ../_static/model/GCNsNet.jpeg
        :align: center
        :alt: GCNs-Net Architecture
        :width: 600px

    .. rubric:: Architectural Overview

    GCNs-Net is a *graph-based* architecture that models EEG channels as nodes
    in a graph. It has been introduced as a novel structure of GCNs to decode
    EEG MI signals. The end-to-end flow is:

    - (i) based on the absolute Pearson's matrix (PCC) of overall signals, the
    graph Laplacian is built up to represent the topological relationship of
    EEG electrodes.
    - (ii) GCNs-Net built on the graph convolutional layers learns the
    generalized features.
    - (iii) followed pooling layers reduce dimensionality.
    - (iv) fully-connected (FC) softmax layer derives the final prediction.

    The framework of this work, as shown in Fig. 1.:

    - (i) 64-channel raw EEG signals are acquired as one of the inputs of the
    GCNs-Net.
    - (ii) The PCC matrix, absolute PCC matrix, adjacency matrix, and graph
    Laplacian are introduced to represent the correlations between electrodes.
    - (iii) The graph representation, another input of the GCNs-Net, is
    represented by the graph Laplacian.
    - (iv) The GCNs-Net is applied to decode EEG MI signals, where N denotes
    the number of electrodes and l denotes the lth graph pooling layer.

    .. rubric:: Macro Components

    An undirected and weighted graph is represented by :math:`G = \{V,E,A\}`,
    in which :math:`V` denotes a set of nodes with the number :math:`|V| = N`,
    :math:`E` denotes a set of edges connecting nodes, and
    :math:`A \in R^{N \times N}` is a weighted adjacency matrix
    representing correlations between two nodes.

    To present the degree matrix of a graph, the scale of the graph weights
    is analysed regardless of the polar relevance, i.e., whether the
    correlations are positive or negative. So the absolute Pearson
    Correlation Coefficient (PCC) matrix :math:`|P| \in [0,1]` is introduced,
    which is the absolute of the PCC matrix :math:`P`, to map the linear
    correlations between signals.

    :math:`A` is represented as

    .. math::

        A = |P| - I

    where :math:`I` is an identity matrix.

    Furthermore, the degree matrix :math:`D` is obtained, which is a diagonal
    matrix, and the i-th diagonal element can be computed by:

    .. math::

        D_{ii} = \sum_{j=1}^{N} A_{ij}

    Finally, the combinatorial Laplacian :math:`L \in R^{N \times N}` is
    represented as

    .. math::

        L = D - A

    And the normalized graph Laplacian

    .. math::

        L = I_N - D^{-1/2} A D^{-1/2}

    represents the correlations between nodes.

    Parameters
    ----------
    chs_info : list of dict, optional
        Information about each channel, typically obtained from
        ``mne.Info['chs']``.
    laplacians : tuple of Tensor, optional
        List of Graph Laplacians. Size M x M. One per coarsening level.
    n_features : tuple of int, default=(16, 32, 64, 128, 256, 512)
        Number of features.
    cheb_orders : tuple of int, default=(2, 2, 2, 2, 2, 2)
        Order :math:`K` of the Chebyshev polynomial approximation.
    pool_sizes : tuple of int, default=(2, 2, 2, 2, 2, 2)
        Pooling size. Should be 1 (no pooling) or a power of 2
        (reduction by 2 at each coarser level). Beware to have
        coarsened enough.

    References
    ----------
    .. [hou2022gcns] Hou, Y., Jia, S., Lun, X., Hao, Z., Shi, Y., Li, Y.,
        Zeng, R., & Lv, J. (2022). GCNs-Net: A Graph Convolutional Neural
        Network Approach for Decoding Time-resolved EEG Motor Imagery
        Signals. IEEE Transactions on Neural Networks and Learning Systems.
        https://ieeexplore.ieee.org/abstract/document/9889159
    """

    def __init__(
        self,
        n_outputs: int | None = None,
        n_chans: int | None = None,
        chs_info: list[dict] | None = None,
        n_times: int | None = None,
        input_window_seconds: float | None = None,
        sfreq: float | None = None,
        laplacians: tuple[torch.Tensor, ...] | None = None,
        n_features: tuple[int, ...] = (16, 32, 64, 128, 256, 512),
        cheb_orders: tuple[int, ...] = (2, 2, 2, 2, 2, 2),
        pool_sizes: tuple[int, ...] = (2, 2, 2, 2, 2, 2),
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

        if not (
            len(n_features) == len(cheb_orders) == len(pool_sizes) != 0
        ):
            raise ValueError(
                "n_features, cheb_orders, and pool_sizes must be of the same, non-zero length. "
                "Each value is for a layer of the corresponding index order. "
                "At least one layer must be present. "
            )

        self.n_features = n_features
        self.cheb_orders = cheb_orders
        self.pool_sizes = pool_sizes
        self.graph_initialized = False

        for i in range(len(n_features)):
            self.register_buffer(f"laplacian_{i}", None)

        if laplacians is not None and len(laplacians) != 0:
            # From input Laplacians, keep the useful Laplacians only.
            # There may be zero Laplacians selected.
            laplacian_index = 0
            useful_laplacians = []

            for pool_size in pool_sizes:
                useful_laplacians.append(laplacians[laplacian_index])
                laplacian_index += int(np.log2(pool_size)) if pool_size > 1 else 0

            for i, laplacian_matrix in enumerate(useful_laplacians):
                laplacian_matrix = self._rescale_laplacian(laplacian_matrix)
                setattr(self, f"laplacian_{i}", laplacian_matrix)

            self.graph_initialized = True

        self.graph_coarsening = _GraphCoarsening()

        # Implement Graph Convolutional Neural Network layers.
        self.graph_convs = nn.ModuleList()

        in_features = 1  # Single scalar input to first layer
        for out_features, cheb_order in zip(n_features, cheb_orders):
            self.graph_convs.append(
                _ChebyshevGraphConvolution(
                    in_features=in_features,
                    out_features=out_features,
                    cheb_order=cheb_order,
                )
            )
            in_features = out_features

        # Bias, batch normalization and activation of each layer.
        self.biases = nn.ParameterList(
            [nn.Parameter(torch.full((1, 1, n_feature), 0.1)) for n_feature in n_features]
        )

        self.batch_norms = nn.ModuleList(
            [nn.BatchNorm1d(n_feature) for n_feature in n_features]
        )

        self.activation = nn.Softplus()

        # Final classification layer.
        self.final_layer = nn.LazyLinear(self.n_outputs)

    def _adjacency(self, x):
        """Compute the adjacency matrix from EEG data."""
        signals = x.transpose(0, 1).flatten(1)

        pcc = torch.corrcoef(signals)

        adjacency = pcc.abs()
        adjacency.fill_diagonal_(0)

        return adjacency

    def _laplacian(self, adjacency):
        """Compute the normalized graph Laplacian from the adjacency matrix."""
        laplacian_matrix = laplacian(
            adjacency.detach().cpu().numpy(),
            normed=True,
        )

        return torch.as_tensor(
            laplacian_matrix,
            dtype=adjacency.dtype,
            device=adjacency.device,
        )

    def _initialize_graph(self, x):
        adjacency = self._adjacency(x)

        adjacencies, permutation = self.graph_coarsening(
            adjacency,
            levels=5,
        )

        for i, adjacency in enumerate(adjacencies):
            laplacian_matrix = self._laplacian(adjacency)
            laplacian_matrix = self._rescale_laplacian(laplacian_matrix)
            setattr(self, f"laplacian_{i}", laplacian_matrix)

        self.graph_initialized = True

        return adjacencies, permutation

    def _bias_norm_softplus(self, x, layer_index):
        """
        Apply bias, batch normalization, and Softplus.
        Output:
        N x M x Fout = Number of samples x Number of nodes x Number of output features
        """
        x = x + self.biases[layer_index]

        x = x.transpose(1, 2)
        x = self.batch_norms[layer_index](x)
        x = x.transpose(1, 2)

        return self.activation(x)

    def _max_pool(self, x, pool_size):
        """Max pooling of size pool_size. Should be a power of 2."""
        if pool_size > 1:
            x = x.transpose(1, 2)
            x = nn.functional.max_pool1d(
                x,
                kernel_size=pool_size,
                stride=pool_size,
                ceil_mode=True,
            )
            x = x.transpose(1, 2)

        return x

    def _rescale_laplacian(self, laplacian_matrix):
        """Rescale Laplacian, without modifying the input Laplacian."""
        identity = torch.eye(
            laplacian_matrix.shape[0],
            dtype=laplacian_matrix.dtype,
            device=laplacian_matrix.device,
        )

        return laplacian_matrix - identity

    def forward(self, x) -> torch.Tensor:
        """
        Forward pass of the GCNs-Net model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, n_chans, n_times).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, n_outputs, n_times).
        """
        if not self.graph_initialized:
            self._initialize_graph(x)

        batch_size, n_chans, n_times = x.shape

        x = x.transpose(1, 2)
        x = x.reshape(batch_size * n_times, n_chans)
        x = x.unsqueeze(-1)

        for i, graph_conv in enumerate(self.graph_convs):
            laplacian_matrix = getattr(self, f"laplacian_{i}")

            x = graph_conv(x, laplacian_matrix)
            x = self._bias_norm_softplus(x, i)
            x = self._max_pool(x, self.pool_sizes[i])

        x = torch.flatten(x, 1)
        x = self.final_layer(x)

        # Restore dimensions
        x = x.reshape(batch_size, n_times, self.n_outputs)
        x = x.transpose(1, 2)

        return x
