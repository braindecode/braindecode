# Authors: Shuyue Jia <shuyuej@ieee.org> (original implementation)
#          Seáic Mac Murchadha <seaicmacmurchadha@yahoo.com> (braindecode adaptation)
#
# License: MIT
# Adapted from https://github.com/SuperBruceJia/EEG-DL (MIT).

from __future__ import annotations

from math import log2

import mne
import numpy as np
import torch
import torch.nn.functional as F
from scipy.sparse.csgraph import laplacian
from torch import nn
from torch.nn.modules.lazy import LazyModuleMixin

from braindecode.models.base import EEGModuleMixin
from braindecode.models.dgcnn import _GraphConvolution


class GCNsNet(EEGModuleMixin, nn.Module, license="mit"):
    r"""GCNs-Net from Hou et al. (2022) [hou2022gcns]_.

    :bdg-light:`Graph Neural Network` :bdg-dark-line:`Channel`

    .. figure:: ../_static/model/gcnsnet_arch.png
        :align: center
        :alt: GCNs-Net Architecture
        :width: 600px

    Every time sample of a ``(n_chans, n_times)`` window is classified as a
    graph signal over the electrodes, and the logits are averaged over time.
    The graph is the absolute Pearson correlation between channels,
    :math:`A = |P| - I`, coarsened with Graclus so that each pooling layer
    merges pairs of nodes. Each layer is a Chebyshev graph convolution on the
    rescaled normalized Laplacian :math:`L = I - D^{-1/2} A D^{-1/2}`,
    followed by a per-node bias, batch normalization, the activation and
    graph max pooling; a linear layer gives the logits.

    `License <https://github.com/SuperBruceJia/EEG-DL/blob/master/LICENSE>`_

    Parameters
    ----------
    adjacency : Tensor, optional
        Adjacency matrix of shape (n_chans, n_chans). If not provided, it is
        computed from the array passed to ``EEGClassifier.fit`` /
        ``EEGRegressor.fit``, otherwise from the first input passed to
        ``forward``.
    n_features : tuple of int, default=(16, 32, 64, 128, 256, 512)
        Number of filters of each graph convolution.
    cheb_orders : tuple of int, default=(2, 2, 2, 2, 2, 2)
        Order :math:`K` of the Chebyshev polynomial of each layer.
    pool_sizes : tuple of int, default=(2, 2, 2, 2, 2, 2)
        Graph pooling size of each layer, 1 (no pooling) or a power of 2.
    activation : nn.Module, default=nn.Softplus
        Activation function class.
    drop_prob : float, default=0.0
        Dropout probability before the final layer.

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
        adjacency: torch.Tensor | None = None,
        n_features: tuple[int, ...] = (16, 32, 64, 128, 256, 512),
        cheb_orders: tuple[int, ...] = (2, 2, 2, 2, 2, 2),
        pool_sizes: tuple[int, ...] = (2, 2, 2, 2, 2, 2),
        activation: type[nn.Module] = nn.Softplus,
        drop_prob: float = 0.0,
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

        if not (len(n_features) == len(cheb_orders) == len(pool_sizes) != 0):
            raise ValueError(
                "n_features, cheb_orders and pool_sizes must have the same, "
                "non-zero length (one value per layer)."
            )
        if any(p < 1 or p & (p - 1) for p in pool_sizes):
            raise ValueError(f"pool_sizes must be 1 or powers of 2, got {pool_sizes}.")

        self.n_features = n_features
        self.cheb_orders = cheb_orders
        self.pool_sizes = pool_sizes
        self.graph_initialized = False
        self.register_buffer("permutation", None)
        self.laplacians = nn.ModuleList([_LaplacianModule() for _ in n_features])
        if adjacency is not None:
            self._initialize_graph(torch.as_tensor(adjacency, dtype=torch.float32))

        in_features = [1, *n_features[:-1]]  # one scalar per node at the input
        self.graph_convs = nn.ModuleList(
            _GraphConvolution(f_in, f_out, k)
            for f_in, f_out, k in zip(in_features, n_features, cheb_orders)
        )
        self.biases = nn.ModuleList([_NodeBias() for _ in n_features])
        self.batch_norms = nn.ModuleList([nn.BatchNorm1d(f) for f in n_features])
        self.activation = activation()
        self.dropout = nn.Dropout(drop_prob)
        self.final_layer = nn.LazyLinear(self.n_outputs)

        self.register_load_state_dict_pre_hook(self._prepare_graph_state)

    @classmethod
    def _infer_model_kwargs(cls, X, y=None):
        """Adjacency of the whole training array (hook of ``EEGClassifier``)."""
        if isinstance(X, mne.BaseEpochs):
            X = X.get_data(units="uV")
        if not isinstance(X, (np.ndarray, torch.Tensor)):
            # Datasets: the first training batch gives the graph in forward.
            return {}
        return {"adjacency": cls._adjacency(torch.as_tensor(X, dtype=torch.float32))}

    @staticmethod
    def _adjacency(x):
        """Absolute Pearson correlation between channels, zero diagonal."""
        adjacency = torch.corrcoef(x.transpose(0, 1).flatten(1)).abs()
        adjacency.fill_diagonal_(0)
        return adjacency

    @staticmethod
    def _compute_pooling_levels(pool_sizes):
        """Graph coarsening level each convolution runs on."""
        levels = [0]
        for pool_size in pool_sizes[:-1]:
            levels.append(levels[-1] + int(log2(pool_size)))
        return levels

    def _initialize_graph(self, adjacency):
        levels = self._compute_pooling_levels(self.pool_sizes)
        n_levels = levels[-1]
        adjacencies, permutation = _coarsen(adjacency, n_levels)
        self.permutation = (
            torch.tensor(permutation, dtype=torch.long, device=adjacency.device)
            if n_levels > 0
            else None
        )
        for laplacian_module, level in zip(self.laplacians, levels):
            normalized = laplacian(
                adjacencies[level].detach().cpu().numpy(), normed=True
            )
            normalized = torch.as_tensor(
                normalized, dtype=adjacency.dtype, device=adjacency.device
            )
            # Rescale the spectrum from [0, 2] to [-1, 1] (lmax = 2).
            identity = torch.eye(
                len(normalized), dtype=normalized.dtype, device=normalized.device
            )
            laplacian_module.matrix = normalized - identity
        self.graph_initialized = True

    def _prepare_graph_state(self, module, state_dict, prefix, *_):
        """Allocate the graph buffers a checkpoint holds before loading it."""
        reference = next(self.parameters())
        keys = [f"{prefix}laplacians.{i}.matrix" for i in range(len(self.laplacians))]
        has_pooling = self._compute_pooling_levels(self.pool_sizes)[-1] > 0
        if has_pooling:
            keys.append(f"{prefix}permutation")
        present = [key in state_dict for key in keys]
        if any(present) and not all(present):
            raise RuntimeError("Incomplete GCNsNet graph state in checkpoint.")
        if not all(present):
            return
        for laplacian_module, key in zip(self.laplacians, keys):
            laplacian_module.matrix = torch.empty_like(
                state_dict[key], device=reference.device, dtype=reference.dtype
            )
        if has_pooling:
            self.permutation = torch.empty_like(
                state_dict[f"{prefix}permutation"], device=reference.device
            )
        self.graph_initialized = True

    def forward(self, x) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, n_chans, n_times).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, n_outputs).
        """
        if torch.jit.is_scripting():
            if not self.graph_initialized:
                raise RuntimeError(
                    "GCNsNet graph must be initialized before scripted inference."
                )
        elif not self.graph_initialized:
            self._initialize_graph(self._adjacency(x))

        batch_size, n_chans, n_times = x.shape
        # Each time sample is one graph signal: (batch * n_times, nodes, 1).
        x = x.transpose(1, 2).reshape(batch_size * n_times, n_chans)
        permutation = self.permutation
        if permutation is not None:
            # Fake nodes added by the coarsening are zero-padded.
            x = F.pad(x, (0, permutation.numel() - n_chans))[:, permutation]
        x = x.unsqueeze(-1)

        for graph_conv, bias, batch_norm, laplacian_module, pool_size in zip(
            self.graph_convs,
            self.biases,
            self.batch_norms,
            self.laplacians,
            self.pool_sizes,
        ):
            matrix = laplacian_module.matrix
            assert matrix is not None
            x = bias(graph_conv(x, matrix))
            x = self.activation(batch_norm(x.transpose(1, 2)))
            if pool_size > 1:
                # Pooling merges the node pairs of the coarsening tree.
                x = F.max_pool1d(x, pool_size, ceil_mode=True)
            x = x.transpose(1, 2)

        x = self.final_layer(self.dropout(x.flatten(1)))
        # Average the per-sample logits over time.
        return x.reshape(batch_size, n_times, self.n_outputs).mean(dim=1)


class _NodeBias(LazyModuleMixin, nn.Module):
    """One bias per node per filter (EEG-DL ``b2relu``), sized at first call."""

    def __init__(self):
        super().__init__()
        self.bias = nn.UninitializedParameter()

    def initialize_parameters(self, x):
        if self.has_uninitialized_params():
            self.bias.materialize((1, x.shape[1], x.shape[2]))
            nn.init.constant_(self.bias, 0.1)

    def forward(self, x):
        return x + self.bias


class _LaplacianModule(nn.Module):
    """Laplacian buffer of one layer (sizes differ across layers)."""

    def __init__(self):
        super().__init__()
        self.register_buffer("matrix", None)


def _coarsen(adjacency, levels):
    """Graclus multilevel coarsening, as ``coarsening.coarsen`` in EEG-DL.

    Returns the adjacency of every level, with the nodes ordered so that
    pooling pairs ``(2i, 2i + 1)`` are siblings (fake nodes padded), and the
    permutation of the input nodes.
    """
    adjacencies, clusters = [adjacency], []
    node_order = torch.randperm(len(adjacency), device=adjacency.device)
    degree = adjacency.sum(dim=0) - adjacency.diagonal()
    for _ in range(levels):
        cluster = _cluster_one_level(adjacency, node_order, degree)
        membership = F.one_hot(cluster).to(adjacency.dtype)
        adjacency = membership.T @ adjacency @ membership
        clusters.append(cluster)
        adjacencies.append(adjacency)
        degree = adjacency.sum(dim=0)
        node_order = torch.argsort(degree)

    permutations = _compute_perm(clusters)
    for i, adjacency in enumerate(adjacencies):
        adjacency.fill_diagonal_(0)
        if i < levels:
            n_pad = len(permutations[i]) - len(adjacency)
            indices = torch.tensor(permutations[i], device=adjacency.device)
            adjacencies[i] = F.pad(adjacency, (0, n_pad, 0, n_pad))[indices][:, indices]
    return adjacencies, permutations[0] if levels > 0 else None


def _cluster_one_level(adjacency, node_order, weights):
    """Greedy Graclus pairing: each unmarked node joins its best neighbor."""
    marked = torch.zeros(len(adjacency), dtype=torch.bool, device=adjacency.device)
    clusters = torch.empty(len(adjacency), dtype=torch.long, device=adjacency.device)
    cluster_id = 0
    for node in node_order.tolist():
        if marked[node]:
            continue
        marked[node] = True
        clusters[node] = cluster_id
        neighbors = torch.nonzero(adjacency[node], as_tuple=True)[0]
        neighbors = neighbors[~marked[neighbors]]
        if neighbors.numel() > 0:
            scores = adjacency[node, neighbors] * (
                1.0 / weights[node] + 1.0 / weights[neighbors]
            )
            best = torch.argmax(scores)
            if scores[best] > 0:
                clusters[neighbors[best]] = cluster_id
                marked[neighbors[best]] = True
        cluster_id += 1
    return clusters


def _compute_perm(parents):
    """Node order of every level so that sibling pairs form a binary tree.

    Singletons get a fake sibling, appended after the real nodes.
    """
    indices = [list(range(int(parents[-1].max()) + 1))] if parents else []
    for parent in parents[::-1]:
        n_nodes = len(parent)
        layer = []
        for i in indices[-1]:
            children = torch.where(parent == i)[0].tolist()
            n_fake = 2 - len(children)
            layer += children + list(range(n_nodes, n_nodes + n_fake))
            n_nodes += n_fake
        indices.append(layer)
    return indices[::-1]
