from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from braindecode.models.base import EEGModuleMixin
from braindecode.models.dgcnn import _ChebyshevGraphConvolution


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

    def forward(self):
        pass