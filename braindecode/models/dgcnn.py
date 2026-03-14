# Authors: Yue Wang <yuewangx@mit.edu>
#
# Code adapted from https://github.com/WangYueFt/dgcnn
#
# License: BSD (3-clause)

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F

from braindecode.models.base import EEGModuleMixin


def knn(x, k):
    # x: (B, n_times, n_chans)
    inner = -2 * torch.matmul(
        einops.rearrange(x, "b t c -> b c t"), x
    )  # (B, n_chans, n_chans)
    xx = torch.sum(x**2, dim=1, keepdim=True)  # (B, 1, n_chans)
    pairwise_distance = (
        -xx - inner - einops.rearrange(xx, "b one c -> b c one")
    )  # (B, n_chans, n_chans)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (B, n_chans, k)
    return idx


def get_graph_feature(x, n_neighbors=20, idx=None):
    # x: (B, n_times, n_chans)
    batch_size, n_times, n_chans = x.shape
    if idx is None:
        idx = knn(x, k=n_neighbors)  # (B, n_chans, n_neighbors)
    device = x.device

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * n_chans
    idx = (idx + idx_base).view(-1)

    x = einops.rearrange(x, "b t c -> b c t").contiguous()  # (B, n_chans, n_times)
    feature = einops.rearrange(x, "b c t -> (b c) t")[idx, :]  # gather neighbors
    feature = einops.rearrange(
        feature, "(b c k) t -> b c k t", b=batch_size, c=n_chans, k=n_neighbors
    )
    x = einops.repeat(x, "b c t -> b c k t", k=n_neighbors)

    feature = einops.rearrange(
        torch.cat((feature - x, x), dim=3), "b c k t -> b t c k"
    ).contiguous()

    return feature


class DGCNN(EEGModuleMixin, nn.Module):
    """DGCNN for EEG classification from Song et al. (2018) [dgcnn]_.

    :bdg-success:`Graph Neural Network`

    .. figure:: https://ar5iv.labs.arxiv.org/html/1801.07829/assets/sections/figure/model_architecture.jpg
        :align: center
        :alt: DGCNN Architecture
        :width: 600px

    Dynamic Graph Convolutional Neural Network (DGCNN) treats EEG electrodes
    as nodes in a graph and dynamically learns inter-channel relationships
    using k-nearest neighbors in feature space. Unlike fixed graph methods,
    the graph is recomputed at each layer based on learned features.

    .. rubric:: Architectural Overview

    1. **Graph Construction (KNN)**: For each electrode, find k nearest
       neighbors based on feature similarity (not physical position).
    2. **EdgeConv Blocks**: Four stacked blocks, each computing edge features
       between neighbors, applying a shared Conv2d, and max-pooling over
       neighbors to get per-electrode features.
    3. **Multi-scale Concatenation**: Outputs of all four EdgeConv blocks are
       concatenated to preserve low-level and high-level features.
    4. **Global Pooling**: Max and average pooling collapse all electrode
       features into a single vector.
    5. **Classification MLP**: Three linear layers produce the final output.

    Parameters
    ----------
    n_outputs : int
        Number of outputs (classes).
    n_chans : int
        Number of EEG channels (electrodes = graph nodes).
    chs_info : list of dict
        Information about each channel. See :class:`mne.Info`.
    n_times : int
        Number of time samples per window. Used as node feature dimension.
    input_window_seconds : float
        Length of input window in seconds.
    sfreq : float
        Sampling frequency of the EEG recording.
    block_dims : tuple[int, ...], default=(64, 64, 128, 256)
        Output dimensionality of each EdgeConv block.
    emb_dims : int, default=1024
        Embedding dimensions after multi-scale concatenation.
    mlp_dims : tuple[int, ...], default=(512, 256)
        Hidden layer sizes of the classification MLP.
    n_neighbors : int, default=20
        Number of nearest neighbors for KNN graph construction.
    drop_prob : float, default=0.5
        Dropout probability in the classification head.
    activation : type[nn.Module], default=nn.LeakyReLU
        Activation function class to use throughout the network.

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
        emb_dims=1024,
        block_dims=(64, 64, 128, 256),
        mlp_dims=(512, 256),
        n_neighbors=20,
        drop_prob=0.5,
        activation: type[nn.Module] = nn.LeakyReLU,
    ):
        super().__init__(
            n_outputs=n_outputs,
            n_chans=n_chans,
            chs_info=chs_info,
            n_times=n_times,
            sfreq=sfreq,
            input_window_seconds=input_window_seconds,
        )

        del n_outputs, n_chans, chs_info, n_times, input_window_seconds, sfreq
        self.n_neighbors = n_neighbors
        self.drop_prob = drop_prob
        edge_in_dims = [self.n_times * 2] + [d * 2 for d in block_dims[:-1]]
        self.edge_convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(in_d, out_d, kernel_size=1, bias=False),
                    nn.BatchNorm2d(out_d),
                    activation(),
                )
                for in_d, out_d in zip(edge_in_dims, block_dims)
            ]
        )

        self.global_proj = nn.Sequential(
            nn.Conv1d(sum(block_dims), emb_dims, kernel_size=1, bias=False),
            nn.BatchNorm1d(emb_dims),
            activation(),
        )

        mlp_in_dims = [emb_dims * 2] + list(mlp_dims)
        mlp_layers = []
        for in_d, out_d in zip(mlp_in_dims[:-1], mlp_dims):
            mlp_layers += [
                nn.Linear(in_d, out_d, bias=False),
                nn.BatchNorm1d(out_d),
                nn.Dropout(p=drop_prob),
                activation(),
            ]
        self.final_layer = nn.Linear(mlp_dims[-1], self.n_outputs)
        mlp_layers.append(self.final_layer)
        self.classifier = nn.Sequential(*mlp_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = einops.rearrange(x, "b c t -> b t c")
        block_outputs = []

        for edge_conv in self.edge_convs:
            x = get_graph_feature(x, n_neighbors=self.n_neighbors)
            x = edge_conv(x)
            x = x.max(dim=-1, keepdim=False)[0]
            block_outputs.append(x)

        x = self.global_proj(torch.cat(block_outputs, dim=1))

        x_max = einops.rearrange(F.adaptive_max_pool1d(x, 1), "b c 1 -> b c")
        x_avg = einops.rearrange(F.adaptive_avg_pool1d(x, 1), "b c 1 -> b c")
        x = torch.cat((x_max, x_avg), dim=1)

        return self.classifier(x)
