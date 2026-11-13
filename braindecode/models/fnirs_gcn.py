# Authors: Bruno Aristimunha <b.aristimunha@gmail.com>
#
# License: BSD (3-clause)
"""FNIRSGCN: functional-connectivity graph convolutional network for fNIRS."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from braindecode.models.base import EEGModuleMixin


class FNIRSGCN(EEGModuleMixin, nn.Module):
    r"""FNIRSGCN, a functional-connectivity graph convolutional network for fNIRS.

    :bdg-success:`Convolution` :bdg-light:`Graph Neural Network`

    .. versionadded:: 2.1

    Each channel is a node: a shared temporal CNN plus a learnable embedding give
    its features. The graph blends the softmax of the window's Pearson
    correlation (flat channels masked) with a learnable static graph. Residual
    graph convolutions [kipf2017gcn]_ and mean, max and learned node pooling
    feed ``final_layer``; no optode coordinates are needed.

    Parameters
    ----------
    node_dim : int, default=64
        Feature dimension per node.
    n_gcn_layers : int, default=2
        Number of residual graph convolution layers.
    temperature : float, default=0.25
        Softmax temperature of the correlation graph.
    drop_prob : float, default=0.3
        Dropout of the graph layers and before ``final_layer``.
    activation : type[nn.Module], default=nn.GELU
        Activation of the stem and the graph layers.

    References
    ----------
    .. [kipf2017gcn] Kipf, T. N., & Welling, M. (2017). Semi-Supervised
       Classification with Graph Convolutional Networks. ICLR.
    """

    def __init__(
        self,
        n_outputs: int | None = None,
        n_chans: int | None = None,
        chs_info: list[dict] | None = None,
        n_times: int | None = None,
        input_window_seconds: float | None = None,
        sfreq: float | None = None,
        *,
        node_dim: int = 64,
        n_gcn_layers: int = 2,
        temperature: float = 0.25,
        drop_prob: float = 0.3,
        activation: type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__(
            n_outputs, n_chans, chs_info, n_times, input_window_seconds, sfreq
        )
        self.temperature = temperature
        mid = max(8, node_dim // 2)
        self.channel_temporal_stem = nn.Sequential(
            nn.Conv1d(1, mid, 7, padding=3),
            activation(),
            nn.MaxPool1d(2, ceil_mode=True),
            nn.Conv1d(mid, node_dim, 5, padding=2),
            activation(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(1),
        )
        self.node_embedding = nn.Parameter(torch.empty(1, self.n_chans, node_dim))
        self.node_pool = nn.Linear(self.n_chans, 2)
        self.static_adj_logits = nn.Parameter(torch.empty(self.n_chans, self.n_chans))
        self.adj_mix_logit = nn.Parameter(torch.empty(()))
        self.graph_steps = nn.ModuleList(
            nn.ModuleDict(
                {
                    "norm": nn.LayerNorm(node_dim),
                    "self_linear": nn.Linear(node_dim, node_dim),
                    "neigh_linear": nn.Linear(node_dim, node_dim, bias=False),
                    "act": activation(),
                }
            )
            for _ in range(n_gcn_layers)
        )
        self.readout_norm = nn.LayerNorm(2 * node_dim)
        self.dropout = nn.Dropout(drop_prob)
        self.final_layer = nn.Linear(2 * node_dim, self.n_outputs)
        self._init_parameters()

    def _init_parameters(self) -> None:
        nn.init.normal_(self.node_embedding, std=0.02)
        nn.init.zeros_(self.static_adj_logits)
        nn.init.zeros_(self.adj_mix_logit)

    def reset_head(self, n_outputs: int) -> None:
        self._set_n_outputs(n_outputs)
        self.final_layer = nn.Linear(self.final_layer.in_features, n_outputs)

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
        ch_mask: torch.Tensor | None = None,
    ) -> torch.Tensor | dict[str, torch.Tensor | None]:
        if ch_mask is not None:
            raise ValueError("FNIRSGCN does not accept ch_mask: nodes span n_chans.")
        x = torch.nan_to_num(x, nan=0.0)  # (batch, n_chans, n_times)
        b, c, t = x.shape
        h = self.channel_temporal_stem(x.reshape(b * c, 1, t)).reshape(b, c, -1)
        h = h + self.node_embedding  # (batch, n_chans, node_dim)
        centered = x - x.mean(dim=-1, keepdim=True)
        norm_sq = (centered * centered).sum(dim=-1, keepdim=True)  # (batch, c, 1)
        active = norm_sq > 1e-20  # flat channels get no correlation edges
        unit = torch.where(active, centered / norm_sq.clamp_min(1e-24).sqrt(), 0.0)
        corr = torch.bmm(unit, unit.transpose(1, 2)) / self.temperature  # (b, c, c)
        corr = corr.masked_fill(active & ~active.transpose(1, 2), float("-inf"))
        dyn = F.softmax(corr, dim=-1)
        static = F.softmax(self.static_adj_logits, dim=-1).unsqueeze(0)
        alpha = torch.sigmoid(self.adj_mix_logit)
        adj = alpha * dyn + (1.0 - alpha) * static  # (batch, n_chans, n_chans)
        for step in self.graph_steps:
            normed = step["norm"](h)
            agg = torch.bmm(adj, step["neigh_linear"](normed))
            h = h + self.dropout(step["act"](agg + step["self_linear"](normed)))
        pooled = self.node_pool(h.transpose(1, 2)).transpose(1, 2).reshape(b, -1)
        pooled = torch.cat((h.mean(dim=1), h.amax(dim=1)), dim=-1) + pooled
        features = self.dropout(self.readout_norm(pooled))  # (batch, 2 * node_dim)
        if return_features:
            return {"features": features, "cls_token": None}
        return self.final_layer(features)
