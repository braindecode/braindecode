# Authors: OpenAI Codex contributors
#
# License: BSD-3
"""NeurIPT: Foundation Model for Neural Interfaces.

This implementation follows the published architecture in Fang et al. (2025).
The official source repository currently contains no model implementation or
released checkpoint, so this module does not claim checkpoint parity or paper
benchmark reproduction.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from braindecode.models.base import EEGModuleMixin


def _sinusoidal_encoding(positions: Tensor, dim: int) -> Tensor:
    """Build a sinusoidal encoding for arbitrary scalar positions."""
    if dim < 1:
        return positions.new_zeros((*positions.shape, 0))
    frequencies = torch.exp(
        torch.arange(0, dim, 2, device=positions.device, dtype=positions.dtype)
        * (-math.log(10000.0) / dim)
    )
    angles = positions.unsqueeze(-1) * frequencies
    encoding = positions.new_zeros((*positions.shape, dim))
    encoding[..., 0::2] = torch.sin(angles)
    if dim > 1:
        encoding[..., 1::2] = torch.cos(angles[..., : encoding[..., 1::2].shape[-1]])
    return encoding


def amplitude_aware_mask(
    x: Tensor,
    mask_ratio: float = 0.5,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Sample per-channel contiguous amplitude-rank masks from an EEG batch.

    For each trial and channel, a percentile center is sampled uniformly and
    an interval of width ``mask_ratio`` is selected around it in the sorted
    amplitudes. Intervals near either boundary are shifted to preserve the
    requested count. The returned boolean tensor has the same shape as ``x``.
    Ties are resolved by a stable sort.
    """
    if x.ndim != 3:
        raise ValueError("x must have shape (batch, channels, time).")
    if not 0.0 < mask_ratio < 1.0:
        raise ValueError("mask_ratio must be strictly between 0 and 1.")
    n_times = x.shape[-1]
    n_mask = min(n_times, max(1, round(n_times * mask_ratio)))
    ranks = torch.argsort(x, dim=-1, stable=True)
    max_start = n_times - n_mask
    centers = (
        torch.rand((*x.shape[:2], 1), device=x.device, generator=generator) * n_times
    ).long()
    starts = (centers - n_mask // 2).clamp(min=0, max=max_start)
    selected_ranks = torch.arange(n_times, device=x.device).view(1, 1, -1)
    selected = (selected_ranks >= starts) & (selected_ranks < starts + n_mask)
    mask = torch.zeros_like(selected)
    mask.scatter_(-1, ranks, selected)
    return mask


class _SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.gate = nn.Linear(dim, hidden_dim)
        self.value = nn.Linear(dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.dropout(self.output(F.silu(self.gate(x)) * self.value(x)))


class _ProgressiveMoE(nn.Module):
    """Top-k routed experts with an always-active shared SwiGLU expert."""

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        n_experts: int,
        top_k_fraction: float,
        dropout: float,
    ):
        super().__init__()
        if n_experts < 0:
            raise ValueError("n_experts must be non-negative.")
        if not 0.0 < top_k_fraction <= 1.0:
            raise ValueError("top_k_fraction must be in (0, 1].")
        self.n_experts = n_experts
        self.top_k = max(1, math.ceil(n_experts * top_k_fraction)) if n_experts else 0
        self.router = nn.Linear(dim, n_experts) if n_experts else None
        self.experts = nn.ModuleList(
            [_SwiGLU(dim, hidden_dim, dropout) for _ in range(n_experts)]
        )
        self.shared = _SwiGLU(dim, dim, dropout)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        shared = self.shared(x)
        if self.n_experts == 0:
            return shared, x.new_zeros(())
        if self.router is None:
            raise RuntimeError("A router is required when routed experts are enabled.")

        flat = x.reshape(-1, x.shape[-1])
        logits = self.router(flat)
        values, indices = logits.topk(self.top_k, dim=-1)
        weights = values.softmax(dim=-1)
        flat_out = torch.zeros_like(flat)
        selected_load = torch.zeros(self.n_experts, device=x.device, dtype=logits.dtype)
        for expert_id, expert in enumerate(self.experts):
            token_ids, topk_slots = torch.where(indices == expert_id)
            # Avoid a data-dependent Python branch so torch.export can capture
            # the variable-size expert batches. Empty batches are valid here.
            expert_values = expert(flat[token_ids])
            expert_weights = weights[token_ids, topk_slots].unsqueeze(-1)
            flat_out.index_add_(0, token_ids, expert_values * expert_weights)
            selected_load[expert_id] = token_ids.numel() / (flat.shape[0] * self.top_k)
        importance = logits.softmax(dim=-1).mean(dim=0)
        auxiliary_loss = self.n_experts * torch.sum(importance * selected_load)
        return shared + flat_out.reshape_as(x), auxiliary_loss


class _TSAStage(nn.Module):
    """Two-stage time/channel attention with progressive MoE feed-forwards."""

    def __init__(
        self,
        dim: int,
        n_heads: int,
        expert_hidden_dim: int,
        n_experts: int,
        top_k_fraction: float,
        dropout: float,
    ):
        super().__init__()
        self.time_norm = nn.LayerNorm(dim)
        self.time_attn = nn.MultiheadAttention(
            dim, n_heads, dropout=dropout, batch_first=True
        )
        self.time_post_norm = nn.LayerNorm(dim)
        self.time_moe = _ProgressiveMoE(
            dim, expert_hidden_dim, n_experts, top_k_fraction, dropout
        )
        self.channel_norm = nn.LayerNorm(dim)
        self.channel_attn = nn.MultiheadAttention(
            dim, n_heads, dropout=dropout, batch_first=True
        )
        self.channel_post_norm = nn.LayerNorm(dim)
        self.channel_moe = _ProgressiveMoE(
            dim, expert_hidden_dim, n_experts, top_k_fraction, dropout
        )

    def forward(
        self,
        x: Tensor,
        time_position: Tensor | None = None,
        spatial_position: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        batch, n_times, n_chans, dim = x.shape
        time_residual = x.permute(0, 2, 1, 3).reshape(batch * n_chans, n_times, dim)
        time_input = x if time_position is None else x + time_position[None, :, None, :]
        time = (
            self.time_norm(time_input)
            .permute(0, 2, 1, 3)
            .reshape(batch * n_chans, n_times, dim)
        )
        time_attn, _ = self.time_attn(time, time, time, need_weights=False)
        # LayerNorm feeds attention; the residual path preserves the signal.
        time = time_residual + time_attn
        time_update, time_aux = self.time_moe(self.time_post_norm(time))
        time = time + time_update
        x = time.reshape(batch, n_chans, n_times, dim).permute(0, 2, 1, 3)

        channel_input = (
            x if spatial_position is None else x + spatial_position[None, None, :, :]
        )
        channel = self.channel_norm(channel_input).reshape(
            batch * n_times, n_chans, dim
        )
        channel_residual = x.reshape(batch * n_times, n_chans, dim)
        channel_attn, _ = self.channel_attn(
            channel, channel, channel, need_weights=False
        )
        # Keep the same pre-normalization residual semantics in the spatial stage.
        channel = channel_residual + channel_attn
        channel_update, channel_aux = self.channel_moe(self.channel_post_norm(channel))
        channel = channel + channel_update
        x = channel.reshape(batch, n_times, n_chans, dim)
        return x, time_aux + channel_aux


class NeurIPT(EEGModuleMixin, nn.Module, license="bsd-3-clause"):
    """NeurIPT EEG foundation-model architecture for downstream classification.

    :bdg-danger:`Foundation Model` :bdg-info:`Attention/Transformer`
    :bdg-warning:`Mixture-of-Experts` :bdg-dark-line:`Channel`

    This paper-based implementation includes 3D electrode positional encoding,
    hierarchical two-stage time/channel attention, progressive Top-k MoE
    feed-forward blocks, and intra/inter-lobe pooling (IILP). It does not ship
    pretrained weights. The official source repository currently contains no
    recoverable implementation, so initialization and data benchmarks have not
    been validated against the authors' released model.

    .. figure:: ../_static/model/neuript_arch.svg
       :align: center
       :alt: NeurIPT downstream path with electrode encoding, hierarchical
             time/channel attention, optional lobe pooling and a classifier.
       :width: 95%

       Architecture implemented here. The masking utility is a separate
       pretraining helper; the masked-reconstruction objective is not included.

    References
    ----------
    .. [1] Fang et al., "NeurIPT: Foundation Model for Neural Interfaces,"
       NeurIPS 2025. https://arxiv.org/abs/2510.16548

    Parameters
    ----------
    n_outputs : int
        Number of downstream classes.
    n_chans : int
        Number of EEG channels.
    chs_info : list of dict | None
        MNE channel information. ``loc[:3]`` supplies electrode coordinates
        when explicit ``channel_positions`` are not passed.
    n_times : int
        Number of time samples in each input window.
    input_window_seconds : float | None
        Duration of the input window. If omitted, it is inferred from
        ``n_times`` and ``sfreq`` by :class:`EEGModuleMixin`.
    sfreq : float
        Sampling frequency in Hz.
    d_model : int, default=96
        Transformer embedding dimension. Must be divisible by ``n_heads``. The
        paper's reported configuration uses 768.
    n_heads : int, default=8
        Number of attention heads in each TSA stage.
    n_layers : int, default=6
        Number of hierarchical TSA layers.
    merge_factors : sequence of int | None
        Temporal merge factor before each layer; defaults to the paper schedule
        ``(1, 4, 1, 2, 1, 2)`` truncated or extended with ones.
    n_experts : sequence of int | None
        Expert count per layer, matching the paper schedule ``(0, 2, 2, 4, 4,
        6)`` by default. A zero means only the shared expert is active.
    expert_hidden_dim : int, default=128
        Hidden width of each routed expert. The paper's reported configuration
        uses 512.
    top_k_fraction : float, default=0.5
        Fraction of routed experts selected per token.
    dropout : float, default=0.1
        Dropout probability in attention and SwiGLU blocks.
    drop_prob : float | None, default=None
        Optional Braindecode-compatible alias for ``dropout``. When provided,
        it overrides ``dropout``.
    channel_positions : Tensor | sequence | None
        Optional ``(n_chans, 3)`` electrode coordinates. If omitted, channel
        coordinates are read from ``chs_info``; when unavailable, spatial
        encodings are zero and the model does not claim montage transfer.
    lobe_groups : sequence of sequences of int | None
        Channel indices for each IILP region. If omitted, one global region is
        used; supply anatomical groups to enable explicit inter-lobe features.
    activation : nn.Module class, default=nn.GELU
        Retained for the standard Braindecode constructor contract. The paper
        uses SwiGLU in its transformer feed-forward blocks.
    """

    def __init__(
        self,
        n_outputs: int,
        n_chans: int | None = None,
        chs_info=None,
        n_times: int | None = None,
        input_window_seconds: float | None = None,
        sfreq: float | None = None,
        d_model: int = 96,
        n_heads: int = 8,
        n_layers: int = 6,
        merge_factors: Sequence[int] | None = None,
        n_experts: Sequence[int] | None = None,
        expert_hidden_dim: int = 128,
        top_k_fraction: float = 0.5,
        dropout: float = 0.1,
        drop_prob: float | None = None,
        channel_positions: Tensor | Sequence[Sequence[float]] | None = None,
        lobe_groups: Sequence[Sequence[int]] | None = None,
        activation: type[nn.Module] = nn.GELU,
    ):
        super().__init__(
            n_outputs=n_outputs,
            n_chans=n_chans,
            chs_info=chs_info,
            n_times=n_times,
            input_window_seconds=input_window_seconds,
            sfreq=sfreq,
        )
        del activation  # The published backbone uses SwiGLU, not GELU.
        if drop_prob is not None:
            dropout = drop_prob
        if not 0.0 <= dropout <= 1.0:
            raise ValueError("dropout must be in [0, 1].")
        if n_heads < 1 or d_model < 1 or d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads.")
        if n_layers < 1:
            raise ValueError("n_layers must be at least 1.")
        self.d_model = d_model
        self.n_layers = n_layers
        paper_merge = (1, 4, 1, 2, 1, 2)
        if merge_factors is None:
            merge_factors = (*paper_merge[:n_layers], *((1,) * max(0, n_layers - 6)))
        if len(merge_factors) != n_layers or any(
            factor < 1 for factor in merge_factors
        ):
            raise ValueError(
                "merge_factors must contain one positive integer per layer."
            )
        if merge_factors[0] != 1:
            raise ValueError("The first TSA layer must not merge temporal tokens.")
        paper_experts = (0, 2, 2, 4, 4, 6)
        if n_experts is None:
            n_experts = (*paper_experts[:n_layers], *((0,) * max(0, n_layers - 6)))
        if len(n_experts) != n_layers or any(count < 0 for count in n_experts):
            raise ValueError(
                "n_experts must contain one non-negative integer per layer."
            )
        self.merge_factors = tuple(merge_factors)
        self.n_experts_per_layer = tuple(n_experts)

        self.input_projection = nn.Linear(1, d_model)
        positions = self._resolve_channel_positions(channel_positions, chs_info)
        self.register_buffer("channel_coordinates", positions, persistent=True)
        self.layers = nn.ModuleList()
        self.mergers = nn.ModuleList()
        for factor, expert_count in zip(self.merge_factors, self.n_experts_per_layer):
            self.mergers.append(
                nn.Linear(factor * d_model, d_model) if factor > 1 else nn.Identity()
            )
            self.layers.append(
                _TSAStage(
                    d_model,
                    n_heads,
                    expert_hidden_dim,
                    expert_count,
                    top_k_fraction,
                    dropout,
                )
            )
        self.lobe_groups = self._resolve_lobe_groups(lobe_groups)
        feature_dim = n_layers * len(self.lobe_groups) * d_model
        self.final_layer = nn.Linear(feature_dim, n_outputs)
        self._init_weights()

    def _resolve_channel_positions(self, channel_positions, chs_info) -> Tensor:
        if channel_positions is None and chs_info is not None:
            channel_positions = [
                channel.get("loc", [0.0, 0.0, 0.0])[:3] for channel in chs_info
            ]
        if channel_positions is None:
            return torch.zeros(self.n_chans, 3, dtype=torch.float32)
        if isinstance(channel_positions, Tensor):
            positions = channel_positions.to(dtype=torch.float32)
        else:
            positions = torch.as_tensor(
                np.asarray(channel_positions), dtype=torch.float32
            )
        if tuple(positions.shape) != (self.n_chans, 3):
            raise ValueError("channel_positions must have shape (n_chans, 3).")
        if not torch.isfinite(positions).all():
            raise ValueError("channel_positions must contain only finite values.")
        return positions

    def _resolve_lobe_groups(self, lobe_groups) -> tuple[tuple[int, ...], ...]:
        if lobe_groups is None:
            return (tuple(range(self.n_chans)),)
        groups = tuple(tuple(int(index) for index in group) for group in lobe_groups)
        if not groups or any(not group for group in groups):
            raise ValueError("lobe_groups must contain at least one non-empty group.")
        flattened = [index for group in groups for index in group]
        if len(flattened) != len(set(flattened)) or any(
            index < 0 or index >= self.n_chans for index in flattened
        ):
            raise ValueError("lobe_groups must use unique, in-range channel indices.")
        return groups

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.input_projection.weight)
        nn.init.zeros_(self.input_projection.bias)
        nn.init.xavier_uniform_(self.final_layer.weight)
        nn.init.zeros_(self.final_layer.bias)

    def _position_encodings(self, x: Tensor) -> tuple[Tensor, Tensor]:
        n_times = x.shape[-1]
        time = torch.arange(n_times, device=x.device, dtype=x.dtype)
        time_encoding = _sinusoidal_encoding(time, self.d_model)
        axis_dim = self.d_model // 3
        dims = (axis_dim, axis_dim, self.d_model - 2 * axis_dim)
        spatial = torch.cat(
            [
                _sinusoidal_encoding(
                    self.channel_coordinates[:, axis].to(x), dims[axis]
                )
                for axis in range(3)
            ],
            dim=-1,
        )
        return time_encoding, spatial

    def _embed(self, x: Tensor) -> Tensor:
        """Project each EEG sample independently, preserving temporal detail."""
        return self.input_projection(x.permute(0, 2, 1).unsqueeze(-1))

    def _merge_time(self, x: Tensor, factor: int, merger: nn.Module) -> Tensor:
        if factor == 1:
            return x
        batch, n_times, n_chans, dim = x.shape
        padding = (-n_times) % factor
        if padding:
            x = F.pad(x, (0, 0, 0, 0, 0, padding))
        n_merged = x.shape[1] // factor
        x = x.reshape(batch, n_merged, factor, n_chans, dim)
        x = x.permute(0, 1, 3, 2, 4).reshape(batch, n_merged, n_chans, factor * dim)
        return merger(x)

    def forward_features(self, x: Tensor) -> tuple[Tensor, Tensor]:
        if x.ndim != 3 or tuple(x.shape[1:]) != (self.n_chans, self.n_times):
            raise ValueError(
                f"Expected input shape (batch, {self.n_chans}, {self.n_times}), "
                f"got {tuple(x.shape)}."
            )
        hidden = self._embed(x)
        time_position, spatial_position = self._position_encodings(x)
        layer_features = []
        auxiliary_losses = []
        for layer_idx, (factor, merger, layer) in enumerate(
            zip(self.merge_factors, self.mergers, self.layers)
        ):
            hidden = self._merge_time(hidden, factor, merger)
            hidden, aux = layer(
                hidden,
                time_position=time_position if layer_idx == 0 else None,
                spatial_position=spatial_position if layer_idx == 0 else None,
            )
            auxiliary_losses.append(aux)
            channel_means = hidden.mean(dim=1)
            regional = [
                channel_means[:, group].mean(dim=1) for group in self.lobe_groups
            ]
            layer_features.append(torch.cat(regional, dim=-1))
        features = torch.cat(layer_features, dim=-1)
        return features, torch.stack(auxiliary_losses).sum()

    def forward(self, x: Tensor, return_features: bool = False):
        """Compute classification logits, optionally returning IILP features."""
        features, auxiliary_loss = self.forward_features(x)
        logits = self.final_layer(features)
        if return_features:
            return {"logits": logits, "features": features, "aux_loss": auxiliary_loss}
        return logits

    def reset_head(self, n_outputs: int) -> None:
        """Replace the task-specific classification layer."""
        self._set_n_outputs(n_outputs)
        old = self.final_layer
        self.final_layer = nn.Linear(old.in_features, n_outputs).to(
            device=old.weight.device, dtype=old.weight.dtype
        )
        self.final_layer.train(old.training)


__all__ = ["NeurIPT", "amplitude_aware_mask"]
