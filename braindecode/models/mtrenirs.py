# Authors: Bruno Aristimunha <b.aristimunha@gmail.com>
#
# License: MIT
#
# Adapted from https://github.com/microsoft/MtReNIRS (MIT).

from __future__ import annotations

import math
from collections import OrderedDict
from typing import Literal

import numpy as np
import torch
import torch.nn as nn

from braindecode.functional import (
    apply_rotary,
    check_ch_mask,
    pad_or_crop_to_patches,
    patch_count,
    rotary_angles,
    rotary_frequencies,
    scaled_dot_product_attention,
)
from braindecode.models.base import EEGModuleMixin
from braindecode.models.util import (
    extract_channel_locations_from_chs_info,
    fnirs_chromophores_from_chs_info,
)
from braindecode.modules import FeedForwardBlock, TimmAttention


class MtReNIRS(EEGModuleMixin, nn.Module, license="mit"):
    r"""MtReNIRS from Chae et al. (2026) [microsoft2026mtrenirs]_.

    :bdg-info:`Attention/Transformer` :bdg-dark-line:`Channel` :bdg-danger:`Foundation Model`

    .. versionadded:: 2.1

    Each channel is cut into windows of ``window_samples`` samples; a window
    is one token, the sum of a projection of its samples and a Fourier-feature
    MLP of the channel's position and chromophore. A pre-norm Transformer with
    per-head QK LayerNorm and a rotary embedding of the window index encodes
    all tokens jointly. Positions (``loc[:3]``, 0 when unknown) and HbO/HbR
    (from ``coil_type``, else the channel name) come from ``chs_info`` or the
    ``coordinates`` argument; a montage without any location needs
    ``coordinates``. Windows with a NaN sample are masked.
    :meth:`load_state_dict` takes the official ``mtrenirs_10s.pt`` checkpoint.

    Parameters
    ----------
    window_samples : int
        Samples per token window.
    d_model, depth, n_heads, mlp_hidden : int
        Width, blocks, heads (even ``d_model // n_heads``) and feed-forward
        width of the Transformer.
    fourier_num_bands : int
        Frequencies ``pi * 2**k``, ``k < fourier_num_bands``, per coordinate.
    spatial_mlp_hidden : int
        Hidden width of the spatial MLP.
    rope_base : float
        Base of the rotary embedding.
    qk_norm : bool
        Per-head LayerNorm of queries and keys.
    pool : {"temporal_mean", "channel_temporal_mean", "flatten"}
        Readout of the ``(batch, n_windows, n_chans, d_model)`` tokens: mean
        over valid windows, over valid windows and channels, or all of them.
    pos_center, pos_scale : tuple of float, float
        Centre and scale (m) standardizing the positions, from the checkpoint.
    drop_prob : float
        Dropout of the attention weights and blocks.
    activation : type[nn.Module]
        Activation of the spatial MLP and the feed-forward blocks.

    References
    ----------
    .. [microsoft2026mtrenirs] Chae, T., Avramidis, K., Emmanouilidou, D., &
       Tashev, I. J. (2026). MtReNIRS: A Self-Supervised Foundation Model for
       Transferable Neurovascular Representations. arXiv preprint
       arXiv:2610.36159.

    `License <https://github.com/microsoft/MtReNIRS/blob/main/LICENSE.txt>`_
    """

    def __init__(
        self,
        n_outputs: int | None = None,
        n_chans: int | None = None,
        chs_info=None,
        n_times: int | None = None,
        input_window_seconds: float | None = None,
        sfreq: float | None = None,
        *,
        window_samples: int = 100,
        d_model: int = 128,
        depth: int = 6,
        n_heads: int = 4,
        mlp_hidden: int = 512,
        fourier_num_bands: int = 6,
        spatial_mlp_hidden: int = 128,
        rope_base: float = 500.0,
        qk_norm: bool = True,
        pool: Literal[
            "temporal_mean", "channel_temporal_mean", "flatten"
        ] = "temporal_mean",
        pos_center: tuple[float, float, float] = (
            0.00021245056996122003,
            -0.022675372660160065,
            0.020025914534926414,
        ),
        pos_scale: float = 0.09265176206827164,
        drop_prob: float = 0.1,
        activation: type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__(
            n_outputs=n_outputs,
            n_chans=n_chans,
            chs_info=chs_info,
            n_times=n_times,
            input_window_seconds=input_window_seconds,
            sfreq=sfreq,
        )
        del n_outputs, n_chans, chs_info, n_times, input_window_seconds, sfreq
        if d_model % n_heads or (d_model // n_heads) % 2:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by n_heads ({n_heads}) "
                "with an even head_dim for rotary position embeddings."
            )
        self.window_samples = window_samples
        self.pool = pool
        self.n_windows = patch_count(self.n_times, window_samples)
        self.embed = _TokenEmbedding(
            window_samples, d_model, fourier_num_bands, spatial_mlp_hidden, activation
        )
        block = (n_heads, mlp_hidden, float(drop_prob), rope_base, qk_norm, activation)
        self.encoder = _Encoder(d_model, depth, *block)
        self.register_buffer(
            "default_coordinates",
            torch.from_numpy(_coordinates(self._chs_info, pos_center, pos_scale)),
            persistent=False,
        )
        self.feature_dim = {
            "temporal_mean": self.n_chans * d_model,
            "channel_temporal_mean": d_model,
            "flatten": self.n_windows * self.n_chans * d_model,
        }[pool]
        self.final_layer = nn.Linear(self.feature_dim, self.n_outputs)

    def encode_windows(
        self,
        x: torch.Tensor,
        coordinates: torch.Tensor | None = None,
        ch_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``(batch, n_chans, n_times)`` -> tokens ``(batch, n_windows, n_chans,
        d_model)`` and their validity ``(batch, n_windows, n_chans)``.
        ``coordinates`` is ``(n_chans, 4)`` or ``(batch, n_chans, 4)``."""
        batch_size, n_chans, _ = x.shape
        x = pad_or_crop_to_patches(x, self.window_samples)  # zero-pads the last window
        n_win = x.shape[-1] // self.window_samples
        # (B, C, T) -> (B, W, C, window_samples)
        windows = x.unflatten(-1, [n_win, self.window_samples]).transpose(1, 2)
        finite = torch.isfinite(windows)
        valid = finite.all(dim=-1)  # (B, W, C): a NaN masks its window
        if ch_mask is not None:
            valid = ch_mask[:, None, :n_chans].to(torch.bool) & valid
        valid_tokens = valid.reshape(batch_size, n_win * n_chans)
        tokens = torch.where(finite, windows, torch.zeros_like(windows)).reshape(
            batch_size, n_win * n_chans, self.window_samples
        )
        coord = (
            self.default_coordinates[:n_chans] if coordinates is None else coordinates
        )
        if coord.numel() == 0:
            raise ValueError(
                "MtReNIRS needs channel positions and chromophores: chs_info "
                "with channel locations (loc[:3]) or the coordinates argument."
            )
        coord = coord.to(device=x.device, dtype=tokens.dtype)
        if coord.ndim == 2:
            coord = coord.unsqueeze(0).expand(batch_size, -1, -1)
        # (B, C, 4) -> (B, W*C, 4); the rotary position of a token is its window
        coord_tokens = coord.unsqueeze(1).expand(-1, n_win, -1, -1).flatten(1, 2)
        window_indices = (
            torch.arange(n_win, device=x.device)
            .repeat_interleave(n_chans)
            .expand(batch_size, -1)
        )
        embedded = self.embed(tokens, coord_tokens)  # (B, W*C, d_model)
        encoded = self.encoder(embedded, window_indices, valid_tokens)
        encoded = encoded * valid_tokens.unsqueeze(-1).to(encoded.dtype)
        return encoded.reshape(batch_size, n_win, n_chans, -1), valid

    def reset_head(self, n_outputs: int) -> None:
        self._set_n_outputs(n_outputs)
        self.final_layer = nn.Linear(self.feature_dim, n_outputs)

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        """Also takes the official checkpoint (``{"model": ...}``); a missing
        head keeps its init."""
        state_dict = dict(state_dict.get("model", state_dict))
        if not any(k.startswith("final_layer.") for k in state_dict):
            state_dict.update(self.final_layer.state_dict(prefix="final_layer."))
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
        coordinates: torch.Tensor | None = None,
        ch_mask: torch.Tensor | None = None,
    ) -> torch.Tensor | dict[str, torch.Tensor | None]:
        """``(batch, n_chans, n_times)`` -> ``(batch, n_outputs)``.

        ``coordinates`` as in :meth:`encode_windows`. ``ch_mask``
        ``(batch, n_chans)`` (:func:`~braindecode.functional.check_ch_mask`)
        needs ``pool="channel_temporal_mean"``; padded channels are left out of
        the attention and the pooling.
        """
        ch_mask = check_ch_mask(ch_mask, x)
        if ch_mask is not None and self.pool != "channel_temporal_mean":
            raise ValueError(
                "ch_mask needs pool='channel_temporal_mean': the other heads "
                "are tied to the channels."
            )
        # (B, C, T) -> (B, W, C, D)
        encoded, valid = self.encode_windows(x, coordinates, ch_mask)
        if self.pool == "flatten":
            features = encoded.flatten(1)
        else:  # mean over the valid windows (and channels)
            dims = [1] if self.pool == "temporal_mean" else [1, 2]
            valid_f = valid.unsqueeze(-1).to(encoded.dtype)
            denom = valid_f.sum(dim=dims).clamp_min(1.0)
            features = ((encoded * valid_f).sum(dim=dims) / denom).flatten(1)
        if return_features:
            return {"features": features, "cls_token": None}
        return self.final_layer(features)


class _TokenEmbedding(nn.Module):
    """``signal(window) + spatial(Fourier features of (x, y, z, chromophore))``."""

    def __init__(self, window_samples, d_model, num_bands, hidden, activation):
        super().__init__()
        # ``signal.proj`` and ``spatial.mlp`` are the checkpoint's key paths
        self.signal = nn.Sequential(
            OrderedDict(proj=nn.Linear(window_samples, d_model))
        )
        freqs = (2.0 ** torch.arange(num_bands).float()) * torch.pi
        self.register_buffer("freqs", freqs, persistent=False)
        mlp = nn.Sequential(
            nn.Linear(4 * num_bands * 2, hidden),
            activation(),
            nn.Linear(hidden, d_model),
        )
        self.spatial = nn.Sequential(OrderedDict(mlp=mlp))

    def forward(self, signal: torch.Tensor, coordinates: torch.Tensor) -> torch.Tensor:
        freqs = self.freqs.to(device=coordinates.device, dtype=coordinates.dtype)
        # (B, N, 4) -> (B, N, 4 * num_bands)
        proj = (coordinates.unsqueeze(-1) * freqs).flatten(start_dim=-2)
        return self.signal(signal) + self.spatial(
            torch.cat([proj.sin(), proj.cos()], dim=-1)
        )


class _QKNormSelfInteraction(TimmAttention):
    """timm attention with per-head QK LayerNorm, a rotary embedding of the
    window index and a key padding mask."""

    def __init__(self, dim, heads, dropout, rope_base, qk_norm):
        super().__init__(dim, heads, qkv_bias=True, attn_drop=dropout)
        self.rope_base = float(rope_base)
        self.q_norm = nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()

    def forward(
        self, values: torch.Tensor, positions: torch.Tensor, valid: torch.Tensor
    ) -> torch.Tensor:
        # (B, N, D) -> 3 x (B, heads, N, head_dim)
        query, key, value = self._heads(values)
        query, key = self.q_norm(query), self.k_norm(key)
        inv_freq = rotary_frequencies(
            self.head_dim, self.rope_base, positions.device, query.dtype
        )
        angles = rotary_angles(positions.to(query.dtype), inv_freq, interleaved=False)
        cosine = angles.cos().unsqueeze(1)
        sine = angles.sin().unsqueeze(1)
        query = apply_rotary(query, cosine, sine, interleaved=False)
        key = apply_rotary(key, cosine, sine, interleaved=False)
        # a sample with no valid token attends to all of them
        safe = valid | (~valid.any(dim=-1, keepdim=True))
        bias = torch.zeros_like(safe[:, None, None, :], dtype=query.dtype)
        bias = bias.masked_fill(~safe[:, None, None, :], float("-inf"))
        output = scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=bias,
            dropout_p=self.attn_drop.p if self.training else 0.0,
            scale=1.0 / math.sqrt(self.head_dim),
        )
        # (B, heads, N, head_dim) -> (B, N, D)
        return self.proj(output.transpose(1, 2).flatten(2))


class _Block(nn.Module):
    """Pre-norm block: QK-norm rotary self-attention, then feed-forward."""

    def __init__(
        self, d_model, n_heads, mlp_hidden, dropout, rope_base, qk_norm, activation
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = _QKNormSelfInteraction(
            d_model, n_heads, dropout, rope_base, qk_norm
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = FeedForwardBlock(
            d_model, 1, dropout, activation, hidden_features=mlp_hidden
        )
        self.drop = nn.Dropout(dropout)

    def forward(
        self, values: torch.Tensor, positions: torch.Tensor, valid: torch.Tensor
    ) -> torch.Tensor:
        values = values + self.drop(self.attn(self.norm1(values), positions, valid))
        return values + self.mlp(self.norm2(values))


class _Encoder(nn.Module):
    """``depth`` blocks and a final LayerNorm."""

    def __init__(self, d_model, depth, *block):
        super().__init__()
        self.blocks = nn.ModuleList([_Block(d_model, *block) for _ in range(depth)])
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self, values: torch.Tensor, positions: torch.Tensor, valid: torch.Tensor
    ) -> torch.Tensor:
        for block in self.blocks:
            values = block(values, positions, valid)
        return self.norm(values)


def _coordinates(chs_info, pos_center, pos_scale) -> np.ndarray:
    """``(n_chans, 4)``: ``(loc[:3] - pos_center) / pos_scale`` (0 when
    unknown) and 0 for HbO, 1 for HbR, as the official ``chromophore_map``;
    ``(0, 4)`` without ``chs_info`` or without any location."""
    if chs_info is None:
        return np.zeros((0, 4), np.float32)
    chromophores = fnirs_chromophores_from_chs_info(chs_info)
    if "unknown" in chromophores:
        name = chs_info[chromophores.index("unknown")].get("ch_name")
        raise ValueError(f"MtReNIRS needs HbO/HbR channels, got {name!r}.")
    loc = extract_channel_locations_from_chs_info(chs_info, fill_missing=True)
    if not np.isfinite(loc).any():  # forward raises unless coordinates are given
        return np.zeros((0, 4), np.float32)
    xyz = (loc - np.asarray(pos_center, dtype=np.float32)) / float(pos_scale)
    xyz = np.where(np.isfinite(loc).all(axis=1, keepdims=True), xyz, 0.0)
    return np.c_[xyz, np.equal(chromophores, "hbr")].astype(np.float32)
