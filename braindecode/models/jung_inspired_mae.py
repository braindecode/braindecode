# Authors: Bruno Aristimunha <b.aristimunha@gmail.com>
#
# License: BSD-3-Clause
# Adapted from https://github.com/GeoNIRS/GeoNIRS model/jung_inspired_mae.py (81f8995;
# no licence file; paper arXiv:2610.17845), the GeoNIRS authors' paper-guided
# implementation of Jung et al. (2025).

from __future__ import annotations

import math
from typing import Literal

import torch
from torch import nn

from braindecode.functional import (
    pad_or_crop_to_patches,
    patch_count,
    sinusoidal_positional_encoding,
)
from braindecode.functional.patches import _check_on_non_divisible
from braindecode.models.base import EEGModuleMixin
from braindecode.models.util import fnirs_channel_order, fnirs_split_hbo_hbr


class JungInspiredMAE(EEGModuleMixin, nn.Module):
    r"""The GeoNIRS authors' Jung-inspired masked autoencoder (Li et al., 2026).

    :bdg-info:`Attention/Transformer` :bdg-danger:`Foundation Model`

    .. versionadded:: 2.1

    The GeoNIRS authors :footcite:`li2026geonirs` implemented, from the paper,
    the fNIRS masked autoencoder of Jung et al. :footcite:`jung2025mae` as a
    baseline (their ``jung_inspired_vanilla_mae``) and trained its weights,
    ``jung_mae.pt``; this is a port of that implementation, not of code by
    Jung et al. Each HbO/HbR pair's samples are cut into patches, embedded
    with sinusoidal positions and encoded by a pre-norm transformer, one pair
    at a time; a linear head reads all the tokens. NaN marks a missing
    sample: patches with too few finite samples are left out of attention
    and zeroed.

    :meth:`load_state_dict` takes the GeoNIRS ``jung_mae.pt`` (decoder
    dropped). The file holds no data configuration; the GeoNIRS benchmark
    feeds it the same rows as :class:`GeoNIRS`: 5 Hz HbO/HbR in 150-sample
    (30 s) windows, each recording's HbO and HbR z-scored separately.

    Parameters
    ----------
    patch_samples : int, default=10
        Samples per patch (2 s at 5 Hz).
    on_non_divisible : {"error", "pad", "crop"}, default="error"
        A ``n_times`` that is not a multiple of ``patch_samples`` is refused,
        as by the reference, zero-padded or cropped.
    encoder_dim : int, default=192
        Token width.
    encoder_layers : int, default=4
        Transformer layers.
    encoder_heads : int, default=8
        Attention heads per layer.
    mlp_ratio : float, default=4.0
        Feed-forward width relative to ``encoder_dim``.
    channel_order : {"stacked", "interleaved"} or None, default=None
        ``[HbO_1..HbO_C, HbR_1..HbR_C]`` or MNE's ``[HbO_1, HbR_1, ...]``;
        ``None`` reads it from ``chs_info``, and is ``"stacked"`` without it.
        ``n_chans`` must be even.
    drop_prob : float, default=0.1
        Dropout in the encoder layers.
    head_drop_prob : float, default=0.1
        Dropout before ``final_layer``.
    minimum_patch_valid_fraction : float, default=0.8
        Fraction of finite samples a patch needs to be valid.
    activation : type[nn.Module], default=nn.GELU
        Activation of the encoder feed-forward sub-layers.

    References
    ----------
    .. footbibliography::
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
        patch_samples: int = 10,
        on_non_divisible: Literal["error", "pad", "crop"] = "error",
        encoder_dim: int = 192,
        encoder_layers: int = 4,
        encoder_heads: int = 8,
        mlp_ratio: float = 4.0,
        channel_order: Literal["stacked", "interleaved"] | None = None,
        drop_prob: float = 0.1,
        head_drop_prob: float = 0.1,
        minimum_patch_valid_fraction: float = 0.8,
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
        self.channel_order = fnirs_channel_order(self._chs_info, channel_order)
        if self.n_chans % 2:
            raise ValueError(
                "JungInspiredMAE needs HbO/HbR pairs: n_chans must be even."
            )
        _check_on_non_divisible(on_non_divisible)
        self.on_non_divisible = on_non_divisible
        self.patch_samples = patch_samples
        self.encoder_dim = encoder_dim
        self.minimum_patch_valid_fraction = minimum_patch_valid_fraction
        n_patches = patch_count(self.n_times, patch_samples, on_non_divisible)
        self.patch_embed = nn.Linear(2 * patch_samples, encoder_dim)
        layer = nn.TransformerEncoderLayer(
            encoder_dim,
            encoder_heads,
            int(encoder_dim * mlp_ratio),
            drop_prob,
            activation(),
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, encoder_layers, nn.LayerNorm(encoder_dim), enable_nested_tensor=False
        )
        self.head_dropout = nn.Dropout(head_drop_prob)
        self.final_layer = nn.Linear(
            self.n_chans // 2 * n_patches * encoder_dim, self.n_outputs
        )
        self._init_parameters()

    def _init_parameters(self) -> None:
        # Xavier-uniform weights, zero biases, as the GeoNIRS reference.
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.MultiheadAttention):
                nn.init.xavier_uniform_(module.in_proj_weight)
                nn.init.zeros_(module.in_proj_bias)

    def encode(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        """``(batch, n_chans, n_times)`` -> ``"foundation"`` ``(batch,
        n_chans // 2, n_patches, encoder_dim)`` (zero at invalid patches) and
        ``"valid"`` ``(batch, n_chans // 2, n_patches)``. ``mask`` (bool, same
        shape as ``x``): True samples count as missing."""
        if mask is not None:
            x = x.masked_fill(mask, float("nan"))
        x = pad_or_crop_to_patches(x, self.patch_samples, self.on_non_divisible)
        # (batch, n_chans, T) -> 2 x (batch, n_pairs, T)
        hbo, hbr = fnirs_split_hbo_hbr(x, self.channel_order)
        batch_size, n_pairs, n_times = hbo.shape
        q = self.patch_samples
        n_patches = n_times // q
        valid = torch.isfinite(hbo) & torch.isfinite(hbr)
        # (batch, n_pairs, T, 2) -> (batch * n_pairs, n_patches, 2 * q)
        patches = torch.where(valid.unsqueeze(-1), torch.stack((hbo, hbr), dim=-1), 0.0)
        patches = patches.reshape(batch_size * n_pairs, n_patches, 2 * q)
        # (batch, n_pairs, T) -> (batch, n_pairs, n_patches)
        counts = valid.reshape(batch_size, n_pairs, n_patches, q).sum(dim=-1)
        patch_valid = counts >= math.ceil(q * self.minimum_patch_valid_fraction)
        flat_valid = patch_valid.reshape(batch_size * n_pairs, n_patches)
        # (batch * n_pairs, n_patches, 2 * q) -> (batch * n_pairs, n_patches, encoder_dim)
        tokens = self.patch_embed(patches)
        tokens = tokens + sinusoidal_positional_encoding(
            n_patches, self.encoder_dim, device=x.device, dtype=tokens.dtype
        )
        # A pair without valid patch attends to all (an all-masked softmax
        # is NaN); its output is zeroed.
        safe = flat_valid | ~flat_valid.any(dim=-1, keepdim=True)
        encoded = self.encoder(tokens, src_key_padding_mask=~safe)
        encoded = encoded * flat_valid.unsqueeze(-1).to(encoded.dtype)
        foundation = encoded.reshape(batch_size, n_pairs, n_patches, self.encoder_dim)
        return {"foundation": foundation, "valid": patch_valid}

    def reset_head(self, n_outputs: int) -> None:
        self._set_n_outputs(n_outputs)
        self.final_layer = nn.Linear(self.final_layer.in_features, n_outputs)
        nn.init.xavier_uniform_(self.final_layer.weight)  # as the constructor
        nn.init.zeros_(self.final_layer.bias)

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        """Also takes the released payload (weights under ``"model"``): the
        decoder is dropped, and without ``final_layer`` the head is kept."""
        state_dict = state_dict.get("model", state_dict)
        drop = ("decoder_embed.", "mask_token", "decoder.", "prediction.")
        sd = {k: v for k, v in state_dict.items() if not k.startswith(drop)}
        if not any(k.startswith("final_layer.") for k in sd):
            own = self.state_dict()
            sd.update({k: v for k, v in own.items() if k.startswith("final_layer.")})
        return super().load_state_dict(sd, strict=strict, assign=assign)

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
        ch_mask: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor | dict[str, torch.Tensor | None]:
        """``x``: ``(batch, n_chans, n_times)``, NaN = missing; ``mask`` as in
        :meth:`encode`. ``ch_mask`` is refused: ``final_layer`` spans every
        channel (mark a missing channel with NaN)."""
        if ch_mask is not None:
            raise ValueError(
                "JungInspiredMAE does not accept ch_mask; use NaN channels."
            )
        # (batch, n_pairs, n_patches, encoder_dim) -> (batch, feature_dim)
        foundation = self.encode(x, mask)["foundation"]
        features = self.head_dropout(foundation.reshape(foundation.shape[0], -1))
        if return_features:
            return {"features": features, "cls_token": None}
        return self.final_layer(features)
