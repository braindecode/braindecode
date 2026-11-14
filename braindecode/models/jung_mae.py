# Authors: E. Jung, H. Lee, J. An (original fNIRS MAE architecture)
#          GeoNIRS Consortium (reference dual-chromophore MAE implementation)
#          Bruno Aristimunha <b.aristimunha@gmail.com>
#
# License: BSD (3-clause)
#
# Adapted from Jung et al. (2025) and https://github.com/GeoNIRS/GeoNIRS.
"""JungMAE: Dual-Chromophore Masked Autoencoder for fNIRS Representation Learning."""

from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn

from braindecode.functional import patch_count, sinusoidal_positional_encoding
from braindecode.models.base import EEGModuleMixin
from braindecode.models.util import fnirs_channel_order


class JungMAE(EEGModuleMixin, nn.Module):
    r"""fNIRS masked autoencoder from Jung et al. (2025) :footcite:`jung2025mae`.

    :bdg-info:`Attention/Transformer` :bdg-danger:`Foundation Model`

    .. versionadded:: 2.1

    Each channel's paired HbO/HbR samples are cut into patches, embedded with
    sinusoidal positions and encoded by a pre-norm transformer, one channel at
    a time; a linear head reads the pooled tokens. NaN marks a missing sample:
    patches with too few finite samples are left out of attention and zeroed.
    :meth:`load_state_dict` takes the GeoNIRS ``jung_mae.pt`` (decoder dropped).

    Parameters
    ----------
    patch_samples : int, default=10
        Samples per patch (2 s at 5 Hz).
    encoder_dim : int, default=192
        Token width.
    encoder_layers : int, default=4
        Transformer layers.
    encoder_heads : int, default=8
        Attention heads per layer.
    mlp_ratio : float, default=4.0
        Feed-forward width relative to ``encoder_dim``.
    pool : {"flatten", "mean"}, default="flatten"
        Keep every (channel, patch) token, or average each channel's valid ones.
    channel_order : {"stacked", "interleaved"} or None, default=None
        ``[HbO_1..HbO_C, HbR_1..HbR_C]`` or MNE's ``[HbO_1, HbR_1, ...]``
        (``None`` reads it from ``chs_info``); an odd ``n_chans`` uses each
        channel as both chromophores.
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
        encoder_dim: int = 192,
        encoder_layers: int = 4,
        encoder_heads: int = 8,
        mlp_ratio: float = 4.0,
        pool: Literal["flatten", "mean"] = "flatten",
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
        self.patch_samples = patch_samples
        self.encoder_dim = encoder_dim
        self.pool = pool
        self.channel_order = fnirs_channel_order(self._chs_info, channel_order)
        self.minimum_patch_valid_fraction = minimum_patch_valid_fraction
        n_phys_chans = self.n_chans if self.n_chans % 2 else self.n_chans // 2
        n_tokens = patch_count(self.n_times, patch_samples) if pool == "flatten" else 1
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
            n_phys_chans * n_tokens * encoder_dim, self.n_outputs
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
        n_phys_chans, n_patches, encoder_dim)`` (zero at invalid patches) and
        ``"valid"`` ``(batch, n_phys_chans, n_patches)``. ``mask`` (bool, same
        shape as ``x``): True samples count as missing."""
        if mask is not None:
            x = x.masked_fill(mask, float("nan"))
        batch_size, n_chans, n_times = x.shape
        q = self.patch_samples
        rem = n_times % q
        if rem:  # NaN padding: the padded samples count as missing
            x = F.pad(x, (0, q - rem), value=float("nan"))
        n_patches = x.shape[-1] // q
        # (batch, n_chans, T) -> 2 x (batch, n_phys, T)
        if n_chans % 2:
            hbo, hbr = x, x
        elif self.channel_order == "interleaved":
            hbo, hbr = x[:, 0::2], x[:, 1::2]
        else:
            hbo, hbr = x[:, : n_chans // 2], x[:, n_chans // 2 :]
        n_phys = hbo.shape[1]
        valid = torch.isfinite(hbo) & torch.isfinite(hbr)
        # (batch, n_phys, T, 2) -> (batch * n_phys, n_patches, 2 * q)
        patches = torch.where(valid.unsqueeze(-1), torch.stack((hbo, hbr), dim=-1), 0.0)
        patches = patches.reshape(batch_size * n_phys, n_patches, 2 * q)
        # (batch, n_phys, T) -> (batch, n_phys, n_patches)
        counts = valid.reshape(batch_size, n_phys, n_patches, q).sum(dim=-1)
        frac = self.minimum_patch_valid_fraction
        if rem:  # the last patch needs the fraction of its in-bounds samples
            in_bounds = torch.arange(n_patches * q, device=x.device) < n_times
            in_bounds = in_bounds.reshape(n_patches, q).sum(dim=-1).to(torch.float32)
            patch_valid = counts >= torch.ceil(in_bounds * frac).clamp_min(1.0)
        else:
            patch_valid = counts >= math.ceil(q * frac)
        flat_valid = patch_valid.reshape(batch_size * n_phys, n_patches)
        # (batch * n_phys, n_patches, 2 * q) -> (batch * n_phys, n_patches, encoder_dim)
        tokens = self.patch_embed(patches)
        tokens = tokens + sinusoidal_positional_encoding(
            n_patches, self.encoder_dim, device=x.device, dtype=tokens.dtype
        )
        # A channel without valid patch attends to all (an all-masked softmax
        # is NaN); its output is zeroed.
        safe = flat_valid | ~flat_valid.any(dim=-1, keepdim=True)
        encoded = self.encoder(tokens, src_key_padding_mask=~safe)
        encoded = encoded * flat_valid.unsqueeze(-1).to(encoded.dtype)
        foundation = encoded.reshape(batch_size, n_phys, n_patches, self.encoder_dim)
        return {"foundation": foundation, "valid": patch_valid}

    def reset_head(self, n_outputs: int) -> None:
        self._set_n_outputs(n_outputs)
        self.final_layer = nn.Linear(self.final_layer.in_features, n_outputs)

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
            raise ValueError("JungMAE does not accept ch_mask; use NaN channels.")
        encoded = self.encode(x, mask)
        foundation = encoded["foundation"]  # (batch, n_phys, n_patches, encoder_dim)
        if self.pool == "mean":  # -> (batch, n_phys, encoder_dim), valid patches
            valid = encoded["valid"].unsqueeze(-1).to(foundation.dtype)
            foundation = (foundation * valid).sum(dim=2)
            foundation = foundation / valid.sum(dim=2).clamp_min(1.0)
        # -> (batch, feature_dim)
        features = self.head_dropout(foundation.reshape(foundation.shape[0], -1))
        if return_features:
            return {"features": features, "cls_token": None}
        return self.final_layer(features)
