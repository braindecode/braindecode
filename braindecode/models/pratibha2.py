# Authors: Mahir Jain (mahir@mannas.ai)
#
# License: Apache-2.0
# Reference implementation and weights: https://huggingface.co/MannasAI/pratibha-2 (Apache-2.0).
"""PRATIBHA-2 (formerly MANAS-2): a dense-attention EEG foundation model."""

from functools import lru_cache, partial
from typing import Optional

import numpy as np
import torch
from mne.channels import make_standard_montage
from torch import nn

from braindecode.models.base import EEGModuleMixin
from braindecode.models.reve import _reve_position_bank
from braindecode.models.util import warn_if_sfreq_differs
from braindecode.modules import (
    FeedForwardBlock,
    PreNormBlock,
    TimmAttention,
    resolve_channel_positions,
)
from braindecode.modules.layers import RMSNorm
from braindecode.util import resolve_montage_name


class Pratibha2(EEGModuleMixin, nn.Module, license="apache-2.0"):
    r"""PRATIBHA-2 (formerly MANAS-2) from Kulkarni et al. (2026) [manas2]_.

    :bdg-danger:`Foundation Model` :bdg-info:`Attention/Transformer` :bdg-dark-line:`Channel`

    .. rubric:: Architecture Overview

    A transformer encoder pretrained with masked autoencoding on clinical and
    research EEG: waveform and spectral reconstruction (Raw-Band Hybrid) with a
    constraint on the local RMS energy of the reconstructed waveform
    (Constrained Reconstruction). Every electrode-patch pair is one token and
    every layer attends densely over all tokens of the window::

        (batch, n_chans, n_times) -> per-channel z-score -> 1 s patches
        (0.1 s overlap) -> patch embedding + 4D position encoding -> dense
        transformer -> mean over tokens -> head -> (batch, n_outputs)

    .. rubric:: Macro Components

    ``Pratibha2.encoder.patch_embed``
        **Operations.** ``Linear(patch_size, embed_dim)`` on each patch.

    ``Pratibha2.encoder.pos_enc``
        **Operations.** Sines and cosines of ``n_freqs ** 4`` frequency
        combinations of the token's 4D position (electrode position in cm,
        patch index), projected to ``embed_dim``, plus a learned gated branch;
        the sum is RMS-normalized and added to the patch embedding.

    ``Pratibha2.encoder.blocks``
        **Operations.** ``depth`` pre-norm blocks (RMSNorm) of multi-head
        self-attention and a gated feed-forward network
        ``fc2(fc1(x) * (swish(g) + gelu_tanh(g)))``, ``g = fc_gate(x)``.
        **Role.** The pretrained backbone; :meth:`encode` returns its token grid.

    ``Pratibha2.final_layer``
        **Operations.** ``Linear -> head_activation -> Dropout -> Linear`` on
        the token mean. Not pretrained.

    .. rubric:: Temporal, Spatial, and Spectral Encoding

    - **Temporal:** 1 s patches with a learned linear embedding, ordered by the
      patch index of the position encoding.
    - **Spatial:** electrodes enter only through their 3D positions, so any
      montage the position table places can be read.
    - **Spectral:** learned by the band-power reconstruction target.

    .. rubric:: Additional Mechanisms

    - **Channel positions.** The pretraining electrode table, REVE's bank plus
      PRATIBHA-2's own ``AFZ`` and ``IZ``, looked up by name (exact spelling
      first, then ignoring case, ``EEG``/``-REF``, ``T3``-style aliases, or the
      midpoint of a bipolar ``A-B``; see
      :func:`~braindecode.modules.resolve_channel_positions`). Other 10-05 names
      take their position in MNE's ``standard_1005`` montage frame, the frame of
      the table's 10-05 entries. Channel ``loc`` values are not used: they are in
      MNE's head frame, a few centimetres from the pretraining frame.
    - **Input scale.** With ``normalize_input=True`` each channel is z-scored
      within each window, so volts and microvolts give the same output.

    .. important::

        Resample to **200 Hz**; windows need at least ``patch_size`` samples.
        The released encoder, converted to these parameter names, is on the
        Hugging Face Hub at ``braindecode/pratibha2-pretrained``; the head is not
        pretrained::

            from braindecode.models import Pratibha2
            model = Pratibha2.from_pretrained(
                "braindecode/pratibha2-pretrained",
                chs_info=raw.info["chs"], n_outputs=4, n_times=1000,
            )

    .. versionadded:: 2.1

    Parameters
    ----------
    patch_size : int
        Number of samples per patch (token length). 200 samples = 1 s at 200 Hz.
    patch_stride : int
        Step between consecutive patches, in samples.
    embed_dim : int
        Token embedding dimension.
    depth : int
        Number of transformer blocks.
    num_heads : int
        Number of attention heads.
    ffn_expansion : int
        Width multiplier of the gated feed-forward network.
    n_freqs : int
        Frequencies per axis of the Fourier position encoding.
    head_hidden_dim : int
        Hidden size of the classification head.
    normalize_input : bool
        If True, z-score each channel within each window before patching.
    head_activation : type[nn.Module]
        Activation of the classification head.
    drop_prob : float
        Dropout probability in the classification head.
    att_drop_prob : float
        Attention dropout probability (0 in the pretrained model).
    on_missing : str
        Channels the position table cannot place: ``"error"`` raises,
        ``"drop"`` leaves them out of the input, ``"nearest"`` takes the table
        entry closest to the channel's ``loc``.
    channel_strategy : str
        How another montage reaches the encoder; see
        :class:`~braindecode.models.base.EEGModuleMixin`. PRATIBHA-2 reads any
        montage natively, so the default ``"native"`` is usually what you want.
    channel_strategy_kwargs : dict or None
        Options of ``channel_strategy``.

    References
    ----------
    .. [manas2] Kulkarni, A., Mishra, A. R., Jain, M., Runwal, P., Saini, L.,
       Panwar, S., & Singh, S. (2026). MANAS-2: Constrained Reconstruction for
       EEG Foundation Models. arXiv:2609.13717.
       https://arxiv.org/abs/2609.13717
    """

    channel_index: Optional[torch.Tensor]

    def __init__(
        self,
        n_outputs=None,
        n_chans=None,
        chs_info=None,
        n_times=None,
        input_window_seconds=None,
        sfreq=None,
        *,
        patch_size: int = 200,
        patch_stride: int = 180,
        embed_dim: int = 512,
        depth: int = 22,
        num_heads: int = 8,
        ffn_expansion: int = 4,
        n_freqs: int = 4,
        head_hidden_dim: int = 128,
        normalize_input: bool = True,
        head_activation: type[nn.Module] = nn.ELU,
        drop_prob: float = 0.3,
        att_drop_prob: float = 0.0,
        on_missing: str = "error",
        channel_strategy: str = "native",
        channel_strategy_kwargs: Optional[dict] = None,
    ):
        super().__init__(
            n_outputs=n_outputs,
            n_chans=n_chans,
            chs_info=chs_info,
            n_times=n_times,
            input_window_seconds=input_window_seconds,
            sfreq=sfreq,
            channel_strategy=channel_strategy,
            channel_strategy_kwargs=channel_strategy_kwargs,
        )
        del n_outputs, n_chans, chs_info, n_times, input_window_seconds, sfreq

        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})."
            )
        warn_if_sfreq_differs("Pratibha2", self._sfreq, 200.0)
        try:
            n_times = self.n_times  # also inferred from input_window_seconds
        except ValueError:
            n_times = None
        if n_times is not None and n_times < patch_size:
            raise ValueError(
                f"n_times ({n_times}) must be at least patch_size ({patch_size})."
            )

        self.embed_dim = embed_dim
        self.head_hidden_dim = head_hidden_dim
        self.head_activation = head_activation
        self.drop_prob = drop_prob
        self.on_missing = on_missing

        positions, kept = resolve_channel_positions(
            self.chs_info, _pratibha2_position_bank(), on_missing=on_missing
        )
        dropped = len(kept) < len(self.chs_info)
        self.register_buffer(
            "channel_index", torch.tensor(kept) if dropped else None, persistent=False
        )
        self.encoder = _Pratibha2Encoder(
            channel_positions=torch.tensor(positions, dtype=torch.float32) * 100.0,
            patch_size=patch_size,
            patch_stride=patch_stride,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            ffn_expansion=ffn_expansion,
            n_freqs=n_freqs,
            normalize_input=normalize_input,
            att_drop_prob=att_drop_prob,
        )
        self.final_layer = self._build_head(self.n_outputs)

    def _build_head(self, n_outputs: int) -> nn.Module:
        return nn.Sequential(
            nn.Linear(self.embed_dim, self.head_hidden_dim),
            self.head_activation(),
            nn.Dropout(self.drop_prob),
            nn.Linear(self.head_hidden_dim, n_outputs),
        )

    def reset_head(self, n_outputs: int) -> None:
        """Replace the classification head for a new number of outputs.

        The encoder is left unchanged; the new head is randomly initialised.

        Parameters
        ----------
        n_outputs : int
            Number of outputs of the new head.
        """
        self._set_n_outputs(n_outputs)
        self.final_layer = self._build_head(n_outputs)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode EEG into the grid of token embeddings.

        This is the encoder output before pooling and the classification head.

        Parameters
        ----------
        x : torch.Tensor
            EEG of shape (batch, n_chans, n_times), sampled at 200 Hz.

        Returns
        -------
        torch.Tensor
            Tokens of shape (batch, n_kept_chans, n_patches, embed_dim), where
            ``n_kept_chans`` excludes channels dropped by ``on_missing="drop"``.
        """
        if x.dim() != 3 or x.shape[1] != self.n_chans:
            raise ValueError(
                f"Expected EEG with {self.n_chans} channels in the configured montage."
            )
        channel_index = self.channel_index
        if channel_index is not None:
            x = x.index_select(1, channel_index)
        return self.encoder(x)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        """Classify a batch of EEG windows.

        Tokens are averaged over electrodes and patches, then passed to the
        classification head ``final_layer``.

        Parameters
        ----------
        x : torch.Tensor
            EEG of shape (batch, n_chans, n_times), sampled at 200 Hz.
        return_features : bool
            If True, return a dict with the pooled embedding (``"features"``,
            shape (batch, embed_dim)) and the token grid (``"tokens"``, shape
            (batch, n_kept_chans, n_patches, embed_dim), see :meth:`encode`)
            instead of logits.

        Returns
        -------
        torch.Tensor or dict
            Logits of shape (batch, n_outputs), or the feature dict when
            ``return_features`` is True.
        """
        tokens = self.encode(x)
        features = tokens.mean(dim=(1, 2))
        logits = self.final_layer(features)

        if return_features:
            if torch.jit.is_scripting():
                return logits
            return {
                "features": features,
                "tokens": tokens,
                "cls_token": None,  # nosec B105
            }
        return logits


class _SwishGelu(nn.Module):
    """``swish(x) + gelu_tanh(x)``, the gate activation of PRATIBHA-2."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x) + nn.functional.gelu(x, approximate="tanh")


# The pretraining electrode table is REVE's bank plus these two entries (m), which
# sit 1.43 and 0.74 cm from 10-05 AFz and Iz (release 863edc9). They come last, so
# a name that matches only ignoring case (``afz``) takes them, as in the release.
_EXTRA_POSITIONS = {
    "AFZ": (0.0010058004409074783, 0.07728150114417076, 0.021591500379145145),
    "IZ": (-3.840029239654541e-05, -0.1144150011241436, -0.029236000031232834),
}


@lru_cache(maxsize=1)
def _pratibha2_position_bank() -> dict[str, np.ndarray]:
    """Name -> position (m) of the pretraining table, in MNE's montage frame.

    The table is REVE's bank plus ``_EXTRA_POSITIONS``. It is laid over MNE's
    ``standard_1005`` montage, which places the 10-05 names the table lacks in the
    frame of its 10-05 entries.
    """
    montage = make_standard_montage(resolve_montage_name("standard_1005"))
    return {
        **montage.get_positions()["ch_pos"],
        **_reve_position_bank(),
        **_EXTRA_POSITIONS,
    }


class _Pratibha2Encoder(nn.Module):
    """Patch embedding, position encoding and the stack of transformer blocks.

    Maps EEG (B, C, T) to the token grid (B, C, P, D).
    """

    def __init__(
        self,
        channel_positions: torch.Tensor,
        patch_size: int,
        patch_stride: int,
        embed_dim: int,
        depth: int,
        num_heads: int,
        ffn_expansion: int,
        n_freqs: int,
        normalize_input: bool,
        att_drop_prob: float,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.embed_dim = embed_dim
        self.normalize_input = normalize_input
        # Not persistent: the pretrained weights must load onto any montage.
        self.register_buffer("channel_positions", channel_positions, persistent=False)

        self.patch_embed = nn.Linear(patch_size, embed_dim, bias=False)
        self.pos_enc = _Pratibha2PositionalEncoding(embed_dim, n_freqs)
        norm = partial(RMSNorm, eps=1e-8)
        self.blocks = nn.ModuleList(
            [
                PreNormBlock(
                    embed_dim,
                    attn=TimmAttention(
                        embed_dim, num_heads, qkv_bias=True, attn_drop=att_drop_prob
                    ),
                    mlp=FeedForwardBlock(
                        embed_dim, ffn_expansion, 0.0, _SwishGelu, gated=True
                    ),
                    norm_layer=norm,
                )
                for _ in range(depth)
            ]
        )
        self.final_norm = norm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.normalize_input:
            x32 = x.to(torch.promote_types(x.dtype, torch.float32))
            mean = x32.mean(dim=-1, keepdim=True)
            std = x32.std(dim=-1, keepdim=True, unbiased=x32.size(-1) > 1)
            x = ((x32 - mean) / std.clamp_min(1e-12)).to(x.dtype)
        patches = x.unfold(-1, self.patch_size, self.patch_stride)  # (B, C, P, S)
        tokens = self.patch_embed(patches)  # (B, C, P, D)
        batch, n_chans, n_patches, _ = tokens.shape

        positions = self.channel_positions
        positions = positions.to(torch.promote_types(positions.dtype, torch.float32))
        time = torch.arange(n_patches, device=tokens.device, dtype=positions.dtype)
        coords = torch.cat(
            [
                positions[:, None, :].expand(-1, n_patches, -1),
                time[None, :, None].expand(n_chans, -1, -1),
            ],
            dim=-1,
        )  # (C, P, 4)
        tokens = tokens + self.pos_enc(coords).unsqueeze(0).to(tokens.dtype)

        tokens = tokens.reshape(batch, n_chans * n_patches, self.embed_dim)
        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.final_norm(tokens)
        return tokens.reshape(batch, n_chans, n_patches, self.embed_dim)


class _Pratibha2PositionalEncoding(nn.Module):
    """4D position encoding: Fourier features plus a learned gated branch."""

    def __init__(self, embed_dim: int, n_freqs: int):
        super().__init__()
        freqs = torch.linspace(1.0, 10.0, n_freqs)
        grid = torch.cartesian_prod(freqs, freqs, freqs, freqs).reshape(-1, 4)
        self.register_buffer("freq_matrix", grid.T.contiguous(), persistent=False)
        self.fourier_linear = nn.Linear(2 * n_freqs**4, embed_dim, bias=False)
        self.learned_linear = nn.Linear(4, 2 * embed_dim, bias=False)
        self.gate_activation = _SwishGelu()
        self.learned_norm = RMSNorm(embed_dim, eps=1e-8)
        self.final_norm = RMSNorm(embed_dim, eps=1e-8)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        # bfloat16 cannot represent the angles.
        angle_dtype = torch.promote_types(coords.dtype, torch.float32)
        # Autocast lowers matmul precision even for two float32 operands.
        phases = (
            coords.to(angle_dtype).unsqueeze(-1) * self.freq_matrix.to(angle_dtype)
        ).sum(dim=-2)
        weight_dtype = self.fourier_linear.weight.dtype
        fourier = self.fourier_linear(
            torch.cat([phases.sin(), phases.cos()], dim=-1).to(weight_dtype)
        )
        value, gate = self.learned_linear(coords.to(weight_dtype)).chunk(2, dim=-1)
        learned = self.learned_norm(value * self.gate_activation(gate))
        return self.final_norm(fourier + learned)
