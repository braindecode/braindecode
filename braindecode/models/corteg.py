# Authors: Julien Gadonneix <juliengado.2001@gmail.com>
#
# License: MIT
# Adapted from https://github.com/LiuyinYang1101/CORTEG (MIT).
# Copyright (c) 2026 Liuyin Yang
"""CORTEG: an ST-EEGFormer adapted to intracranial recordings."""

import math
from typing import Dict, Optional, Union

import mne
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from braindecode.functional import check_ch_mask, masked_mean
from braindecode.models.base import EEGModuleMixin
from braindecode.models.steegformer import (
    _channel_order,
    _ChannelPositionalEmbed,
    _init_weights,
    _TemporalPositionalEncoding,
    _TransformerEncoderBlock,
)
from braindecode.models.util import (
    extract_channel_locations_from_chs_info,
    has_valid_locations,
    valid_location_mask,
)
from braindecode.modules import PatchTokenizer
from braindecode.util import resolve_montage_name

def _vocab_positions() -> torch.Tensor:
    """``(142, 3)`` positions (m) of ST-EEGFormer's positioned vocabulary slots.

    The slot names come from STEEGFormer's channel vocabulary, their positions
    from MNE's Colin27 10-05 montage (the reference's table, to 6 decimals).
    """
    montage = mne.channels.make_standard_montage(resolve_montage_name("standard_1005"))
    ch_pos = montage.get_positions()["ch_pos"]
    positions = np.stack([ch_pos[name] for name in _channel_order()])
    return torch.as_tensor(positions, dtype=torch.float32)


class CORTEG(EEGModuleMixin, nn.Module):
    r"""CORTEG from Yang et al. (2026) [Yang2026b]_.

    :bdg-info:`Attention/Transformer` :bdg-danger:`Foundation Model` :bdg-dark-line:`Channel`

    .. figure:: https://raw.githubusercontent.com/LiuyinYang1101/CORTEG/main/docs/assets/figure1_overview_scheme.png
        :align: center
        :width: 1000px

        CORTEG overview, reproduced from the official repository
        (``LiuyinYang1101/CORTEG``).

    .. versionadded:: 2.1

    .. rubric:: Architecture Overview

    A scalp-EEG foundation model, :class:`~braindecode.models.STEEGFormer`,
    adapted to intracranial (ECoG) recordings. The transformer stays the
    pretrained one; around it, a spatial adapter places each electrode by its
    3D position, and a second token stream carries the high-gamma activity the
    scalp model was never trained on. The paper trains only the spatial
    adapter, the high-gamma patch embedding, LoRA on the last
    ``n_merge_blocks`` blocks with their LayerNorms, and the head (about 297 K
    parameters for the small backbone); this module builds the network, and
    LoRA and freezing are left to the training code.

    .. rubric:: Macro Components

    - **Dual-stream tokenizer** (``CORTEG.patch_embed``,
      ``CORTEG.patch_embed_hi``). *Operations:* the low-frequency stream ``x``
      is cut into patches of ``patch_size`` samples, the high-gamma stream
      ``x_hi`` into patches of ``hi_patch_size``, each linearly embedded and
      the high-gamma tokens average-pooled to the low-frequency count. The
      high-gamma embedding starts from the pretrained low-frequency one,
      linearly interpolated to its patch size.
    - **KNNSoftFourier spatial adapter** (``CORTEG.channel_adapter``).
      *Operations:* a soft branch maps each electrode position to softmax
      weights over the 142 positioned slots of the pretrained channel
      embedding (``CORTEG.channel_pos``, kept frozen) and returns their
      weighted sum; the last bias starts from a Gaussian ``knn_k``-nearest-
      neighbour kernel over the slots' scalp positions, so the soft branch
      starts as a nearest-electrode interpolation. A Fourier residual
      (``n_fourier_freqs`` log-spaced frequencies per axis, up to
      ``fourier_max_freq``, zero-initialized output) adds a position-specific
      correction. *Role:* replace the scalp channel
      embedding, which knows no intracranial electrode.
    - **Transformer encoder** (``CORTEG.encoder``). *Operations:* the
      ``depth`` pretrained pre-norm ViT blocks of ST-EEGFormer, over a CLS
      token and the (time-patch, channel) tokens of the low-frequency stream.
    - **Late fusion** (``merge_strategy``). *Operations:* ``"average"``
      replaces the low-frequency tokens by their mean with the high-gamma
      tokens before the last ``n_merge_blocks`` blocks; ``"layerwise_gate"``
      adds the high-gamma tokens after every block, scaled per block and per
      sample by a small gate network that starts at zero.
    - **Read-out** (``CORTEG.norm``, ``CORTEG.final_layer``). *Operations:*
      layer norm, mean over all tokens (CLS included), linear head.

    .. rubric:: Inputs

    The paper decomposes the raw signal into two streams, both z-scored per
    channel with training-set statistics:

    - ``x``: the low-frequency stream, resampled to 128 Hz (the 16-sample
      patches of the pretrained model then span 125 ms);
    - ``x_hi``: the 70-200 Hz Hilbert envelope at 200 Hz, so that its
      25-sample patches line up with those of ``x``.

    With MNE, ``raw.copy().filter(70, 200).apply_hilbert(envelope=True)
    .resample(200)`` gives ``x_hi``. Without ``x_hi`` the model reads ``x``
    alone, as the paper's low-frequency-only ablation.

    Electrode positions are 3D coordinates in metres (the paper's MNI
    coordinates), passed as ``pos`` or read from ``chs_info``.

    .. rubric:: Pre-trained weights

    The ST-EEGFormer small backbone, with the high-gamma embedding
    warm-started from its low-frequency one, loads from the re-hosted
    checkpoint; the adapter, gate and head keep their initialization::

        model = CORTEG.from_pretrained(
            "braindecode/corteg-small-pretrained", n_outputs=5, n_times=128,
            chs_info=chs_info,
        )

    The authors' Stanford model (finger flexion, nine patients pooled) is
    re-hosted with its LoRA factors folded in::

        model = CORTEG.from_pretrained(
            "braindecode/corteg-stanford-pooled", chs_info=chs_info
        )

    .. note::
        The outputs match the reference implementation for the same weights
        and inputs, for both merge strategies. ``ch_mask`` is an addition:
        the reference batches one patient at a time.

    Parameters
    ----------
    patch_size : int
        Patch size of the low-frequency stream (16 for the released
        backbones).
    hi_patch_size : int
        Patch size of the high-gamma stream.
    embed_dim : int
        Token embedding dimension (512 for the small backbone).
    depth : int
        Number of Transformer encoder blocks.
    num_heads : int
        Number of attention heads.
    mlp_ratio : int
        Hidden-to-embedding ratio of the MLP blocks.
    drop_prob : float
        Dropout rate.
    drop_path : float
        Stochastic-depth rate (max of a linear schedule over ``depth``).
    activation : type[nn.Module]
        Activation layer class of the feed-forward blocks and of the MLPs of
        the spatial adapter and the gate.
    n_chans_pos : int
        Size of the pretrained channel vocabulary (145 for small/base/large).
    merge_strategy : {"average", "layerwise_gate"}
        Fusion of the high-gamma stream: the paper's default mean before the
        last ``n_merge_blocks`` blocks (the released Stanford adapter), or the
        layer-wise gate (its BrainTreebank results).
    n_merge_blocks : int
        Number of blocks after the ``"average"`` fusion.
    knn_k : int
        Number of nearest scalp slots of the soft branch's initial kernel.
    adapter_hidden : int
        Hidden width of the spatial adapter's MLPs.
    n_fourier_freqs : int
        Number of Fourier frequencies per axis of the residual branch.
    fourier_max_freq : float
        Highest of those frequencies, log-spaced from 1, in radians per metre.
    gate_bottleneck : int
        Hidden width of the layer-wise gate.
    on_non_divisible : {"pad", "crop", "error"}
        How a stream whose length is not a multiple of its patch size is
        handled (right zero-pad, crop, or raise); the reference crops.

    References
    ----------
    .. [Yang2026b] Yang, L., Sun, Q., Van Dyck, B., Calvo Merino, E. &
       Van Hulle, M. M. (2026). CORTEG: Foundation models enable
       cross-modality representation transfer from scalp to intracranial
       brain recordings. Advances in Neural Information Processing Systems
       (NeurIPS 2026). https://arxiv.org/abs/2605.10337
    """

    def __init__(
        self,
        # --- signal-related (handled by EEGModuleMixin) ---
        n_outputs=None,
        n_chans=None,
        chs_info=None,
        n_times=None,
        input_window_seconds=None,
        sfreq=None,
        # --- model hyperparameters (defaults: small backbone) ---
        patch_size: int = 16,
        hi_patch_size: int = 25,
        embed_dim: int = 512,
        depth: int = 8,
        num_heads: int = 8,
        mlp_ratio: int = 4,
        drop_prob: float = 0.0,
        drop_path: float = 0.1,
        activation: type[nn.Module] = nn.GELU,
        n_chans_pos: int = 145,
        merge_strategy: str = "average",
        n_merge_blocks: int = 4,
        knn_k: int = 8,
        adapter_hidden: int = 128,
        n_fourier_freqs: int = 32,
        fourier_max_freq: float = 50.0,
        gate_bottleneck: int = 16,
        on_non_divisible: str = "pad",
    ):
        super().__init__(
            n_outputs=n_outputs,
            n_chans=n_chans,
            chs_info=chs_info,
            n_times=n_times,
            input_window_seconds=input_window_seconds,
            sfreq=sfreq,
        )
        del n_outputs, n_chans, chs_info, n_times, input_window_seconds, sfreq

        if merge_strategy not in ("average", "layerwise_gate"):
            raise ValueError(
                "merge_strategy must be 'average' or 'layerwise_gate', got "
                f"{merge_strategy!r}."
            )
        if not 1 <= n_merge_blocks <= depth:
            raise ValueError(
                f"n_merge_blocks must be in [1, depth={depth}], got {n_merge_blocks}."
            )
        vocab_positions = _vocab_positions()  # (n_slots, 3)
        n_slots = len(vocab_positions)
        if n_chans_pos < n_slots:
            raise ValueError(
                f"n_chans_pos must be at least {n_slots}, the "
                f"positioned slots of the vocabulary, got {n_chans_pos}."
            )
        self.embed_dim = embed_dim
        self.depth = depth
        self.merge_strategy = merge_strategy
        self.merge_at = depth - n_merge_blocks
        self.knn_k = knn_k

        default_pos: Optional[torch.Tensor] = None
        chs_info = self._chs_info
        if has_valid_locations(chs_info):
            locations = extract_channel_locations_from_chs_info(
                chs_info, fill_missing=True
            )
            default_pos = torch.as_tensor(locations, dtype=torch.float32)
            unplaced = (~valid_location_mask(default_pos))[:, 0].tolist()
            if any(unplaced):
                names = [ch["ch_name"] for ch, u in zip(chs_info, unplaced) if u]
                raise ValueError(
                    f"CORTEG places each electrode by its 3D position, but chs_info "
                    f"has none for {names}; drop these channels, or pass pos at "
                    "each call."
                )
        # Montage-specific, rebuilt from chs_info: not saved with the weights.
        self.register_buffer("default_pos", default_pos, persistent=False)
        self.register_buffer("vocab_positions", vocab_positions, persistent=False)

        # The ST-EEGFormer backbone, under the names of STEEGFormer's modules
        # so that its checkpoints load as they are.
        self.patch_embed = PatchTokenizer(
            patch_size=patch_size,
            n_times=self.n_times,
            emb_dim=embed_dim,
            learnable=True,
            on_non_divisible=on_non_divisible,
            projection="linear",
            output_order="patch_channel",
        )
        self.temporal_pos = _TemporalPositionalEncoding(embed_dim)
        self.channel_pos = _ChannelPositionalEmbed(n_chans_pos, embed_dim)
        # The soft branch's codebook, frozen as in the paper.
        self.channel_pos.embedding.weight.requires_grad_(False)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_drop = nn.Dropout(drop_prob)
        dpr = torch.linspace(0, drop_path, depth).tolist()
        self.encoder = nn.Sequential(
            *[
                _TransformerEncoderBlock(
                    embed_dim, num_heads, drop_prob, int(mlp_ratio), dpr[i], activation
                )
                for i in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

        # x_hi of this length gives one high-gamma token per low-frequency token.
        self.patch_embed_hi = PatchTokenizer(
            patch_size=hi_patch_size,
            n_times=self.patch_embed.n_patches(self.n_times) * hi_patch_size,
            emb_dim=embed_dim,
            learnable=True,
            on_non_divisible=on_non_divisible,
            projection="linear",
            output_order="patch_channel",
        )
        self.channel_adapter = _KNNSoftFourierAdapter(
            n_slots,
            embed_dim,
            adapter_hidden,
            n_fourier_freqs,
            fourier_max_freq,
            activation,
        )
        self.layerwise_gate: Optional[_LayerwiseGate] = (
            _LayerwiseGate(embed_dim, depth, gate_bottleneck, activation)
            if merge_strategy == "layerwise_gate"
            else None
        )
        self.final_layer = nn.Linear(embed_dim, self.n_outputs)

        self._init_parameters()

    def _init_parameters(self) -> None:
        """Model-specific initialization (see ``reset_parameters``)."""
        for module in (self.patch_embed, self.encoder, self.norm, self.final_layer):
            module.apply(_init_weights)
        nn.init.normal_(self.cls_token, std=0.02)
        with torch.no_grad():
            lo, hi = self.patch_embed.proj, self.patch_embed_hi.proj
            assert lo is not None and hi is not None
            size = self.patch_embed_hi.patch_size
            hi.weight.copy_(F.interpolate(lo.weight[None], size=size, mode="linear")[0])
            hi.bias.copy_(lo.bias)
        prior = None
        if self.default_pos is not None:
            prior = _knn_log_prior(self.default_pos, self.vocab_positions, self.knn_k)
        self.channel_adapter.init_parameters(prior)
        if self.layerwise_gate is not None:
            self.layerwise_gate.init_parameters()

    def reset_head(self, n_outputs):
        """Replace the linear head for a new ``n_outputs``."""
        self._n_outputs = n_outputs
        self._update_init_kwargs(n_outputs=n_outputs)
        self.final_layer = nn.Linear(self.embed_dim, n_outputs)
        _init_weights(self.final_layer)

    def forward(
        self,
        x: torch.Tensor,
        x_hi: Optional[torch.Tensor] = None,
        pos: Optional[torch.Tensor] = None,
        ch_mask: Optional[torch.Tensor] = None,
        return_features: bool = False,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Encode a batch of intracranial recordings.

        Parameters
        ----------
        x : torch.Tensor
            Low-frequency stream, ``(batch, n_chans, n_times)``.
        x_hi : torch.Tensor, optional
            High-gamma stream of the same channels, ``(batch, n_chans,
            n_times_hi)``. Without it the model reads ``x`` alone.
        pos : torch.Tensor, optional
            Electrode positions in metres, ``(batch, n_chans, 3)`` or
            ``(n_chans, 3)``; by default those of ``chs_info``.
        ch_mask : torch.Tensor, optional
            ``(batch, n_chans)`` mask of real channels, see
            :func:`~braindecode.functional.check_ch_mask`. Padded channels
            take part in no attention and no pooling.
        return_features : bool
            If ``True``, return ``{"features", "cls_token", "tokens"}``: the
            layer-normalized patch tokens ``(batch, n_patches * n_chans,
            embed_dim)``, the CLS token, and the patch tokens as ``(batch,
            n_chans, n_patches, embed_dim)``; padded channels' tokens are zero.

        Returns
        -------
        torch.Tensor | dict
            Outputs ``(batch, n_outputs)``, or the feature dict.
        """
        # Shapes: B batch, C channels, P patches, D embed_dim, N = P * C tokens.
        batch_size, n_chans, n_times = x.shape
        n_patches = self.patch_embed.n_patches(n_times)
        mask = check_ch_mask(ch_mask, x)  # (B, C), None without padding
        if pos is None:
            pos = self.default_pos
        hi: Optional[torch.Tensor] = None
        gates: Optional[torch.Tensor] = None
        token_mask: Optional[torch.Tensor] = None
        key_mask: Optional[torch.Tensor] = None
        attn_mask: Optional[torch.Tensor] = None

        if pos is None:
            raise ValueError(
                "CORTEG places each electrode by its 3D position: pass pos, or "
                "chs_info with the electrode positions."
            )
        if pos.dim() == 2:
            pos = pos.unsqueeze(0)  # (1, C, 3), shared by the batch
        if pos.shape[1] != n_chans or pos.shape[2] != 3:
            raise ValueError(
                f"pos must have shape (batch, {n_chans}, 3) or ({n_chans}, 3), "
                f"got {list(pos.shape)}."
            )
        if mask is not None:
            # Padding may hold NaN, which the embeddings would spread.
            x = x.masked_fill(~mask[:, :, None], 0.0)
            pos = pos.masked_fill(~mask[:, :, None], 0.0)
            if x_hi is not None:
                x_hi = x_hi.masked_fill(~mask[:, :, None], 0.0)
            # (B, N), patch-major like the tokens
            token_mask = mask.repeat(1, n_patches)
            key_mask = F.pad(token_mask, (1, 0), value=True)  # (B, 1 + N), CLS first
            attn_mask = key_mask[:, None, None, :]  # (B, 1, 1, 1 + N)

        codebook = self.channel_pos.embedding.weight[: self.vocab_positions.shape[0]]
        channel_emb = self.channel_adapter(pos, codebook)  # (B or 1, C, D)
        # (B or 1, P, C, D)
        embedding = self.temporal_pos(n_patches) + channel_emb[:, None]
        tokens = self.patch_embed(x)  # (B, P, C, D)
        lo = (tokens + embedding).flatten(1, 2)  # (B, N, D)
        if x_hi is not None:
            hi_tokens = self.patch_embed_hi(x_hi)  # (B, P_hi, C, D)
            if hi_tokens.shape[1] != n_patches:
                pooled = F.adaptive_avg_pool1d(
                    hi_tokens.permute(0, 2, 3, 1).flatten(0, 1), n_patches
                )  # (B * C, D, P)
                hi_tokens = pooled.reshape(
                    batch_size, n_chans, self.embed_dim, n_patches
                ).permute(0, 3, 1, 2)  # (B, P, C, D)
            hi = (hi_tokens + embedding).flatten(1, 2)  # (B, N, D)

        cls = self.cls_token + self.temporal_pos.cls_token_encoding()  # (1, 1, D)
        z = torch.cat([cls.expand(batch_size, -1, -1), lo], dim=1)  # (B, 1 + N, D)
        z = self.pos_drop(z)
        if hi is not None and self.layerwise_gate is not None:
            gates = self.layerwise_gate(
                masked_mean(lo, token_mask, [1]), masked_mean(hi, token_mask, [1])
            )  # (B, depth)
        for i, block in enumerate(self.encoder):
            if hi is not None and gates is None and i == self.merge_at:
                # "average" fusion: the CLS token is kept, the others averaged.
                z = torch.cat([z[:, :1], 0.5 * (z[:, 1:] + hi)], dim=1)  # (B, 1 + N, D)
            z = block(z, attn_mask)  # (B, 1 + N, D)
            if hi is not None and gates is not None:
                # "layerwise_gate" fusion: hi added after each block, not to CLS;
                # (B, 1 + N, D)
                z = z + gates[:, i, None, None] * F.pad(hi, (0, 0, 1, 0))
        z = self.norm(z)  # (B, 1 + N, D)

        if return_features:
            patches = z[:, 1:]  # (B, N, D)
            if token_mask is not None:
                patches = patches.masked_fill(~token_mask[:, :, None], 0.0)
            per_channel = patches.reshape(
                batch_size, n_patches, n_chans, self.embed_dim
            ).transpose(1, 2)  # (B, C, P, D)
            return {"features": patches, "cls_token": z[:, 0], "tokens": per_channel}
        return self.final_layer(masked_mean(z, key_mask, [1]))  # (B, n_outputs)


def _knn_log_prior(
    positions: torch.Tensor, vocab_positions: torch.Tensor, k: int
) -> torch.Tensor:
    """Log Gaussian k-NN weight of each vocabulary slot, averaged over ``positions``.

    The bandwidth is the median distance of the ``k`` nearest slots.
    """
    distances, slots = torch.cdist(positions, vocab_positions).topk(
        k, dim=-1, largest=False
    )
    sigma = distances.median().clamp(min=1e-6)
    weights = torch.exp(-0.5 * (distances / sigma) ** 2)
    weights = weights / weights.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    dense = torch.zeros(len(positions), len(vocab_positions)).scatter_(
        1, slots, weights
    )
    return torch.log(dense + 1e-8).mean(dim=0)


class _KNNSoftFourierAdapter(nn.Module):
    """Channel embedding of a 3D position: a soft branch over the pretrained
    codebook plus a Fourier-feature residual."""

    def __init__(
        self,
        n_slots: int,
        embed_dim: int,
        hidden: int,
        n_freqs: int,
        max_freq: float,
        activation: type[nn.Module],
    ):
        super().__init__()
        self.soft = _SoftCodebookBranch(n_slots, hidden, activation)
        self.residual = _FourierBranch(embed_dim, hidden, n_freqs, max_freq, activation)

    def init_parameters(self, prior: Optional[torch.Tensor]) -> None:
        """Start the soft branch at ``prior`` (uniform if ``None``) and the
        residual at zero."""
        with torch.no_grad():
            bias = self.soft.net[-1].bias
            bias.copy_(torch.zeros_like(bias) if prior is None else prior)
            self.soft.scale.fill_(1.0)
        nn.init.zeros_(self.residual.net[-1].weight)

    def forward(self, pos: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
        return self.soft(pos, codebook) + self.residual(pos)


class _SoftCodebookBranch(nn.Module):
    """Softmax attention of a position over the codebook slots."""

    def __init__(self, n_slots: int, hidden: int, activation: type[nn.Module]):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden),
            activation(),
            nn.Linear(hidden, hidden),
            activation(),
            nn.Linear(hidden, n_slots),
        )
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, pos: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
        return self.scale * torch.matmul(self.net(pos).softmax(dim=-1), codebook)


class _FourierBranch(nn.Module):
    """MLP on sinusoidal features of a position, at log-spaced frequencies."""

    freqs: torch.Tensor

    def __init__(
        self,
        embed_dim: int,
        hidden: int,
        n_freqs: int,
        max_freq: float,
        activation: type[nn.Module],
    ):
        super().__init__()
        self.register_buffer(
            "freqs", torch.logspace(0, math.log10(max_freq), n_freqs), persistent=False
        )
        # sin and cos of each frequency on each of the 3 axes
        self.net = nn.Sequential(
            nn.Linear(6 * n_freqs, hidden),
            activation(),
            nn.Linear(hidden, hidden),
            activation(),
            nn.Linear(hidden, embed_dim, bias=False),
        )

    def forward(self, pos: torch.Tensor) -> torch.Tensor:
        angles = (pos[..., None] * self.freqs).flatten(-2)
        return self.net(torch.cat([angles.sin(), angles.cos()], dim=-1))


class _LayerwiseGate(nn.Module):
    """One tanh gate per block and sample, from the mean low and high tokens."""

    def __init__(
        self, embed_dim: int, depth: int, bottleneck: int, activation: type[nn.Module]
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(2 * embed_dim),
            nn.Linear(2 * embed_dim, bottleneck),
            activation(),
            nn.Linear(bottleneck, depth),
        )

    def init_parameters(self) -> None:
        """Start every gate closed, at the low-frequency-only model."""
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(torch.cat([lo, hi], dim=-1)))
