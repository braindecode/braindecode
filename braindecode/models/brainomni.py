# Authors: Qian Xiao and OpenTSLab BrainOmni contributors
#          Bruno Aristimunha <b.aristimunha@gmail.com> (braindecode adaptation)
#
# License: MIT
#
# Adapted from https://github.com/OpenTSLab/BrainOmni (MIT); the SEANet codec
# derives from Meta's EnCodec (MIT).
from __future__ import annotations

import warnings

import mne
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from mne.io.constants import FIFF
from torch.nn import RMSNorm
from torch.nn.utils.parametrizations import weight_norm

from braindecode.functional import rotate_pairs
from braindecode.models.base import EEGModuleMixin
from braindecode.models.util import extract_channel_locations_from_chs_info
from braindecode.modules import FeedForwardBlock
from braindecode.modules.quantization import ResidualVectorQuantizer


class BrainTokenizer(EEGModuleMixin, nn.Module, license="mit"):
    r"""BrainTokenizer from Xiao et al. (2025) [brainomni]_.

    :bdg-danger:`Foundation Model` :bdg-info:`Attention/Transformer`

    .. rubric:: Architecture Overview

    A VQ-VAE over raw EEG/MEG conditioned on sensor geometry::

        (batch, n_chans, n_times) -> windows -> SEANet encoder -> cross-attention
        to n_neuro sources -> residual VQ -> cross-attention back to n_chans ->
        SEANet decoder -> (batch, n_chans, n_times)

    .. rubric:: Macro Components

    ``BrainTokenizer.sensor_embed``
        **Operations.** MLP on each channel's position and orientation plus an
        EEG/MAG/GRAD type embedding, then RMSNorm. **Role.** Montage-agnostic
        channel identity.

    ``BrainTokenizer.encoder``
        **Operations.** SEANet encodes each ``(channel, window)``; ``n_neuro``
        learned queries attend over the sensor-conditioned channels.
        **Role.** ``(batch, n_neuro, n_windows, n_tokens, emb_dim)`` latents.

    ``BrainTokenizer.quantizer``
        **Operations.** Residual vector quantization with EMA codebooks.
        **Role.** ``num_quantizers`` codebook indices per token.

    ``BrainTokenizer.final_layer``
        **Operations.** The sensor embeddings query the quantized sources, then
        the SEANet decoder rebuilds each window. **Role.** Reconstruction.

    .. rubric:: Temporal, Spatial, and Spectral Encoding

    - **Temporal:** SEANet's strided convolutions and LSTM downsample each
      window by ``prod(ratios)``.
    - **Spatial:** cross-attention between channels and ``n_neuro`` sources,
      keyed by sensor geometry.
    - **Spectral:** learned implicitly by the convolutions.

    .. rubric:: Additional Mechanisms

    - Windows of ``window_length`` samples; a shorter input is zero-padded and
      an incomplete non-overlapping tail is dropped (zero-filled in the
      reconstruction).
    - :meth:`tokenize` runs in ``eval`` mode without gradients, so the
      codebooks are never EMA-updated.

    .. important::

        Weights converted from the released checkpoint are on the Hugging Face
        Hub at ``braindecode/braintokenizer-pretrained``::

            from braindecode.models import BrainTokenizer
            model = BrainTokenizer.from_pretrained("braindecode/braintokenizer-pretrained", chs_info=raw.info["chs"])

        Input is expected at 256 Hz, preprocessed as in the authors' code.

    .. versionadded:: 1.8

    Parameters
    ----------
    emb_dim : int
        Embedding dimension.
    n_neuro : int
        Number of latent source tokens.
    window_length : int
        Samples per analysis window.
    n_filters : int
        Base number of SEANet filters.
    ratios : tuple of int
        SEANet downsampling ratios.
    kernel_size : int
        SEANet kernel size.
    last_kernel_size : int
        Kernel size of the first and last SEANet convolutions.
    tokenizer_num_heads : int
        Heads of the cross-attention blocks.
    codebook_dim : int
        Codebook dimension.
    codebook_size : int
        Entries per codebook.
    num_quantizers : int
        Number of residual VQ stages.
    rotation_trick : bool
        Pass encoder gradients through the rotation trick instead of the
        straight-through estimator.
    drop_prob : float
        Attention dropout.

    References
    ----------
    .. [brainomni] Xiao, Q., Cui, Z., Zhang, C., Chen, S., Wu, W.,
       Thwaites, A., Woolgar, A., Zhou, B., Zhang, C. (2025).
       BrainOmni: A Brain Foundation Model for Unified EEG and MEG Signals.
       NeurIPS 2025. https://arxiv.org/abs/2505.18185
    """

    def __init__(
        self,
        n_outputs=None,
        n_chans=None,
        chs_info=None,
        n_times=None,
        input_window_seconds=None,
        sfreq=None,
        *,
        emb_dim: int = 256,
        n_neuro: int = 16,
        window_length: int = 512,
        n_filters: int = 32,
        ratios: tuple[int, ...] = (8, 4, 2),
        kernel_size: int = 5,
        last_kernel_size: int = 5,
        tokenizer_num_heads: int = 4,
        codebook_dim: int = 256,
        codebook_size: int = 512,
        num_quantizers: int = 4,
        rotation_trick: bool = True,
        drop_prob: float = 0.0,
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
        if self.sfreq != 256:
            warnings.warn(f"BrainTokenizer weights expect 256 Hz, got {self.sfreq}.")
        if n_filters < 2:  # the first residual block has n_filters // 2 channels
            raise ValueError(f"n_filters must be at least 2, got {n_filters}.")
        if emb_dim <= 0 or tokenizer_num_heads <= 0:
            raise ValueError(
                f"emb_dim ({emb_dim}) and tokenizer_num_heads "
                f"({tokenizer_num_heads}) must be positive."
            )
        if emb_dim % tokenizer_num_heads:
            raise ValueError(
                f"emb_dim ({emb_dim}) must be divisible by "
                f"tokenizer_num_heads ({tokenizer_num_heads})."
            )
        # Derived from chs_info, so not saved: one checkpoint serves any montage.
        pos, sensor_type = _geometry_from_chs_info(self.chs_info)
        self.register_buffer("pos", torch.from_numpy(pos), persistent=False)
        self.register_buffer(
            "sensor_type", torch.from_numpy(sensor_type), persistent=False
        )
        self.window_length = window_length

        self.sensor_embed = _SensorEmbedding(emb_dim)
        self.encoder = _TokenizerEncoder(
            n_filters,
            ratios,
            kernel_size,
            last_kernel_size,
            emb_dim,
            tokenizer_num_heads,
            drop_prob,
            n_neuro,
        )
        self.quantizer = ResidualVectorQuantizer(
            dim=emb_dim,
            codebook_dim=codebook_dim,
            codebook_size=codebook_size,
            num_quantizers=num_quantizers,
            rotation_trick=rotation_trick,
        )
        self.final_layer = _TokenizerDecoder(
            emb_dim,
            tokenizer_num_heads,
            n_filters,
            ratios,
            kernel_size,
            last_kernel_size,
            drop_prob,
        )
        self.apply(_init_weights)

    def _encode_quantize(self, x: torch.Tensor, overlap_ratio: float = 0.0):
        step = int(self.window_length * (1 - overlap_ratio))
        x = F.pad(x, (0, max(self.window_length - x.shape[-1], 0)))
        if step < self.window_length:  # overlap pads the last window, as released
            x = F.pad(x, (0, -(x.shape[-1] - self.window_length) % step))
        x = x.unfold(-1, self.window_length, step)  # (batch, chans, nwin, wlen)
        se = self.sensor_embed(self.pos, self.sensor_type)
        se = se.unsqueeze(0).expand(x.shape[0], -1, -1)  # (batch, chans, emb_dim)
        feat_q, indices, commit = self.quantizer(self.encoder(x, se))
        return feat_q, indices, commit, se

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Reconstruct ``x``."""
        return self.encode_decode(x)[0]

    def encode_decode(self, x: torch.Tensor):
        """Return the reconstruction, commitment loss and codebook indices.

        A dropped non-overlapping tail is zero-filled.
        """
        feat_q, indices, commit, se = self._encode_quantize(x)
        recon = self.final_layer(feat_q, se)[..., : self.window_length]
        recon = recon.reshape(recon.shape[0], recon.shape[1], -1)
        n_times = x.shape[-1]
        recon = F.pad(recon, (0, max(n_times - recon.shape[-1], 0)))[..., :n_times]
        return recon, commit, indices

    @torch.no_grad()
    def tokenize(self, x: torch.Tensor, overlap_ratio: float = 0.0):
        """Quantized features and codebook indices of ``x``, in eval mode.

        Parameters
        ----------
        x : torch.Tensor
            ``(batch, n_chans, n_times)``.
        overlap_ratio : float
            Overlap between consecutive windows, in ``[0, 1)``.

        Returns
        -------
        feat : torch.Tensor
            ``(batch, n_neuro, n_windows * n_tokens, emb_dim)``.
        indices : torch.Tensor
            ``(batch, n_neuro, n_windows * n_tokens, num_quantizers)``.
        """
        if overlap_ratio < 0 or int(self.window_length * (1 - overlap_ratio)) < 1:
            raise ValueError(
                f"overlap_ratio must be >= 0 and leave a stride of at least one "
                f"sample for window_length={self.window_length}, got {overlap_ratio}."
            )
        was_training = self.training
        self.eval()
        try:
            feat, indices, _, _ = self._encode_quantize(x, overlap_ratio)
        finally:
            self.train(was_training)
        flat = "batch chans nwin tok dim -> batch chans (nwin tok) dim"
        return rearrange(feat, flat), rearrange(indices, flat)


class BrainOmni(EEGModuleMixin, nn.Module, license="mit"):
    r"""BrainOmni from Xiao et al. (2025) [brainomni]_.

    :bdg-danger:`Foundation Model` :bdg-info:`Attention/Transformer`

    .. rubric:: Architecture Overview

    A frozen :class:`BrainTokenizer` followed by factored spatial-temporal
    attention blocks and a classification head::

        (batch, n_chans, n_times) -> BrainTokenizer.tokenize -> projection ->
        spatial-temporal blocks -> mean over time -> (batch, n_outputs)

    .. rubric:: Macro Components

    ``BrainOmni.tokenizer``
        **Operations.** :meth:`BrainTokenizer.tokenize` with windows overlapping
        by ``overlap_ratio``, plus the learned source embeddings. **Role.**
        Frozen feature extractor (no gradients, no codebook updates).

    ``BrainOmni.projection``
        **Operations.** ``Linear(emb_dim, lm_dim)``, identity when equal.

    ``BrainOmni.blocks``
        **Operations.** Half of the features attend over time (RoPE), the other
        half over the ``n_neuro`` sources, then a feed-forward layer. The last
        block is part of the pretrained stack but unused downstream, as released.
        **Role.** Space-time contextualization.

    ``BrainOmni.final_layer``
        **Operations.** ``Dropout(0.1) -> Linear -> activation -> Linear`` on the
        flattened ``n_neuro * lm_dim`` features. **Role.** Classification head.

    .. rubric:: Temporal, Spatial, and Spectral Encoding

    - **Temporal:** RoPE attention over the token sequence of the windows.
    - **Spatial:** attention over the ``n_neuro`` sources.
    - **Spectral:** inherited from the tokenizer's convolutions.

    .. rubric:: Additional Mechanisms

    - The block outputs are L2-normalized before pooling, as released.
    - The RoPE cache holds cosines only for the first 240 positions, as in the
      released checkpoints; longer sequences rebuild it from ``freqs``.

    .. important::

        Weights converted from the released tiny and base checkpoints are on
        the Hugging Face Hub at ``braindecode/brainomni-tiny-pretrained`` and
        ``braindecode/brainomni-base-pretrained``; the head is not pretrained::

            from braindecode.models import BrainOmni
            model = BrainOmni.from_pretrained("braindecode/brainomni-tiny-pretrained", chs_info=raw.info["chs"], n_outputs=2)

        Input is expected at 256 Hz, preprocessed as in the authors' code.

    .. versionadded:: 1.8

    Parameters
    ----------
    emb_dim : int
        Tokenizer embedding dimension.
    n_neuro : int
        Number of latent source tokens.
    window_length : int
        Samples per tokenizer window.
    overlap_ratio : float
        Overlap between tokenizer windows, in ``[0, 1)``.
    n_filters : int
        Base number of SEANet filters.
    ratios : tuple of int
        SEANet downsampling ratios.
    kernel_size : int
        SEANet kernel size.
    last_kernel_size : int
        Kernel size of the first and last SEANet convolutions.
    tokenizer_num_heads : int
        Heads of the tokenizer cross-attention.
    codebook_dim : int
        Codebook dimension.
    codebook_size : int
        Entries per codebook.
    num_quantizers : int
        Number of residual VQ stages.
    rotation_trick : bool
        Rotation trick in the tokenizer quantizer.
    tokenizer_drop_prob : float
        Tokenizer attention dropout.
    lm_dim : int
        Transformer dimension.
    num_heads : int
        Transformer heads (even: half temporal, half spatial).
    depth : int
        Number of transformer blocks, the last one unused.
    drop_prob : float
        Transformer dropout.
    activation : type[nn.Module]
        Activation of the classification head.

    References
    ----------
    .. [brainomni] Xiao, Q., Cui, Z., Zhang, C., Chen, S., Wu, W.,
       Thwaites, A., Woolgar, A., Zhou, B., Zhang, C. (2025).
       BrainOmni: A Brain Foundation Model for Unified EEG and MEG Signals.
       NeurIPS 2025. https://arxiv.org/abs/2505.18185
    """

    def __init__(
        self,
        n_outputs=None,
        n_chans=None,
        chs_info=None,
        n_times=None,
        input_window_seconds=None,
        sfreq=None,
        *,
        emb_dim: int = 256,
        n_neuro: int = 16,
        window_length: int = 512,
        overlap_ratio: float = 0.25,
        n_filters: int = 32,
        ratios: tuple[int, ...] = (8, 4, 2),
        kernel_size: int = 5,
        last_kernel_size: int = 5,
        tokenizer_num_heads: int = 4,
        codebook_dim: int = 256,
        codebook_size: int = 512,
        num_quantizers: int = 4,
        rotation_trick: bool = True,
        tokenizer_drop_prob: float = 0.0,
        lm_dim: int = 256,
        num_heads: int = 8,
        depth: int = 12,
        drop_prob: float = 0.1,
        activation: type[nn.Module] = nn.SELU,
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
        if not 0 <= overlap_ratio < 1:
            raise ValueError(f"overlap_ratio must be in [0, 1), got {overlap_ratio}.")
        if lm_dim <= 0 or num_heads <= 0:
            raise ValueError(
                f"lm_dim ({lm_dim}) and num_heads ({num_heads}) must be positive."
            )
        if num_heads % 2 or lm_dim % num_heads or (lm_dim // num_heads) % 2:
            raise ValueError(
                f"num_heads ({num_heads}) must be even and divide lm_dim ({lm_dim}) "
                "into even head dimensions."
            )
        self.lm_dim = lm_dim
        self.n_neuro = n_neuro
        self.overlap_ratio = overlap_ratio
        self.activation = activation

        self.tokenizer = BrainTokenizer(
            chs_info=self.chs_info,
            n_times=self.n_times,
            sfreq=self.sfreq,
            emb_dim=emb_dim,
            n_neuro=n_neuro,
            window_length=window_length,
            n_filters=n_filters,
            ratios=ratios,
            kernel_size=kernel_size,
            last_kernel_size=last_kernel_size,
            tokenizer_num_heads=tokenizer_num_heads,
            codebook_dim=codebook_dim,
            codebook_size=codebook_size,
            num_quantizers=num_quantizers,
            rotation_trick=rotation_trick,
            drop_prob=tokenizer_drop_prob,
        )
        self.projection = (
            nn.Linear(emb_dim, lm_dim) if emb_dim != lm_dim else nn.Identity()
        )
        self.blocks = nn.ModuleList(
            [_SpatialTemporalBlock(lm_dim, num_heads, drop_prob) for _ in range(depth)]
        )
        self.apply(_init_weights)
        self.final_layer = self._make_head(self.n_outputs)  # default init, as released
        self.tokenizer.requires_grad_(False)

    def _make_head(self, n_outputs: int) -> nn.Module:
        return nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(self.n_neuro * self.lm_dim, self.lm_dim),
            self.activation(),
            nn.Linear(self.lm_dim, n_outputs),
        )

    def reset_head(self, n_outputs: int) -> None:
        """Replace the classification head for ``n_outputs`` classes."""
        self._set_n_outputs(n_outputs)
        reference = next(self.parameters())
        self.final_layer = self._make_head(n_outputs)
        self.final_layer.to(device=reference.device, dtype=reference.dtype)
        self.final_layer.train(self.training)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """L2-normalized ``(batch, n_neuro, n_tokens, lm_dim)`` embedding."""
        feat, _ = self.tokenizer.tokenize(x, overlap_ratio=self.overlap_ratio)
        neuro = self.tokenizer.encoder.neuros.detach().to(feat.dtype)
        h = self.projection(feat + neuro.view(1, feat.shape[1], 1, -1))
        for block in self.blocks[:-1]:
            h = block(h)
        return F.normalize(h, p=2.0, dim=-1, eps=1e-6)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        """Classify ``x`` or return the pooled features."""
        feat = self.encode(x).mean(dim=2).flatten(1)  # (batch, n_neuro * lm_dim)
        if return_features:
            return {"features": feat, "cls_token": None}  # nosec B105
        return self.final_layer(feat)


def _init_weights(module: nn.Module) -> None:
    """Released init: truncated normal (std 0.02) linears and embeddings."""
    if isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)
    elif isinstance(module, nn.Embedding):
        nn.init.trunc_normal_(module.weight, std=0.02)
    elif isinstance(module, RMSNorm):
        nn.init.constant_(module.weight, 1.0)


_SENSOR_CODE = {"eeg": 0, "mag": 1, "grad": 2}
# FIFF coil names by value, so plain-int coil types (serialized chs_info) resolve.
_COIL_NAMES = {
    int(value): name for name, value in FIFF.items() if name.startswith("FIFFV_COIL_")
}


def _coil_name(ch) -> str:
    return _COIL_NAMES.get(int(ch.get("coil_type", 0)), "")


def _sensor_type_of(chs_info, index: int) -> str:
    """EEG, MAG or GRAD as released: an MEG coil is MAG iff its name has "MAG".

    So CTF/KIT axial gradiometers are GRAD although MNE types them ``mag``.
    """
    ch = chs_info[index]
    kind = ch.get("kind")
    if kind is not None and not isinstance(kind, str):
        if int(kind) == FIFF.FIFFV_EEG_CH:
            return "eeg"
        if int(kind) == FIFF.FIFFV_MEG_CH and "coil_type" in ch:
            return "mag" if "MAG" in _coil_name(ch) else "grad"
        return mne.channel_type({"chs": chs_info}, index)
    resolved_type = ch.get("ch_type", kind)
    if resolved_type is None:
        resolved_type = mne.channel_type({"chs": chs_info}, index)
    return str(resolved_type).lower()


def _geometry_from_chs_info(chs_info):
    """``(pos (n_chans, 6) float32, sensor_type (n_chans,) int64)``, as released.

    Positions are centred and scaled per modality (EEG, MEG); the orientation
    is the coil x-axis for planar gradiometers, the coil normal for other MEG
    coils and zero for EEG.
    """
    types = [_sensor_type_of(chs_info, index) for index in range(len(chs_info))]
    xyz = extract_channel_locations_from_chs_info(chs_info)
    if xyz is None or len(xyz) != len(chs_info) or not np.isfinite(xyz).all():
        raise ValueError(
            "chs_info lacks finite sensor positions; call raw.set_montage(...)."
        )
    unsupported = set(types) - set(_SENSOR_CODE)
    if unsupported:
        raise ValueError(
            f"Unsupported channel type(s) {sorted(unsupported)}; pass only EEG/MEG."
        )
    sensor_type = np.array([_SENSOR_CODE[t] for t in types], dtype=np.int64)

    ori = np.zeros((len(chs_info), 3))
    for index in np.flatnonzero(sensor_type > 0):
        planar = "PLANAR" in _coil_name(chs_info[index])
        axis = np.asarray(chs_info[index]["loc"], dtype=np.float64)
        axis = axis[3:6] if sensor_type[index] == 2 and planar else axis[9:12]
        if axis.shape != (3,) or not np.isfinite(axis).all():
            raise ValueError("chs_info lacks a finite MEG coil orientation.")
        ori[index] = axis

    pos = np.concatenate([xyz, ori], axis=1).astype(np.float32)
    for mask in (sensor_type == 0, sensor_type > 0):
        if mask.any():
            centred = pos[mask, :3] - pos[mask, :3].mean(axis=0, keepdims=True)
            scale = np.sqrt(3 * np.mean(np.sum(centred**2, axis=1)))
            pos[mask, :3] = centred / (scale if scale > 0 else 1.0)
    return pos, sensor_type


def _attend(q, k, v, n_head: int, dropout_p: float, rope=None) -> torch.Tensor:
    """SDPA over ``(batch, seq, n_head * head_dim)`` inputs."""
    q, k, v = (t.unflatten(-1, (n_head, -1)) for t in (q, k, v))
    if rope is not None:
        q, k = rope(q, k)
    out = F.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), dropout_p=dropout_p
    )
    return out.transpose(1, 2).flatten(2)


class _SpatialTemporalBlock(nn.Module):
    """Temporal attention (RoPE) on one feature half, spatial on the other."""

    def __init__(self, n_dim, n_head, dropout):
        super().__init__()
        self.pre_attn_norm = RMSNorm(n_dim, eps=1e-6)
        self.time_attn = _MultiHeadAttentionRoPE(n_dim // 2, n_head // 2, dropout, True)
        self.spatial_attn = _MultiHeadAttentionRoPE(
            n_dim // 2, n_head // 2, dropout, False
        )
        self.pre_ff_norm = RMSNorm(n_dim, eps=1e-6)
        # Released FeedForward: Linear -> SELU -> Linear -> Dropout.
        self.ff = FeedForwardBlock(n_dim, 4, 0.0, nn.SELU, output_drop_p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, _, dim = x.shape
        h = self.pre_attn_norm(x)
        xs = rearrange(h[..., dim // 2 :], "b c t d -> (b t) c d")
        xt = rearrange(h[..., : dim // 2], "b c t d -> (b c) t d")
        xs = rearrange(self.spatial_attn(xs), "(b t) c d -> b c t d", b=batch)
        xt = rearrange(self.time_attn(xt), "(b c) t d -> b c t d", b=batch)
        # Spatial half first: the halves swap, as released.
        x = x + torch.cat([xs, xt], dim=-1)
        return x + self.ff(self.pre_ff_norm(x))


class _RotaryPositionalEmbedding(nn.Module):
    """RoPE of the released BrainOmni.

    ``rotate`` caches 240 positions as ``(cos, sin)`` pairs; the released
    checkpoints hold cosines only (zero sine). A longer sequence rebuilds the
    cache from ``freqs`` and later calls keep it (not saved). The caches stay
    float32 under a dtype cast, ``freqs`` follows it.
    """

    def __init__(self, n_dim, base=10000):
        super().__init__()
        self.n_dim = n_dim
        exponent = torch.arange(0, n_dim, 2).float() / n_dim
        self.register_buffer("freqs", 1.0 / (base**exponent))
        self.register_buffer("rotate", self._polar(240))
        self.register_buffer("rebuilt", None, persistent=False)

    def _polar(self, seq: int) -> torch.Tensor:
        positions = torch.arange(seq, device=self.freqs.device).type_as(self.freqs)
        angles = torch.outer(positions, self.freqs).float()
        return torch.stack((angles.cos(), angles.sin()), dim=-1)

    def _apply(self, fn, recurse=True):
        before = {name: self._buffers[name] for name in ("rotate", "rebuilt")}
        super()._apply(fn, recurse)
        for name, buffer in before.items():
            moved = self._buffers[name]
            if buffer is not None and moved is not None and moved.dtype != buffer.dtype:
                self._buffers[name] = buffer.to(device=moved.device)
        return self

    def _load_from_state_dict(self, *args, **kwargs):
        self.rebuilt = None
        super()._load_from_state_dict(*args, **kwargs)

    def forward(self, q, k):
        """Rotate ``q`` and ``k`` of shape ``(batch, seq, n_heads, head_dim)``."""
        _, seq, heads, _ = q.shape
        rotate = self.rotate if self.rebuilt is None else self.rebuilt
        if seq > rotate.shape[0]:
            self.rebuilt = rotate = self._polar(seq)
        rotate = rotate[:seq].repeat_interleave(2, dim=1)
        rotate = rearrange(rotate, "s (h d) two -> s h d two", h=heads)
        cos, sin = rotate[..., 0], rotate[..., 1]
        q_float, k_float = q.float(), k.float()
        q_out = q_float * cos + rotate_pairs(q_float) * sin
        k_out = k_float * cos + rotate_pairs(k_float) * sin
        return q_out.type_as(q), k_out.type_as(k)


class _MultiHeadAttentionRoPE(nn.Module):
    """Self-attention with a fused ``qkv`` projection and optional RoPE."""

    def __init__(self, n_dim, n_head, dropout, rope):
        super().__init__()
        self.dropout = dropout
        self.n_dim = n_dim
        self.n_head = n_head
        self.qkv = nn.Linear(n_dim, 3 * n_dim)
        self.proj = nn.Linear(n_dim, n_dim)
        self.rope_embedding_layer = _RotaryPositionalEmbedding(n_dim) if rope else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q, k, v = torch.split(self.qkv(x), self.n_dim, dim=-1)
        dropout_p = self.dropout if self.training else 0.0
        out = _attend(q, k, v, self.n_head, dropout_p, self.rope_embedding_layer)
        return self.proj(out)


class _ForwardSolution(nn.Module):
    """Cross-attention: sensor embeddings query the source tokens."""

    def __init__(self, n_dim: int, n_head: int, dropout: float) -> None:
        super().__init__()
        self.n_dim = n_dim
        self.n_head = n_head
        self.dropout = dropout
        self.kv = nn.Linear(n_dim, 2 * n_dim)
        self.proj = nn.Linear(n_dim, n_dim)

    def forward(self, sensor_embedding, neurons):
        k, v = torch.split(self.kv(neurons), self.n_dim, dim=-1)
        dropout_p = self.dropout if self.training else 0.0
        return self.proj(_attend(sensor_embedding, k, v, self.n_head, dropout_p))


class _BackwardSolution(nn.Module):
    """Cross-attention: source queries attend to the channels."""

    def __init__(self, n_dim: int, n_head: int, dropout: float) -> None:
        super().__init__()
        self.n_head = n_head
        self.dropout = dropout
        self.v = nn.Linear(n_dim, n_dim)
        self.proj = nn.Linear(n_dim, n_dim)

    def forward(self, neuros, k, x):
        dropout_p = self.dropout if self.training else 0.0
        return self.proj(_attend(neuros, k, self.v(x), self.n_head, dropout_p))


class _SensorEmbedding(nn.Module):
    """``(n_chans, 6)`` geometry and ``(n_chans,)`` type -> ``(n_chans, n_dim)``."""

    def __init__(self, n_dim: int) -> None:
        super().__init__()
        self.sensor_embedding_layer = nn.Embedding(3, n_dim)
        self.pos_embedding_layer = nn.Sequential(
            nn.Linear(6, n_dim // 2), nn.SELU(), nn.Linear(n_dim // 2, n_dim)
        )
        self.aggregate_mlp = FeedForwardBlock(n_dim, 4, 0.0, nn.SELU)
        self.norm = RMSNorm(n_dim, eps=1e-6)

    def forward(self, pos: torch.Tensor, sensor_type: torch.Tensor) -> torch.Tensor:
        x = self.pos_embedding_layer(pos)
        x = x + self.sensor_embedding_layer(sensor_type).type_as(x)
        x = x + self.aggregate_mlp(x)
        return self.norm(x)


class _TokenizerEncoder(nn.Module):
    """SEANet-encode each window, then attend ``n_neuro`` queries over channels."""

    def __init__(
        self,
        n_filters,
        ratios,
        kernel_size,
        last_kernel_size,
        n_dim,
        n_head,
        dropout,
        n_neuro,
    ) -> None:
        super().__init__()
        self.seanet_encoder = _seanet_encoder(
            n_dim, n_filters, ratios, kernel_size, last_kernel_size
        )
        self.neuros = nn.Parameter(torch.randn(n_neuro, n_dim))
        self.backwardsolution = _BackwardSolution(n_dim, n_head, dropout)
        self.k_proj = nn.Linear(n_dim, n_dim)

    def forward(self, x: torch.Tensor, sensor_embedding: torch.Tensor):
        batch, chans, nwin, _ = x.shape
        x = self.seanet_encoder(rearrange(x, "b c w l -> (b c w) 1 l"))
        x = rearrange(x, "(b c w) d t -> b c (w t) d", b=batch, c=chans, w=nwin)
        tokens = x.shape[2]
        sensor_embedding = rearrange(
            sensor_embedding.unsqueeze(2).repeat(1, 1, tokens, 1),
            "b c t d -> (b t) c d",
        )
        x = rearrange(x, "b c t d -> (b t) c d")
        neuros = self.neuros.type_as(x).unsqueeze(0).repeat(x.shape[0], 1, 1)
        x = self.backwardsolution(neuros, self.k_proj(x + sensor_embedding), x)
        x = rearrange(x, "(b w t) c d -> b c (w t) d", b=batch, w=nwin)
        return rearrange(x, "b c (w t) d -> b c w t d", w=nwin)


class _TokenizerDecoder(nn.Module):
    """Attend channels over the sources, then SEANet-decode each window."""

    def __init__(
        self,
        n_dim,
        n_head,
        n_filters,
        ratios,
        kernel_size,
        last_kernel_size,
        dropout,
    ) -> None:
        super().__init__()
        self.forwardsolution = _ForwardSolution(n_dim, n_head, dropout)
        self.seanet_decoder = _seanet_decoder(
            n_dim, n_filters, ratios, kernel_size, last_kernel_size
        )

    def forward(self, x: torch.Tensor, sensor_embedding: torch.Tensor):
        batch, _, nwin, tok, dim = x.shape
        x = rearrange(x, "b c w t d -> (b w t) c d")
        sensor_embedding = rearrange(
            sensor_embedding.view(batch, -1, 1, 1, dim).repeat(1, 1, nwin, tok, 1),
            "b c w t d -> (b w t) c d",
        )
        x = self.forwardsolution(sensor_embedding, x)
        x = rearrange(x, "(b w t) c d -> (b c w) d t", b=batch, w=nwin, t=tok)
        x = self.seanet_decoder(x)
        return rearrange(x, "(b c w) 1 l -> b c w l", b=batch, w=nwin)


class _SEANetConv1d(nn.Module):
    """Weight-normed Conv1d with EnCodec's non-causal reflect padding."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super().__init__()
        self.conv = weight_norm(
            nn.Conv1d(in_channels, out_channels, kernel_size, stride)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        kernel_size, stride = self.conv.kernel_size[0], self.conv.stride[0]
        padding_total = kernel_size - stride
        # Round the length up so the last strided window is full.
        extra = (kernel_size - padding_total - x.shape[-1]) % stride
        right = padding_total // 2
        padding = (padding_total - right, right + extra)
        # Zero-extend a signal too short to reflect, as EnCodec does.
        short = max(max(padding) - x.shape[-1] + 1, 0)
        x = F.pad(F.pad(x, (0, short)), padding, mode="reflect")
        return self.conv(x[..., : x.shape[-1] - short])


class _SEANetConvTranspose1d(nn.Module):
    """Weight-normed ConvTranspose1d trimmed as EnCodec (non-causal)."""

    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.convtr = weight_norm(
            nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride)
        )
        self.padding_total = kernel_size - stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.convtr(x)
        right = self.padding_total // 2
        return y[..., self.padding_total - right : y.shape[-1] - right]


class _SEANetLSTM(nn.Module):
    """Two-layer LSTM with a skip connection over ``(batch, dim, time)``."""

    def __init__(self, dimension: int):
        super().__init__()
        self.lstm = nn.LSTM(dimension, dimension, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(2, 0, 1)
        y, _ = self.lstm(x)
        return (y + x).permute(1, 2, 0)


class _SEANetResBlock(nn.Module):
    """``ELU -> conv3 -> ELU -> conv1`` plus a 1x1 conv shortcut."""

    def __init__(self, dim: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.ELU(),
            _SEANetConv1d(dim, dim // 2, 3),
            nn.ELU(),
            _SEANetConv1d(dim // 2, dim, 1),
        )
        self.shortcut = _SEANetConv1d(dim, dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.shortcut(x) + self.block(x)


def _seanet_encoder(dim, n_filters, ratios, kernel_size, last_kernel_size):
    """SEANet encoder of the release: ``(N, 1, L) -> (N, dim, L / prod(ratios))``."""
    layers = [_SEANetConv1d(1, n_filters, kernel_size)]
    mult = 1
    for ratio in reversed(ratios):
        layers += [
            _SEANetResBlock(mult * n_filters),
            nn.ELU(),
            _SEANetConv1d(mult * n_filters, 2 * mult * n_filters, 2 * ratio, ratio),
        ]
        mult *= 2
    layers += [
        _SEANetLSTM(mult * n_filters),
        nn.ELU(),
        _SEANetConv1d(mult * n_filters, dim, last_kernel_size),
    ]
    return nn.Sequential(*layers)


def _seanet_decoder(dim, n_filters, ratios, kernel_size, last_kernel_size):
    """Mirror of :func:`_seanet_encoder` with transposed convolutions."""
    mult = 2 ** len(ratios)
    layers = [_SEANetConv1d(dim, mult * n_filters, kernel_size)]
    layers += [_SEANetLSTM(mult * n_filters)]
    for ratio in ratios:
        out = mult * n_filters // 2
        layers += [
            nn.ELU(),
            _SEANetConvTranspose1d(mult * n_filters, out, 2 * ratio, ratio),
            _SEANetResBlock(out),
        ]
        mult //= 2
    layers += [nn.ELU(), _SEANetConv1d(n_filters, 1, last_kernel_size)]
    return nn.Sequential(*layers)
