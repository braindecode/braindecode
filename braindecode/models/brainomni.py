# Authors: Qian Xiao and OpenTSLab BrainOmni contributors
#          Bruno Aristimunha <b.aristimunha@gmail.com> (braindecode adaptation)
#
# License: MIT
#
# Adapted from https://github.com/OpenTSLab/BrainOmni (MIT); the SEANet codec
# derives from Meta's EnCodec (MIT).
from __future__ import annotations

from functools import partial
from typing import Dict, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import RMSNorm

from braindecode.functional import check_n_chans, init_weights_
from braindecode.models.base import EEGModuleMixin
from braindecode.models.util import (
    geometry_from_chs_info,
    sensor_type_of,
    warn_if_sfreq_differs,
)
from braindecode.modules import (
    FeedForwardBlock,
    SEANetDecoder,
    SEANetEncoder,
    SpatialTemporalBlock,
)
from braindecode.modules.quantization import ResidualVectorQuantizer
from braindecode.modules.spatial_temporal import _SELU, _attend


class BrainTokenizer(EEGModuleMixin, nn.Module, license="mit"):
    r"""BrainTokenizer from Xiao et al. (2025) [brainomni]_.

    :bdg-danger:`Foundation Model` :bdg-info:`Attention/Transformer`

    .. figure:: ../_static/model/braintokenizer_arch.png
       :align: center
       :alt: BrainTokenizer architecture
       :width: 1000px

       BrainTokenizer training as an autoencoder, with its encoder,
       reconstructor and sensor encoder, reproduced from [brainomni]_
       (CC BY 4.0).

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

    .. versionadded:: 2.0

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
    channel_strategy : str
        How another EEG montage reaches the encoder; see
        :class:`~braindecode.models.base.EEGModuleMixin`. The channel layer maps
        electrodes only, so a ``chs_info`` with MEG sensors takes only
        ``"native"``. BrainTokenizer builds on any EEG/MEG montage from the sensor
        geometry of ``chs_info`` (the weights do not depend on it).
    channel_strategy_kwargs : dict or None
        Options of ``channel_strategy``.

    References
    ----------
    .. [brainomni] Xiao, Q., Cui, Z., Zhang, C., Chen, S., Wu, W.,
       Thwaites, A., Woolgar, A., Zhou, B., Zhang, C. (2025).
       BrainOmni: A Brain Foundation Model for Unified EEG and MEG Signals.
       NeurIPS 2025. https://arxiv.org/abs/2505.18185
    """

    _time_axis = "free"
    _channel_axis = "fixed"  # fixed: pos is sized by n_chans

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
        channel_strategy: str = "native",
        channel_strategy_kwargs: dict | None = None,
    ):
        _check_channel_strategy(self, chs_info, channel_strategy)
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
        warn_if_sfreq_differs("BrainTokenizer", self.sfreq, 256)
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
        pos, sensor_type = geometry_from_chs_info(self.chs_info)
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
        self._min_n_times = 1
        self._init_parameters()

    def _init_parameters(self) -> None:
        """Model-specific initialization (see ``reset_parameters``)."""
        self.apply(_init_weights)

    def _encode_quantize(self, x: torch.Tensor, overlap_ratio: float = 0.0):
        check_n_chans(
            x,
            self.n_chans,
            "BrainTokenizer",
            "build the model with the chs_info of this montage.",
        )
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
        return feat.flatten(2, 3), indices.flatten(2, 3)


class BrainOmni(EEGModuleMixin, nn.Module, license="mit"):
    r"""BrainOmni from Xiao et al. (2025) [brainomni]_.

    :bdg-danger:`Foundation Model` :bdg-info:`Attention/Transformer`

    .. figure:: ../_static/model/brainomni_arch.png
       :align: center
       :alt: BrainOmni architecture
       :width: 1000px

       BrainOmni pretraining: masked BrainTokenizer tokens are predicted by
       stacked criss-cross Transformer blocks, reproduced from [brainomni]_
       (CC BY 4.0).

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

    .. versionadded:: 2.0

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
    channel_strategy : str
        How another EEG montage reaches the encoder; see
        :class:`~braindecode.models.base.EEGModuleMixin`. The channel layer maps
        electrodes only, so a ``chs_info`` with MEG sensors takes only
        ``"native"``. BrainOmni builds on any EEG/MEG montage from the sensor
        geometry of ``chs_info`` (the weights do not depend on it).
    channel_strategy_kwargs : dict or None
        Options of ``channel_strategy``.

    References
    ----------
    .. [brainomni] Xiao, Q., Cui, Z., Zhang, C., Chen, S., Wu, W.,
       Thwaites, A., Woolgar, A., Zhou, B., Zhang, C. (2025).
       BrainOmni: A Brain Foundation Model for Unified EEG and MEG Signals.
       NeurIPS 2025. https://arxiv.org/abs/2505.18185
    """

    _time_axis = "free"
    _channel_axis = "fixed"  # fixed: tokenizer.pos is sized by n_chans

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
        channel_strategy: str = "native",
        channel_strategy_kwargs: dict | None = None,
    ):
        _check_channel_strategy(self, chs_info, channel_strategy)
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
            [SpatialTemporalBlock(lm_dim, num_heads, drop_prob) for _ in range(depth)]
        )
        self._init_parameters()
        self.final_layer = self._make_head(self.n_outputs)  # default init, as released
        self.tokenizer.requires_grad_(False)
        self._min_n_times = 1

    def _init_parameters(self) -> None:
        """Model-specific initialization (see ``reset_parameters``)."""
        # The head is built after this, with PyTorch's default init, as released.
        for name, child in self.named_children():
            if name != "final_layer":
                child.apply(_init_weights)

    def _make_head(self, n_outputs: int) -> nn.Module:
        return nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(self.n_neuro * self.lm_dim, self.lm_dim),
            # nn.SELU (the released config) computed as _SELU, the scale == 1 ELU form
            (_SELU if self.activation is nn.SELU else self.activation)(),
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
        check_n_chans(
            x,
            self.n_chans,
            "BrainOmni",
            "build the model with the chs_info of this montage.",
        )
        if torch.jit.is_scripting():
            # TorchScript cannot switch modes, so tokenize() is not scriptable.
            if self.tokenizer.training:
                raise RuntimeError("Scripted BrainOmni needs tokenizer.eval().")
            with torch.no_grad():
                feat = self.tokenizer._encode_quantize(x, self.overlap_ratio)[0]
            feat = feat.flatten(2, 3)
        else:
            feat, _ = self.tokenizer.tokenize(x, overlap_ratio=self.overlap_ratio)
        neuro = self.tokenizer.encoder.neuros.detach().to(feat.dtype)
        h = self.projection(feat + neuro.view(1, feat.shape[1], 1, -1))
        for i, block in enumerate(self.blocks):  # the last block is unused, as released
            if i < len(self.blocks) - 1:
                h = block(h)
        return F.normalize(h, p=2.0, dim=-1, eps=1e-6)

    def forward(
        self, x: torch.Tensor, return_features: bool = False
    ) -> Union[torch.Tensor, Dict[str, Optional[torch.Tensor]]]:
        """Classify ``x`` or return the pooled features."""
        feat = self.encode(x).mean(dim=2).flatten(1)  # (batch, n_neuro * lm_dim)
        if return_features:
            out: Dict[str, Optional[torch.Tensor]] = {
                "features": feat,
                "cls_token": None,  # nosec B105
            }
            return out
        return self.final_layer(feat)


# Released init: truncated normal (std 0.02) linears and embeddings.
_init_weights = partial(
    init_weights_,
    weights={(nn.Linear, nn.Embedding): partial(nn.init.trunc_normal_, std=0.02)},
    norms=(RMSNorm,),
)


def _check_channel_strategy(model, chs_info, channel_strategy):
    """The channel layer maps electrodes only: a montage with MEG stays native."""
    if channel_strategy == "native" or not chs_info:
        return
    other = [
        ch["ch_name"]
        for index, ch in enumerate(chs_info)
        if sensor_type_of(chs_info, index) != "eeg"
    ]
    if other:
        raise ValueError(
            f"{type(model).__name__} reads EEG and MEG, but the channel layer maps "
            f"electrodes only and chs_info has other sensors ({other}): only "
            f"channel_strategy='native' is supported for them. "
            f"{type(model).__name__} builds on any EEG/MEG montage from the sensor "
            f"geometry of chs_info."
        )


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
            nn.Linear(6, n_dim // 2), _SELU(), nn.Linear(n_dim // 2, n_dim)
        )
        self.aggregate_mlp = FeedForwardBlock(n_dim, 4, 0.0, _SELU)
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
        self.seanet_encoder = SEANetEncoder(
            n_dim, n_filters, ratios, kernel_size, last_kernel_size
        )
        self.neuros = nn.Parameter(torch.randn(n_neuro, n_dim))
        self.backwardsolution = _BackwardSolution(n_dim, n_head, dropout)
        self.k_proj = nn.Linear(n_dim, n_dim)

    def forward(self, x: torch.Tensor, sensor_embedding: torch.Tensor):
        batch, chans, nwin, length = x.shape
        x = self.seanet_encoder(x.reshape(batch * chans * nwin, 1, length))
        # (b c w) d t -> b c (w t) d
        x = x.reshape(batch, chans, nwin, x.shape[1], -1).permute(0, 1, 2, 4, 3)
        x = x.reshape(batch, chans, -1, x.shape[-1])
        tokens, dim = x.shape[2], x.shape[3]
        # b c t d -> (b t) c d
        sensor_embedding = sensor_embedding.unsqueeze(2).repeat(1, 1, tokens, 1)
        sensor_embedding = sensor_embedding.permute(0, 2, 1, 3).reshape(-1, chans, dim)
        x = x.permute(0, 2, 1, 3).reshape(-1, chans, dim)
        neuros = self.neuros.type_as(x).unsqueeze(0).repeat(x.shape[0], 1, 1)
        x = self.backwardsolution(neuros, self.k_proj(x + sensor_embedding), x)
        # (b w t) c d -> b c (w t) d -> b c w t d
        x = x.reshape(batch, nwin, -1, x.shape[1], dim).permute(0, 3, 1, 2, 4)
        x = x.reshape(batch, x.shape[1], -1, dim)
        return x.reshape(batch, x.shape[1], nwin, -1, dim)


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
        self.seanet_decoder = SEANetDecoder(
            n_dim, n_filters, ratios, kernel_size, last_kernel_size
        )

    def forward(self, x: torch.Tensor, sensor_embedding: torch.Tensor):
        batch, sources, nwin, tok, dim = x.shape
        # b c w t d -> (b w t) c d
        x = x.permute(0, 2, 3, 1, 4).reshape(-1, sources, dim)
        sensor_embedding = sensor_embedding.view(batch, -1, 1, 1, dim)
        chans = sensor_embedding.shape[1]
        sensor_embedding = sensor_embedding.repeat(1, 1, nwin, tok, 1)
        sensor_embedding = sensor_embedding.permute(0, 2, 3, 1, 4).reshape(
            -1, chans, dim
        )
        x = self.forwardsolution(sensor_embedding, x)
        # (b w t) c d -> (b c w) d t
        x = x.reshape(batch, nwin, tok, chans, dim).permute(0, 3, 1, 4, 2)
        x = self.seanet_decoder(x.reshape(-1, dim, tok))
        # (b c w) 1 l -> b c w l
        return x.reshape(batch, chans, nwin, x.shape[-1])
