# Authors: Hamdi Altaheri <haltaheri@uwaterloo.ca> (original implementation)
#          Bruno Aristimunha <b.aristimunha@gmail.com> (braindecode adaptation)
#
# License: MIT

from __future__ import annotations

import torch
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import Tensor, nn

from braindecode.models.base import EEGModuleMixin
from braindecode.modules import CausalConv1d, Conv1dWithConstraint, DropPath


class TCFormer(EEGModuleMixin, nn.Module):
    r"""TCFormer from Altaheri et al (2025) [tcformer]_.

    :bdg-success:`Convolution` :bdg-info:`Attention/Transformer`

    .. figure:: https://braindecode.org/dev/_static/model/tcformer.png
        :align: center
        :alt: TCFormer Architecture

    Temporal Convolutional Transformer for EEG-based motor-imagery decoding. It
    couples a multi-kernel convolutional front-end, a grouped-query Transformer
    encoder with rotary positional embeddings, and a grouped temporal
    convolutional network head, reaching state-of-the-art accuracy on BCIC IV-2a,
    IV-2b and the High-Gamma Dataset while remaining compact (~78k parameters)
    [tcformer]_.

    .. rubric:: Architecture Overview

    The raw trial ``(batch, n_chans, n_times)`` flows through three stages:
    (1) :class:`_MultiKernelConvBlock` extracts multi-scale spatiotemporal
    features and emits a short sequence of ``d_model`` tokens; (2) a stack of
    :class:`_TransformerBlock` layers models global temporal context with
    grouped-query attention and rotary embeddings; (3) the Transformer output is
    reduced and concatenated with the convolutional tokens, then a grouped
    :class:`_TCN` head with a per-group :class:`_ClassificationHead`
    (``final_layer``) produces class logits.

    .. rubric:: Macro Components

    ``TCFormer.conv_block`` (Multi-Kernel Convolutional Embedding)
        **Operations.** Parallel temporal convolutions with kernels
        ``temp_kernel_lengths`` (each ``n_filters_time`` filters, batch-normed),
        concatenated, then a depthwise spatial convolution over electrodes
        (``depth_multiplier``), average pooling, grouped 1x1 channel reduction to
        ``d_model = group_dim * n_groups``, a grouped temporal convolution, a
        grouped squeeze-and-excitation gate, and a second pooling.
        **Role.** Turns the raw montage into ``Tc`` compact feature tokens, one
        group per temporal kernel, encoding band-specific rhythms.

    ``TCFormer.transformer`` (Grouped-Query Transformer Encoder)
        **Operations.** ``n_transformer_layers`` pre-norm blocks, each applying
        grouped-query self-attention (``q_heads`` queries sharing ``kv_heads``
        key/value groups) with rotary positional embedding, then a position-wise
        MLP (expansion ``mlp_ratio``); residual connections use a quadratic
        DropPath schedule up to ``drop_path_max``.
        **Role.** Adds global temporal context efficiently (fewer K/V projections
        than full multi-head attention).

    ``TCFormer.tcn`` + ``TCFormer.final_layer`` (Grouped TCN Head)
        **Operations.** The reduced Transformer output is concatenated with the
        convolutional tokens (``d_fused = group_dim * (n_groups + 1)`` channels,
        ``n_groups + 1`` groups), processed by ``tcn_depth`` dilated causal
        residual blocks (kernel ``tcn_kernel_length``, dilations ``2**i``), the
        last timestep is taken, and a grouped 1x1 conv produces per-group logits
        averaged across groups.
        **Role.** Long-range causal temporal decoding and read-out.

    .. rubric:: Temporal, Spatial, and Spectral Encoding

    - **Temporal:** multi-kernel and dilated-causal convolutions plus rotary
      self-attention capture short- and long-range temporal dependencies.
    - **Spatial:** a depthwise convolution spanning all electrodes mixes channels
      per feature map.
    - **Spectral:** the three temporal kernel sizes target distinct EEG bands
      (short kernels -> high frequencies, long kernels -> low frequencies).

    .. rubric:: Additional Mechanisms

    - Grouped-query attention and rotary embeddings reduce attention cost while
      preserving relative-position information.
    - Group structure is preserved end-to-end: each temporal kernel forms a
      channel group that stays separated through the grouped reductions, grouped
      TCN, and per-group classifier.

    Parameters
    ----------
    n_filters_time : int, optional
        Number of temporal filters per kernel (``F1``). Default ``32``.
    temp_kernel_lengths : tuple of int, optional
        Temporal kernel lengths; their count is the number of feature groups.
        Default ``(20, 32, 64)``.
    depth_multiplier : int, optional
        Depthwise spatial expansion factor (``D``). Default ``2``.
    pool_length_1, pool_length_2 : int, optional
        Average-pool factors after the depthwise and the second temporal conv.
        Defaults ``8`` and ``7``.
    temp_kernel_length_2 : int, optional
        Kernel length of the grouped second temporal convolution. Default ``16``.
    group_dim : int, optional
        Channels per group (``d_group``); ``d_model = group_dim * n_groups``.
        Default ``16``.
    se_reduction : int, optional
        Reduction ratio of the grouped squeeze-and-excitation block. Default ``4``.
    n_transformer_layers : int, optional
        Number of encoder layers (``N``). Default ``2`` (the 77.8k-parameter
        headline configuration). Use ``5`` for the deeper ~131k variant.
    q_heads, kv_heads : int, optional
        Number of query heads and key/value groups for grouped-query attention
        (``q_heads`` must be divisible by ``kv_heads``). Defaults ``4`` and ``2``.
    mlp_ratio : int, optional
        Feed-forward expansion ratio in each encoder block. Default ``2``.
    drop_path_max : float, optional
        Maximum stochastic-depth rate (quadratic schedule over depth). Default
        ``0.25``.
    tcn_depth : int, optional
        Number of TCN residual blocks (``L``). Default ``2``.
    tcn_kernel_length : int, optional
        Kernel length of the TCN causal convolutions (``KT``). Default ``4``.
    classifier_max_norm : float, optional
        Max-norm constraint on the classifier convolution weights. Default
        ``0.25``.
    drop_prob_conv, drop_prob_trans, drop_prob_tcn : float, optional
        Dropout probabilities of the conv front-end, the Transformer, and the TCN
        head. Defaults ``0.4``, ``0.4``, ``0.3``.
    activation : nn.Module, optional
        Activation class for the convolutional and TCN blocks. Default
        :class:`torch.nn.ELU`.
    activation_ffn : nn.Module, optional
        Activation class for the Transformer feed-forward sublayer. Default
        :class:`torch.nn.GELU`.

    Notes
    -----
    This implementation is adapted from the original source code [tcformercode]_
    to comply with braindecode's model conventions. The default configuration
    reproduces the paper's headline (Table 1) setup; the released training config
    additionally uses Adam (lr 0.0009, weight_decay 1e-3), linear warm-up + cosine
    decay, per-channel z-scoring, and segmentation-and-reconstruction augmentation
    (handled outside the model, in the training pipeline).

    .. versionadded:: 1.6.1

    References
    ----------
    .. [tcformer] Altaheri, H., Karray, F., & Karimi, A.-H. (2025). Temporal
       convolutional transformer for EEG based motor imagery decoding. Scientific
       Reports, 15, 32959. https://doi.org/10.1038/s41598-025-16219-7
    .. [tcformercode] Altaheri, H. (2025). TCFormer source code.
       https://github.com/Altaheri/TCFormer
    """

    def __init__(
        self,
        # Signal related parameters
        n_outputs=None,
        n_chans=None,
        chs_info=None,
        n_times=None,
        input_window_seconds=None,
        sfreq=None,
        # Model parameters
        n_filters_time: int = 32,
        temp_kernel_lengths: tuple[int, ...] = (20, 32, 64),
        depth_multiplier: int = 2,
        pool_length_1: int = 8,
        pool_length_2: int = 7,
        temp_kernel_length_2: int = 16,
        group_dim: int = 16,
        se_reduction: int = 4,
        n_transformer_layers: int = 2,
        q_heads: int = 4,
        kv_heads: int = 2,
        mlp_ratio: int = 2,
        drop_path_max: float = 0.25,
        tcn_depth: int = 2,
        tcn_kernel_length: int = 4,
        classifier_max_norm: float = 0.25,
        drop_prob_conv: float = 0.4,
        drop_prob_trans: float = 0.4,
        drop_prob_tcn: float = 0.3,
        activation: type[nn.Module] = nn.ELU,
        activation_ffn: type[nn.Module] = nn.GELU,
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

        n_groups = len(temp_kernel_lengths)
        d_model = group_dim * n_groups
        d_fused = group_dim * (n_groups + 1)

        self.conv_block = _MultiKernelConvBlock(
            self.n_chans,
            temp_kernel_lengths,
            n_filters_time,
            depth_multiplier,
            pool_length_1,
            pool_length_2,
            temp_kernel_length_2,
            drop_prob_conv,
            group_dim,
            se_reduction,
            activation,
        )

        self.mix = nn.Sequential(
            nn.Conv1d(d_model, d_model, 1, bias=False),
            nn.BatchNorm1d(d_model),
            nn.SiLU(),
        )
        self.to_tokens = Rearrange("batch feat time -> batch time feat")

        drop_rates = torch.linspace(0, 1, n_transformer_layers) ** 2 * drop_path_max
        self.transformer = nn.ModuleList(
            [
                _TransformerBlock(
                    d_model,
                    q_heads,
                    kv_heads,
                    mlp_ratio,
                    drop_prob_trans,
                    float(drop_rates[i]),
                    activation_ffn,
                )
                for i in range(n_transformer_layers)
            ]
        )

        # RoPE tables are built per forward from the actual token length.
        self.head_dim = d_model // q_heads

        self.reduce = nn.Sequential(
            Rearrange("batch time feat -> batch feat time"),
            nn.Conv1d(d_model, group_dim, 1, bias=False),
            nn.BatchNorm1d(group_dim),
            nn.SiLU(),
        )

        self.tcn = _TCN(
            tcn_depth,
            tcn_kernel_length,
            d_fused,
            n_groups + 1,
            drop_prob_tcn,
            activation,
        )
        self.final_layer = _ClassificationHead(
            d_fused, n_groups + 1, self.n_outputs, classifier_max_norm
        )
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            params = getattr(module, "parametrizations", None)
            if params is not None and "weight" in params:
                # max-norm layers expose a computed .weight; init the leaf
                nn.init.xavier_uniform_(params.weight.original)
            elif isinstance(getattr(module, "weight", None), nn.Parameter):
                nn.init.xavier_uniform_(module.weight)
            if getattr(module, "bias", None) is not None:
                nn.init.constant_(module.bias, 0.0)

    def forward(self, x: Tensor) -> Tensor:
        conv_features = self.conv_block(x)  # (batch, d_model, n_tokens)
        tokens = self.to_tokens(self.mix(conv_features))  # (batch, n_tokens, d_model)
        seq_len = tokens.shape[1]
        cos, sin = _build_rotary_cache(
            self.head_dim, seq_len, device=tokens.device, dtype=tokens.dtype
        )
        for block in self.transformer:
            tokens = block(tokens, cos, sin)
        transformer_features = self.reduce(tokens)  # (batch, group_dim, n_tokens)
        fused = torch.cat(
            (conv_features, transformer_features), dim=1
        )  # (batch, d_fused, n_tokens)
        decoded = self.tcn(fused)
        last_step = decoded[:, :, -1:]  # (batch, d_fused, 1)
        return self.final_layer(last_step)  # (batch, n_outputs)


# ----------------------------------------------------------------------------- #
# Rotary positional embedding (verbatim port of the reference; see plan note on
# the deliberate NeoX-cache / interleaved-rotate mismatch).
def _build_rotary_cache(
    head_dim: int, seq_len: int, device=None, dtype=None
) -> tuple[Tensor, Tensor]:
    """Return ``(cos, sin)`` each of shape ``(seq_len, head_dim)``.

    Angles are computed in float32 for precision; the returned cos/sin are
    cast to ``dtype`` (the token dtype) so attention stays in the working
    precision under autocast.
    """
    theta = 1.0 / (
        10000 ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim)
    )
    seq_idx = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(seq_idx, theta)  # (seq_len, head_dim/2)
    emb = torch.cat((freqs, freqs), dim=-1)  # (seq_len, head_dim)
    cos, sin = emb.cos(), emb.sin()
    if dtype is not None:
        cos, sin = cos.to(dtype), sin.to(dtype)
    return cos, sin


def _apply_rope(
    query: Tensor, key: Tensor, cos: Tensor, sin: Tensor
) -> tuple[Tensor, Tensor]:
    """Apply rotary embedding to ``query`` and ``key``.

    Both have shape ``(batch, n_heads, seq_len, head_dim)``.
    """

    def rotate(tensor: Tensor) -> Tensor:
        even = tensor[..., 0::2]
        odd = tensor[..., 1::2]
        pairs = torch.stack((-odd, even), dim=-1)
        return rearrange(pairs, "... half two -> ... (half two)")

    query_rotated = (query * cos) + (rotate(query) * sin)
    key_rotated = (key * cos) + (rotate(key) * sin)
    return query_rotated, key_rotated


# ----------------------------------------------------------------------------- #
class _GroupedQueryAttention(nn.Module):
    """Grouped-query self-attention (``q_heads >= kv_heads``) with RoPE."""

    def __init__(
        self, d_model: int, q_heads: int, kv_heads: int, drop_prob: float = 0.4
    ):
        super().__init__()
        assert d_model % q_heads == 0, "d_model must be divisible by q_heads"
        assert q_heads % kv_heads == 0, "q_heads must be a multiple of kv_heads"
        self.n_query_heads = q_heads
        self.n_kv_heads = kv_heads
        self.head_dim = d_model // q_heads
        assert self.head_dim % 2 == 0, (
            "head_dim (d_model // q_heads) must be even for rotary embeddings"
        )
        self.scale = self.head_dim**-0.5
        self.query_proj = nn.Linear(d_model, d_model, bias=False)
        self.key_value_proj = nn.Linear(
            d_model, 2 * kv_heads * self.head_dim, bias=False
        )
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.drop = nn.Dropout(drop_prob)

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        seq_len = x.shape[1]
        query = rearrange(
            self.query_proj(x),
            "batch seq (heads dim) -> batch heads seq dim",
            heads=self.n_query_heads,
        )
        key, value = rearrange(
            self.key_value_proj(x),
            "batch seq (heads two dim) -> two batch heads seq dim",
            two=2,
            heads=self.n_kv_heads,
        )
        # share each key/value group across its query heads
        groups_per_kv = self.n_query_heads // self.n_kv_heads
        key = repeat(
            key,
            "batch heads seq dim -> batch (heads repeats) seq dim",
            repeats=groups_per_kv,
        )
        value = repeat(
            value,
            "batch heads seq dim -> batch (heads repeats) seq dim",
            repeats=groups_per_kv,
        )
        query, key = _apply_rope(query, key, cos[:seq_len], sin[:seq_len])
        key_t = rearrange(key, "batch heads seq dim -> batch heads dim seq")
        attention = ((query @ key_t) * self.scale).softmax(dim=-1)
        attention = self.drop(attention)
        context = rearrange(
            attention @ value, "batch heads seq dim -> batch seq (heads dim)"
        )
        return self.out_proj(context)


# ----------------------------------------------------------------------------- #
class _TransformerBlock(nn.Module):
    """Pre-norm encoder block: GQA + position-wise MLP, both with DropPath."""

    def __init__(
        self,
        d_model: int,
        q_heads: int,
        kv_heads: int,
        mlp_ratio: int = 2,
        drop_prob: float = 0.4,
        drop_path_rate: float = 0.0,
        activation: type[nn.Module] = nn.GELU,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = _GroupedQueryAttention(d_model, q_heads, kv_heads, drop_prob)
        self.drop_path = DropPath(drop_path_rate)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_ratio * d_model),
            activation(),
            nn.Linear(mlp_ratio * d_model, d_model),
            nn.Dropout(drop_prob),
        )

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x), cos, sin))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# ----------------------------------------------------------------------------- #
class _ChannelGroupAttention(nn.Module):
    """Grouped Squeeze-and-Excitation: one sigmoid gate per channel group."""

    def __init__(self, in_channels: int, num_groups: int, reduction: int = 4):
        super().__init__()
        assert in_channels % num_groups == 0
        self.in_channels = in_channels
        self.num_groups = num_groups
        self.group_size = in_channels // num_groups
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.att_fc1 = nn.Conv2d(
            in_channels, in_channels // reduction, 1, groups=num_groups, bias=False
        )
        self.att_fc2 = nn.Conv2d(
            in_channels // reduction, num_groups, 1, groups=num_groups, bias=False
        )
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        descriptor = self.pool(x)
        gate = self.sigmoid(self.att_fc2(self.relu(self.att_fc1(descriptor))))
        grouped = rearrange(
            x,
            "batch (groups chans) height width -> batch groups chans height width",
            groups=self.num_groups,
        )
        gate = rearrange(
            gate, "batch groups height width -> batch groups 1 height width"
        )
        return rearrange(
            grouped * gate,
            "batch groups chans height width -> batch (groups chans) height width",
        )


# ----------------------------------------------------------------------------- #
class _MultiKernelConvBlock(nn.Module):
    """Multi-kernel temporal + depthwise spatial CNN producing feature tokens."""

    def __init__(
        self,
        n_chans: int,
        temp_kernel_lengths: tuple[int, ...],
        n_filters_time: int,
        depth_multiplier: int,
        pool_length_1: int,
        pool_length_2: int,
        temp_kernel_length_2: int,
        drop_prob: float,
        group_dim: int,
        se_reduction: int,
        activation: type[nn.Module],
    ):
        super().__init__()
        n_groups = len(temp_kernel_lengths)
        self.d_model = group_dim * n_groups
        self.rearrange = Rearrange("batch nchans time -> batch 1 nchans time")

        # one temporal conv per kernel ('same' time padding via ConstantPad2d),
        # BN only -- NO activation here (matches reference).
        self.temporal_convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.ConstantPad2d(
                        (k // 2 - 1, k // 2, 0, 0)
                        if k % 2 == 0
                        else (k // 2, k // 2, 0, 0),
                        0,
                    ),
                    nn.Conv2d(1, n_filters_time, (1, k), bias=False),
                    nn.BatchNorm2d(n_filters_time),
                )
                for k in temp_kernel_lengths
            ]
        )

        in_dw = n_filters_time * n_groups
        f2 = in_dw * depth_multiplier
        self.channel_dw_conv = nn.Sequential(
            nn.Conv2d(in_dw, f2, (n_chans, 1), bias=False, groups=in_dw),
            nn.BatchNorm2d(f2),
            activation(),
        )
        self.pool1 = nn.AvgPool2d((1, pool_length_1))
        self.drop1 = nn.Dropout(drop_prob)

        # grouped 1x1 channel reduction f2 -> d_model (one group per kernel)
        self.use_channel_reduction = self.d_model != f2
        if self.use_channel_reduction:
            self.channel_reduction = nn.Sequential(
                nn.Conv2d(f2, self.d_model, (1, 1), bias=False, groups=n_groups),
                nn.BatchNorm2d(self.d_model),
            )

        self.temporal_conv_2 = nn.Sequential(
            nn.Conv2d(
                self.d_model,
                self.d_model,
                (1, temp_kernel_length_2),
                padding="same",
                bias=False,
                groups=n_groups,
            ),
            nn.BatchNorm2d(self.d_model),
            activation(),
        )

        self.use_group_attn = n_groups > 1
        if self.use_group_attn:
            self.group_attn = _ChannelGroupAttention(
                self.d_model, n_groups, se_reduction
            )

        self.pool2 = nn.AvgPool2d((1, pool_length_2))
        self.drop2 = nn.Dropout(drop_prob)

    def forward(self, x: Tensor) -> Tensor:
        x = self.rearrange(x)
        x = torch.cat([conv(x) for conv in self.temporal_convs], dim=1)
        x = self.channel_dw_conv(x)
        x = self.drop1(self.pool1(x))
        if self.use_channel_reduction:
            x = self.channel_reduction(x)
        x = self.temporal_conv_2(x)
        if self.use_group_attn:
            x = x + self.group_attn(x)
        x = self.drop2(self.pool2(x))
        # drop the singleton spatial axis -> (batch, d_model, n_tokens)
        return rearrange(x, "batch feat 1 time -> batch feat time")


# ----------------------------------------------------------------------------- #
class _TCNResidualBlock(nn.Module):
    """Two grouped dilated-causal convs (BN + activation + dropout) + residual."""

    def __init__(
        self,
        n_filters: int,
        kernel_length: int,
        dilation: int,
        n_groups: int,
        drop_prob: float,
        activation: type[nn.Module],
    ):
        super().__init__()
        self.conv1 = CausalConv1d(
            n_filters, n_filters, kernel_length, dilation=dilation, groups=n_groups
        )
        self.bn1 = nn.BatchNorm1d(n_filters)
        self.act1 = activation()
        self.drop1 = nn.Dropout(drop_prob)
        self.conv2 = CausalConv1d(
            n_filters, n_filters, kernel_length, dilation=dilation, groups=n_groups
        )
        self.bn2 = nn.BatchNorm1d(n_filters)
        self.act2 = activation()
        self.drop2 = nn.Dropout(drop_prob)
        self.act3 = activation()

    def forward(self, x: Tensor) -> Tensor:
        out = self.drop1(self.act1(self.bn1(self.conv1(x))))
        out = self.drop2(self.act2(self.bn2(self.conv2(out))))
        return self.act3(x + out)


class _TCN(nn.Module):
    """Stack of ``depth`` residual blocks with exponentially growing dilation."""

    def __init__(
        self,
        depth: int,
        kernel_length: int,
        n_filters: int,
        n_groups: int,
        drop_prob: float,
        activation: type[nn.Module],
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                _TCNResidualBlock(
                    n_filters, kernel_length, 2**i, n_groups, drop_prob, activation
                )
                for i in range(depth)
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        for blk in self.blocks:
            x = blk(x)
        return x


# ----------------------------------------------------------------------------- #
class _ClassificationHead(nn.Module):
    """Grouped 1x1 conv producing per-group logits, averaged across groups."""

    def __init__(
        self, d_features: int, n_groups: int, n_outputs: int, max_norm: float = 0.25
    ):
        super().__init__()
        self.n_groups = n_groups
        self.n_outputs = n_outputs
        self.conv = Conv1dWithConstraint(
            d_features,
            n_outputs * n_groups,
            kernel_size=1,
            groups=n_groups,
            max_norm=max_norm,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x).squeeze(-1)  # (B, n_outputs*n_groups)
        return x.view(x.size(0), self.n_groups, self.n_outputs).mean(dim=1)
