# Authors: Vandit Shah <shahvanditt@gmail.com>
#
# Code adapted from https://github.com/jingyingma01/CodeBrain
#
# License: BSD (3-clause)

import math
from typing import Optional

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn
from torch.nn.utils.parametrizations import weight_norm

contract = torch.einsum

from braindecode.models.base import EEGModuleMixin


class _GConv(nn.Module):
    """Sparse Global Convolution (SGConv) structured state-space layer.

    Implements the SGConv layer from Section 3.3 of [codebrain]_. As described
    in the paper, SGConv improves the convolution kernel by introducing two
    features: *sparse parameterization* and *kernel decay*, making it "easier
    and more efficient to compute compared to the traditional S4 kernel."

    The convolution is computed efficiently in :math:`O(N \\log N)` via FFT:

    .. math::

        y = \\mathcal{F}^{-1}_N D_k \\mathcal{F}_N u, \\quad D_k = \\text{diag}(K \\mathcal{F}_N)

    The multi-scale kernel :math:`K` is composed of ``num_scales`` sub-kernels
    of base dimension ``kernel_dim``, each upsampled and decayed by a learnable
    multiplier :math:`\\alpha`:

    .. math::

        K = \\frac{1}{Z} [k_0, k_1, \\ldots, k_{N-1}], \\quad
        k_i = \\alpha^i \\, \\text{Upsample}_{2^{\\max(i-1,0) d}}(w_i)

    where :math:`Z` is a normalisation constant that ensures the convolution
    does not change the scale of the input.

    Parameters
    ----------
    d_model : int
        Number of hidden features (model width).
    d_state : int, default=64
        SSM state dimension (unused in forward, kept for API compatibility).
    l_max : int, default=1
        Maximum sequence length; determines the number of multi-scale
        sub-kernels as ``1 + ceil(log2(l_max / kernel_dim)) - init_scale``.
    channels : int, default=1
        Number of output channels (multiplies ``d_model``).
    bidirectional : bool, default=False
        If ``True``, applies the kernel in both forward and backward directions
        and sums the results.
    activation : str, default="gelu"
        Activation function after the convolution output (``"gelu"`` or
        ``"relu"``).
    dropout : float, default=0.0
        Dropout probability applied after activation.
    transposed : bool, default=True
        If ``True``, expects input of shape ``(batch, d_model, seq_len)``;
        otherwise ``(batch, seq_len, d_model)``.
    linear : bool, default=False
        If ``True``, skips the activation, dropout, norm, and output linear
        projection — returns the raw convolution output.
    mode : str, default="cat_randn"
        Kernel initialisation and concatenation strategy. ``"cat_randn"``
        initialises sub-kernels with random normal values and concatenates
        them along the last dimension.
    layer_norm : bool, default=False
        If ``True``, applies LayerNorm before the output linear projection.
    **kernel_args
        Optional keyword arguments: ``init_scale`` (int, default 0),
        ``kernel_dim`` (int, default 2), ``n_scales`` (int, overrides
        automatic scale count), ``decay_min`` (float, default 2),
        ``decay_max`` (float, default 2).

    References
    ----------
    .. [codebrain] Ding et al. (2025). CodeBrain: Scalable Code EEG
       Pre-Training for Unified Downstream BCI Tasks.
       https://arxiv.org/abs/2506.09110
    """

    def __init__(
        self,
        d_model,
        d_state=64,
        l_max=1,
        channels=1,
        bidirectional=False,
        activation="gelu",
        dropout=0.0,
        transposed=True,
        shift=False,
        linear=False,
        mode="cat_randn",
        layer_norm=False,
        init_scale: int = 0,
        kernel_dim: int = 2,
        n_scales: Optional[int] = None,
        decay_min: float = 2.0,
        decay_max: float = 2.0,
    ):
        super().__init__()
        self.h = d_model
        self.n = d_state
        self.bidirectional = bidirectional
        self.channels = channels
        self.transposed = transposed
        self.shift = shift
        self.linear = linear
        self.mode = mode
        self.l_max = l_max

        self.D = nn.Parameter(torch.randn(channels, self.h))

        if self.bidirectional:
            channels *= 2

        if not self.linear:
            self.activation = nn.GELU() if activation == "gelu" else nn.ReLU()
            self.dropout = nn.Dropout2d(dropout) if dropout > 0.0 else nn.Identity()
            self.norm = (
                nn.LayerNorm(self.h * self.channels) if layer_norm else nn.Identity()
            )

        if not self.linear:
            self.output_linear = nn.Linear(self.h * self.channels, self.h)

        self.init_scale = init_scale
        self.kernel_dim = kernel_dim
        self.num_scales = (
            n_scales
            if n_scales is not None
            else 1 + math.ceil(math.log2(l_max / self.kernel_dim)) - self.init_scale
        )

        self.kernel_list = nn.ParameterList()
        for _ in range(self.num_scales):
            if "randn" in mode:
                kernel = torch.randn(channels, self.h, self.kernel_dim)
            elif "cos" in mode:
                kernel = torch.cat(
                    [
                        torch.cos(
                            torch.linspace(0, 2 * i * math.pi, self.kernel_dim)
                        ).expand(channels, 1, self.kernel_dim)
                        for i in range(self.h)
                    ],
                    dim=1,
                )[:, torch.randperm(self.h), :]
            else:
                raise ValueError(f"Unknown mode {mode}")
            self.kernel_list.append(nn.Parameter(kernel))

        self.register_buffer(
            "multiplier",
            torch.linspace(decay_min, decay_max, self.h).view(1, -1, 1),
        )
        self.register_buffer("kernel_norm", torch.ones(channels, self.h, 1))
        self.register_buffer(
            "kernel_norm_initialized", torch.tensor(0, dtype=torch.bool)
        )

    def forward(self, u, return_kernel=False):
        if not self.transposed:
            u = u.transpose(-1, -2)
        L = u.size(-1)

        kernel_list = []
        interpolate_mode = "nearest" if "nearest" in self.mode else "linear"
        multiplier = self.multiplier

        if "cat" in self.mode:
            for i in range(self.num_scales):
                kernel = F.interpolate(
                    self.kernel_list[i],
                    scale_factor=2 ** (max(0, i - 1) + self.init_scale),
                    mode=interpolate_mode,
                ) * multiplier ** (self.num_scales - i - 1)
                kernel_list.append(kernel)
            k = torch.cat(kernel_list, dim=-1)
        else:
            raise ValueError(f"Unknown mode {self.mode}")

        if not self.kernel_norm_initialized:
            self.kernel_norm = k.norm(dim=-1, keepdim=True).detach()
            self.kernel_norm_initialized = torch.tensor(
                1, dtype=torch.bool, device=k.device
            )

        if k.size(-1) > L:
            k = k[..., :L]
        elif k.size(-1) < L:
            k = F.pad(k, (0, L - k.size(-1)))

        k = k / self.kernel_norm

        if self.bidirectional:
            k0, k1 = rearrange(k, "(s c) h l -> s c h l", s=2)
            k = F.pad(k0, (0, L)) + F.pad(k1.flip(-1), (L, 0))

        k_f = torch.fft.rfft(k.float(), n=2 * L)
        u_f = torch.fft.rfft(u.float(), n=2 * L)
        y_f = contract("bhl,chl->bchl", u_f, k_f)
        y = torch.fft.irfft(y_f, n=2 * L)[..., :L]

        y = y + contract("bhl,ch->bchl", u, self.D)
        y = rearrange(y, "... c h l -> ... (c h) l")

        if not self.linear:
            y = self.dropout(self.activation(y))
            y = rearrange(y, "b c l -> b l c")
            y = self.norm(y)
            y = self.output_linear(y)
            y = rearrange(y, "b l c -> b c l")

        if not self.transposed:
            y = y.transpose(-1, -2)

        if return_kernel:
            return y, k
        return y, None


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super(Conv, self).__init__()
        self.padding = dilation * (kernel_size - 1) // 2
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            dilation=dilation,
            padding=self.padding,
        )
        self.conv = weight_norm(self.conv)
        nn.init.kaiming_normal_(self.conv.parametrizations.weight.original1)

    def forward(self, x):
        out = self.conv(x)
        return out


class ZeroConv1d(nn.Module):
    # initializing the conv layers
    def __init__(self, in_channel, out_channel):
        super(ZeroConv1d, self).__init__()
        self.conv = nn.Conv1d(in_channel, out_channel, kernel_size=1, padding=0)

    def forward(self, x):
        out = self.conv(x)
        return out


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super(RMSNorm, self).__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(1, dim, 1))

    def forward(self, x):
        norm = x.norm(dim=1, keepdim=True)
        rms = norm / (x.shape[1] ** 0.5)
        x_normed = x / (rms + self.eps)
        return self.scale * x_normed


class ResidualBlock(nn.Module):
    def __init__(
        self,
        res_channels,
        skip_channels,
        s4_lmax,
        s4_d_state,
        s4_dropout,
        s4_bidirectional,
        s4_layernorm,
        swa_window_size: int = 1,
    ):
        super(ResidualBlock, self).__init__()
        self.res_channels = res_channels
        self.swa_window_size = swa_window_size

        self.sn = RMSNorm(res_channels)

        self.s41 = _GConv(
            d_model=2 * self.res_channels,
            channels=4,
            l_max=s4_lmax,
            d_state=s4_d_state,
            dropout=s4_dropout,
            bidirectional=s4_bidirectional,
            layer_norm=s4_layernorm,
        )

        self.conv_layer = Conv(self.res_channels, 2 * self.res_channels, kernel_size=3)

        self.attention = nn.MultiheadAttention(
            embed_dim=2 * self.res_channels,
            num_heads=4,
            dropout=s4_dropout,
            bias=True,
            batch_first=True,
        )

        self.gelu = nn.GELU()  # used between conv and SSM layers

        self.res_conv = nn.Conv1d(res_channels, res_channels, kernel_size=1)
        self.res_conv = weight_norm(self.res_conv)
        nn.init.kaiming_normal_(self.res_conv.parametrizations.weight.original1)

        self.skip_conv = nn.Conv1d(res_channels, skip_channels, kernel_size=1)
        self.skip_conv = weight_norm(self.skip_conv)
        nn.init.kaiming_normal_(self.skip_conv.parametrizations.weight.original1)

    def generate_local_window_mask(self, seq_len, window_size):
        assert window_size % 2 == 1, "window_size should be odd number, like 7, 9, 11"

        half_window = window_size // 2

        mask = torch.full((seq_len, seq_len), float("-inf"))

        for i in range(seq_len):
            start = max(0, i - half_window)
            end = min(seq_len, i + half_window + 1)
            mask[i, start:end] = 0

        return mask

    def forward(self, input_data):
        x, original = input_data
        h = x
        _, C, L = x.shape
        x = self.sn(x)
        assert C == self.res_channels

        part_t = rearrange(original, "b c l -> b c l")
        h = h + part_t

        h = self.conv_layer(h)

        h = self.gelu(h)
        h_t, _ = self.s41(h)

        h_s = rearrange(h_t, "b c l -> b l c")
        swa_mask = self.generate_local_window_mask(L, self.swa_window_size).to(x.device)
        h_s, _ = self.attention(h_s, h_s, h_s, attn_mask=swa_mask)
        h_s = rearrange(h_s, "b l c -> b c l")

        h = h_t + h_s

        out = torch.tanh(h[:, : self.res_channels, :]) * torch.sigmoid(
            h[:, self.res_channels :, :]
        )

        res = self.res_conv(out)
        assert x.shape == res.shape
        skip = self.skip_conv(out)

        return (x + res) * math.sqrt(0.5), skip


class ResidualGroup(nn.Module):
    """Stack of EEGSSM residual blocks with aggregated skip connections.

    Implements the multi-block EEGSSM backbone described in Section 3.3 of
    [codebrain]_. Each :class:`ResidualBlock` receives both the running hidden
    state ``h`` and the original input ``x`` (passed as ``noise`` below),
    processes them through RMSNorm -> Conv -> SGConv -> SWA -> gating, and emits
    a residual update and a skip signal.

    All skip signals are summed across blocks and normalised by
    :math:`\\sqrt{1 / \\text{num\\_res\\_layers}}` to keep the output scale
    stable regardless of depth, following the WaveNet convention.

    Parameters
    ----------
    res_channels : int
        Number of channels in the residual stream.
    skip_channels : int
        Number of channels in the aggregated skip stream.
    num_res_layers : int
        Number of stacked :class:`ResidualBlock` modules.
    s4_lmax : int
        Maximum sequence length for the SGConv kernel inside each block.
    s4_d_state : int
        State dimension of the SGConv layer.
    s4_dropout : float
        Dropout probability inside SGConv and sliding-window attention.
    s4_bidirectional : bool
        Whether SGConv processes the sequence bidirectionally.
    s4_layernorm : bool
        Whether to apply LayerNorm inside SGConv.
    swa_window_size : int, default=1
        Window size for sliding-window attention in each block.
    """

    def __init__(
        self,
        res_channels,
        skip_channels,
        num_res_layers,
        s4_lmax,
        s4_d_state,
        s4_dropout,
        s4_bidirectional,
        s4_layernorm,
        swa_window_size: int = 1,
    ):
        super(ResidualGroup, self).__init__()
        self.num_res_layers = num_res_layers

        self.residual_blocks = nn.ModuleList()
        for _ in range(self.num_res_layers):
            self.residual_blocks.append(
                ResidualBlock(
                    res_channels,
                    skip_channels,
                    s4_lmax=s4_lmax,
                    s4_d_state=s4_d_state,
                    s4_dropout=s4_dropout,
                    s4_bidirectional=s4_bidirectional,
                    s4_layernorm=s4_layernorm,
                    swa_window_size=swa_window_size,
                )
            )

    def forward(self, input_data):
        noise = input_data
        h = noise
        skip = 0
        for n in range(self.num_res_layers):
            h, skip_n = self.residual_blocks[n]((h, noise))
            skip = skip_n + skip

        return skip * math.sqrt(1.0 / self.num_res_layers)


class PatchEmbedding(nn.Module):
    """Dual temporal-spectral patch embedding for EEG signals.

    Converts a windowed EEG tensor of shape
    ``(batch, n_chans, seq_len, patch_size)`` into a patch embedding of shape
    ``(batch, n_chans, seq_len, emb_dim)`` by combining two complementary
    feature streams, following the TFDual-Tokenizer design of [codebrain]_:

    1. **Temporal projection** (``proj_in``): A three-layer Conv2d stack
       projects each patch into ``emb_dim = conv_out_chans * _t`` features,
       where ``_t`` is the temporal output length of the first strided
       convolution. GroupNorm and GELU are applied after each layer.

    2. **Spectral projection** (``spectral_proj``): The magnitude spectrum of
       each patch is computed via ``torch.fft.rfft`` and projected to
       ``emb_dim`` with a linear layer and dropout.

    The two embeddings are summed element-wise, then a depth-wise Conv2d
    (``positional_encoding``) adds dynamic positional information that captures
    inter-channel spatial relationships, as described in Section 3.3:
    *"we first learn dynamic positional embeddings that capture inter-channel
    relationships through a lightweight convolutional module."*

    If a binary ``mask`` is provided, masked patches are replaced with a
    learnable ``mask_encoding`` vector before temporal projection (used during
    pre-training).

    Parameters
    ----------
    conv_out_chans : int
        Number of output channels for each Conv2d layer in ``proj_in``.
        Also used as the stride of the first conv, so the temporal output
        length is ``(patch_size + 2*proj_padding - proj_kernel_size) //
        conv_out_chans + 1``.
    patch_size : int
        Number of time samples per patch.
    conv_groups : int
        Number of groups for GroupNorm in ``proj_in``.
    proj_kernel_size : int, default=49
        Kernel size of the first Conv2d in ``proj_in``.
    proj_padding : int, default=24
        Padding of the first Conv2d in ``proj_in``.
    proj_refine_kernel : int, default=3
        Kernel size for the two refinement Conv2d layers in ``proj_in``.
    pos_kernel : tuple of int, default=(19, 7)
        Kernel size ``(height, width)`` for the depth-wise positional
        encoding Conv2d.
    spectral_dropout : float, default=0.1
        Dropout probability in the spectral projection.

    Attributes
    ----------
    emb_dim : int
        Embedding dimension, derived from ``conv_out_chans`` and
        ``proj_kernel_size``/``proj_padding``.
    """

    def __init__(
        self,
        conv_out_chans,
        patch_size,
        conv_groups,
        proj_kernel_size: int = 49,
        proj_padding: int = 24,
        proj_refine_kernel: int = 3,
        pos_kernel: tuple = (19, 7),
        spectral_dropout: float = 0.1,
    ):
        super().__init__()
        proj_refine_padding = proj_refine_kernel // 2
        pos_padding = (pos_kernel[0] // 2, pos_kernel[1] // 2)
        # emb_dim derived from first Conv2d output: any change to proj_kernel_size,
        # proj_padding, or conv_out_chans (used as stride) must update this formula
        _t = (patch_size + 2 * proj_padding - proj_kernel_size) // conv_out_chans + 1
        self.emb_dim = conv_out_chans * _t
        self.d_model = self.emb_dim
        self.positional_encoding = nn.Sequential(
            nn.Conv2d(
                in_channels=self.emb_dim,
                out_channels=self.emb_dim,
                kernel_size=pos_kernel,
                stride=(1, 1),
                padding=pos_padding,
                groups=self.emb_dim,
            ),
        )
        self.mask_encoding = nn.Parameter(
            torch.zeros(self.emb_dim), requires_grad=False
        )

        self.proj_in = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=conv_out_chans,
                kernel_size=(1, proj_kernel_size),
                stride=(1, conv_out_chans),
                padding=(0, proj_padding),
            ),
            nn.GroupNorm(conv_groups, conv_out_chans),
            nn.GELU(),
            nn.Conv2d(
                in_channels=conv_out_chans,
                out_channels=conv_out_chans,
                kernel_size=(1, proj_refine_kernel),
                stride=(1, 1),
                padding=(0, proj_refine_padding),
            ),
            nn.GroupNorm(conv_groups, conv_out_chans),
            nn.GELU(),
            nn.Conv2d(
                in_channels=conv_out_chans,
                out_channels=conv_out_chans,
                kernel_size=(1, proj_refine_kernel),
                stride=(1, 1),
                padding=(0, proj_refine_padding),
            ),
            nn.GroupNorm(conv_groups, conv_out_chans),
            nn.GELU(),
        )
        self.spectral_proj = nn.Sequential(
            nn.Linear(patch_size // 2 + 1, self.emb_dim),
            nn.Dropout(spectral_dropout),
        )

    def forward(self, x, mask=None):
        batch, n_chans, seq_len, patch_size = x.shape
        if mask is None:
            mask_x = x
        else:
            mask_x = x.clone()
            mask_x[mask == 1] = self.mask_encoding

        # Merge channel and patch dims so Conv2d sees one image row per (channel, patch)
        # (batch, 1, n_chans * seq_len, patch_size)
        mask_x = mask_x.contiguous().view(batch, 1, n_chans * seq_len, patch_size)

        # Temporal projection: conv stack outputs (batch, conv_ch, n_chans * seq_len, emb)
        patch_emb = self.proj_in(mask_x)

        # Restore channel and patch dims, flatten conv channels into embedding dim
        # (batch, n_chans, seq_len, conv_ch * emb) = (batch, n_chans, seq_len, emb_dim)
        patch_emb = rearrange(
            patch_emb,
            "batch conv_ch (n_chans seq_len) emb -> batch n_chans seq_len (conv_ch emb)",
            n_chans=n_chans,
            seq_len=seq_len,
        )

        # Flatten patches for FFT: (batch * n_chans * seq_len, patch_size)
        patches_flat = rearrange(
            x, "batch n_chans seq_len patch_size -> (batch n_chans seq_len) patch_size"
        )

        # Spectral projection: rfft gives (batch * n_chans * seq_len, patch_size // 2 + 1)
        spectral = torch.abs(
            torch.fft.rfft(patches_flat.float(), dim=-1, norm="forward")
        )

        # Restore batch/channel/patch dims: (batch, n_chans, seq_len, freq_bins)
        spectral = rearrange(
            spectral,
            "(batch n_chans seq_len) freq -> batch n_chans seq_len freq",
            batch=batch,
            n_chans=n_chans,
            seq_len=seq_len,
        )

        # Project frequency features to emb_dim and add to temporal embedding
        spectral_emb = self.spectral_proj(spectral)
        patch_emb = patch_emb + spectral_emb

        # Positional encoding expects (batch, emb_dim, n_chans, seq_len)
        patch_emb = patch_emb + rearrange(
            self.positional_encoding(
                rearrange(
                    patch_emb,
                    "batch n_chans seq_len emb_dim -> batch emb_dim n_chans seq_len",
                )
            ),
            "batch emb_dim n_chans seq_len -> batch n_chans seq_len emb_dim",
        )

        return patch_emb


class CodeBrain(EEGModuleMixin, nn.Module):
    r"""CodeBrain: Scalable Code EEG Pre-Training for Unified Downstream BCI Tasks.

    :bdg-danger:`Foundation Model` :bdg-info:`Attention/Transformer`

    .. figure:: https://raw.githubusercontent.com/jingyingma01/CodeBrain/refs/heads/main/assets/intro.png
        :align: center
        :alt: CodeBrain pre-training overview
        :width: 1000px

    CodeBrain is a foundation model for EEG that pre-trains on large unlabelled
    corpora using a two-stage vector-quantised masking strategy, then fine-tunes
    on downstream BCI tasks. It segments EEG signals into fixed-size patches,
    embeds them with convolutional and spectral projections, and processes them
    through stacked residual blocks that combine a multi-scale convolutional
    structured state-space model (``_GConv``) with sliding-window self-attention.

    .. rubric:: Stage 2: EEGSSM Backbone (this implementation)

    This class implements Stage 2 of CodeBrain — the EEGSSM backbone described
    in Section 3.3 of [codebrain]_. Following :class:`Labram`, CodeBrain
    discretises EEG patches into codebook tokens via VQ-VAE (Stage 1, not
    implemented here), then trains the backbone to predict masked token indices
    via cross-entropy. CodeBrain extends this with a *dual* tokenizer that
    decouples temporal and frequency representations, as stated in the paper:
    *"the TFDual-Tokenizer, which decouples heterogeneous temporal and frequency
    EEG signals into discrete tokens to enhance discriminative power."*

    .. rubric:: Macro Components

    - **PatchEmbedding**: Splits ``(batch, n_chans, n_times)`` into
      ``(batch, n_chans, seq_len, patch_size)`` patches, projects each patch
      with a 2-D convolutional stack, adds FFT-based spectral embeddings, and
      applies depth-wise convolutional positional encoding.
    - **Residual blocks** (``ResidualGroup``): Each block applies RMSNorm,
      a ``_GConv`` SSM layer, and sliding-window multi-head attention, with
      gated activation and separate residual/skip paths.
    - **Classification head** (``final_layer``): Flattens the output and maps
      to ``n_outputs`` classes.

    .. rubric:: Pre-training vs Fine-tuning

    Set ``pretrain_mode=True`` during pre-training to return ``(lm_head_t,
    lm_head_f)`` codebook logits. Set ``pretrain_mode=False`` (default) for
    fine-tuning to return ``(batch, n_outputs)`` class logits.

    Parameters
    ----------
    patch_size : int, default=200
        Number of time samples per patch. Input length is trimmed to the
        nearest multiple of ``patch_size``.
    res_channels : int, default=200
        Width of the residual stream inside each ``ResidualBlock``.
    skip_channels : int, default=200
        Width of the skip-connection stream aggregated across blocks.
    out_channels : int, default=200
        Output channels of ``final_conv`` before the classification head.
    num_res_layers : int, default=8
        Number of stacked ``ResidualBlock`` modules.
    drop_prob : float, default=0.1
        Dropout rate used inside the ``_GConv`` SSM and attention layers.
    s4_bidirectional : bool, default=True
        Whether the ``_GConv`` SSM processes the sequence bidirectionally.
    s4_layernorm : bool, default=False
        Whether to apply layer normalisation inside the ``_GConv`` SSM.
        Set to ``False`` to match the released pretrained checkpoint.
    s4_lmax : int, default=570
        Maximum sequence length for the ``_GConv`` SSM kernel. Also determines
        the patch embedding dimension as ``s4_lmax // n_chans``.
    s4_d_state : int, default=64
        State dimension of the ``_GConv`` SSM.
    conv_out_chans : int, default=25
        Number of output channels in the patch projection convolutions.
    conv_groups : int, default=5
        Number of groups for ``GroupNorm`` in the patch projection.
    codebook_size_t : int, default=4096
        Vocabulary size for the temporal codebook head (pre-training only).
    codebook_size_f : int, default=4096
        Vocabulary size for the spectral codebook head (pre-training only).
    pretrain_mode : bool, default=False
        If ``True``, returns ``(lm_head_t, lm_head_f)`` logits for masked
        pre-training. If ``False``, returns ``(batch, n_outputs)`` logits.
    activation : type[nn.Module], default=nn.ReLU
        Non-linear activation class used in ``init_conv`` and ``final_conv``.

    References
    ----------
    .. [codebrain] Yi Ding, Xuyang Chen, Yong Li, Rui Yan, Tao Wang, Le Wu (2025).
       CodeBrain: Scalable Code EEG Pre-Training for Unified Downstream BCI Tasks.
       https://arxiv.org/abs/2506.09110
    """

    def __init__(
        self,
        n_outputs=None,
        n_chans=None,
        chs_info=None,
        n_times=None,
        input_window_seconds=None,
        sfreq=None,
        # Model specific arguments
        patch_size: int = 200,
        res_channels: int = 200,
        skip_channels: int = 200,
        out_channels: int = 200,
        num_res_layers: int = 8,
        drop_prob: float = 0.1,
        s4_bidirectional: bool = True,
        s4_layernorm: bool = False,
        s4_lmax: int = 570,
        s4_d_state: int = 64,
        conv_out_chans: int = 25,
        conv_groups: int = 5,
        proj_kernel_size: int = 49,
        proj_padding: int = 24,
        proj_refine_kernel: int = 3,
        pos_kernel: tuple = (19, 7),
        spectral_dropout: float = 0.1,
        mlp_hidden_multiplier: int = 4,
        swa_window_size: int = 1,
        codebook_size_t: int = 4096,
        codebook_size_f: int = 4096,
        pretrain_mode: bool = False,
        activation: type[nn.Module] = nn.ReLU,
    ):
        super().__init__(
            n_outputs=n_outputs,
            n_chans=n_chans,
            chs_info=chs_info,
            n_times=n_times,
            input_window_seconds=input_window_seconds,
            sfreq=sfreq,
        )
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.activation = activation
        self.patch_embedding = PatchEmbedding(
            patch_size=patch_size,
            conv_out_chans=conv_out_chans,
            conv_groups=conv_groups,
            proj_kernel_size=proj_kernel_size,
            proj_padding=proj_padding,
            proj_refine_kernel=proj_refine_kernel,
            pos_kernel=pos_kernel,
            spectral_dropout=spectral_dropout,
        )
        emb_dim = self.patch_embedding.emb_dim

        self.init_conv = nn.Sequential(
            Conv(emb_dim, res_channels, kernel_size=1), self.activation()
        )

        self.residual_layer = ResidualGroup(
            res_channels=res_channels,
            skip_channels=skip_channels,
            num_res_layers=num_res_layers,
            s4_lmax=s4_lmax,
            s4_d_state=s4_d_state,
            s4_dropout=drop_prob,
            s4_bidirectional=s4_bidirectional,
            s4_layernorm=s4_layernorm,
            swa_window_size=swa_window_size,
        )
        self.final_conv = nn.Sequential(
            Conv(skip_channels, skip_channels, kernel_size=1),
            self.activation(),
            ZeroConv1d(skip_channels, out_channels),
        )
        self.lm_head_t = nn.Linear(out_channels, codebook_size_t, bias=False)
        self.lm_head_f = nn.Linear(out_channels, codebook_size_f, bias=False)
        self.pretrain_mode = pretrain_mode
        self.norm = nn.LayerNorm(out_channels)
        # 3-layer MLP classifier as described in the paper (Section 3.3)
        _flat = self.n_chans * (self.n_times // self.patch_size) * self.out_channels
        self.final_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(_flat, mlp_hidden_multiplier * out_channels),
            nn.ELU(),
            nn.Dropout(drop_prob),
            nn.Linear(mlp_hidden_multiplier * out_channels, out_channels),
            nn.ELU(),
            nn.Dropout(drop_prob),
            nn.Linear(out_channels, self.n_outputs),
        )

    def forward(self, inputs, mask=None):
        batch, n_chans, n_times = inputs.shape
        patch_size = self.patch_size
        seq_len = n_times // patch_size
        inputs = inputs[:, :, : seq_len * patch_size].reshape(
            batch, n_chans, seq_len, patch_size
        )
        inputs = self.patch_embedding(inputs, mask=mask)
        x = rearrange(
            inputs, "batch n_chans seq_len emb_dim -> batch emb_dim (n_chans seq_len)"
        )
        x = self.init_conv(x)
        x = self.residual_layer(x)
        x = self.final_conv(x)
        x = rearrange(
            x,
            "batch out_channels (n_chans seq_len) -> batch n_chans seq_len out_channels",
            n_chans=n_chans,
            seq_len=seq_len,
        )
        x = self.norm(x)
        if self.pretrain_mode:
            if mask is not None:
                x = x[mask == 1]
            x_t = self.lm_head_t(x)
            x_f = self.lm_head_f(x)
            return (x_t, x_f)
        else:
            return self.final_layer(x)
