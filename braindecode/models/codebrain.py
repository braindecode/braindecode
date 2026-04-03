# Authors: Vandit Shah <shahvanditt@gmail.com>
#
# Code adapted from https://github.com/jingyingma01/CodeBrain
#
# License: BSD (3-clause)

import math
import re
from collections import OrderedDict
from typing import Optional

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn
from torch.nn.utils.parametrizations import weight_norm

from braindecode.models.base import EEGModuleMixin


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

    .. important::
       **Pre-trained Weights Available**

       This model has pre-trained weights available on the Hugging Face Hub.
       You can load them using:

       .. code-block:: python

           from braindecode.models import CodeBrain

           # Load pre-trained model from Hugging Face Hub
           model = CodeBrain.from_pretrained("braindecode/codebrain-pretrained")

       To push your own trained model to the Hub:

       .. code-block:: python

           model.push_to_hub("my-username/my-codebrain")

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

        # ========== Parameters ==========
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.activation = activation
        self.pretrain_mode = pretrain_mode

        # ========== Layers ==========
        # Dual temporal-spectral patch embedding
        self.patch_embedding = _PatchEmbedding(
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

        # Input projection
        self.init_conv = nn.Sequential(
            _Conv(emb_dim, res_channels, kernel_size=1), self.activation()
        )

        # EEGSSM residual backbone
        self.residual_layer = _ResidualGroup(
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

        # Output projection
        self.final_conv = nn.Sequential(
            _Conv(skip_channels, skip_channels, kernel_size=1),
            self.activation(),
            _ZeroConv1d(skip_channels, out_channels),
        )
        self.norm = nn.LayerNorm(out_channels)

        # Pre-training heads (codebook prediction)
        self.lm_head_t = nn.Linear(out_channels, codebook_size_t, bias=False)
        self.lm_head_f = nn.Linear(out_channels, codebook_size_f, bias=False)

        # Classification head (3-layer MLP, Section 3.3)
        flat_dim = self.n_chans * (self.n_times // self.patch_size) * self.out_channels
        self.final_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, mlp_hidden_multiplier * out_channels),
            nn.ELU(),
            nn.Dropout(drop_prob),
            nn.Linear(mlp_hidden_multiplier * out_channels, out_channels),
            nn.ELU(),
            nn.Dropout(drop_prob),
            nn.Linear(out_channels, self.n_outputs),
        )

    def load_state_dict(self, state_dict, *args, **kwargs):
        """Remap upstream checkpoint keys to braindecode attribute names.

        Handles four kinds of key differences from the original CodeBrain
        checkpoint (``YjMajy/CodeBrain`` on HuggingFace):

        1. ``module.`` prefix from ``DataParallel`` saving.
        2. Attribute renames: ``S41`` -> ``sgconv``, ``sn`` -> ``rms_norm``,
           ``.D`` -> ``.skip_weight``.
        3. ``KernelModule`` wrapper: ``.kernel_list.N.kernel`` -> ``.kernel_list.N``.
        4. Old ``weight_norm`` API: ``weight_g``/``weight_v`` ->
           ``parametrizations.weight.original0``/``original1``.
        """
        remapped = OrderedDict()
        for key, value in state_dict.items():
            new_key = key.removeprefix("module.")
            new_key = new_key.replace(".S41.", ".sgconv.")
            new_key = new_key.replace(".sn.", ".rms_norm.")
            if new_key.endswith(".D"):
                new_key = new_key[:-2] + ".skip_weight"
            new_key = re.sub(
                r"\.kernel_list\.(\d+)\.kernel$", r".kernel_list.\1", new_key
            )
            new_key = new_key.replace(".weight_g", ".parametrizations.weight.original0")
            new_key = new_key.replace(".weight_v", ".parametrizations.weight.original1")
            remapped[new_key] = value
        return super().load_state_dict(remapped, *args, **kwargs)

    def forward(self, inputs, mask=None, return_features=False):
        # inputs: (batch, n_chans, n_times)
        batch, n_chans, n_times = inputs.shape
        patch_size = self.patch_size
        seq_len = n_times // patch_size

        # Trim to nearest multiple of patch_size and reshape into patches
        # (batch, n_chans, seq_len, patch_size)
        x = inputs[:, :, : seq_len * patch_size].reshape(
            batch, n_chans, seq_len, patch_size
        )
        # Dual temporal-spectral patch embedding
        # (batch, n_chans, seq_len, emb_dim)
        x = self.patch_embedding(x, mask=mask)

        # Flatten channel and patch dims for 1-D convolution backbone
        # (batch, emb_dim, n_chans * seq_len)
        x = rearrange(
            x, "batch n_chans seq_len emb_dim -> batch emb_dim (n_chans seq_len)"
        )
        # (batch, res_channels, n_chans * seq_len)
        x = self.init_conv(x)
        # Residual SSM + attention blocks → aggregated skip connections
        # (batch, skip_channels, n_chans * seq_len)
        x = self.residual_layer(x)
        # (batch, out_channels, n_chans * seq_len)
        x = self.final_conv(x)

        # Restore channel and patch dims
        # (batch, n_chans, seq_len, out_channels)
        x = rearrange(
            x,
            "batch out_channels (n_chans seq_len) -> batch n_chans seq_len out_channels",
            n_chans=n_chans,
            seq_len=seq_len,
        )
        x = self.norm(x)

        if return_features:
            return {"features": x, "cls_token": None}
        if self.pretrain_mode:
            if mask is not None:
                x = x[mask == 1]
            x_t = self.lm_head_t(x)
            x_f = self.lm_head_f(x)
            return (x_t, x_f)
        # (batch, n_outputs)
        return self.final_layer(x)


# =============================================================================
# Private helper modules
# =============================================================================


class _GConv(nn.Module):
    """Sparse Global Convolution (SGConv) structured state-space layer.

    Implements the SGConv layer from Section 3.3 of [codebrain_sgconv]_. As
    described in the paper, SGConv improves the convolution kernel by introducing
    two features: *sparse parameterization* and *kernel decay*, making it "easier
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
    activation : type[nn.Module], default=nn.GELU
        Activation function class applied after the convolution output.
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
    .. [codebrain_sgconv] Ding et al. (2025). CodeBrain: Scalable Code EEG
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
        activation: type[nn.Module] = nn.GELU,
        dropout=0.0,
        transposed=True,
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

        # ========== Parameters ==========
        self.d_model = d_model
        self.channels = channels
        self.bidirectional = bidirectional
        self.transposed = transposed
        self.linear = linear
        self.mode = mode
        self.l_max = l_max
        self.init_scale = init_scale
        self.kernel_dim = kernel_dim

        # ========== Derived attributes ==========
        n_conv_channels = channels * 2 if self.bidirectional else channels
        self.num_scales = (
            n_scales
            if n_scales is not None
            else 1 + math.ceil(math.log2(l_max / self.kernel_dim)) - self.init_scale
        )

        # ========== Buffers ==========
        self.register_buffer(
            "multiplier",
            torch.linspace(decay_min, decay_max, self.d_model).view(1, -1, 1),
        )
        self.register_buffer(
            "kernel_norm",
            torch.ones(n_conv_channels, self.d_model, 1),
        )
        self.register_buffer(
            "kernel_norm_initialized", torch.tensor(0, dtype=torch.bool)
        )

        # ========== Learnable parameters ==========
        self.skip_weight = nn.Parameter(torch.randn(channels, self.d_model))

        self.kernel_list = nn.ParameterList()
        for _ in range(self.num_scales):
            if "randn" in mode:
                kernel = torch.randn(n_conv_channels, self.d_model, self.kernel_dim)
            elif "cos" in mode:
                kernel = torch.cat(
                    [
                        torch.cos(
                            torch.linspace(0, 2 * i * math.pi, self.kernel_dim)
                        ).expand(n_conv_channels, 1, self.kernel_dim)
                        for i in range(self.d_model)
                    ],
                    dim=1,
                )[:, torch.randperm(self.d_model), :]
            else:
                raise ValueError(f"Unknown mode {mode}")
            self.kernel_list.append(nn.Parameter(kernel))

        # ========== Layers ==========
        if not self.linear:
            self.activation = activation()
            self.dropout = nn.Dropout2d(dropout) if dropout > 0.0 else nn.Identity()
            self.norm = (
                nn.LayerNorm(self.d_model * self.channels)
                if layer_norm
                else nn.Identity()
            )
            self.output_linear = nn.Linear(self.d_model * self.channels, self.d_model)

    def forward(self, x, return_kernel=False):
        # x: (batch, d_model, seq_len) if transposed, else (batch, seq_len, d_model)
        if not self.transposed:
            x = x.transpose(-1, -2)
        # x: (batch, d_model, seq_len)
        seq_len = x.size(-1)

        # Build multi-scale kernel by upsampling and decaying each sub-kernel
        kernel_list = []
        interpolate_mode = "nearest" if "nearest" in self.mode else "linear"
        multiplier = self.multiplier  # (1, d_model, 1)

        if "cat" in self.mode:
            for i in range(self.num_scales):
                kernel = F.interpolate(
                    self.kernel_list[i],
                    scale_factor=2 ** (max(0, i - 1) + self.init_scale),
                    mode=interpolate_mode,
                ) * multiplier ** (self.num_scales - i - 1)
                kernel_list.append(kernel)
            # kernel: (channels, d_model, kernel_len)
            kernel = torch.cat(kernel_list, dim=-1)
        else:
            raise ValueError(f"Unknown mode {self.mode}")

        # Lazy-init kernel normalisation on first forward pass
        if not self.kernel_norm_initialized:
            self.kernel_norm = kernel.norm(dim=-1, keepdim=True).detach()
            self.kernel_norm_initialized = torch.tensor(
                1, dtype=torch.bool, device=kernel.device
            )

        # Pad or truncate kernel to match seq_len
        if kernel.size(-1) > seq_len:
            kernel = kernel[..., :seq_len]
        elif kernel.size(-1) < seq_len:
            kernel = F.pad(kernel, (0, seq_len - kernel.size(-1)))

        # kernel: (channels, d_model, seq_len) — normalised
        kernel = kernel / self.kernel_norm

        if self.bidirectional:
            k_fwd, k_bwd = rearrange(
                kernel,
                "(s channels) d_model seq_len -> s channels d_model seq_len",
                s=2,
            )
            # Combine forward and time-reversed backward kernels
            kernel = F.pad(k_fwd, (0, seq_len)) + F.pad(k_bwd.flip(-1), (seq_len, 0))

        # FFT-based convolution: O(N log N)
        # kernel_freq: (channels, d_model, freq_bins)
        kernel_freq = torch.fft.rfft(kernel.float(), n=2 * seq_len)
        # x_freq: (batch, d_model, freq_bins)
        x_freq = torch.fft.rfft(x.float(), n=2 * seq_len)
        # out_freq: (batch, channels, d_model, freq_bins)
        out_freq = torch.einsum("bhl,chl->bchl", x_freq, kernel_freq)
        # out: (batch, channels, d_model, seq_len)
        out = torch.fft.irfft(out_freq, n=2 * seq_len)[..., :seq_len]

        # Skip connection via learnable D matrix
        # (batch, channels, d_model, seq_len)
        out = out + torch.einsum("bhl,ch->bchl", x, self.skip_weight)
        # Merge channels and d_model: (batch, channels * d_model, seq_len)
        out = rearrange(out, "... c h l -> ... (c h) l")

        if not self.linear:
            out = self.dropout(self.activation(out))
            # (batch, seq_len, channels * d_model)
            out = rearrange(out, "b c l -> b l c")
            out = self.norm(out)
            # (batch, seq_len, d_model)
            out = self.output_linear(out)
            # (batch, d_model, seq_len)
            out = rearrange(out, "b l c -> b c l")

        if not self.transposed:
            out = out.transpose(-1, -2)

        if return_kernel:
            return out, kernel
        return out, None


class _Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super().__init__()

        # ========== Layers ==========
        padding = dilation * (kernel_size - 1) // 2
        self.conv = weight_norm(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                dilation=dilation,
                padding=padding,
            )
        )
        nn.init.kaiming_normal_(self.conv.parametrizations.weight.original1)

    def forward(self, x):
        return self.conv(x)


class _ZeroConv1d(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()

        # ========== Layers ==========
        self.conv = nn.Conv1d(in_channel, out_channel, kernel_size=1, padding=0)

    def forward(self, x):
        return self.conv(x)


class _RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()

        # ========== Parameters ==========
        self.eps = eps

        # ========== Learnable parameters ==========
        self.scale = nn.Parameter(torch.ones(1, dim, 1))

    def forward(self, x):
        norm = x.norm(dim=1, keepdim=True)
        rms = norm / (x.shape[1] ** 0.5)
        x_normed = x / (rms + self.eps)
        return self.scale * x_normed


class _ResidualBlock(nn.Module):
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
        super().__init__()

        # ========== Parameters ==========
        self.res_channels = res_channels
        self.swa_window_size = swa_window_size

        # ========== Layers ==========
        self.rms_norm = _RMSNorm(res_channels)
        self.conv_layer = _Conv(res_channels, 2 * res_channels, kernel_size=3)
        self.gelu = nn.GELU()

        self.sgconv = _GConv(
            d_model=2 * res_channels,
            channels=4,
            l_max=s4_lmax,
            d_state=s4_d_state,
            dropout=s4_dropout,
            bidirectional=s4_bidirectional,
            layer_norm=s4_layernorm,
        )
        self.attention = nn.MultiheadAttention(
            embed_dim=2 * res_channels,
            num_heads=4,
            dropout=s4_dropout,
            bias=True,
            batch_first=True,
        )

        self.res_conv = weight_norm(
            nn.Conv1d(res_channels, res_channels, kernel_size=1)
        )
        nn.init.kaiming_normal_(self.res_conv.parametrizations.weight.original1)

        self.skip_conv = weight_norm(
            nn.Conv1d(res_channels, skip_channels, kernel_size=1)
        )
        nn.init.kaiming_normal_(self.skip_conv.parametrizations.weight.original1)

    def generate_local_window_mask(self, seq_len, window_size):
        if window_size % 2 != 1:
            raise ValueError(
                f"window_size must be odd (e.g. 7, 9, 11), got {window_size}"
            )

        half_window = window_size // 2
        idx = torch.arange(seq_len)
        dist = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs()
        return torch.where(
            dist <= half_window, torch.zeros(1), torch.full((1,), float("-inf"))
        )

    def forward(self, input_data):
        x, original = input_data
        # x, original: (batch, res_channels, seq_len)
        hidden = x
        _, n_channels, seq_len = x.shape
        x = self.rms_norm(x)
        assert n_channels == self.res_channels

        # Add original input as residual shortcut
        hidden = hidden + original

        # Conv expansion: (batch, res_channels, seq_len) -> (batch, 2*res_channels, seq_len)
        hidden = self.conv_layer(hidden)

        # SGConv SSM branch
        hidden = self.gelu(hidden)
        # h_ssm: (batch, 2*res_channels, seq_len)
        h_ssm, _ = self.sgconv(hidden)

        # Sliding-window attention branch
        # (batch, 2*res_channels, seq_len) -> (batch, seq_len, 2*res_channels)
        h_attn = rearrange(h_ssm, "b c l -> b l c")
        swa_mask = self.generate_local_window_mask(seq_len, self.swa_window_size).to(
            x.device
        )
        h_attn, _ = self.attention(h_attn, h_attn, h_attn, attn_mask=swa_mask)
        # (batch, seq_len, 2*res_channels) -> (batch, 2*res_channels, seq_len)
        h_attn = rearrange(h_attn, "b l c -> b c l")

        # Combine SSM and attention
        # combined: (batch, 2*res_channels, seq_len)
        combined = h_ssm + h_attn

        # Gated activation: split into two halves along channel dim
        # out: (batch, res_channels, seq_len)
        out = torch.tanh(combined[:, : self.res_channels, :]) * torch.sigmoid(
            combined[:, self.res_channels :, :]
        )

        # Residual and skip projections: (batch, res_channels, seq_len)
        res = self.res_conv(out)
        assert x.shape == res.shape
        skip = self.skip_conv(out)

        return (x + res) * math.sqrt(0.5), skip


class _ResidualGroup(nn.Module):
    """Stack of EEGSSM residual blocks with aggregated skip connections.

    Implements the multi-block EEGSSM backbone described in Section 3.3 of
    [codebrain]_. Each :class:`_ResidualBlock` receives both the running hidden
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
        Number of stacked :class:`_ResidualBlock` modules.
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
        super().__init__()

        # ========== Parameters ==========
        self.num_res_layers = num_res_layers

        # ========== Layers ==========
        self.residual_blocks = nn.ModuleList()
        for _ in range(self.num_res_layers):
            self.residual_blocks.append(
                _ResidualBlock(
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


class _PatchEmbedding(nn.Module):
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

        # ========== Derived attributes ==========
        proj_refine_padding = proj_refine_kernel // 2
        pos_padding = (pos_kernel[0] // 2, pos_kernel[1] // 2)
        # emb_dim = conv_out_chans * temporal_output_len of first strided conv
        temporal_out_len = (
            patch_size + 2 * proj_padding - proj_kernel_size
        ) // conv_out_chans + 1
        self.emb_dim = conv_out_chans * temporal_out_len
        self.d_model = self.emb_dim

        # ========== Learnable parameters ==========
        self.mask_encoding = nn.Parameter(
            torch.zeros(self.emb_dim), requires_grad=False
        )

        # ========== Layers ==========
        # Temporal projection: 3-layer Conv2d stack
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

        # Spectral projection: FFT magnitude -> emb_dim
        self.spectral_proj = nn.Sequential(
            nn.Linear(patch_size // 2 + 1, self.emb_dim),
            nn.Dropout(spectral_dropout),
        )

        # Depth-wise positional encoding
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

    def forward(self, x, mask=None):
        batch, n_chans, seq_len, patch_size = x.shape
        if mask is None:
            mask_x = x
        else:
            mask_x = x.clone()
            mask_x[mask == 1] = self.mask_encoding

        # Merge channel and patch dims so Conv2d sees one image row per (channel, patch)
        # (batch, 1, n_chans * seq_len, patch_size)
        mask_x_conv = mask_x.contiguous().view(batch, 1, n_chans * seq_len, patch_size)

        # Temporal projection: conv stack outputs (batch, conv_ch, n_chans * seq_len, emb)
        patch_emb = self.proj_in(mask_x_conv)

        # Restore channel and patch dims, flatten conv channels into embedding dim
        # (batch, n_chans, seq_len, conv_ch * emb) = (batch, n_chans, seq_len, emb_dim)
        patch_emb = rearrange(
            patch_emb,
            "batch conv_ch (n_chans seq_len) emb -> batch n_chans seq_len (conv_ch emb)",
            n_chans=n_chans,
            seq_len=seq_len,
        )

        # Flatten masked patches for FFT: (batch * n_chans * seq_len, patch_size)
        patches_flat = rearrange(
            mask_x,
            "batch n_chans seq_len patch_size -> (batch n_chans seq_len) patch_size",
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
