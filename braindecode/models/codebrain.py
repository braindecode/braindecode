# Authors: Vandit Shah <shahvanditt@gmail.com>
#
# Code adapted from https://github.com/jingyingma01/CodeBrain
#
# License: BSD (3-clause)

import math
from functools import partial
from typing import Optional

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

try:
    import opt_einsum as oe
    contract = oe.contract
except ImportError:
    contract = torch.einsum

from braindecode.models.base import EEGModuleMixin

class _KernelModule(nn.Module):
    def __init__(self, kernel):
        super().__init__()
        self.kernel = nn.Parameter(kernel)


class _GConv(nn.Module):
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
        **kernel_args,
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
            self.norm = nn.LayerNorm(self.h * self.channels) if layer_norm else nn.Identity()

        if not self.linear:
            self.output_linear = nn.Linear(self.h * self.channels, self.h)

        self.init_scale = kernel_args.get("init_scale", 0)
        self.kernel_dim = kernel_args.get("kernel_dim", 2)
        self.num_scales = kernel_args.get(
            "n_scales",
            1 + math.ceil(math.log2(l_max / self.kernel_dim)) - self.init_scale,
        )
        if self.num_scales is None:
            self.num_scales = (
                1 + math.ceil(math.log2(l_max / self.kernel_dim)) - self.init_scale
            )

        self.kernel_list = nn.ModuleList()
        decay_min = kernel_args.get("decay_min", 2)
        decay_max = kernel_args.get("decay_max", 2)
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
            self.kernel_list.append(_KernelModule(kernel))

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
                    self.kernel_list[i].kernel,
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
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, padding=self.padding)
        self.conv = nn.utils.weight_norm(self.conv)
        nn.init.kaiming_normal_(self.conv.weight)

    def forward(self, x):
        out = self.conv(x)
        return out
    
class ZeroConv1d(nn.Module):
    # initializeing the conv layers
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


class Residual_block(nn.Module):
    def __init__(self, res_channels, skip_channels,
                 s4_lmax,
                 s4_d_state,
                 s4_dropout,
                 s4_bidirectional,
                 s4_layernorm):
        super(Residual_block, self).__init__()
        self.res_channels = res_channels

        self.sn = RMSNorm(res_channels)

        self.S41 = _GConv(d_model=2 * self.res_channels,
                         channels=4,
                         l_max=s4_lmax,
                         d_state=s4_d_state,
                         dropout=s4_dropout,
                         bidirectional=s4_bidirectional,
                         layer_norm=s4_layernorm)

        self.conv_layer = Conv(self.res_channels, 2 * self.res_channels, kernel_size=3)

        self.attention = nn.MultiheadAttention(embed_dim=2 * self.res_channels, num_heads=4, dropout=s4_dropout,
                                               bias=True, batch_first=True)

        self.gelu = nn.GELU()

        self.res_conv = nn.Conv1d(res_channels, res_channels, kernel_size=1)
        self.res_conv = nn.utils.weight_norm(self.res_conv)
        nn.init.kaiming_normal_(self.res_conv.weight)


        self.skip_conv = nn.Conv1d(res_channels, skip_channels, kernel_size=1)
        self.skip_conv = nn.utils.weight_norm(self.skip_conv)
        nn.init.kaiming_normal_(self.skip_conv.weight)


    def generate_local_window_mask(self, seq_len, window_size):
        assert window_size % 2 == 1, "window_size shoule be odd number, like 7, 9, 11"

        half_window = window_size // 2

        mask = torch.full((seq_len, seq_len), float('-inf'))

        for i in range(seq_len):
            start = max(0, i - half_window)
            end = min(seq_len, i + half_window + 1)
            mask[i, start:end] = 0

        return mask

    def forward(self, input_data):
        x, original = input_data
        h = x
        B, C, L = x.shape
        x = self.sn(x)
        assert C == self.res_channels

        part_t = rearrange(original, 'b c l -> b c l')
        h = h + part_t

        h = self.conv_layer(h)

        h = self.gelu(h)
        h_t, _ = self.S41(h)

        h_s = rearrange(h_t, "b c l -> b l c")
        SWA_mask = self.generate_local_window_mask(L, 1).to(x.device)
        h_s, _ = self.attention(h_s, h_s, h_s, attn_mask=SWA_mask)
        h_s = rearrange(h_s, "b l c -> b c l")

        h = h_t + h_s

        out = torch.tanh(h[:, :self.res_channels, :]) * torch.sigmoid(h[:, self.res_channels:, :])

        res = self.res_conv(out)
        assert x.shape == res.shape
        skip = self.skip_conv(out)

        return (x + res) * math.sqrt(0.5), skip


class Residual_group(nn.Module):
    def __init__(self, res_channels, skip_channels, num_res_layers,
                 s4_lmax,
                 s4_d_state,
                 s4_dropout,
                 s4_bidirectional,
                 s4_layernorm):
        super(Residual_group, self).__init__()
        self.num_res_layers = num_res_layers

        self.residual_blocks = nn.ModuleList()
        for n in range(self.num_res_layers):
            self.residual_blocks.append(Residual_block(res_channels, skip_channels,
                                                       s4_lmax=s4_lmax,
                                                       s4_d_state=s4_d_state,
                                                       s4_dropout=s4_dropout,
                                                       s4_bidirectional=s4_bidirectional,
                                                       s4_layernorm=s4_layernorm))

    def forward(self, input_data):
        noise = input_data
        h = noise
        skip = 0
        for n in range(self.num_res_layers):
            h, skip_n = self.residual_blocks[n]((h, noise))
            skip = skip_n + skip

        return skip * math.sqrt(1.0 / self.num_res_layers)

class PatchEmbedding(nn.Module):
    def __init__(self, conv_out_chans, patch_size, conv_groups):
        super().__init__()
        # Compute actual conv output dim from the first Conv2d:
        # kernel=49, stride=conv_out_chans, padding=24
        _t = (patch_size + 2 * 24 - 49) // conv_out_chans + 1
        self.emb_dim = conv_out_chans * _t
        self.d_model = self.emb_dim
        self.positional_encoding = nn.Sequential(
            nn.Conv2d(in_channels=self.emb_dim, out_channels=self.emb_dim, kernel_size=(19, 7), stride=(1, 1),
                      padding=(9, 3),
                      groups=self.emb_dim),
        )
        self.mask_encoding = nn.Parameter(torch.zeros(self.emb_dim), requires_grad=False)

        self.proj_in = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=conv_out_chans, kernel_size=(1, 49), stride=(1, conv_out_chans), padding=(0, 24)),
            nn.GroupNorm(conv_groups, conv_out_chans),
            nn.GELU(),

            nn.Conv2d(in_channels=conv_out_chans, out_channels=conv_out_chans, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)),
            nn.GroupNorm(conv_groups, conv_out_chans),
            nn.GELU(),

            nn.Conv2d(in_channels=conv_out_chans, out_channels=conv_out_chans, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)),
            nn.GroupNorm(conv_groups, conv_out_chans),
            nn.GELU(),
        )
        self.spectral_proj = nn.Sequential(
            nn.Linear(patch_size // 2 + 1, self.emb_dim),
            nn.Dropout(0.1),
        )

    def forward(self, x, mask=None):
        bz, ch_num, patch_num, patch_size = x.shape
        if mask is None:
            mask_x = x
        else:
            mask_x = x.clone()
            mask_x[mask == 1] = self.mask_encoding

        mask_x = mask_x.contiguous().view(bz, 1, ch_num * patch_num, patch_size)
        patch_emb = self.proj_in(mask_x)
        patch_emb = rearrange(patch_emb, 'b c (ch p) t -> b ch p (c t)', ch=ch_num, p=patch_num)


        mask_x = rearrange(x, 'b ch p t -> (b ch p) t')

        spectral = torch.abs(torch.fft.rfft(mask_x, dim=-1, norm='forward'))
        spectral = rearrange(spectral, '(b ch p) f -> b ch p f', b=bz, ch=ch_num, p=patch_num)

        spectral_emb = self.spectral_proj(spectral)
        patch_emb = patch_emb + spectral_emb

        positional_embedding = self.positional_encoding(patch_emb.permute(0, 3, 1, 2))
        positional_embedding = positional_embedding.permute(0, 2, 3, 1)

        patch_emb = patch_emb + positional_embedding

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

    .. rubric:: Key Innovation: Code-Based Pre-Training

    Rather than reconstructing raw EEG values, CodeBrain first discretises EEG
    patches into temporal and spectral codebook tokens via VQ-VAE, then trains
    the backbone to predict these codes from masked input. This avoids the
    low signal-to-noise issues of direct waveform reconstruction.

    .. rubric:: Macro Components

    - **PatchEmbedding**: Splits ``(batch, n_chans, n_times)`` into
      ``(batch, n_chans, seq_len, patch_size)`` patches, projects each patch
      with a 2-D convolutional stack, adds FFT-based spectral embeddings, and
      applies depth-wise convolutional positional encoding.
    - **Residual blocks** (``Residual_group``): Each block applies RMSNorm,
      a ``_GConv`` SSM layer, and sliding-window multi-head attention, with
      gated activation and separate residual/skip paths.
    - **Classification head** (``final_layer``): Flattens the output and maps
      to ``n_outputs`` classes.

    .. rubric:: Pre-training vs Fine-tuning

    Set ``if_codebook=True`` during pre-training to return ``(lm_head_t,
    lm_head_f)`` codebook logits. Set ``if_codebook=False`` (default) for
    fine-tuning to return ``(batch, n_outputs)`` class logits.

    Parameters
    ----------
    patch_size : int, default=200
        Number of time samples per patch. Input length is trimmed to the
        nearest multiple of ``patch_size``.
    res_channels : int, default=200
        Width of the residual stream inside each ``Residual_block``.
    skip_channels : int, default=200
        Width of the skip-connection stream aggregated across blocks.
    out_channels : int, default=200
        Output channels of ``final_conv`` before the classification head.
    num_res_layers : int, default=8
        Number of stacked ``Residual_block`` modules.
    drop_prob : float, default=0.1
        Dropout rate used inside the ``_GConv`` SSM and attention layers.
    s4_bidirectional : bool, default=True
        Whether the ``_GConv`` SSM processes the sequence bidirectionally.
    s4_layernorm : bool, default=True
        Whether to apply layer normalisation inside the ``_GConv`` SSM.
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
    if_codebook : bool, default=False
        If ``True``, returns ``(lm_head_t, lm_head_f)`` logits for masked
        pre-training. If ``False``, returns ``(batch, n_outputs)`` logits.
    activation : type[nn.Module], default=nn.ReLU
        Non-linear activation class used in ``init_conv`` and ``final_conv``.

    References
    ----------
    .. [codebrain] Yi Ding, Xuyang Chen, Yong Li, Rui Yan, Tao Wang, Le Wu (2024).
       CodeBrain: Scalable Code EEG Pre-Training for Unified Downstream BCI Tasks.
       Advances in Neural Information Processing Systems (NeurIPS), 2024.
       https://arxiv.org/abs/2412.04083
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
        res_channels:int = 200,
        skip_channels:int = 200,
        out_channels:int = 200,
        num_res_layers:int=8,
        drop_prob:float = 0.1,
        s4_bidirectional:bool=True,
        s4_layernorm:bool=True,
        s4_lmax: int = 570,
        s4_d_state: int = 64,
        conv_out_chans: int = 25,
        conv_groups: int = 5,
        codebook_size_t: int = 4096,
        codebook_size_f: int = 4096,
        if_codebook: bool = False,
        activation: type[nn.Module] = nn.ReLU,
        ):
            super().__init__(n_outputs=n_outputs,
                n_chans=n_chans,
                chs_info=chs_info,
                n_times=n_times,
                input_window_seconds=input_window_seconds,
                sfreq=sfreq,
            )
            self.patch_size = patch_size
            self.activation = activation
            self.patch_embedding = PatchEmbedding(
                patch_size=patch_size,
                conv_out_chans=conv_out_chans,
                conv_groups=conv_groups,
            )
            emb_dim = self.patch_embedding.emb_dim

            self.init_conv = nn.Sequential(Conv(emb_dim, res_channels, kernel_size=1), self.activation())

            self.residual_layer = Residual_group(res_channels=res_channels,
                                                skip_channels=skip_channels,
                                                num_res_layers=num_res_layers,
                                                s4_lmax=s4_lmax,
                                                s4_d_state=s4_d_state,
                                                s4_dropout=drop_prob,
                                                s4_bidirectional=s4_bidirectional,
                                                s4_layernorm=s4_layernorm)
            self.final_conv = nn.Sequential(Conv(skip_channels, skip_channels, kernel_size=1),
                                        self.activation(),
                                        ZeroConv1d(skip_channels, out_channels))
            self.lm_head_t = nn.Linear(out_channels, codebook_size_t, bias=False)
            self.lm_head_f = nn.Linear(out_channels, codebook_size_f, bias=False)
            self.if_codebook = if_codebook
            self.norm = nn.LayerNorm(out_channels)
            # 3-layer MLP classifier as described in the paper (Section 3.3)
            self.final_layer = nn.Sequential(
                nn.Flatten(),
                nn.LazyLinear(4 * out_channels),
                nn.ELU(),
                nn.Dropout(drop_prob),
                nn.LazyLinear(out_channels),
                nn.ELU(),
                nn.Dropout(drop_prob),
                nn.LazyLinear(self.n_outputs),
            )


    def load_state_dict(self, state_dict, strict=True):
        # Strip DataParallel 'module.' prefix if present (checkpoint from YjMajy/CodeBrain)
        if all(k.startswith("module.") for k in state_dict.keys()):
            state_dict = {k[len("module."):]: v for k, v in state_dict.items()}
        return super().load_state_dict(state_dict, strict=strict)

    def forward(self, inputs, mask=None):
        bz, ch_num, n_times = inputs.shape
        patch_size = self.patch_size
        seq_len = n_times // patch_size
        inputs = inputs[:, :, :seq_len * patch_size].reshape(bz, ch_num, seq_len, patch_size)
        inputs = self.patch_embedding(inputs, mask=mask)
        x = rearrange(inputs, 'b c s p -> b p (c s)')
        x = self.init_conv(x)
        x = self.residual_layer(x)
        x = self.final_conv(x)
        x = rearrange(x, 'b p (c s)  -> b c s p', p=patch_size, s=seq_len, c=ch_num)
        x = self.norm(x)
        if self.if_codebook:
            if mask is not None:
                x = x[mask == 1]
            x_t = self.lm_head_t(x)
            x_f = self.lm_head_f(x)
            return (x_t, x_f)
        else:
            return self.final_layer(x)

