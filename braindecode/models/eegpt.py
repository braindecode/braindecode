# Authors: Young Truong <dt.young112@gmail.com>
#          Kuntal Kokate <kukokate@ucsd.edu>
#
# License: BSD-3

from functools import partial
from typing import Optional

import torch
from torch import nn

from braindecode.models.base import EEGModuleMixin


class EEGPT(EEGModuleMixin, nn.Module):
    r"""
    EEGPT: Pretrained Transformer for Universal and Reliable Representation of EEG Signals from Tang et al. (2024) [eegpt]_.

    :bdg-danger:`Foundation Model` :bdg-info:`Attention/Transformer`

    .. figure:: https://github.com/BINE022/EEGPT/raw/main/figures/EEGPT.jpg
        :align: center
        :alt: EEGPT Architecture
        :width: 1000px

    EEGPT is a novel 10-million-parameter pretrained transformer model designed for universal EEG feature extraction.
    In EEGPT, a mask-based dual self-supervised learning method for efficient feature extraction is designed.
    Compared to other mask-based self-supervised learning methods, it adds spatio-temporal representation alignment,
    constructing a self-supervised task on EEG representations with high SNR and rich semantic information instead
    of raw signals, thus avoiding poor feature quality extracted from low SNR signals.

    .. rubric:: Pretrained Weights

    You can download pretrained models here:
    `EEG_large <https://figshare.com/s/e37df4f8a907a866df4b`_ (in the ``Files/EEGPT/checkpoint/eegpt_mcae_58chs_4s_large4E.ckpt``)
    trained on mixed dataset (58-channels, 256Hz, 4s time length EEG) using patch size 64.

    Parameters
    ----------
    n_outputs : int
        Number of outputs of the model. This is the number of classes in the
        case of classification.
    n_chans : int
        Number of EEG channels.
    chs_info : list of dict
        Information about the channels, as returned by `mne.Info`.
    n_times : int
        Number of time samples.
    input_window_seconds : float
        Length of the input window in seconds.
    sfreq : float
        Sampling frequency of the EEG signals.
    return_encoder_output : bool, default=False
        Whether to return the encoder output or the classifier output.
    channel_names : list of str, optional
        List of channel names. If None, it will be extracted from `chs_info`.
    patch_size : int, default=64
        Size of the patches for the transformer.
    patch_stride : int, default=32
        Stride of the patches for the transformer.
    embed_num : int, default=4
        Number of embeddings.
    embed_dim : int, default=512
        Dimension of the embeddings.
    depth : int, default=8
        Number of transformer layers.
    num_heads : int, default=8
        Number of attention heads.
    mlp_ratio : float, default=4.0
        Ratio of the MLP hidden dimension to the embedding dimension.
    drop_prob : float, default=0.0
        Dropout probability.
    attn_drop_rate : float, default=0.0
        Attention dropout rate.
    drop_path_rate : float, default=0.0
        Drop path rate.
    init_std : float, default=0.02
        Standard deviation for weight initialization.
    qkv_bias : bool, default=True
        Whether to use bias in the QKV projection.
    norm_layer : torch.nn.Module, default=partial(nn.LayerNorm, eps=1e-6)
        Normalization layer.

    References
    ----------
    .. [eegpt] Wang, G., Liu, W., He, Y., Xu, C., Ma, L., & Li, H. (2024).
       EEGGPT: Pretrained transformer for universal and reliable representation of eeg signals.
       Advances in Neural Information Processing Systems, 37, 39249-39280.
       Online: https://proceedings.neurips.cc/paper_files/paper/2024/file/4540d267eeec4e5dbd9dae9448f0b739-Paper-Conference.pdf
    """

    def __init__(
        self,
        # braindecode parameters
        n_outputs=None,
        n_chans=None,
        chs_info=None,
        n_times=None,
        input_window_seconds=None,
        sfreq=None,
        # model parameters
        patch_size: int = 64,
        patch_stride: int = 32,
        embed_num: int = 4,
        embed_dim: int = 512,
        depth: int = 8,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        drop_prob: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        init_std: float = 0.02,
        qkv_bias: bool = True,
        norm_layer: Optional[nn.Module] = None,
        return_encoder_output: bool = False,
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

        # model parameters
        self.return_encoder_output = return_encoder_output
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.embed_num = embed_num
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.drop_prob = drop_prob
        self.attn_drop_rate = attn_drop_rate
        self.drop_path_rate = drop_path_rate
        self.init_std = init_std
        self.qkv_bias = qkv_bias
        self.norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        self.target_encoder = EEGTransformer(
            img_size=[self.n_chans, self.n_times],
            patch_size=self.patch_size,
            patch_stride=self.patch_stride,
            embed_num=self.embed_num,
            embed_dim=self.embed_dim,
            depth=self.depth,
            num_heads=self.num_heads,
            mlp_ratio=self.mlp_ratio,
            drop_rate=self.drop_prob,
            attn_drop_rate=self.attn_drop_rate,
            drop_path_rate=self.drop_path_rate,
            init_std=self.init_std,
            qkv_bias=self.qkv_bias,
            norm_layer=self.norm_layer,
        )

        if self.chs_info is not None:
            self.channel_names = [ch["ch_name"] for ch in self.chs_info]  # type: ignore

        self.chans_id = self.target_encoder.prepare_chan_ids(self.channel_names)

        self.flattened_encoder_output_dim = (
            self.target_encoder.num_patches[1] * self.embed_num * self.embed_dim
        )

        if not return_encoder_output:
            self.final_layer = nn.Linear(
                self.flattened_encoder_output_dim, self.n_outputs
            )
        else:
            self.final_layer = nn.Identity()

    def forward(self, x):
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            EEG data of shape (batch_size, n_chans, n_times)
        """

        # B, N, embed_num, D
        z = self.target_encoder(x, self.chans_id.to(x.device))

        if self.return_encoder_output:
            return z

        h = z.flatten(1)
        if self.flattened_encoder_output_dim != h.shape[1]:
            raise ValueError(
                f"Expected output dim {self.flattened_encoder_output_dim}, got {h.shape[1]}"
            )

        h = self.final_layer(h)

        return h


CHANNEL_DICT = {
    k.upper(): v
    for v, k in enumerate(
        [
            "FP1",
            "FPZ",
            "FP2",
            "AF7",
            "AF3",
            "AF4",
            "AF8",
            "F7",
            "F5",
            "F3",
            "F1",
            "FZ",
            "F2",
            "F4",
            "F6",
            "F8",
            "FT7",
            "FC5",
            "FC3",
            "FC1",
            "FCZ",
            "FC2",
            "FC4",
            "FC6",
            "FT8",
            "T7",
            "C5",
            "C3",
            "C1",
            "CZ",
            "C2",
            "C4",
            "C6",
            "T8",
            "TP7",
            "CP5",
            "CP3",
            "CP1",
            "CPZ",
            "CP2",
            "CP4",
            "CP6",
            "TP8",
            "P7",
            "P5",
            "P3",
            "P1",
            "PZ",
            "P2",
            "P4",
            "P6",
            "P8",
            "PO7",
            "PO5",
            "PO3",
            "POZ",
            "PO4",
            "PO6",
            "PO8",
            "O1",
            "OZ",
            "O2",
        ]
    )
}


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample."""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def extra_repr(self) -> str:
        return f"p={self.drop_prob}"

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        return output


class MLP(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        attn_drop=0.0,
        proj_drop=0.0,
        is_causal=False,
        use_rope=False,
        return_attention=False,
    ):
        super().__init__()
        self.use_rope = use_rope

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.is_causal = is_causal
        self.return_attention = return_attention

    def forward(self, x):
        B, T, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, T, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )  # 3, B, nh, T, d
        q, k, v = qkv[0], qkv[1], qkv[2]  # B, nh, T, d

        if self.use_rope:  # RoPE
            q = apply_rotary_emb(freqs, q)
            k = apply_rotary_emb(freqs, k)
        if self.return_attention:
            if self.is_causal:
                attn_mask = torch.ones(q.size(-2), q.size(-2), dtype=torch.bool).tril(
                    diagonal=0
                )
                attn_maak = torch.zeros(q.size(-2), q.size(-2))
                attn_mask = attn_maak.masked_fill(
                    torch.logical_not(attn_mask), -float("inf")
                )
                attn_weight = torch.softmax(
                    (q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))) + attn_mask,
                    dim=-1,
                )
            else:
                attn_weight = torch.softmax(
                    (q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))), dim=-1
                )
            return attn_weight
        # efficient attention using Flash Attention CUDA kernels
        y = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.attn_drop if self.training else 0,
            is_causal=self.is_causal,
        )
        x = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # (B, nh, T, hs) -> (B, T, hs*nh)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        is_causal=False,
        use_rope=False,
        return_attention=False,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            is_causal=is_causal,
            use_rope=use_rope,
            return_attention=return_attention,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x, freqs=None):
        y = self.attn(self.norm1(x), freqs)
        if self.return_attention:
            return y
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """Image to Patch Embedding"""

    def __init__(
        self, img_size=(64, 1000), patch_size=16, patch_stride=None, embed_dim=768
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        if patch_stride is None:
            self.num_patches = ((img_size[0]), (img_size[1] // patch_size))
        else:
            self.num_patches = (
                (img_size[0]),
                ((img_size[1] - patch_size) // patch_stride + 1),
            )

        self.proj = nn.Conv2d(
            1,
            embed_dim,
            kernel_size=(1, patch_size),
            stride=(1, patch_size if patch_stride is None else patch_stride),
        )

    def forward(self, x):
        # x: B, C, T
        x = x.unsqueeze(1)  # B, 1, C, T
        x = self.proj(x).transpose(1, 3)  # B, T, C, D
        return x


class PatchNormEmbed(nn.Module):
    """Image to Patch Embedding"""

    def __init__(
        self, img_size=(64, 1000), patch_size=16, patch_stride=None, embed_dim=768
    ):
        super().__init__()

        assert img_size[1] % patch_size == 0

        self.img_size = img_size
        self.patch_size = patch_size
        self.patch_stride = patch_stride

        if patch_stride is None:
            self.num_patches = ((img_size[0]), (img_size[1] // patch_size))
        else:
            self.num_patches = (
                (img_size[0]),
                ((img_size[1] - patch_size) // patch_stride + 1),
            )

        self.unfold = torch.nn.Unfold(
            kernel_size=(1, patch_size),
            stride=(1, patch_stride if patch_stride is not None else patch_size),
        )

        self.proj = nn.Linear(patch_size, embed_dim)  # +2

    def forward(self, x):
        # x: B,C,T
        B, C, T = x.shape
        x = x.unsqueeze(1)  # B 1 C T

        x = self.unfold(x)

        x = x.transpose(-1, -2)

        x = x.view(B, C, -1, self.patch_size).contiguous()
        x = x.transpose(1, 2)

        x = torch.layer_norm(x, (self.patch_size,))

        x = self.proj(x)  # B, T, C, D

        return x


class EEGTransformer(nn.Module):
    """EEG Transformer"""

    def __init__(
        self,
        img_size=(64, 1000),
        patch_size=64,
        patch_stride=None,
        embed_dim=768,
        embed_num=1,
        predictor_embed_dim=384,
        depth=12,
        predictor_depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        patch_module=PatchEmbed,  # PatchNormEmbed
        init_std=0.02,
        interpolate_factor=2.0,
        return_attention_layer=-1,
        **kwargs,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.embed_num = embed_num
        self.num_heads = num_heads

        self.patch_embed = patch_module(
            img_size=img_size,
            patch_size=patch_size,
            patch_stride=patch_stride,
            embed_dim=embed_dim,
        )
        self.num_patches = self.patch_embed.num_patches

        self.chan_embed = nn.Embedding(len(CHANNEL_DICT), embed_dim)

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule

        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    is_causal=False,
                    use_rope=False,
                    return_attention=(i + 1) == return_attention_layer,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)
        self.init_std = init_std
        self.summary_token = nn.Parameter(torch.zeros(1, embed_num, embed_dim))

        nn.init.trunc_normal_(self.summary_token, std=self.init_std)
        self.apply(self._init_weights)
        self.fix_init_weight()

    def prepare_chan_ids(self, channels):
        chan_ids = []
        for ch in channels:
            ch_normalised = ch.upper().strip(".").replace("Z", "z")
            # Revert Z to z only if it's the last character?
            # The dictionary uses UPPERCASE.
            # Original code: ch.upper().strip(".")
            # Dictionary keys are UPPERCASE.
            # Wait, original code CHANNEL_DICT had keys like 'FP1', 'FPZ' (upper Z).
            # But standard 10-20 often uses 'Fp1', 'Fpz'.
            # MNE uses Fpz.
            # ch.upper() 'FPZ' -> matches 'FPZ' in dict.

            ch_upper = ch.upper().strip(".")
            assert ch_upper in CHANNEL_DICT, (
                f"Channel {ch} not found in EEGPT channel list."
            )
            chan_ids.append(CHANNEL_DICT[ch_upper])

        return torch.tensor(chan_ids).unsqueeze_(0).long()

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(torch.sqrt(torch.tensor(2.0 * layer_id)))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Embedding):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, x, chan_ids=None, mask_x=None, mask_t=None):
        # x.shape B, C, T

        # -- patchify x
        x = self.patch_embed(x)
        B, N, C, D = x.shape

        assert N == self.num_patches[1] and C == self.num_patches[0], (
            f"{N}=={self.num_patches[1]} and {C}=={self.num_patches[0]}"
        )

        if chan_ids is None:
            chan_ids = torch.arange(0, C)
        chan_ids = chan_ids.to(x.device)

        # -- add channels positional embedding to x
        # -- add channels positional embedding to x
        x = x + self.chan_embed(chan_ids.long()).unsqueeze(0)  # (1,C) -> (1,1,C,D)

        if mask_x is not None:
            mask_x = mask_x.to(x.device)
            x = apply_mask(mask_x, x)  # B, mN, mC, D
            B, N, C, D = x.shape

        x = x.flatten(0, 1)  # BmN, mC, D

        # -- concat summary token
        summary_token = self.summary_token.repeat((x.shape[0], 1, 1))
        x = torch.cat([x, summary_token], dim=1)  # BmN, mC+embed_num, D

        # -- fwd prop
        for i, blk in enumerate(self.blocks):
            x = blk(x)  # B*N, mC+1, D
            if blk.return_attention == True:
                return x

        x = x[:, -summary_token.shape[1] :, :]

        if self.norm is not None:
            x = self.norm(x)

        x = x.flatten(-2)
        x = x.reshape((B, N, -1))
        # -- reshape back

        if mask_t is not None:
            mask_t = mask_t.to(x.device)
            x = apply_mask_t(mask_t, x)  # B, mN, D

        x = x.reshape((B, N, self.embed_num, -1))

        return x
