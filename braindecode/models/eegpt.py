# Authors: Young Truong <dt.young112@gmail.com>
#          Kuntal Kokate <kukokate@ucsd.edu>
#
# License: BSD-3

from functools import partial
from typing import Optional

import torch
from einops import rearrange
from torch import nn

from braindecode.models.base import EEGModuleMixin
from braindecode.modules import DropPath


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

    .. rubric:: Dual Self-Supervised Learning

    Unlike standard masked autoencoders that operate directly on raw signals (which often contain low SNR),
    EEGPT is pretrained with two objectives:

    1.  **Masked Signal Modeling**: Reconstructing masked patches of the raw EEG signal.
    2.  **Spatio-Temporal Representation Alignment**: Aligning the representations of masked signals with
        features from unmasked signals, focusing on high-level semantic information.

    This dual approach ensures the model captures both low-level temporal dynamics and high-level
    spatio-temporal semantics, making it robust to noise and varying electrode montages.

    .. rubric:: Architecture Components

    The model consists of:

    -   **Patch Embedding**: Splits the EEG signal into overlapping patches.
    -   **Channel Embedding**: Learnable embeddings added to distinguish different electrodes.
    -   **Transformer Encoder**: A stack of standard Transformer blocks with multi-head self-attention.
    -   **Channel-Invariance**: The model is designed to handle variable numbers of channels through its
        patch-based approach and channel embeddings.

    .. rubric:: Usage

    .. code-block:: python

        from braindecode.models import EEGPT

        model = EEGPT(
            n_chans=22,
            n_times=1000,
            sfreq=200,
            n_outputs=4,  # For classification tasks
            patch_size=64,
            depth=8,
            embed_dim=512,
        )

        # Forward pass
        # Input shape: (batch_size, n_chans, n_times)
        y = model(x)

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
        Number of summary tokens used for the global representation.
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
    .. [eegpt] Tang, G., Liu, W., He, Y., Xu, C., Ma, L., & Li, H. (2024).
       EEGPT: Pretrained transformer for universal and reliable representation of eeg signals.
       Advances in Neural Information Processing Systems, 37, 39249-39280.
       Online: https://proceedings.neurips.cc/paper_files/paper/2024/file/4540d267eeec4e5dbd9dae9448f0b739-Paper-Conference.pdf

    Notes
    -----
    When loading pretrained weights from the original EEGPT checkpoint (e.g., for
    fine-tuning), you may encounter "unexpected keys" related to the `predictor`
    and `reconstructor` modules (e.g., `predictor.mask_token`, `reconstructor.time_embed`).
    These components are used only during the self-supervised pre-training phase
    (Masked Auto-Encoder) and are not part of this encoder-only model used for
    downstream tasks. It is safe to ignore them.
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

        self.target_encoder = _EEGTransformer(
            n_chans=self.n_chans,
            n_times=self.n_times,
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

        if self._chs_info is not None:
            self.channel_names = [ch["ch_name"] for ch in self.chs_info]  # type: ignore
        else:
            self.channel_names = None  # type: ignore

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
            EEG data of shape (batch, n_chans, n_times).

        Returns
        -------
        torch.Tensor
            Model output. Shape depends on `n_outputs` and `return_encoder_output`.
        """

        # z shape: (batch, n_patches, embed_num, embed_dim)
        z = self.target_encoder(x, self.chans_id.to(x.device))

        if self.return_encoder_output:
            return z

        # Flatten encoder output for classification
        h = z.flatten(1)
        # h shape: (batch, n_patches * embed_num * embed_dim)

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


class _MLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) with GELU activation and dropout.

    This block consists of two linear layers with a GELU activation and dropout
    in between, and another dropout after the second linear layer.

    Parameters
    ----------
    in_features : int
        Number of input features.
    hidden_features : int, optional
        Number of hidden features. If None, defaults to `in_features`.
    out_features : int, optional
        Number of output features. If None, defaults to `in_features`.
    act_layer : nn.Module, default=nn.GELU
        Activation function.
    drop : float, default=0.0
        Dropout probability.
    """

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


class _Attention(nn.Module):
    """
    Multi-head Self-Attention with optional RoPE and Causal Masking.

    This layer implements multi-head self-attention using PyTorch's
    scaled_dot_product_attention (Flash Attention) for efficiency.
    It supports Rotary Positional Embeddings (RoPE) and causal masking.

    Parameters
    ----------
    dim : int
        Input and output embedding dimension.
    num_heads : int, default=8
        Number of attention heads.
    qkv_bias : bool, default=False
        If True, add a learnable bias to query, key, value projections.
    attn_drop : float, default=0.0
        Dropout probability for attention weights.
    proj_drop : float, default=0.0
        Dropout probability for output projection.
    is_causal : bool, default=False
        If True, applies a causal mask (temporal causality).
    use_rope : bool, default=False
        If True, applies Rotary Positional Embeddings (RoPE) to queries and keys.
    return_attention : bool, default=False
        If True, returns the attention weights instead of the output tensor.
    """

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

    def forward(self, x, freqs=None):
        """
        Forward pass of the attention layer.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, seq_len, embed_dim).
        freqs : torch.Tensor, optional
            Frequencies for Rotary Positional Embeddings (RoPE).
        """
        # qkv: (batch, seq_len, 3 * num_heads * head_dim)
        qkv = self.qkv(x)

        # Reshape to (3, batch, num_heads, seq_len, head_dim)
        qkv = rearrange(
            qkv,
            "batch seq_len (three num_heads head_dim) -> three batch num_heads seq_len head_dim",
            three=3,
            num_heads=self.num_heads,
        )
        q, k, v = qkv[0], qkv[1], qkv[2]  # batch, num_heads, seq_len, head_dim

        # 1. Rotary Positional Embeddings (RoPE)
        # Unlike standard absolute positional encodings, RoPE rotates the
        # query and key vectors to encode relative positions.
        if self.use_rope:
            q = apply_rotary_emb(freqs, q)
            k = apply_rotary_emb(freqs, k)

        # 2. Return Attention Weights
        # If return_attention is True, we manually compute attention scores
        # because F.scaled_dot_product_attention doesn't return weights.
        if self.return_attention:
            if self.is_causal:
                attn_mask = torch.ones(q.size(-2), q.size(-2), dtype=torch.bool).tril(
                    diagonal=0
                )
                attn_zeros = torch.zeros(q.size(-2), q.size(-2))
                attn_mask = attn_zeros.masked_fill(
                    torch.logical_not(attn_mask), -float("inf")
                )
                # Naive attention computation for visualization/debugging
                attn_weight = torch.softmax(
                    (q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))) + attn_mask,
                    dim=-1,
                )
            else:
                attn_weight = torch.softmax(
                    (q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))), dim=-1
                )
            return attn_weight

        # 3. Flash Attention
        # Use PyTorch's optimized scaled_dot_product_attention (SDPA)
        # which internally uses FlashAttention or EfficientAttention kernels.
        y = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.attn_drop.p if self.training else 0,
            is_causal=self.is_causal,
        )

        # Reshape back: (batch, num_heads, seq_len, head_dim) -> (batch, seq_len, embed_dim)
        x = rearrange(
            y,
            "batch num_heads seq_len head_dim -> batch seq_len (num_heads head_dim)",
        )

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class _Block(nn.Module):
    """
    Transformer Block consisting of Self-Attention and MLP.

    This is a standard Transformer encoder block that applies:
    1. Layer Normalization
    2. Multi-Head Self-Attention (with optional RoPE)
    3. Residual Connection with DropPath
    4. Layer Normalization
    5. MLP
    6. Residual Connection with DropPath

    Parameters
    ----------
    dim : int
        Input and output embedding dimension.
    num_heads : int
        Number of attention heads.
    mlp_ratio : float, default=4.0
        Ratio of MLP hidden dimension to embedding dimension.
    qkv_bias : bool, default=False
        If True, add a learnable bias to query, key, value projections.
    drop : float, default=0.0
        Dropout probability for MLP and projection layers.
    attn_drop : float, default=0.0
        Dropout probability for attention weights.
    drop_path : float, default=0.0
        Stochastic depth rate.
    act_layer : nn.Module, default=nn.GELU
        Activation function.
    norm_layer : nn.Module, default=nn.LayerNorm
        Normalization layer.
    is_causal : bool, default=False
        If True, applies a causal mask to attention.
    use_rope : bool, default=False
        If True, applies Rotary Positional Embeddings (RoPE).
    return_attention : bool, default=False
        If True, returns attention weights instead of the output tensor.
    """

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
        self.return_attention = return_attention
        self.norm1 = norm_layer(dim)
        self.attn = _Attention(
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
        self.mlp = _MLP(
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
    """
    Splits the EEG signal into patches and embeds them.

    This layer transforms the input EEG signal (2D: channels x time) into a sequence of
    patch embeddings using a 2D convolution. It treats the channel dimension effectively
    as height=1 for the convolution, sliding over the time dimension.

    Parameters
    ----------
    n_chans : int, default=64
        Number of input channels.
    n_times : int, default=1000
        Number of time samples.
    patch_size : int, default=16
        Size of each patch along the time dimension.
    patch_stride : int, optional
        Stride between patches. If None, defaults to `patch_size` (non-overlapping).
    embed_dim : int, default=768
        Dimension of the output embeddings.
    """

    def __init__(
        self,
        n_chans=64,
        n_times=1000,
        patch_size=16,
        patch_stride=None,
        embed_dim=768,
    ):
        super().__init__()
        self.n_chans = n_chans
        self.n_times = n_times
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        if patch_stride is None:
            self.num_patches = (n_chans, n_times // patch_size)
        else:
            self.num_patches = (
                n_chans,
                (n_times - patch_size) // patch_stride + 1,
            )

        self.proj = nn.Conv2d(
            1,
            embed_dim,
            kernel_size=(1, patch_size),
            stride=(1, patch_size if patch_stride is None else patch_stride),
        )

    def forward(self, x):
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input EEG signal of shape (batch, n_chans, n_times).

        Returns
        -------
        torch.Tensor
            Patch embeddings of shape (batch, n_patches, n_chans, embed_dim).
        """
        # x: batch, n_chans, n_times
        x = rearrange(x, "batch n_chans n_times -> batch 1 n_chans n_times")
        # Convolve and rearrange:
        # (batch, embed_dim, n_chans, n_patches) -> (batch, n_patches, n_chans, embed_dim)
        x = self.proj(x)
        x = rearrange(
            x,
            "batch embed_dim n_chans n_patches -> batch n_patches n_chans embed_dim",
        )
        return x


class _PatchNormEmbed(nn.Module):
    """
    Alternative Patch Embedding with Unfold and LayerNorm.

    This layer splits the signal into patches using `torch.nn.Unfold`, applies
    layer normalization to each patch, and then projects it to the embedding dimension.
    This is an alternative to the Convolution-based `PatchEmbed`.

    Parameters
    ----------
    n_chans : int, default=64
        Number of input channels.
    n_times : int, default=1000
        Number of time samples.
    patch_size : int, default=16
        Size of each patch along the time dimension.
    patch_stride : int, optional
        Stride between patches. If None, defaults to `patch_size`.
    embed_dim : int, default=768
        Dimension of the output embeddings.
    """

    def __init__(
        self,
        n_chans=64,
        n_times=1000,
        patch_size=16,
        patch_stride=None,
        embed_dim=768,
    ):
        super().__init__()

        assert n_times % patch_size == 0

        self.n_chans = n_chans
        self.n_times = n_times
        self.patch_size = patch_size
        self.patch_stride = patch_stride

        if patch_stride is None:
            self.num_patches = (n_chans, n_times // patch_size)
        else:
            self.num_patches = (
                n_chans,
                (n_times - patch_size) // patch_stride + 1,
            )

        self.unfold = torch.nn.Unfold(
            kernel_size=(1, patch_size),
            stride=(1, patch_stride if patch_stride is not None else patch_size),
        )

        self.proj = nn.Linear(patch_size, embed_dim)  # +2

    def forward(self, x):
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input EEG signal of shape (batch, n_chans, n_times).

        Returns
        -------
        torch.Tensor
            Patch embeddings of shape (batch, n_patches, n_chans, embed_dim).
        """
        # x: batch, n_chans, n_times
        x = rearrange(x, "batch n_chans n_times -> batch 1 n_chans n_times")

        x = self.unfold(x)
        # (batch, patch_size, n_patches * n_chans)

        # Rearrange using Einops:
        # 1. Split dim 2 into (n_chans, n_patches) - Unfold iterates time then channel
        # 2. Transpose to (batch, n_patches, n_chans, patch_size)
        x = rearrange(
            x,
            "batch patch_size (n_chans n_patches) -> batch n_patches n_chans patch_size",
            n_chans=self.n_chans,
        )

        x = torch.layer_norm(x, (self.patch_size,))

        x = self.proj(x)  # (batch, n_patches, n_chans, embed_dim)

        return x


class _EEGTransformer(nn.Module):
    """
    Backbone of the EEGPT model processing the sequence of patch embeddings.

    This standard Transformer encoder processes the sequence of patch embeddings
    augmented with channel embeddings and a summary token.

    Parameters
    ----------
    n_chans : int
        Number of input channels.
    n_times : int
        Number of time samples.
    patch_size : int
        Size of each patch.
    patch_stride : int
        Stride between patches.
    embed_dim : int
        Embedding dimension.
    embed_num : int
        Number of summary tokens.
    depth : int
        Number of Transformer blocks.
    num_heads : int
        Number of attention heads.
    mlp_ratio : float
        Ratio of MLP hidden dimension to embedding dimension.
    qkv_bias : bool
        If True, add bias to QKV projections.
    drop_rate : float
        Dropout rate.
    attn_drop_rate : float
        Attention dropout rate.
    drop_path_rate : float
        Stochastic depth rate.
    norm_layer : nn.Module
        Normalization layer.
    patch_module : nn.Module
        Module used for patch embedding (e.g., `PatchEmbed`).
    init_std : float
        Standard deviation for weight initialization.
    """

    def __init__(
        self,
        n_chans=64,
        n_times=1000,
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
            n_chans=n_chans,
            n_times=n_times,
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
                _Block(
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
        if channels is None:
            # If no channel names are provided, use sequential IDs
            return torch.arange(self.patch_embed.n_chans)

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
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input EEG data of shape (batch, n_chans, n_times).
        chan_ids : torch.Tensor, optional
            Channel IDs for channel embedding.
        mask_x : torch.Tensor, optional
            Mask for input patches.
        mask_t : torch.Tensor, optional
            Mask for temporal patches.

        Returns
        -------
        torch.Tensor
            Transformer output.
        """
        # x shape: (batch, n_chans, n_times)

        # -- patchify x
        x = self.patch_embed(x)
        # x shape: (batch, n_patches, n_chans, embed_dim)
        batch, n_patches, n_chans, embed_dim = x.shape

        assert n_patches == self.num_patches[1] and n_chans == self.num_patches[0], (
            f"{n_patches}=={self.num_patches[1]} and {n_chans}=={self.num_patches[0]}"
        )

        if chan_ids is None:
            chan_ids = torch.arange(0, n_chans)
        chan_ids = chan_ids.to(x.device)

        # -- add channels positional embedding to x
        # chan_embed shape: (1, 1, n_chans, embed_dim)
        x = x + self.chan_embed(chan_ids.long()).unsqueeze(0)

        if mask_x is not None:
            mask_x = mask_x.to(x.device)
            x = apply_mask(mask_x, x)
            # x shape might change here if masking removes tokens
            batch, n_patches, n_chans, embed_dim = x.shape

        # Flatten batch and patches dimensions for Transformer processing
        # (batch, n_patches, n_chans, embed_dim) -> (batch * n_patches, n_chans, embed_dim)
        x = rearrange(
            x,
            "batch n_patches n_chans embed_dim -> (batch n_patches) n_chans embed_dim",
        )

        # -- concat summary token
        # summary_token shape: (batch * n_patches, embed_num, embed_dim)
        summary_token = self.summary_token.repeat((x.shape[0], 1, 1))
        x = torch.cat([x, summary_token], dim=1)
        # x shape: (batch * n_patches, n_chans + embed_num, embed_dim)

        # -- fwd prop
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if blk.return_attention:
                return x

        # Extract only the summary tokens (used for global representation)
        # x shape: (batch * n_patches, embed_num, embed_dim)
        x = x[:, -summary_token.shape[1] :, :]

        if self.norm is not None:
            x = self.norm(x)

        # Flatten summary tokens
        # x = x.flatten(-2)
        # x shape: (batch * n_patches, embed_num * embed_dim)
        # -- reshape back to separate batch and patches
        # x = x.reshape((batch, n_patches, -1))
        # x shape: (batch, n_patches, embed_num * embed_dim)

        # Instead of flatten+reshape, let's just rearrange back to separate batch/patches explicitly
        x = rearrange(
            x,
            "(batch n_patches) embed_num embed_dim -> batch n_patches (embed_num embed_dim)",
            batch=batch,
        )

        if mask_t is not None:
            mask_t = mask_t.to(x.device)
            x = apply_mask_t(mask_t, x)

        # Reshape to final output format: (batch, n_patches, embed_num, embed_dim)
        x = rearrange(
            x,
            "batch n_patches (embed_num embed_dim) -> batch n_patches embed_num embed_dim",
            embed_num=self.embed_num,
        )

        return x
