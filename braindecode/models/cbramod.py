# Authors: Pierre Guetschel
#
# Code adapted from https://github.com/wjq-learning/CBraMod
#
# License: BSD (3-clause)


import copy
import logging
from typing import Optional

import torch
from einops.layers.torch import Rearrange
from torch import Tensor, nn

from braindecode.models.base import EEGModuleMixin

logger = logging.getLogger(__name__)


class CBraMod(EEGModuleMixin, nn.Module):
    r"""
    **C**\ riss-\ **C**\ ross **Bra**\ in **Mod**\ el for EEG Decoding from Wang et al. (2025) [cbramod]_.

    :bdg-danger:`Foundation Model` :bdg-info:`Attention/Transformer`

    .. figure:: https://raw.githubusercontent.com/wjq-learning/CBraMod/refs/heads/main/figure/model.png
        :align: center
        :alt:  CBraMod pre-training overview
        :width: 1000px

    CBraMod is a foundation model for EEG decoding that leverages a novel criss-cross transformer
    architecture to effectively model the unique spatial and temporal characteristics of EEG signals.
    Pre-trained on the Temple University Hospital EEG Corpus (TUEG)—the largest public EEG corpus—
    using masked EEG patch reconstruction, CBraMod achieves state-of-the-art performance across
    diverse downstream BCI and clinical applications.

    .. rubric:: Key Innovation: Criss-Cross Attention

    Unlike existing EEG foundation models that use full attention to model all spatial and temporal
    dependencies together, CBraMod separates spatial and temporal dependencies through a
    **criss-cross transformer** architecture:

    - **Spatial Attention**: Models dependencies between channels while keeping patches separate
    - **Temporal Attention**: Models dependencies between temporal patches while keeping channels separate

    This design is inspired by criss-cross strategies from computer vision and effectively
    leverages the inherent structural characteristics of EEG signals. The criss-cross approach
    reduces computational complexity (FLOPs reduced by ~32% compared to full attention) while
    improving performance and enabling faster convergence.

    .. rubric:: Asymmetric Conditional Positional Encoding (ACPE)

    Rather than using fixed positional embeddings, CBraMod employs **Asymmetric Conditional
    Positional Encoding** that dynamically generates positional embeddings using a convolutional
    network. This enables the model to:

    - Capture relative positional information adaptively
    - Handle diverse EEG channel formats (different channel counts and reference schemes)
    - Generalize to arbitrary downstream EEG formats without retraining
    - Support various reference schemes (earlobe, average, REST, bipolar)

    .. rubric:: Pretraining & Generalization

    - **Pretraining Dataset**: Temple University Hospital EEG Corpus (TUEG), the largest public EEG corpus
    - **Pretraining Task**: Self-supervised masked EEG patch reconstruction from both time-domain
      and frequency-domain EEG signals
    - **Model Parameters**: ~4.0M parameters (very compact compared to other foundation models)
    - **Fast Convergence**: Achieves decent results in first epoch on downstream tasks,
      full convergence within ~10 epochs (vs. ~30 for supervised models like EEGConformer)

    CBraMod has been comprehensively evaluated on **10 downstream BCI tasks across 12 public datasets**
    sourced from institutions different from the pretraining dataset, demonstrating strong
    generalization capabilities:

    - Motor imagery (MI) classification
    - Emotion recognition
    - Seizure detection
    - Sleep staging
    - And others

    Consistently outperforms strong baselines including EEGNet, EEGConformer, BIOT, and LaBraM.

    .. rubric:: Macro Components

    - **Patch Encoding Network**: Converts raw EEG patches into embeddings
    - **Asymmetric Conditional Positional Encoding (ACPE)**: Generates spatial-temporal positional
      embeddings adaptively from input EEG format
    - **Criss-Cross Transformer Blocks** (12 layers): Alternates spatial and temporal attention
      to learn EEG representations
    - **Reconstruction Head**: Reconstructs masked EEG patches during pretraining

    The model is highly efficient, requiring only ~318.9M FLOPs on a typical 16-channel, 10-second
    EEG recording (significantly lower than full attention baselines).

    .. rubric:: Known Limitations

    - **Data Quality**: TUEG corpus contains "dirty data"; pretraining used crude filtering,
      reducing available pre-training data
    - **Channel Dependency**: Performance degrades with very sparse electrode setups (e.g., <4 channels)
    - **Computational Resources**: While efficient, foundation models have higher deployment
      requirements than lightweight models
    - **Limited Scaling Exploration**: Future work should explore scaling laws at billion-parameter levels
      and integration with large pre-trained vision/language models

    .. rubric:: Usage Example

    .. code-block:: python

        from braindecode.models import CBraMod

        # Create model
        model = CBraMod(
            n_outputs=4,  # e.g., 4-class motor imagery
            n_chans=22,  # e.g., 22 channels
            n_times=1000,  # e.g., 5 seconds at 200 Hz
        )

        # Forward pass: (batch, n_chans, n_times) -> (batch, n_outputs)
        x = torch.randn(batch_size, 22, 1000)
        output = model(x)

    Parameters
    ----------
    patch_size : int, default=200
        Temporal patch size in samples (200 samples = 1 second at 200 Hz).
    d_model : int, default=200
        Dimension of the embedding space.
    dim_feedforward : int, default=800
        Dimension of the feedforward network in Transformer layers.
    n_layer : int, default=22
        Number of Transformer layers.
    nhead : int, default=8
        Number of attention heads.
    activation : type[nn.Module], default=nn.GELU
        Activation function used in Transformer feedforward layers.
    emb_dim : int, default=200
        Output embedding dimension.

    References
    ----------
    .. [cbramod] Wang, J., Zhao, S., Luo, Z., Zhou, Y., Jiang, H., Li, S., Li, T., & Pan, G. (2025).
       CBraMod: A Criss-Cross Brain Foundation Model for EEG Decoding.
       In The Thirteenth International Conference on Learning Representations (ICLR 2025).
       https://arxiv.org/abs/2412.07236

    Notes
    -----
    - Pretraining uses masked EEG patch reconstruction on TUEG corpus
    - Model supports arbitrary EEG channel counts through flexible positional encoding
    - Compatible with various EEG reference schemes
    - The criss-cross attention mechanism provides interpretability through visualizable
      attention weights showing spatial and temporal patterns learned by the model
    """

    def __init__(
        self,
        n_outputs=None,
        n_chans=None,
        chs_info=None,
        n_times=None,
        input_window_seconds=None,
        sfreq=None,
        patch_size: int = 200,
        d_model: int = 200,
        dim_feedforward: int = 800,
        n_layer: int = 12,
        nhead: int = 8,
        activation: type[nn.Module] = nn.GELU,
        emb_dim: int = 200,
        drop_prob: float = 0.1,
    ):
        super().__init__(
            n_outputs=n_outputs,
            n_chans=n_chans,
            chs_info=chs_info,
            n_times=n_times,
            input_window_seconds=input_window_seconds,
            sfreq=sfreq,
        )
        del n_chans, chs_info, n_times, input_window_seconds, sfreq, n_outputs
        self.rearrange = Rearrange("b c (n p) -> b c n p", p=patch_size)
        self.patch_embedding = PatchEmbedding(patch_size, d_model, drop_prob=drop_prob)
        encoder_layer = CrissCrossTransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            norm_first=True,
            activation=activation,
            dropout=drop_prob,
        )
        self.encoder = TransformerEncoder(
            encoder_layer, num_layers=n_layer, enable_nested_tensor=False
        )
        self.proj_out = nn.Sequential(nn.Linear(d_model, emb_dim))

        self.apply(_weights_init)

    def forward(self, x, mask=None):
        x = self.rearrange(x)
        patch_emb = self.patch_embedding(x, mask)
        feats = self.encoder(patch_emb)
        out = self.proj_out(feats)
        return out


class PatchEmbedding(nn.Module):
    def __init__(self, patch_size, d_model, drop_prob=0.1):
        super().__init__()
        self.d_model = d_model
        self.positional_encoding = nn.Sequential(
            nn.Conv2d(
                in_channels=d_model,
                out_channels=d_model,
                kernel_size=(19, 7),
                stride=(1, 1),
                padding=(9, 3),
                groups=d_model,
            ),
        )
        self.mask_encoding = nn.Parameter(torch.zeros(patch_size), requires_grad=False)

        self.proj_in = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=25,
                kernel_size=(1, 49),
                stride=(1, 25),
                padding=(0, 24),
            ),
            nn.GroupNorm(5, 25),
            nn.GELU(),
            nn.Conv2d(
                in_channels=25,
                out_channels=25,
                kernel_size=(1, 3),
                stride=(1, 1),
                padding=(0, 1),
            ),
            nn.GroupNorm(5, 25),
            nn.GELU(),
            nn.Conv2d(
                in_channels=25,
                out_channels=25,
                kernel_size=(1, 3),
                stride=(1, 1),
                padding=(0, 1),
            ),
            nn.GroupNorm(5, 25),
            nn.GELU(),
        )
        self.spectral_proj = nn.Sequential(
            nn.Linear(101, d_model),
            nn.Dropout(drop_prob),
        )

    def forward(self, x, mask=None):
        bz, ch_num, patch_num, patch_size = x.shape
        if mask == None:
            mask_x = x
        else:
            mask_x = x.clone()
            mask_x[mask == 1] = self.mask_encoding

        mask_x = mask_x.contiguous().view(bz, 1, ch_num * patch_num, patch_size)
        patch_emb = self.proj_in(mask_x)
        patch_emb = (
            patch_emb.permute(0, 2, 1, 3)
            .contiguous()
            .view(bz, ch_num, patch_num, self.d_model)
        )

        mask_x = mask_x.contiguous().view(bz * ch_num * patch_num, patch_size)
        spectral = torch.fft.rfft(mask_x, dim=-1, norm="forward")
        spectral = torch.abs(spectral).contiguous().view(bz, ch_num, patch_num, 101)
        spectral_emb = self.spectral_proj(spectral)
        # print(patch_emb[5, 5, 5, :])
        # print(spectral_emb[5, 5, 5, :])
        patch_emb = patch_emb + spectral_emb

        positional_embedding = self.positional_encoding(patch_emb.permute(0, 3, 1, 2))
        positional_embedding = positional_embedding.permute(0, 2, 3, 1)

        patch_emb = patch_emb + positional_embedding

        return patch_emb


def _weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
    if isinstance(m, nn.Conv1d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        encoder_layer,
        num_layers,
        norm=None,
        enable_nested_tensor=True,
        mask_check=True,
    ):
        super().__init__()
        torch._C._log_api_usage_once(f"torch.nn.modules.{self.__class__.__name__}")
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(
        self,
        src: Tensor,
        mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        is_causal: Optional[bool] = None,
    ) -> Tensor:
        output = src
        for mod in self.layers:
            output = mod(output, src_mask=mask)
        if self.norm is not None:
            output = self.norm(output)
        return output


class CrissCrossTransformerEncoderLayer(nn.Module):
    __constants__ = ["norm_first"]

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: type[nn.Module] = nn.GELU,
        layer_norm_eps: float = 1e-5,
        batch_first: bool = False,
        norm_first: bool = False,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.self_attn_s = nn.MultiheadAttention(
            d_model // 2,
            nhead // 2,
            dropout=dropout,
            bias=bias,
            batch_first=batch_first,
            **factory_kwargs,
        )
        self.self_attn_t = nn.MultiheadAttention(
            d_model // 2,
            nhead // 2,
            dropout=dropout,
            bias=bias,
            batch_first=batch_first,
            **factory_kwargs,
        )

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, bias=bias, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = activation()

    def forward(
        self,
        src: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        is_causal: bool = False,
    ) -> Tensor:
        x = src
        x = x + self._sa_block(
            self.norm1(x), src_mask, src_key_padding_mask, is_causal=is_causal
        )
        x = x + self._ff_block(self.norm2(x))
        return x

    # self-attention block
    def _sa_block(
        self,
        x: Tensor,
        attn_mask: Optional[Tensor],
        key_padding_mask: Optional[Tensor],
        is_causal: bool = False,
    ) -> Tensor:
        bz, ch_num, patch_num, patch_size = x.shape
        xs = x[:, :, :, : patch_size // 2]
        xt = x[:, :, :, patch_size // 2 :]
        xs = (
            xs.transpose(1, 2)
            .contiguous()
            .view(bz * patch_num, ch_num, patch_size // 2)
        )
        xt = xt.contiguous().view(bz * ch_num, patch_num, patch_size // 2)
        xs = self.self_attn_s(
            xs,
            xs,
            xs,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        xs = (
            xs.contiguous().view(bz, patch_num, ch_num, patch_size // 2).transpose(1, 2)
        )
        xt = self.self_attn_t(
            xt,
            xt,
            xt,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        xt = xt.contiguous().view(bz, ch_num, patch_num, patch_size // 2)
        x = torch.concat((xs, xt), dim=3)
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


def _get_clones(module, N):
    # FIXME: copy.deepcopy() is not defined on nn.module
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
