# Authors: Pierre Guetschel
#
# Code adapted from https://github.com/wjq-learning/CBraMod
#
# License: BSD (3-clause)


import copy
import logging
from typing import Optional, Sequence

import torch
from einops import rearrange
from einops.layers.torch import Rearrange
from torch import Tensor, nn

from braindecode.models.base import EEGModuleMixin
from braindecode.modules import CrissCrossTransformerEncoderLayer

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

    .. rubric:: Pretraining Highlights

    - **Pretraining Dataset**: Temple University Hospital EEG Corpus (TUEG), the largest public EEG corpus
    - **Pretraining Task**: Self-supervised masked EEG patch reconstruction from both time-domain
      and frequency-domain EEG signals
    - **Model Parameters**: ~4.0M parameters (very compact compared to other foundation models)
    - **Fast Convergence**: Achieves decent results in first epoch on downstream tasks,
      full convergence within ~10 epochs (vs. ~30 for supervised models like EEGConformer)

    .. rubric:: Macro Components

    - **Patch Encoding Network**: Converts raw EEG patches into embeddings
    - **Asymmetric Conditional Positional Encoding (ACPE)**: Generates spatial-temporal positional
      embeddings adaptively from input EEG format
    - **Criss-Cross Transformer Blocks** (12 layers): Alternates spatial and temporal attention
      to learn EEG representations
    - **Reconstruction Head**: Reconstructs masked EEG patches during pretraining
    - **Task head** (``final_layer``): flatten summary tokens across patches and map to
       ``n_outputs``; if ``return_encoder_output=True``, return the encoder features instead.

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

    .. important::
       **Pre-trained Weights Available**

       This model has pre-trained weights available on the Hugging Face Hub.
       You can load them using:

       .. code-block:: python

           from braindecode.models import CBraMod

           # Load pre-trained model from Hugging Face Hub
           model = CBraMod.from_pretrained(
               "braindecode/cbramod-pretrained", return_encoder_output=True
           )

       To push your own trained model to the Hub:

       .. code-block:: python

           # After training your model
           model.push_to_hub(
               repo_id="username/my-cbramod-model", commit_message="Upload trained CBraMod model"
           )

       Requires installing ``braindecode[hug]`` for Hub integration.

    Parameters
    ----------
    patch_size : int, default=200
        Temporal patch size in samples (200 samples = 1 second at 200 Hz).
    dim_feedforward : int, default=800
        Dimension of the feedforward network in Transformer layers.
    n_layer : int, default=12
        Number of Transformer layers.
    nhead : int, default=8
        Number of attention heads.
    activation : type[nn.Module], default=nn.GELU
        Activation function used in Transformer feedforward layers.
    emb_dim : int, default=200
        Output embedding dimension.
    drop_prob : float, default=0.1
        Dropout probability.
    return_encoder_output : bool, default=False
        If false (default), the features are flattened and passed through a final linear layer
        to produce class logits of size ``n_outputs``.
        If True, the model returns the encoder output features.

    References
    ----------
    .. [cbramod] Wang, J., Zhao, S., Luo, Z., Zhou, Y., Jiang, H., Li, S., Li, T., & Pan, G. (2025).
       CBraMod: A Criss-Cross Brain Foundation Model for EEG Decoding.
       In The Thirteenth International Conference on Learning Representations (ICLR 2025).
       https://arxiv.org/abs/2412.07236
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
        dim_feedforward: int = 800,
        n_layer: int = 12,
        nhead: int = 8,
        activation: type[nn.Module] = nn.GELU,
        emb_dim: int = 200,
        channels_kernel_stride_padding_norm: Sequence[
            tuple[int, int, int, int, tuple[int, int]]
        ] = (
            (25, 49, 25, 24, (5, 25)),
            (25, 3, 1, 1, (5, 25)),
            (25, 3, 1, 1, (5, 25)),
        ),
        drop_prob: float = 0.1,
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
        del n_chans, chs_info, n_times, input_window_seconds, sfreq, n_outputs
        self.rearrange = Rearrange("b c (n p) -> b c n p", p=patch_size)
        self.patch_embedding = _PatchEmbedding(
            patch_size,
            channels_kernel_stride_padding_norm,
            drop_prob=drop_prob,
        )
        d_model = self.patch_embedding.d_model
        encoder_layer = CrissCrossTransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            norm_first=True,
            activation=activation,
            dropout=drop_prob,
        )
        self.encoder = _TransformerEncoder(encoder_layer, num_layers=n_layer)
        self.proj_out = nn.Sequential(nn.Linear(d_model, emb_dim))

        self._weights_init()

        self.final_layer = (
            nn.Identity()
            if return_encoder_output
            else nn.Sequential(nn.Flatten(), nn.LazyLinear(self.n_outputs))
        )

    def _weights_init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, mask=None):
        x = self.rearrange(x)
        patch_emb = self.patch_embedding(x, mask)
        feats = self.encoder(patch_emb)
        out = self.proj_out(feats)
        return self.final_layer(out)


class _PatchEmbedding(nn.Module):
    def __init__(
        self,
        patch_size,
        channels_kernel_stride_padding_norm,
        drop_prob=0.1,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.channels_kernel_stride_padding_norm = channels_kernel_stride_padding_norm
        self.positional_encoding = nn.Sequential(
            nn.Conv2d(
                in_channels=self.d_model,
                out_channels=self.d_model,
                kernel_size=(19, 7),
                stride=(1, 1),
                padding=(9, 3),
                groups=self.d_model,
            ),
        )
        self.mask_encoding = nn.Parameter(torch.zeros(patch_size), requires_grad=False)

        last_channels = 1
        proj_in_layers = []
        for (
            channels,
            kernel,
            stride,
            padding,
            norm,
        ) in channels_kernel_stride_padding_norm:
            proj_in_layers.extend(
                [
                    nn.Conv2d(
                        in_channels=last_channels,
                        out_channels=channels,
                        kernel_size=(1, kernel),
                        stride=(1, stride),
                        padding=(0, padding),
                    ),
                    nn.GroupNorm(*norm),
                    nn.GELU(),
                ]
            )
            last_channels = channels
        self.proj_in = nn.Sequential(*proj_in_layers)
        self.spectral_proj = nn.Sequential(
            nn.Linear(patch_size // 2 + 1, self.d_model),
            nn.Dropout(drop_prob),
        )

    @property
    def d_model(self):
        last_channels = self.channels_kernel_stride_padding_norm[-1][0]
        patch_size = self.patch_size
        for _, kernel, stride, padding, _ in self.channels_kernel_stride_padding_norm:
            patch_size = int((patch_size + 2 * padding - kernel) / stride + 1)
        return last_channels * patch_size

    def forward(self, x, mask=None):
        bz, ch_num, patch_num, patch_size = x.shape
        if mask is None:
            mask_x = x
        else:
            mask_x = x.clone()
            mask_x[mask == 1] = self.mask_encoding

        mask_x = rearrange(mask_x, "b c n p -> b 1 (c n) p")
        patch_emb = self.proj_in(mask_x)
        patch_emb = rearrange(patch_emb, "b d (c n) p2 -> b c n (d p2)", c=ch_num)

        mask_x = rearrange(mask_x, "b 1 (c n) p -> (b c n) p", c=ch_num)
        spectral = torch.fft.rfft(mask_x, dim=-1, norm="forward")
        spectral = rearrange(
            torch.abs(spectral), "(b c n) p -> b c n p", b=bz, c=ch_num, p=101
        )
        spectral_emb = self.spectral_proj(spectral)

        patch_emb = patch_emb + spectral_emb

        positional_embedding = self.positional_encoding(
            rearrange(patch_emb, "b c n p -> b p c n", p=self.d_model)
        )  # d for sanity check
        positional_embedding = rearrange(positional_embedding, "b p c n -> b c n p")

        patch_emb = patch_emb + positional_embedding

        return patch_emb


class _TransformerEncoder(nn.Module):
    def __init__(
        self,
        encoder_layer,
        num_layers,
        norm=None,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [copy.deepcopy(encoder_layer) for i in range(num_layers)]
        )
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
