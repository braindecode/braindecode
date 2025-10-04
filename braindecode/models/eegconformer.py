# Authors: Yonghao Song <eeyhsong@gmail.com>
#
# License: BSD (3-clause)
import warnings

import torch
from einops.layers.torch import Rearrange
from torch import Tensor, nn

from braindecode.models.base import EEGModuleMixin
from braindecode.modules import FeedForwardBlock, MultiHeadAttention


class EEGConformer(EEGModuleMixin, nn.Module):
    """EEG Conformer from Song et al. (2022) [song2022]_.

    :bdg-success:`Convolution` :bdg-info:`Small Attention`

    .. figure:: https://raw.githubusercontent.com/eeyhsong/EEG-Conformer/refs/heads/main/visualization/Fig1.png
        :align: center
        :alt: EEGConformer Architecture
        :width: 600px


    .. rubric:: Architectural Overview

    EEG-Conformer is a *convolution-first* model augmented with a *lightweight transformer
    encoder*. The end-to-end flow is:

    - (i) :class:`_PatchEmbedding` converts the continuous EEG into a compact sequence of tokens via a
      :class:`ShallowFBCSPNet` temporal–spatial conv stem and temporal pooling;
    - (ii) :class:`_TransformerEncoder` applies small multi-head self-attention to integrate
      longer-range temporal context across tokens;
    - (iii) :class:`_ClassificationHead` aggregates the sequence and performs a linear readout.
      This preserves the strong inductive biases of shallow CNN filter banks while adding
      just enough attention to capture dependencies beyond the pooling horizon [song2022]_.

    .. rubric:: Macro Components

    - :class:`_PatchEmbedding` **(Shallow conv stem → tokens)**

        - *Operations.*
        - A temporal convolution (`:class:torch.nn.Conv2d`) ``(1 x L_t)`` forms a data-driven "filter bank";
        - A spatial convolution (`:class:torch.nn.Conv2d`) (n_chans x 1)`` projects across electrodes,
          collapsing the channel axis into a virtual channel.
        - **Normalization function** :class:`torch.nn.BatchNorm`
        - **Activation function** :class:`torch.nn.ELU`
        - **Average Pooling** :class:`torch.nn.AvgPool` along time (kernel ``(1, P)`` with stride ``(1, S)``)
        -  final ``1x1`` :class:`torch.nn.Linear` projection.

    The result is rearranged to a token sequence ``(B, S_tokens, D)``, where ``D = n_filters_time``.

    *Interpretability/robustness.* Temporal kernels can be inspected as FIR filters;
    the spatial conv yields channel projections analogous to :class:`ShallowFBCSPNet`’s learned
    spatial filters. Temporal pooling stabilizes statistics and reduces sequence length.

    - :class:`_TransformerEncoder` **(context over temporal tokens)**

        - *Operations.*
        - A stack of ``att_depth`` encoder blocks. :class:`_TransformerEncoderBlock`
        - Each block applies LayerNorm :class:`torch.nn.LayerNorm`
        - Multi-Head Self-Attention (``att_heads``) with dropout + residual :class:`MultiHeadAttention` (:class:`torch.nn.Dropout`)
        - LayerNorm :class:`torch.nn.LayerNorm`
        - 2-layer feed-forward (≈4x expansion, :class:`torch.nn.GELU`) with dropout + residual.

    Shapes remain ``(B, S_tokens, D)`` throughout.

    *Role.* Small attention focuses on interactions among *temporal patches* (not channels),
    extending effective receptive fields at modest cost.

    - :class:`ClassificationHead` **(aggregation + readout)**

        - *Operations*.
        - Flatten, :class:`torch.nn.Flatten` the sequence ``(B, S_tokens·D)`` -
        - MLP (:class:`torch.nn.Linear` → activation (default: :class:`torch.nn.ELU`) → :class:`torch.nn.Dropout` → :class:`torch.nn.Linear`)
        - final Linear to classes.

    With ``return_features=True``, features before the last Linear can be exported for
    linear probing or downstream tasks.

    .. rubric:: Convolutional Details

    - **Temporal (where time-domain patterns are learned).**
        The initial ``(1 x L_t)`` conv per channel acts as a *learned filter bank* for oscillatory
        bands and transients. Subsequent **AvgPool** along time performs local integration,
        converting activations into “patches” (tokens). Pool length/stride control the
        token rate and set the lower bound on temporal context within each token.

    - **Spatial (how electrodes are processed).**
        A single conv with kernel ``(n_chans x 1)`` spans the full montage to learn spatial
        projections for each temporal feature map, collapsing the channel axis into a
        virtual channel before tokenization. This mirrors the shallow spatial step in
        :class:`ShallowFBCSPNet` (temporal filters → spatial projection → temporal condensation).

    - **Spectral (how frequency content is captured).**
        No explicit Fourier/wavelet stage is used. Spectral selectivity emerges implicitly
        from the learned temporal kernels; pooling further smooths high-frequency noise.
        The effective spectral resolution is thus governed by ``L_t`` and the pooling
        configuration.

    .. rubric:: Attention / Sequential Modules

    - **Type.** Standard multi-head self-attention (MHA) with ``att_heads`` heads over the token sequence.
    - **Shapes.** Input/Output: ``(B, S_tokens, D)``; attention operates along the ``S_tokens`` axis.
    - **Role.** Re-weights and integrates evidence across pooled windows, capturing dependencies
      longer than any single token while leaving channel relationships to the convolutional stem.
      The design is intentionally *small*—attention refines rather than replaces convolutional feature extraction.

    .. rubric:: Additional Mechanisms

    - **Parallel with ShallowFBCSPNet.** Both begin with a learned temporal filter bank,
        spatial projection across electrodes, and early temporal condensation.
        :class:`ShallowFBCSPNet` then computes band-power (via squaring/log-variance), whereas
        EEG-Conformer applies BN/ELU and **continues with attention** over tokens to
        refine temporal context before classification.

    - **Tokenization knob.** ``pool_time_length`` and especially ``pool_time_stride`` set
        the number of tokens ``S_tokens``. Smaller strides → more tokens and higher attention
        capacity (but higher compute); larger strides → fewer tokens and stronger inductive bias.

    - **Embedding dimension = filters.** ``n_filters_time`` serves double duty as both the
        number of temporal filters in the stem and the transformer’s embedding size ``D``,
        simplifying dimensional alignment.

    .. rubric:: Usage and Configuration

    - **Instantiation.** Choose ``n_filters_time`` (embedding size ``D``) and
        ``filter_time_length`` to match the rhythms of interest. Tune
        ``pool_time_length/stride`` to trade temporal resolution for sequence length.
        Keep ``att_depth`` modest (e.g., 4–6) and set ``att_heads`` to divide ``D``.
        ``final_fc_length="auto"`` infers the flattened size from PatchEmbedding.

    Notes
    -----
    The authors recommend using data augmentation before using Conformer,
    e.g. segmentation and recombination,
    Please refer to the original paper and code for more details [ConformerCode]_.

    The model was initially tuned on 4 seconds of 250 Hz data.
    Please adjust the scale of the temporal convolutional layer,
    and the pooling layer for better performance.

    .. versionadded:: 0.8

    We aggregate the parameters based on the parts of the models, or
    when the parameters were used first, e.g. ``n_filters_time``.

    .. versionadded:: 1.1


    Parameters
    ----------
    n_filters_time: int
        Number of temporal filters, defines also embedding size.
    filter_time_length: int
        Length of the temporal filter.
    pool_time_length: int
        Length of temporal pooling filter.
    pool_time_stride: int
        Length of stride between temporal pooling filters.
    drop_prob: float
        Dropout rate of the convolutional layer.
    att_depth: int
        Number of self-attention layers.
    att_heads: int
        Number of attention heads.
    att_drop_prob: float
        Dropout rate of the self-attention layer.
    final_fc_length: int | str
        The dimension of the fully connected layer.
    return_features: bool
        If True, the forward method returns the features before the
        last classification layer. Defaults to False.
    activation: nn.Module
        Activation function as parameter. Default is nn.ELU
    activation_transfor: nn.Module
        Activation function as parameter, applied at the FeedForwardBlock module
        inside the transformer. Default is nn.GeLU

    References
    ----------
    .. [song2022] Song, Y., Zheng, Q., Liu, B. and Gao, X., 2022. EEG
       conformer: Convolutional transformer for EEG decoding and visualization.
       IEEE Transactions on Neural Systems and Rehabilitation Engineering,
       31, pp.710-719. https://ieeexplore.ieee.org/document/9991178
    .. [ConformerCode] Song, Y., Zheng, Q., Liu, B. and Gao, X., 2022. EEG
       conformer: Convolutional transformer for EEG decoding and visualization.
       https://github.com/eeyhsong/EEG-Conformer.
    """

    def __init__(
        self,
        n_outputs=None,
        n_chans=None,
        n_filters_time=40,
        filter_time_length=25,
        pool_time_length=75,
        pool_time_stride=15,
        drop_prob=0.5,
        att_depth=6,
        att_heads=10,
        att_drop_prob=0.5,
        final_fc_length="auto",
        return_features=False,
        activation: nn.Module = nn.ELU,
        activation_transfor: nn.Module = nn.GELU,
        n_times=None,
        chs_info=None,
        input_window_seconds=None,
        sfreq=None,
    ):
        super().__init__(
            n_outputs=n_outputs,
            n_chans=n_chans,
            chs_info=chs_info,
            n_times=n_times,
            input_window_seconds=input_window_seconds,
            sfreq=sfreq,
        )
        self.mapping = {
            "classification_head.fc.6.weight": "final_layer.final_layer.0.weight",
            "classification_head.fc.6.bias": "final_layer.final_layer.0.bias",
        }

        del n_outputs, n_chans, chs_info, n_times, input_window_seconds, sfreq
        if not (self.n_chans <= 64):
            warnings.warn(
                "This model has only been tested on no more "
                + "than 64 channels. no guarantee to work with "
                + "more channels.",
                UserWarning,
            )

        self.return_features = return_features

        self.patch_embedding = _PatchEmbedding(
            n_filters_time=n_filters_time,
            filter_time_length=filter_time_length,
            n_channels=self.n_chans,
            pool_time_length=pool_time_length,
            stride_avg_pool=pool_time_stride,
            drop_prob=drop_prob,
            activation=activation,
        )

        if final_fc_length == "auto":
            assert self.n_times is not None
            self.final_fc_length = self.get_fc_size()
        else:
            self.final_fc_length = final_fc_length

        self.transformer = _TransformerEncoder(
            att_depth=att_depth,
            emb_size=n_filters_time,
            att_heads=att_heads,
            att_drop=att_drop_prob,
            activation=activation_transfor,
        )

        self.fc = _FullyConnected(
            final_fc_length=self.final_fc_length, activation=activation
        )

        self.final_layer = nn.Linear(self.fc.hidden_channels, self.n_outputs)

    def forward(self, x: Tensor) -> Tensor:
        x = torch.unsqueeze(x, dim=1)  # add one extra dimension
        x = self.patch_embedding(x)
        feature = self.transformer(x)

        if self.return_features:
            return feature

        x = self.fc(feature)
        x = self.final_layer(x)
        return x

    def get_fc_size(self):
        out = self.patch_embedding(torch.ones((1, 1, self.n_chans, self.n_times)))
        size_embedding_1 = out.cpu().data.numpy().shape[1]
        size_embedding_2 = out.cpu().data.numpy().shape[2]

        return size_embedding_1 * size_embedding_2


class _PatchEmbedding(nn.Module):
    """Patch Embedding.

    The authors used a convolution module to capture local features,
    instead of position embedding.

    Parameters
    ----------
    n_filters_time: int
        Number of temporal filters, defines also embedding size.
    filter_time_length: int
        Length of the temporal filter.
    n_channels: int
        Number of channels to be used as number of spatial filters.
    pool_time_length: int
        Length of temporal poling filter.
    stride_avg_pool: int
        Length of stride between temporal pooling filters.
    drop_prob: float
        Dropout rate of the convolutional layer.

    Returns
    -------
    x: torch.Tensor
        The output tensor of the patch embedding layer.
    """

    def __init__(
        self,
        n_filters_time,
        filter_time_length,
        n_channels,
        pool_time_length,
        stride_avg_pool,
        drop_prob,
        activation: nn.Module = nn.ELU,
    ):
        super().__init__()

        self.shallownet = nn.Sequential(
            nn.Conv2d(1, n_filters_time, (1, filter_time_length), (1, 1)),
            nn.Conv2d(n_filters_time, n_filters_time, (n_channels, 1), (1, 1)),
            nn.BatchNorm2d(num_features=n_filters_time),
            activation(),
            nn.AvgPool2d(
                kernel_size=(1, pool_time_length), stride=(1, stride_avg_pool)
            ),
            # pooling acts as slicing to obtain 'patch' along the
            # time dimension as in ViT
            nn.Dropout(p=drop_prob),
        )

        self.projection = nn.Sequential(
            nn.Conv2d(
                n_filters_time, n_filters_time, (1, 1), stride=(1, 1)
            ),  # transpose, conv could enhance fiting ability slightly
            Rearrange("b d_model 1 seq -> b seq d_model"),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.shallownet(x)
        x = self.projection(x)
        return x


class _ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        res = x
        x = self.fn(x)
        x += res
        return x


class _TransformerEncoderBlock(nn.Sequential):
    def __init__(
        self,
        emb_size,
        att_heads,
        att_drop,
        forward_expansion=4,
        activation: nn.Module = nn.GELU,
    ):
        super().__init__(
            _ResidualAdd(
                nn.Sequential(
                    nn.LayerNorm(emb_size),
                    MultiHeadAttention(emb_size, att_heads, att_drop),
                    nn.Dropout(att_drop),
                )
            ),
            _ResidualAdd(
                nn.Sequential(
                    nn.LayerNorm(emb_size),
                    FeedForwardBlock(
                        emb_size,
                        expansion=forward_expansion,
                        drop_p=att_drop,
                        activation=activation,
                    ),
                    nn.Dropout(att_drop),
                )
            ),
        )


class _TransformerEncoder(nn.Sequential):
    """Transformer encoder module for the transformer encoder.

    Similar to the layers used in ViT.

    Parameters
    ----------
    att_depth : int
        Number of transformer encoder blocks.
    emb_size : int
        Embedding size of the transformer encoder.
    att_heads : int
        Number of attention heads.
    att_drop : float
        Dropout probability for the attention layers.

    """

    def __init__(
        self, att_depth, emb_size, att_heads, att_drop, activation: nn.Module = nn.GELU
    ):
        super().__init__(
            *[
                _TransformerEncoderBlock(
                    emb_size, att_heads, att_drop, activation=activation
                )
                for _ in range(att_depth)
            ]
        )


class _FullyConnected(nn.Module):
    def __init__(
        self,
        final_fc_length,
        drop_prob_1=0.5,
        drop_prob_2=0.3,
        out_channels=256,
        hidden_channels=32,
        activation: nn.Module = nn.ELU,
    ):
        """Fully-connected layer for the transformer encoder.

        Parameters
        ----------
        final_fc_length : int
            Length of the final fully connected layer.
        n_classes : int
            Number of classes for classification.
        drop_prob_1 : float
            Dropout probability for the first dropout layer.
        drop_prob_2 : float
            Dropout probability for the second dropout layer.
        out_channels : int
            Number of output channels for the first linear layer.
        hidden_channels : int
            Number of output channels for the second linear layer.
        return_features : bool
            Whether to return input features.
        """

        super().__init__()
        self.hidden_channels = hidden_channels
        self.fc = nn.Sequential(
            nn.Linear(final_fc_length, out_channels),
            activation(),
            nn.Dropout(drop_prob_1),
            nn.Linear(out_channels, hidden_channels),
            activation(),
            nn.Dropout(drop_prob_2),
        )

    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1)
        out = self.fc(x)
        return out
