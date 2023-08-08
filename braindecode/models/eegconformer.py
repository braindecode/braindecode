# Authors: Yonghao Song <eeyhsong@gmail.com>
#
# License: BSD (3-clause)

import torch
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange, Reduce
from torch import nn, Tensor


class EEGConformer(nn.Sequential):
    """EEG Conformer.

    Convolutional Transformer for EEG decoding.

    The paper and original code with more details about the methodological
    choices are availible at the [EEG Conformer]_ and [EEG Conformer Code]_.

    This neural network architecture recieves a traditional braindecode input.
    The input shape should be three-dimensional matrix representing the EEG
    signals.

         (batch_size, n_channels, n_timesteps)`.

    The EEG Conformer architecture is composed of three modules:
        - PatchEmbedding
        - TransformerEncoder
        - ClassificationHead

    Notes
    -----
    The authors recommend using augment data before using Conformer, e.g. S&R,
    at the end of the code.
    Please refer to the original paper and code for more details.

    .. versionadded:: 0.8

    We aggregate the parameters based on the parts of the models, or
    when the parameters were used first, e.g. n_filters_conv.

    Parameters PatchEmbedding
    -------------------------
    - n_filters_conv: int
        Length of kernels for the temporal convolution layer (first layer).
    - n_filters_time: int
        Number of temporal filters.
    - filter_time_length: int
        Length of the temporal filter.
    - n_filters_spat: int
        Number of spatial filters.
    - pool_time_length: int
        Length of temporal poling filter.
    - pool_time_stride: int
        Length of stride between temporal pooling filters.
    - drop_prob: float
        Dropout rate of the convolutional layer.

    Parameters TransformerEncoder
    -----------------------------
    - att_depth: int
        Number of self-attention layers.
    - att_heads: int
        Number of attention heads.
    - att_drop_prob: float
        Dropout rate of the self-attention layer.

    Parameters ClassificationHead
    -----------------------------
    - final_fc_length: int
        The dimension of the fully connected layer.
    - n_classes: int
        Number of classes to predict (number of output filters of last layer).

    References
    ----------
    .. [EEG Conformer] Song, Y., Zheng, Q., Liu, B. and Gao, X., 2022. EEG
       conformer: Convolutional transformer for EEG decoding and visualization.
       IEEE Transactions on Neural Systems and Rehabilitation Engineering,
       31, pp.710-719. https://ieeexplore.ieee.org/document/9991178
    .. [EEG Conformer Code] Song, Y., Zheng, Q., Liu, B. and Gao, X., 2022. EEG
       conformer: Convolutional transformer for EEG decoding and visualization.
       https://github.com/eeyhsong/EEG-Conformer.
    """

    def __init__(
            self,
            n_classes,
            n_filters_conv=40,
            n_filters_time=25,
            filter_time_length=25,
            n_filters_spat=22,
            n_kernel_avg_pool=75,
            pool_time_stride=15,
            drop_prob=0.5,
            att_depth=6,
            att_heads=10,
            att_drop_prob=0.5,
            final_fc_length=2440,

    ):
        super().__init__(
            PatchEmbedding(
                n_filters_conv=n_filters_conv,
                n_filters_time=n_filters_time,
                filter_time_length=filter_time_length,
                n_filters_spat=n_filters_spat,
                pool_time_length=n_kernel_avg_pool,
                stride_avg_pool=pool_time_stride,
                drop_prob=drop_prob,
            ),
            _TransformerEncoder(att_depth, n_filters_conv, att_heads,
                                att_drop_prob),
            ClassificationHead(n_filters_conv, final_fc_length,
                               n_classes),
        )


class PatchEmbedding(nn.Module):
    """Patch Embedding.

    The authors used a convolution moduleto capture local features,
    instead of postion embedding.
    """

    def __init__(
            self,
            n_filters_conv,
            n_filters_time,
            filter_time_length,
            n_filters_spat,
            pool_time_length,
            stride_avg_pool,
            drop_prob,
    ):
        super().__init__()

        self.shallownet = nn.Sequential(
            nn.Conv2d(1, n_filters_time,
                      (1, filter_time_length), (1, 1)),
            nn.Conv2d(n_filters_time, n_filters_time,
                      (n_filters_spat, 1), (1, 1)),

            nn.BatchNorm2d(num_features=n_filters_conv),
            nn.ELU(),
            nn.AvgPool2d(
                kernel_size=(1, pool_time_length),
                stride=(1, stride_avg_pool)
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
        x = x.unsqueeze(dim=1)  # add one extra dimension
        x = self.shallownet(x)
        x = self.projection(x)
        return x


class _MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        queries = rearrange(
            self.queries(x), "b n (h d) -> b h n d", h=self.num_heads
        )
        keys = rearrange(
            self.keys(x), "b n (h d) -> b h n d", h=self.num_heads
        )
        values = rearrange(
            self.values(x), "b n (h d) -> b h n d", h=self.num_heads
        )
        energy = torch.einsum("bhqd, bhkd -> bhqk", queries, keys)
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum("bhal, bhlv -> bhav ", att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out


class _ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class _FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )


class _TransformerEncoderBlock(nn.Sequential):
    def __init__(self, emb_size, att_heads, att_drop, forward_expansion=4):
        super().__init__(
            _ResidualAdd(
                nn.Sequential(
                    nn.LayerNorm(emb_size),
                    _MultiHeadAttention(emb_size, att_heads, att_drop),
                    nn.Dropout(att_drop),
                )
            ),
            _ResidualAdd(
                nn.Sequential(
                    nn.LayerNorm(emb_size),
                    _FeedForwardBlock(
                        emb_size, expansion=forward_expansion,
                        drop_p=att_drop
                    ),
                    nn.Dropout(att_drop),
                )
            ),
        )


class _TransformerEncoder(nn.Sequential):
    """Transformer encoder module for the transformer encoder.

    Similar to the layers used in ViT.
    """
    def __init__(self, att_depth, emb_size, att_heads, att_drop):
        super().__init__(
            *[
                _TransformerEncoderBlock(emb_size, att_heads, att_drop)
                for _ in range(att_depth)
            ]
        )


class ClassificationHead(nn.Module):
    def __init__(self, emb_size, final_fc_length, n_classes,
                 drop_prob_1=0.5, drop_prob_2=0.3, out_channels=256,
                 hidden_channels=32):
        """"Classification head for the transformer encoder.

        Parameters
        ----------
        emb_size : int
            Embedding size of the transformer encoder.
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
        """

        super().__init__()

        # global average pooling
        self.clshead = nn.Sequential(
            Reduce("b n e -> b e", reduction="mean"),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, n_classes),
        )
        self.fc = nn.Sequential(
            nn.Linear(final_fc_length, out_channels),
            nn.ELU(),
            nn.Dropout(drop_prob_1),
            nn.Linear(out_channels, hidden_channels),
            nn.ELU(),
            nn.Dropout(drop_prob_2),
            nn.Linear(hidden_channels, n_classes),
        )

    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1)
        out = self.fc(x)
        return x, out
