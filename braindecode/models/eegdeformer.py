"""
CBCR License 1.0

Copyright 2022 Centre for Brain Computing Research (CBCR)

Redistribution and use for non-commercial purpose in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials
provided with the distribution.
3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior
written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTER-
RUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
# Authors: Yi Ding
#          Bruno Aristimunha <b.aristimunha@gmail.com> (braindecode adaptation)

import torch
from torch import nn

from einops import rearrange
from einops.layers.torch import Rearrange

from braindecode.models.eegnet import Conv2dWithConstraint
from braindecode.models.base import EEGModuleMixin


class _FeedForward(nn.Module):
    """Feedforward network with Layer Normalization, activation, and dropout.

    This module applies a sequence of operations:
    1. Layer Normalization
    2. Linear transformation from ``in_features` to ``out_features``
    3. Activation function
    4. Dropout
    5. Linear transformation from ``out_features`` back to ``in_features``
    6. Dropout

    Parameters
    ----------
    in_features : int
        Size of each input sample.
    out_features : int
        Size of each output sample.
    drop_prob : float, optional
        Dropout probability. Default is `0.0`.
    activation : nn.Module, optional
        Activation function to apply after the first linear layer.
        Default is `nn.GELU`.

    Attributes
    ----------
    net : nn.Module
        Sequential container holding the layers of the feedforward network.

    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        drop_prob: float = 0.0,
        activation: nn.Module = nn.GELU,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Linear(in_features=in_features, out_features=out_features),
            activation(),
            nn.Dropout(p=drop_prob),
            nn.Linear(in_features=out_features, out_features=in_features),
            nn.Dropout(p=drop_prob),
        )

    def forward(self, x):
        return self.net(x)


class _Attention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        dim_head: int = 64,
        drop_prob: float = 0.0,
    ):
        super().__init__()
        inner_dim = dim_head * num_heads
        project_out = not (num_heads == 1 and dim_head == embed_dim)

        self.heads = num_heads
        self.scale = dim_head**-0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(embed_dim, inner_dim * 3, bias=False)

        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, embed_dim), nn.Dropout(drop_prob))
            if project_out
            else nn.Identity()
        )

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        query, key, value = map(
            lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv
        )

        dots = torch.matmul(query, key.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, value)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class _HierarchicalCoarseTransformer(nn.Module):
    """
    Hierarchical Coarse-to-Fine Transformer for EEG signal processing.

    This transformer captures both coarse-grained and fine-grained temporal dynamics
    in EEG data by combining multi-head self-attention mechanisms with CNN-based
    feature extraction within each transformer block.

    Parameters
    ----------
    embed_dim : int
        Dimension of the embedding vectors.
    heads : int
        Number of attention heads.
    dim_head : int
        Dimension of each attention head.
    dim_feedforward : int
        Dimension of the feedforward network.
    drop_prob : float
        Dropout probability.
    in_features : int
        Number of input channels.
    fine_grained_kernel : int, optional, default 11
        Kernel size for fine-grained temporal feature extraction.
    activation: nn.Module, default nn.ELU
        Activation function class to apply. Should be a PyTorch activation
        module class like ``nn.ReLU`` or ``nn.ELU``. Default is ``nn.ELU``.
    """

    def __init__(
        self,
        embed_dim: int,
        depth: int,
        heads: int,
        dim_head: int,
        dim_feedforward: int,
        in_features: int,
        fine_grained_kernel: int = 11,
        drop_prob: float = 0.0,
        activation: nn.Module = nn.ELU,
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        for i in range(depth):
            embed_dim = int(embed_dim * 0.5)
            self.layers.append(
                nn.ModuleList(
                    [
                        _Attention(embed_dim, heads, dim_head, drop_prob=drop_prob),
                        _FeedForward(embed_dim, dim_feedforward, drop_prob=drop_prob),
                        nn.Sequential(
                            nn.Dropout(p=drop_prob),
                            nn.Conv1d(
                                in_channels=in_features,
                                out_channels=in_features,
                                kernel_size=fine_grained_kernel,
                                padding=int(0.5 * (fine_grained_kernel - 1)),
                            ),
                            nn.BatchNorm1d(in_features),
                            activation(),
                            nn.MaxPool1d(kernel_size=2, stride=2),
                        ),
                    ]
                )
            )
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

    def forward(self, x):
        dense_feature = []
        for attn, ff, cnn in self.layers:
            x_cg = self.pool(x)
            x_cg = attn(x_cg) + x_cg
            x_fg = cnn(x)
            x_info = self.get_info(x_fg)  # (b, in_chan)
            dense_feature.append(x_info)
            x = ff(x_cg) + x_fg
        x_dense = torch.cat(dense_feature, dim=-1)  # b, in_chan*depth
        x = x.view(x.size(0), -1)  # b, in_chan*d_hidden_last_layer
        emd = torch.cat(
            (x, x_dense), dim=-1
        )  # b, in_chan*(depth + d_hidden_last_layer)
        return emd

    @staticmethod
    def get_info(x):
        # x: b, k, l
        x = torch.log(torch.mean(x.pow(2), dim=-1))
        return x


class EEGDeformer(EEGModuleMixin, nn.Module):
    """Deformer model from [ding2024]_.

    The model integrates CNN-based shallow feature encoding with a hierarchical
    coarse-to-fine Transformer architecture to effectively capture both shallow
    temporal and spatial information, as well as coarse and fine-grained temporal
     dynamics in EEG data.

    EEGDeformer starts with a CNN-based shallow feature encoder that processes
    the input EEG signals through temporal and spatial convolutional layers,
    followed by batch normalization, ELU activation, and max pooling.
    The output of the CNN encoder is then rearranged and augmented with
    learnable positional embeddings to form tokens for the Transformer.

    The Transformer consists of multiple Hierarchical Coarse-to-fine Transformer
    (HCT) blocks, each combining a multi-head self-attention mechanism to capture
    coarse-grained temporal dynamics and a CNN-based branch to learn fine-grained
    temporal features. Additionally, dense information purification modules are
    employed to extract discriminative information from multiple HCT layers,
    enhancing the model’s ability to capture multi-level temporal information.

    Finally, the model aggregates the feature and passes it through
    a linear layer to produce the final classification or regression output.


    Parameters
    ----------
    temporal_kernel : int, optional
        Size of the temporal convolution kernel. Determines the width of the convolution
        applied along the time dimension. Default is `11`.
    num_kernel : int, optional
        Number of convolutional kernels (output channels) in the CNN encoder. Controls
        the capacity of the convolutional layers. Default is `64`.
    n_layers : int, optional
        Number of transformer layers in the transformer encoder. Each layer consists of
        multi-head self-attention and feedforward networks. Default is `4`.
    heads : int, optional
        Number of attention heads in each transformer layer. More heads allow the model
        to attend to information from different representation subspaces. Default is `16`.
    dim_feedforward : int, optional
        Dimension of the feedforward network within each transformer layer. Determines
        the size of the intermediate representations. Default is `16`.
    dim_head : int, optional
        Dimension of each attention head in the transformer. Affects the capacity of
        the self-attention mechanism. Default is `16`.
    drop_prob : float, optional
        Dropout probability applied after each dropout layer in the model. Helps prevent
        overfitting by randomly zeroing some of the elements. Default is `0.0`.
    activation : nn.Module, optional
        Activation function to use in the CNN encoder and transformer layers. Default is
        `nn.ELU`.

    Notes
    -----
    This implementation was adapted to braindecode from pytorch code [ding2024code]_.
    During the process, some functionality and reproducibility may or may not
    have been lost. If in doubt, consult the original code and
    contact the original authors.

    References
    ----------
    .. [ding2024] Ding, Y., Li, Y., Sun, H., Liu, R., Tong, C., & Guan, C. (2024).
       EEG-Deformer: A Dense Convolutional Transformer for Brain-computer
       Interfaces. arXiv preprint arXiv:2405.00719.
    .. [ding2024code] Ding, Y., Li, Y., Sun, H., Liu, R., Tong, C., & Guan, C.
       (2024). EEG-Deformer: A Dense Convolutional Transformer for Brain-computer
       Interfaces. https://github.com/yi-ding-cs/EEG-Deformer
    """

    def __init__(
        self,
        temporal_kernel: int = 11,
        num_kernel: int = 64,
        n_layers: int = 4,
        heads: int = 16,
        dim_feedforward: int = 16,
        dim_head: int = 16,
        drop_prob: float = 0.0,
        activation: nn.Module = nn.ELU,
        # braindecode
        chs_info=None,
        input_window_seconds=None,
        sfreq=None,
        n_chans=None,
        n_times=None,
        n_outputs=None,
    ):
        super().__init__(
            n_chans=n_chans,
            n_times=n_times,
            n_outputs=n_outputs,
            chs_info=chs_info,
            input_window_seconds=input_window_seconds,
            sfreq=sfreq,
        )
        del n_outputs, n_chans, chs_info, n_times, input_window_seconds, sfreq
        # Variables
        self.n_layers = n_layers
        self.drop_prob = drop_prob
        self.embed_dim = int(0.5 * self.n_times)
        self.hidden_size = int(
            num_kernel * int(self.embed_dim * (0.5**n_layers))
        ) + int(num_kernel * n_layers)
        self.dim_feedforward = dim_feedforward
        self.dim_head = dim_head
        self.heads = heads
        self.num_kernel = num_kernel
        self.temporal_kernel = temporal_kernel
        # Parameters
        self.pos_embedding = nn.Parameter(
            torch.randn(1, self.num_kernel, self.embed_dim)
        )

        # Layers
        self.ensuredim = Rearrange("batch chan time -> batch 1 chan time")

        self.cnn_encoder = nn.Sequential(
            Conv2dWithConstraint(
                in_channels=1,
                out_channels=self.num_kernel,
                kernel_size=(1, self.temporal_kernel),
                padding=(0, int(0.5 * (self.temporal_kernel - 1))),
                max_norm=2,
            ),
            Conv2dWithConstraint(
                in_channels=self.num_kernel,
                out_channels=self.num_kernel,
                kernel_size=(self.n_chans, 1),
                padding=0,
                max_norm=2,
            ),
            nn.BatchNorm2d(num_kernel),
            activation(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
        )

        self.to_patch_embedding = Rearrange(
            "batch kernel chans filter -> batch kernel (chans filter)"
        )
        self.transformer = _HierarchicalCoarseTransformer(
            embed_dim=self.embed_dim,
            depth=self.n_layers,
            heads=self.heads,
            dim_head=self.dim_head,
            dim_feedforward=self.dim_feedforward,
            drop_prob=self.drop_prob,
            in_features=self.num_kernel,
            fine_grained_kernel=self.temporal_kernel,
            activation=self.activation,
        )

        self.final_layer = nn.Linear(self.hidden_size, self.n_outputs)

    def forward(self, x):
        x = self.ensuredim(x)
        x = self.cnn_encoder(x)
        x = self.to_patch_embedding(x)
        x = x + self.pos_embedding
        x = self.transformer(x)
        return self.final_layer(x)
