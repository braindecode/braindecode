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
from torch import nn, Tensor

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
    def __init__(self, embed_dim, num_heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * num_heads
        project_out = not (num_heads == 1 and dim_head == embed_dim)

        self.heads = num_heads
        self.scale = dim_head**-0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(embed_dim, inner_dim * 3, bias=False)

        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, embed_dim), nn.Dropout(dropout))
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


class _Transformer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        heads,
        dim_head,
        mlp_dim,
        in_chan,
        fine_grained_kernel=11,
        dropout=0.0,
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        for i in range(depth):
            dim = int(dim * 0.5)
            self.layers.append(
                nn.ModuleList(
                    [
                        _Attention(dim, heads, dim_head, dropout=dropout),
                        _FeedForward(dim, mlp_dim, drop_prob=dropout),
                        nn.Sequential(
                            nn.Dropout(p=dropout),
                            nn.Conv1d(
                                in_channels=in_chan,
                                out_channels=in_chan,
                                kernel_size=fine_grained_kernel,
                                padding=int(0.5 * (fine_grained_kernel - 1)),
                            ),
                            nn.BatchNorm1d(in_chan),
                            nn.ELU(),
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

    XXXX.

    Parameters
    ----------
    temporal_kernel : int
        Size of the temporal convolutional kernel.
    num_kernel : int, optional
        Number of kernels (filters) in the convolutional layer. Default is 64.
    depth : int, optional
        Depth of the transformer (number of layers). Default is 4.
    heads : int, optional
        Number of attention heads in the transformer. Default is 16.
    mlp_dim : int, optional
        Dimension of the hidden layer in the feedforward network. Default is 16.
    dim_head : int, optional
        Dimension of each attention head. Default is 16.
    drop_prob : float, optional
        Dropout rate. Default is 0.0

    References
    ----------
    .. [ding2024] Ding, Y., Li, Y., Sun, H., Liu, R., Tong, C., & Guan, C. (2024).
       EEG-Deformer: A Dense Convolutional Transformer for Brain-computer
       Interfaces. arXiv preprint arXiv:2405.00719.
    """

    def __init__(
        self,
        temporal_kernel: int = 11,
        num_kernel: int = 64,
        depth: int = 4,
        heads: int = 16,
        mlp_dim: int = 16,
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
        self.drop_prob = drop_prob
        self.dim = int(0.5 * self.n_times)
        # embedding size after the first cnn encoder
        self.hidden_size = int(num_kernel * int(self.dim * (0.5**depth))) + int(
            num_kernel * depth
        )
        # Parameters
        self.pos_embedding = nn.Parameter(torch.randn(1, num_kernel, self.dim))

        # Layers
        self.ensuredim = Rearrange("batch chan time -> batch 1 chan time")

        self.cnn_encoder = nn.Sequential(
            Conv2dWithConstraint(
                in_channels=1,
                out_channels=num_kernel,
                kernel_size=(1, temporal_kernel),
                padding=(0, int(0.5 * (temporal_kernel - 1))),
                max_norm=2,
            ),
            Conv2dWithConstraint(
                in_channels=num_kernel,
                out_channels=num_kernel,
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
        nn.TransformerDecoderLayer
        self.transformer = _Transformer(
            dim=self.dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            mlp_dim=mlp_dim,
            dropout=self.drop_prob,
            in_chan=num_kernel,
            fine_grained_kernel=temporal_kernel,
        )

        self.final_layer = nn.Linear(self.hidden_size, self.n_outputs)

    def forward(self, x):
        x = self.ensuredim(x)
        x = self.cnn_encoder(x)
        x = self.to_patch_embedding(x)
        x = x + self.pos_embedding
        x = self.transformer(x)
        return self.final_layer(x)
