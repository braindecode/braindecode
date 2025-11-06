# Authors: Yihe Wang <ywang145@charlotte.edu>
#          Nan Huang <nhuang1@charlotte.edu>
#          Taida Li <tli14@charlotte.edu>
#
# License: MIT

"""Medformer: A Multi-Granularity Patching Transformer for Medical Time-Series Classification."""

import math
from math import sqrt
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from braindecode.models.base import EEGModuleMixin


class MEDFormer(EEGModuleMixin, nn.Module):
    """Medformer from Wang et al. (2024) [Medformer2024]_.

    :bdg-success:`Convolution` :bdg-danger:`Large Brain Model`

    .. figure:: https://raw.githubusercontent.com/DL4mHealth/Medformer/refs/heads/main/figs/medformer_architecture.png
        :align: center
        :alt: MEDFormer Architecture.

        a) Workflow. b) For the input sample :math:`{x}_{\textrm{in}}`, the authors apply :math:`n` different patch lengths in parallel to create patched features :math:`{x}_p^{(i)}`, where :math:`i` ranges from 1 to :math:`n`.
        Each patch length represents a different granularity. These patched features are then linearly transformed into :math:`{x}_e^{(i)}`, which are subsequently augmented into :math:`\\widetilde{x}_e^{(i)}`.
        c) We obtain the final patch embedding :math:`{x}^{(i)}` by fusing augmented :math:`\\widetilde{{x}}_e^{(i)}` with the positional embedding :math:`{W}_{\text{pos}}` and the granularity embedding :math:`{W}_{\text{gr}}^{(i)}`.
        Additionally, we design a granularity-specific router :math:`{u}^{(i)}` to capture integrated information for its respective granularity.
        The authors compute both intra-granularity attention, which concentrates within individual granularities, and inter-granularity attention,
        which leverages the routers to focus across different granularities, for extensive representation learning.

    Medformer: A Multi-Granularity Patching Transformer for Medical Time-Series Classification.

    The Medformer architecture is designed for medical time series classification
    tasks, particularly for EEG and ECG data. It uses multi-granularity patching
    to capture features at different temporal scales through cross-channel patching,
    multi-granularity embedding, and two-stage multi-granularity self-attention.

    .. versionadded:: 1.3

    .. rubric:: Architecture Overview

    Medformer incorporates three novel mechanisms:

    - **Cross-channel patching**: Leverages inter-channel correlations by creating
      patches that span multiple channels.
    - **Multi-granularity embedding**: Captures features at different temporal scales
      using multiple patch lengths.
    - **Two-stage multi-granularity self-attention**: Learns features and correlations
      within (intra-granularity) and among (inter-granularity) different temporal scales.

    Parameters
    ----------
    patch_len_list : list of int, optional
        List of patch lengths for multi-granularity patching. Each value specifies
        a different temporal scale for feature extraction. For example, [2, 8, 16]
        will create patches of 2, 8, and 16 channels respectively.
        Default is [2, 8, 16].
    d_model : int, optional
        Dimension of the model embeddings. Default is 128.
    num_heads : int, optional
        Number of attention heads. Must divide d_model evenly. Default is 8.
    drop_prob : float, optional
        Dropout probability. Default is 0.1.
    no_inter_attn : bool, optional
        If True, disables inter-granularity attention. Default is False.
    n_layers : int, optional
        Number of encoder layers. Default is 6.
    dim_feedforward : int, optional
        Dimension of the feedforward network. Default is 256.
    activation : nn.Module, optional
        Activation function module. Default is nn.ReLU().
    single_channel : bool, optional
        If True, treats each channel independently. This increases model capacity
        but also computational cost. Default is False.
    output_attention : bool, optional
        If True, model can output attention weights (useful for interpretability).
        Default is True.

    References
    ----------
    .. [Medformer2024] Wang, Y., Huang, N., Li, T., Yan, Y., & Zhang, X. (2024).
        Medformer: A Multi-Granularity Patching Transformer for Medical Time-Series Classification.
        In A. Globerson, L. Mackey, D. Belgrave, A. Fan, U. Paquet, J. Tomczak, & C. Zhang (Eds),
        Advances in Neural Information Processing Systems (Vol. 37, pp. 36314-36341).
        doi:10.52202/079017-1145

    """

    def __init__(
        self,
        # Signal related parameters
        n_chans=None,
        n_outputs=None,
        n_times=None,
        chs_info=None,
        input_window_seconds=None,
        sfreq=None,
        # Model parameters
        patch_len_list: Optional[List[int]] = None,
        d_model: int = 128,
        num_heads: int = 8,
        drop_prob: float = 0.1,
        no_inter_attn: bool = False,
        n_layers: int = 6,
        dim_feedforward: int = 256,
        activation_trans: Optional[nn.Module] = nn.ReLU,
        single_channel: bool = False,
        output_attention: bool = True,
        activation_class: Optional[nn.Module] = nn.GELU,
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

        # In the original Medformer paper:
        # - seq_len refers to the number of channels
        # - enc_in refers to the number of time points

        # Save model parameters as instance variables
        self.d_model = d_model
        self.num_heads = num_heads
        self.drop_prob = drop_prob
        self.no_inter_attn = no_inter_attn
        self.n_layers = n_layers
        self.dim_feedforward = dim_feedforward
        self.activation_trans = activation_trans
        self.output_attention = output_attention
        self.single_channel = single_channel
        self.activation_class = activation_class

        # Process the sequence and patch configurations.
        if patch_len_list is None:
            patch_len_list = [2, 8, 16]

        self.patch_len_list = patch_len_list
        stride_list = patch_len_list  # Using the same values for strides.
        self.stride_list = stride_list
        patch_num_list = [
            int((self.n_chans - patch_len) / stride + 2)
            for patch_len, stride in zip(patch_len_list, stride_list)
        ]
        self.patch_num_list = patch_num_list

        # Initialize the embedding layer.
        self.enc_embedding = _ListPatchEmbedding(
            enc_in=self.n_times,
            d_model=self.d_model,
            seq_len=self.n_chans,
            patch_len_list=self.patch_len_list,
            stride_list=self.stride_list,
            dropout=self.drop_prob,
            single_channel=self.single_channel,
            n_chans=self.n_chans,
            n_times=self.n_times,
        )
        # Build the encoder with multiple layers.
        self.encoder = _Encoder(
            [
                _EncoderLayer(
                    attention=_MedformerLayer(
                        num_blocks=len(self.patch_len_list),
                        d_model=self.d_model,
                        num_heads=self.num_heads,
                        dropout=self.drop_prob,
                        output_attention=self.output_attention,
                        no_inter=self.no_inter_attn,
                    ),
                    d_model=self.d_model,
                    dim_feedforward=self.dim_feedforward,
                    dropout=self.drop_prob,
                    activation=self.activation_trans()
                    if self.activation_trans is not None
                    else nn.ReLU(),
                )
                for _ in range(self.n_layers)
            ],
            norm_layer=torch.nn.LayerNorm(self.d_model),
        )

        # For classification tasks, add additional layers.
        self.activation_layer = (
            self.activation_class() if self.activation_class is not None else nn.GELU()
        )
        self.dropout = nn.Dropout(self.drop_prob)
        self.final_layer = nn.Linear(
            self.d_model
            * len(self.patch_num_list)
            * (1 if not self.single_channel else self.n_chans),
            self.n_outputs,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the Medformer model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, n_chans, n_times).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, n_outputs).
        """
        # Embedding
        enc_out = self.enc_embedding(x)
        enc_out, _ = self.encoder(enc_out, attn_mask=None)

        if self.single_channel:
            # Reshape back from (batch_size * n_chans, ...) to (batch_size, n_chans, ...)
            enc_out = torch.reshape(enc_out, (-1, self.n_chans, *enc_out.shape[-2:]))

        # Output
        output = self.activation_layer(enc_out)
        output = self.dropout(output)
        output = output.reshape(
            output.shape[0], -1
        )  # (batch_size, seq_length * d_model)
        output = self.final_layer(output)  # (batch_size, num_classes)
        return output


class _PositionalEmbedding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        # If d_model is odd, temporarily work with d_model + 1.
        if d_model % 2 == 1:
            d_model_adj = d_model + 1
        else:
            d_model_adj = d_model
        self.d_model = d_model  # store the original dimension

        # Create a pe tensor of size (max_len, d_model_adj)
        pe = torch.zeros(max_len, d_model_adj).float()
        pe.requires_grad = False

        # Compute the sinusoidal factors.
        position = torch.arange(0, max_len).float().unsqueeze(1)
        # Use d_model_adj in the denominator so that the frequencies are computed over an even number.
        div_term = torch.exp(
            torch.arange(0, d_model_adj, 2).float() * (-math.log(10000.0) / d_model_adj)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Unsqueeze to shape (1, max_len, d_model_adj)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is assumed to have shape (B, L, d_model_target)
        # We return the first self.d_model columns from the computed pe.
        return self.pe[:, : x.size(1), : self.d_model]


class _CrossChannelTokenEmbedding(nn.Module):
    def __init__(
        self, c_in: int, l_patch: int, d_model: int, stride: Optional[int] = None
    ):
        super().__init__()
        if stride is None:
            stride = l_patch
        self.token_conv = nn.Conv2d(
            in_channels=1,
            out_channels=d_model,
            kernel_size=(c_in, l_patch),
            stride=(1, stride),
            padding=0,
            padding_mode="circular",
            bias=False,
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_in", nonlinearity="leaky_relu"
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.token_conv(x)
        return x


class _ListPatchEmbedding(nn.Module):
    def __init__(
        self,
        enc_in: int,
        d_model: int,
        seq_len: int,
        patch_len_list: List[int],
        stride_list: List[int],
        dropout: float,
        single_channel: bool = False,
        n_chans: Optional[int] = None,
        n_times: Optional[int] = None,
    ):
        super().__init__()
        self.patch_len_list = patch_len_list
        self.stride_list = stride_list
        self.paddings = [nn.ReplicationPad1d((0, stride)) for stride in stride_list]
        self.single_channel = single_channel
        self.n_chans = n_chans
        self.n_times = n_times

        linear_layers = [
            _CrossChannelTokenEmbedding(
                c_in=enc_in if not single_channel else 1,
                l_patch=patch_len,
                d_model=d_model,
            )
            for patch_len in patch_len_list
        ]
        self.value_embeddings = nn.ModuleList(linear_layers)
        self.position_embedding = _PositionalEmbedding(d_model=d_model)
        self.channel_embedding = _PositionalEmbedding(d_model=seq_len)
        self.dropout = nn.Dropout(dropout)

        self.learnable_embeddings = nn.ParameterList(
            [nn.Parameter(torch.randn(1, d_model)) for _ in patch_len_list]
        )

    def forward(
        self, x: torch.Tensor
    ) -> List[torch.Tensor]:  # (batch_size, seq_len, enc_in)
        x = x.permute(0, 2, 1)  # (batch_size, enc_in, seq_len)
        if self.single_channel:
            # After permute: x.shape = (batch_size, n_times, n_chans)
            # We want to process each channel independently
            batch_size = x.shape[0]
            # Permute to get channels in the middle: (batch_size, n_chans, n_times)
            x = x.permute(0, 2, 1)
            # Reshape to treat each channel independently: (batch_size * n_chans, 1, n_times)
            x = torch.reshape(x, (batch_size * self.n_chans, 1, self.n_times))

        x_list = []
        for padding, value_embedding in zip(self.paddings, self.value_embeddings):
            x_copy = x.clone()
            # add positional embedding to tag each channel (only when not single_channel)
            if not self.single_channel:
                x_new = x_copy + self.channel_embedding(x_copy)
            else:
                x_new = x_copy
            x_new = padding(x_new).unsqueeze(
                1
            )  # (batch_size, 1, enc_in, seq_len+stride)
            x_new = value_embedding(x_new)  # (batch_size, d_model, 1, patch_num)
            x_new = x_new.squeeze(2).transpose(1, 2)  # (batch_size, patch_num, d_model)
            x_list.append(x_new)

        x = [
            x + cxt + self.position_embedding(x)
            for x, cxt in zip(x_list, self.learnable_embeddings)
        ]  # (batch_size, patch_num_1, d_model), (batch_size, patch_num_2, d_model), ...
        return x


class _AttentionLayer(nn.Module):
    def __init__(
        self,
        attention: nn.Module,
        d_model: int,
        num_heads: int,
        d_keys: Optional[int] = None,
        d_values: Optional[int] = None,
    ):
        super().__init__()

        d_keys = d_keys or (d_model // num_heads)
        d_values = d_values or (d_model // num_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * num_heads)
        self.key_projection = nn.Linear(d_model, d_keys * num_heads)
        self.value_projection = nn.Linear(d_model, d_values * num_heads)
        self.out_projection = nn.Linear(d_values * num_heads, d_model)
        self.num_heads = num_heads

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        tau: Optional[torch.Tensor] = None,
        delta: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size, query_len, _ = queries.shape
        _, key_len, _ = keys.shape
        num_heads = self.num_heads

        queries = self.query_projection(queries).view(
            batch_size, query_len, num_heads, -1
        )  # multi-head
        keys = self.key_projection(keys).view(batch_size, key_len, num_heads, -1)
        values = self.value_projection(values).view(batch_size, key_len, num_heads, -1)

        out, attn = self.inner_attention(
            queries, keys, values, attn_mask, tau=tau, delta=delta
        )
        out = out.view(batch_size, query_len, -1)

        return self.out_projection(out), attn


class _TriangularCausalMask:
    def __init__(self, batch_size: int, seq_len: int, device: str = "cpu"):
        mask_shape = [batch_size, 1, seq_len, seq_len]
        with torch.no_grad():
            self._mask = torch.triu(
                torch.ones(mask_shape, dtype=torch.bool), diagonal=1
            ).to(device)

    @property
    def mask(self) -> torch.Tensor:
        return self._mask


class _FullAttention(nn.Module):
    def __init__(
        self,
        mask_flag: bool = True,
        scale: Optional[float] = None,
        attention_dropout: float = 0.1,
        output_attention: bool = False,
    ):
        super().__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        super().__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attn_mask: Optional[_TriangularCausalMask],
        tau: Optional[torch.Tensor] = None,
        delta: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size, query_len, _, embed_dim = queries.shape
        _, _, _, _ = values.shape
        scale = self.scale or 1.0 / sqrt(embed_dim)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = _TriangularCausalMask(
                    batch_size, query_len, device=queries.device
                )

            scores.masked_fill_(attn_mask.mask, -np.inf)

        attention_weights = self.dropout(
            torch.softmax(scale * scores, dim=-1)
        )  # Scaled Dot-Product Attention
        output_values = torch.einsum("bhls,bshd->blhd", attention_weights, values)

        if self.output_attention:
            return output_values.contiguous(), attention_weights
        else:
            return output_values.contiguous(), None


class _MedformerLayer(nn.Module):
    def __init__(
        self,
        num_blocks: int,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
        output_attention: bool = False,
        no_inter: bool = False,
    ):
        super().__init__()

        self.intra_attentions = nn.ModuleList(
            [
                _AttentionLayer(
                    _FullAttention(
                        mask_flag=False,
                        attention_dropout=dropout,
                        output_attention=output_attention,
                    ),
                    d_model,
                    num_heads,
                )
                for _ in range(num_blocks)
            ]
        )
        if no_inter or num_blocks <= 1:
            # print("No inter attention for time")
            self.inter_attention = None
        else:
            self.inter_attention = _AttentionLayer(
                _FullAttention(
                    mask_flag=False,
                    attention_dropout=dropout,
                    output_attention=output_attention,
                ),
                d_model,
                num_heads,
            )

    def forward(
        self,
        x: List[torch.Tensor],
        attn_mask: Optional[List[Optional[torch.Tensor]]] = None,
        tau: Optional[torch.Tensor] = None,
        delta: Optional[torch.Tensor] = None,
    ) -> Tuple[List[torch.Tensor], List[Optional[torch.Tensor]]]:
        attn_mask = attn_mask or ([None] * len(x))
        # Intra attention
        x_intra = []
        attn_out = []
        for x_in, layer, mask in zip(x, self.intra_attentions, attn_mask):
            x_out_temp, attn_temp = layer(
                x_in, x_in, x_in, attn_mask=mask, tau=tau, delta=delta
            )
            x_intra.append(x_out_temp)  # (B, Li, D)
            attn_out.append(attn_temp)
        if self.inter_attention is not None:
            # Inter attention
            routers = torch.cat([x[:, -1:] for x in x_intra], dim=1)  # (B, N, D)
            x_inter, attn_inter = self.inter_attention(
                routers, routers, routers, attn_mask=None, tau=tau, delta=delta
            )
            x_out = [
                torch.cat([x[:, :-1], x_inter[:, i : i + 1]], dim=1)  # (B, Li, D)
                for i, x in enumerate(x_intra)
            ]
            attn_out += [attn_inter]
        else:
            x_out = x_intra
        return x_out, attn_out


class _EncoderLayer(nn.Module):
    def __init__(
        self,
        attention: nn.Module,
        d_model: int,
        dim_feedforward: Optional[int],
        dropout: float,
        activation: Optional[nn.Module] = None,
    ):
        super().__init__()
        dim_feedforward = dim_feedforward or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(
            in_channels=d_model, out_channels=dim_feedforward, kernel_size=1
        )
        self.conv2 = nn.Conv1d(
            in_channels=dim_feedforward, out_channels=d_model, kernel_size=1
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = activation if activation is not None else nn.ReLU()

    def forward(
        self,
        x: List[torch.Tensor],
        attn_mask: Optional[List[Optional[torch.Tensor]]] = None,
        tau: Optional[torch.Tensor] = None,
        delta: Optional[torch.Tensor] = None,
    ) -> Tuple[List[torch.Tensor], List[Optional[torch.Tensor]]]:
        new_x, attn = self.attention(x, attn_mask=attn_mask, tau=tau, delta=delta)
        x = [x_orig + self.dropout(x_new) for x_orig, x_new in zip(x, new_x)]

        y = x = [self.norm1(x_val) for x_val in x]
        y = [
            self.dropout(self.activation(self.conv1(y_val.transpose(-1, 1))))
            for y_val in y
        ]
        y = [self.dropout(self.conv2(y_val).transpose(-1, 1)) for y_val in y]

        return [self.norm2(x_val + y_val) for x_val, y_val in zip(x, y)], attn


class _Encoder(nn.Module):
    def __init__(
        self, attn_layers: List[nn.Module], norm_layer: Optional[nn.Module] = None
    ):
        super().__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(
        self,
        x: List[torch.Tensor],
        attn_mask: Optional[List[Optional[torch.Tensor]]] = None,
        tau: Optional[torch.Tensor] = None,
        delta: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[List[Optional[torch.Tensor]]]]:
        # x [[B, L1, D], [B, L2, D], ...]
        attns = []
        for attn_layer in self.attn_layers:
            x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
            attns.append(attn)

        # concat all the outputs
        """x = torch.cat(
            x, dim=1
        )  # (batch_size, patch_num_1 + patch_num_2 + ... , d_model)"""

        # concat all the routers
        x = torch.cat(
            [x[:, -1, :].unsqueeze(1) for x in x], dim=1
        )  # (batch_size, len(patch_len_list), d_model)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns
