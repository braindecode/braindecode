# Authors: Cedric Rommel <cedric.rommel@inria.fr>
#
# License: BSD (3-clause)
import numpy as np

import torch
from torch import nn

from .modules import Expression, Ensure4d, MaxNormLinear, CausalConv1d
from .functions import transpose_time_to_spat


class ATCNet(nn.Module):
    """ATCNet model from [1]_

    Pytorch implementation based on official tensorflow code [2]_.

    Parameters
    ----------
    n_channels : int
        Number of EEG channels.
    n_classes : int
        Number of target classes.
    input_size_s : float, optional
        Time length of inputs, in secods. Defaults to 4.5 s, as in BCI-IV 2a
        dataset.
    sfreq : int, optional
        Sampling frequency of the inputs, in Hz. Default to 250 Hz, as in
        BCI-IV 2a dataset.
    conv_block_n_filters : int
        Number temporal filters in the first convolutional layer of the
        convolutional block, denoted F1 in figure 2 of the paper [1]_. Defaults
        to 16 as in [1]_.
    conv_block_kernel_length_1 : int
        Length of temporal filters in the first convolutional layer of the
        convolutional block, denoted Kc in table 1 of the paper [1]_. Defaults
        to 64 as in [1]_.
    conv_block_kernel_length_2 : int
        Length of temporal filters in the last convolutional layer of the
        convolutional block. Defaults to 16 as in [1]_.
    conv_block_pool_size_1 : int
        Length of first average pooling kernel in the convolutional block.
        Defaults to 8 as in [1]_.
    conv_block_pool_size_2 : int
        Length of first average pooling kernel in the convolutional block,
        denoted P2 in table 1 of the paper [1]_. Defaults to 7 as in [1]_.
    conv_block_depth_mult : int
        Depth multiplier of depthwise convolution in the convolutional block,
        denoted D in table 1 of the paper [1]_. Defaults to 2 as in [1]_.
    conv_block_dropout : float
        Dropout probability used in the convolution block, denoted pc in
        table 1 of the paper [1]_. Defaults to 0.3 as in [1]_.
    n_windows : int
        Number of sliding windows, denoted n in [1]_. Defaults to 5 as in [1]_.
    att_head_dim : int
        Embedding dimension used in each self-attention head, denoted dh in
        table 1 of the paper [1]_. Defaults to 8 as in [1]_.
    att_num_heads : int
        Number of attention heads, denoted H in table 1 of the paper [1]_.
        Defaults to 2 as in [1_.
    att_dropout : float
        Dropout probability used in the attention block, denoted pa in table 1
        of the paper [1]_. Defaults to 0.5 as in [1]_.
    tcn_depth : int
        Depth of Temporal Convolutional Network block (i.e. number of TCN
        Residual blocks), denoted L in table 1 of the paper [1]_. Defaults to 2
        as in [1]_.
    tcn_kernel_size : int
        Temporal kernel size used in TCN block, denoted Kt in table 1 of the
        paper [1]_. Defaults to 4 as in [1]_.
    tcn_n_filters : int
        Number of filters used in TCN convolutional layers (Ft). Defaults to
        32 as in [1]_.
    tcn_dropout : float
        Dropout probability used in the TCN block, denoted pt in table 1
        of the paper [1]_. Defaults to 0.3 as in [1]_.
    tcn_activation : torch.nn.Module
        Nonlinear activation to use. Defaults to nn.ELU().
    concat : bool
        When ``True``, concatenates each slidding window embedding before
        feeding it to a fully-connected layer, as done in [1]_. When ``False``,
        maps each slidding window to `n_classes` logits and average them.
        Defaults to ``False`` contrary to what is reported in [1]_, but
        matching what the official code does [2]_.
    max_norm_const : float
        Maximum L2-norm constraint imposed on weights of the last
        fully-connected layer. Defaults to 0.25.

    References
    ----------
    .. [1] H. Altaheri, G. Muhammad and M. Alsulaiman, "Physics-informed
           attention temporal convolutional network for EEG-based motor imagery
           classification," in IEEE Transactions on Industrial Informatics,
           2022, doi: 10.1109/TII.2022.3197419.
    .. [2] https://github.com/Altaheri/EEG-ATCNet/blob/main/models.py
    """
    def __init__(
        self,
        n_channels,
        n_classes,
        input_size_s=4.5,
        sfreq=250,
        conv_block_n_filters=16,
        conv_block_kernel_length_1=64,
        conv_block_kernel_length_2=16,
        conv_block_pool_size_1=8,
        conv_block_pool_size_2=7,
        conv_block_depth_mult=2,
        conv_block_dropout=0.3,
        n_windows=5,
        att_head_dim=8,
        att_num_heads=2,
        att_dropout=0.5,
        tcn_depth=2,
        tcn_kernel_size=4,
        tcn_n_filters=32,
        tcn_dropout=0.3,
        tcn_activation=nn.ELU(),
        concat=False,
        max_norm_const=0.25,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.input_size_s = input_size_s
        self.sfreq = sfreq
        self.conv_block_n_filters = conv_block_n_filters
        self.conv_block_kernel_length_1 = conv_block_kernel_length_1
        self.conv_block_kernel_length_2 = conv_block_kernel_length_2
        self.conv_block_pool_size_1 = conv_block_pool_size_1
        self.conv_block_pool_size_2 = conv_block_pool_size_2
        self.conv_block_depth_mult = conv_block_depth_mult
        self.conv_block_dropout = conv_block_dropout
        self.n_windows = n_windows
        self.att_head_dim = att_head_dim
        self.att_num_heads = att_num_heads
        self.att_dropout = att_dropout
        self.tcn_depth = tcn_depth
        self.tcn_kernel_size = tcn_kernel_size
        self.tcn_n_filters = tcn_n_filters
        self.tcn_dropout = tcn_dropout
        self.tcn_activation = tcn_activation
        self.concat = concat
        self.max_norm_const = max_norm_const

        self.ensuredims = Ensure4d()
        self.dimshuffle = Expression(transpose_time_to_spat)

        self.conv_block = _ConvBlock(
            n_channels=n_channels,  # input shape: (batch_size, 1, T, C)
            n_filters=conv_block_n_filters,
            kernel_length_1=conv_block_kernel_length_1,
            kernel_length_2=conv_block_kernel_length_2,
            pool_size_1=conv_block_pool_size_1,
            pool_size_2=conv_block_pool_size_2,
            depth_mult=conv_block_depth_mult,
            dropout=conv_block_dropout
        )

        self.F2 = int(conv_block_depth_mult * conv_block_n_filters)
        self.Tc = int(input_size_s * sfreq / (
            conv_block_pool_size_1 * conv_block_pool_size_2))
        self.Tw = self.Tc - self.n_windows + 1

        self.attention_blocks = nn.ModuleList([
            _AttentionBlock(
                in_shape=self.F2,
                head_dim=self.att_head_dim,
                num_heads=att_num_heads,
                dropout=att_dropout,
            ) for _ in range(self.n_windows)
        ])

        self.temporal_conv_nets = nn.ModuleList([
            nn.Sequential(
                *[_TCNResidualBlock(
                    in_channels=self.F2,
                    kernel_size=tcn_kernel_size,
                    n_filters=tcn_n_filters,
                    dropout=tcn_dropout,
                    activation=tcn_activation,
                    dilation=2**i
                ) for i in range(tcn_depth)]
            ) for _ in range(self.n_windows)
        ])

        if self.concat:
            self.max_norm_linears = nn.ModuleList([
                MaxNormLinear(
                    in_features=self.F2 * self.n_windows,
                    out_features=self.n_classes,
                    max_norm_val=self.max_norm_const
                )
            ])
        else:
            self.max_norm_linears = nn.ModuleList([
                MaxNormLinear(
                    in_features=self.F2,
                    out_features=self.n_classes,
                    max_norm_val=self.max_norm_const
                ) for _ in range(self.n_windows)
            ])

        self.sfmx = nn.LogSoftmax(dim=1)

    def forward(self, X):
        # Dimension: (batch_size, C, T)
        X = self.ensuredims(X)
        # Dimension: (batch_size, C, T, 1)
        X = self.dimshuffle(X)
        # Dimension: (batch_size, 1, T, C)

        # ----- Sliding window -----
        conv_feat = self.conv_block(X)
        # Dimension: (batch_size, F2, Tc, 1)
        conv_feat = conv_feat.view(-1, self.F2, self.Tc)
        # Dimension: (batch_size, F2, Tc)

        # ----- Sliding window -----
        sw_concat = []  # to store sliding window outputs
        for w in range(self.n_windows):
            conv_feat_w = conv_feat[..., w:w + self.Tw]
            # Dimension: (batch_size, F2, Tw)

            # ----- Attention block -----
            att_feat = self.attention_blocks[w](conv_feat_w)
            # Dimension: (batch_size, F2, Tw)

            # ----- Temporal convolutional network (TCN) -----
            tcn_feat = self.temporal_conv_nets[w](att_feat)[..., -1]
            # Dimension: (batch_size, F2)

            # Outputs of sliding window can be either averaged after being
            # mapped by dense layer or concatenated then mapped by a dense
            # layer
            if not self.concat:
                tcn_feat = self.max_norm_linears[w](tcn_feat)

            sw_concat.append(tcn_feat)

        # ----- Aggregation and prediction -----
        if self.concat:
            sw_concat = torch.cat(sw_concat, dim=1)
            sw_concat = self.max_norm_linears[0](sw_concat)
        else:
            if len(sw_concat) > 1:  # more than one window
                sw_concat = torch.stack(sw_concat, dim=0)
                sw_concat = torch.mean(sw_concat, dim=0)
            else:  # one window (# windows = 1)
                sw_concat = sw_concat[0]

        return self.sfmx(sw_concat)


class _ConvBlock(nn.Module):
    """ Convolutional block proposed in ATCNet [1]_, inspired by the EEGNet
    architecture [2]_.

    References
    ----------
    .. [1] H. Altaheri, G. Muhammad and M. Alsulaiman, "Physics-informed
           attention temporal convolutional network for EEG-based motor imagery
           classification," in IEEE Transactions on Industrial Informatics,
           2022, doi: 10.1109/TII.2022.3197419.
    .. [2] Lawhern, V. J., Solon, A. J., Waytowich, N. R., Gordon,
           S. M., Hung, C. P., & Lance, B. J. (2018).
           EEGNet: A Compact Convolutional Network for EEG-based
           Brain-Computer Interfaces.
           arXiv preprint arXiv:1611.08024.
    """
    def __init__(
        self,
        n_channels,
        n_filters=16,
        kernel_length_1=64,
        kernel_length_2=16,
        pool_size_1=8,
        pool_size_2=7,
        depth_mult=2,
        dropout=0.3,
    ):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=n_filters,
            kernel_size=(kernel_length_1, 1),
            padding="same",
            bias=False,
        )

        self.bn1 = nn.BatchNorm2d(num_features=n_filters, eps=1e-4)

        n_depth_kernels = n_filters * depth_mult
        self.conv2 = nn.Conv2d(
            in_channels=n_filters,
            out_channels=n_depth_kernels,
            groups=n_filters,
            kernel_size=(1, n_channels),
            padding="valid",
            bias=False,
        )

        self.bn2 = nn.BatchNorm2d(num_features=n_depth_kernels, eps=1e-4)

        self.activation2 = nn.ELU()

        self.pool2 = nn.AvgPool2d(kernel_size=(pool_size_1, 1))

        self.drop2 = nn.Dropout2d(dropout)

        self.conv3 = nn.Conv2d(
            in_channels=n_depth_kernels,
            out_channels=n_depth_kernels,
            kernel_size=(kernel_length_2, 1),
            padding="same",
            bias=False,
        )

        self.bn3 = nn.BatchNorm2d(num_features=n_depth_kernels, eps=1e-4)

        self.activation3 = nn.ELU()

        self.pool3 = nn.AvgPool2d(kernel_size=(pool_size_2, 1))

        self.drop3 = nn.Dropout2d(dropout)

    def forward(self, X):
        # ----- Temporal convolution -----
        # Dimension: (batch_size, 1, T, C)
        X = self.conv1(X)
        X = self.bn1(X)
        # Dimension: (batch_size, F1, T, C)

        # ----- Depthwise channels convolution -----
        X = self.conv2(X)
        X = self.bn2(X)
        X = self.activation2(X)
        # Dimension: (batch_size, F1*D, T, 1)
        X = self.pool2(X)
        X = self.drop2(X)
        # Dimension: (batch_size, F1*D, T/P1, 1)

        # ----- "Spatial" convolution -----
        X = self.conv3(X)
        X = self.bn3(X)
        X = self.activation3(X)
        # Dimension: (batch_size, F1*D, T/P1, 1)
        X = self.pool3(X)
        X = self.drop3(X)
        # Dimension: (batch_size, F1*D, T/(P1*P2), 1)

        return X


class _AttentionBlock(nn.Module):
    """Multi Head self Attention (MHA) block used in ATCNet [1]_, inspired from
    [2]_.

    References
    ----------
    .. [1] H. Altaheri, G. Muhammad and M. Alsulaiman, "Physics-informed
           attention temporal convolutional network for EEG-based motor imagery
           classification," in IEEE Transactions on Industrial Informatics,
           2022, doi: 10.1109/TII.2022.3197419.
    .. [2] Vaswani, A. et al., "Attention is all you need",
           in Advances in neural information processing systems, 2017.
    """
    def __init__(
        self,
        in_shape=32,
        head_dim=8,
        num_heads=2,
        dropout=0.5,
    ):
        super().__init__()
        self.in_shape = in_shape
        self.head_dim = head_dim
        self.num_heads = num_heads

        # Puts time dimension at -2 and feature dim at -1
        self.dimshuffle = Expression(lambda x: x.permute(0, 2, 1))

        # Layer normalization
        self.ln = nn.LayerNorm(normalized_shape=in_shape, eps=1e-6)

        # Multi-head self-attention layer
        # (We had to reimplement it since the original code is in tensorflow,
        # where it is possible to have an embedding dimension different than
        # the input and output dimensions, which is not possible in pytorch.)
        self.mha = _MHA(
            input_dim=in_shape,
            head_dim=head_dim,
            output_dim=in_shape,
            num_heads=num_heads,
            dropout=dropout,
        )

        # XXX: This line in the official code is weird, as there is already
        # dropout in the MultiheadAttention layer. They also don't mention
        # any additional dropout between the attention block and TCN in the
        # paper. We are adding it here however to follo so we are removing this
        # for now.
        self.drop = nn.Dropout(0.3)

    def forward(self, X):
        # Dimension: (batch_size, F2, Tw)
        X = self.dimshuffle(X)
        # Dimension: (batch_size, Tw, F2)

        # ----- Layer norm -----
        out = self.ln(X)

        # ----- Self-Attention -----
        out = self.mha(out, out, out)
        # Dimension: (batch_size, Tw, F2)

        # XXX In the paper fig. 1, it is drawn that layer normalization is
        # performed before the skip connection, while it is done afterwards
        # in the official code. Here we follow the code.

        # ----- Skip connection -----
        out = X + self.drop(out)

        # Move back to shape (batch_size, F2, Tw) from the beginning
        return self.dimshuffle(out)


class _TCNResidualBlock(nn.Module):
    """ Modified TCN Residual block as proposed in [1]_. Inspired from
    Temporal Convolutional Networks (TCN) [2]_.

    References
    ----------
    .. [1] H. Altaheri, G. Muhammad and M. Alsulaiman, "Physics-informed
           attention temporal convolutional network for EEG-based motor imagery
           classification," in IEEE Transactions on Industrial Informatics,
           2022, doi: 10.1109/TII.2022.3197419.
    .. [2] Bai, S., Kolter, J. Z., & Koltun, V.
           "An empirical evaluation of generic convolutional and recurrent
           networks for sequence modeling", 2018.
    """
    def __init__(
        self,
        in_channels,
        kernel_size=4,
        n_filters=32,
        dropout=0.3,
        activation=nn.ELU(),
        dilation=1
    ):
        super().__init__()
        self.activation = activation
        self.dilation = dilation
        self.dropout = dropout
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.in_channels = in_channels

        self.conv1 = CausalConv1d(
            in_channels=in_channels,
            out_channels=n_filters,
            kernel_size=kernel_size,
            dilation=dilation,
        )
        nn.init.kaiming_uniform_(self.conv1.weight)

        self.bn1 = nn.BatchNorm1d(n_filters)

        self.drop1 = nn.Dropout(dropout)

        self.conv2 = CausalConv1d(
            in_channels=n_filters,
            out_channels=n_filters,
            kernel_size=kernel_size,
            dilation=dilation,
        )
        nn.init.kaiming_uniform_(self.conv2.weight)

        self.bn2 = nn.BatchNorm1d(n_filters)

        self.drop2 = nn.Dropout(dropout)

        # Reshape the input for the residual connection when necessary
        if in_channels != n_filters:
            self.reshaping_conv = nn.Conv1d(
                n_filters,
                kernel_size=1,
                padding='same',
            )
        else:
            self.reshaping_conv = nn.Identity()

    def forward(self, X):
        # Dimension: (batch_size, F2, Tw)
        # ----- Double dilated convolutions -----
        out = self.conv1(X)
        out = self.bn1(out)
        out = self.activation(out)
        out = self.drop1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.activation(out)
        out = self.drop2(out)

        out = self.reshaping_conv(out)

        # ----- Residual connection -----
        out = X + out

        return self.activation(out)


class _MHA(nn.Module):
    def __init__(
        self,
        input_dim: int,
        head_dim: int,
        output_dim: int,
        num_heads: int,
        dropout: float = 0.,
    ):
        """Multi-head Attention

        The difference between this module and torch.nn.MultiheadAttention is
        that this module supports embedding dimensions different then input
        and output ones. It also does not support sequences of different
        length.

        Parameters
        ----------
        input_dim : int
            Dimension of query, key and value inputs.
        head_dim : int
            Dimension of embed query, key and value in each head,
            before computing attention.
        output_dim : int
            Output dimension.
        num_heads : int
            Number of heads in the multi-head architecture.
        dropout : float, optional
            Dropout probability on output weights. Default: 0.0 (no dropout).
        """

        super(_MHA, self).__init__()

        self.input_dim = input_dim
        self.head_dim = head_dim
        # typical choice for the split dimension of the heads
        self.embed_dim = head_dim * num_heads

        # embeddings for multi-head projections
        self.fc_q = nn.Linear(input_dim, self.embed_dim)
        self.fc_k = nn.Linear(input_dim, self.embed_dim)
        self.fc_v = nn.Linear(input_dim, self.embed_dim)

        # output mapping
        self.fc_o = nn.Linear(self.embed_dim, output_dim)

        # dropout
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor
    ) -> torch.Tensor:
        """ Compute MHA(Q, K, V)

        Parameters
        ----------
        Q: torch.Tensor of size (batch_size, seq_len, input_dim)
            Input query (Q) sequence.
        K: torch.Tensor of size (batch_size, seq_len, input_dim)
            Input key (K) sequence.
        V: torch.Tensor of size (batch_size, seq_len, input_dim)
            Input value (V) sequence.

        Returns
        -------
        O: torch.Tensor of size (batch_size, seq_len, output_dim)
            Output MHA(Q, K, V)
        """
        assert Q.shape[-1] == K.shape[-1] == V.shape[-1] == self.input_dim

        batch_size, _, _ = Q.shape

        # embedding for multi-head projections (masked or not)
        Q = self.fc_q(Q)  # (B, S, D)
        K, V = self.fc_k(K), self.fc_v(V)  # (B, S, D)

        # Split into num_head vectors (num_heads * batch_size, n/m, head_dim)
        Q_ = torch.cat(Q.split(self.head_dim, -1), 0)  # (B', S, D')
        K_ = torch.cat(K.split(self.head_dim, -1), 0)  # (B', S, D')
        V_ = torch.cat(V.split(self.head_dim, -1), 0)  # (B', S, D')

        # Attention weights of size (num_heads * batch_size, n, m):
        # measures how similar each pair of Q and K is.
        W = torch.softmax(
            Q_.bmm(
                K_.transpose(-2, -1)  # (B', D', S)
            )
            / np.sqrt(self.head_dim),
            -1
        )  # (B', N, M)

        # Multihead output (batch_size, seq_len, dim):
        # weighted sum of V where a value gets more weight if its corresponding
        # key has larger dot product with the query.
        H = torch.cat(
            (
                W  # (B', S, S)
                .bmm(V_)  # (B', S, D')
            ).split(batch_size, 0),  # [(B, S, D')] * num_heads
            -1
        )  # (B, S, D)

        out = self.fc_o(H)

        return self.dropout(out)
