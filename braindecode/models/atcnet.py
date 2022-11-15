# Authors: Cedric Rommel <cedric.rommel@inria.fr>
#
# License: BSD (3-clause)
import torch
from torch import nn

from .modules import Expression, Ensure4d
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
    att_embedding_dim : int
        Embedding dimension used in self-attention layers, denoted dh in
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
        Defaults to ``True``.
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
        att_embedding_dim=8,
        att_num_heads=2,
        att_dropout=0.5,
        tcn_depth=2,
        tcn_kernel_size=4,
        tcn_n_filters=32,
        tcn_dropout=0.3,
        tcn_activation=nn.ELU(),
        concat=True,
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
        self.att_embedding_dim = att_embedding_dim
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
                embedding_dim=self.att_embedding_dim,
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
            self.max_norm_linear = MaxNormLinear(
                in_features=self.F2 * self.n_windows,
                out_features=self.n_classes,
                max_norm_val=self.max_norm_const
            )
        else:
            self.max_norm_linear = MaxNormLinear(
                in_features=self.F2,
                out_features=self.n_classes,
                max_norm_val=self.max_norm_const
            )

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
        # TODO: This could be optimized by creating a super-batch, doing a
        # single forward and splitting
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
                tcn_feat = self.max_norm_linear(tcn_feat)

            sw_concat.append(tcn_feat)

        # ----- Aggregation and prediction -----
        if self.concat:
            sw_concat = torch.cat(sw_concat, dim=1)
            sw_concat = self.max_norm_linear(sw_concat)
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
        embedding_dim=8,
        num_heads=2,
        dropout=0.5,
    ):
        super().__init__()

        # Puts time dimension at -2 and feature dim at -1
        self.dimshuffle = Expression(lambda x: x.permute(0, 2, 1))

        # Layer normalization
        self.ln = nn.LayerNorm(normalized_shape=in_shape, eps=1e-6)

        # Projection into embedding space of dimension 8
        # (because it seems pytorch MHA is contrained to have square weight
        # matrices)
        self.key_map = nn.Linear(
            in_features=in_shape,
            out_features=embedding_dim,
            bias=True,
        )
        self.query_map = nn.Linear(
            in_features=in_shape,
            out_features=embedding_dim,
            bias=True,
        )
        self.value_map = nn.Linear(
            in_features=in_shape,
            out_features=embedding_dim,
            bias=True,
        )

        self.mha = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Linear mapping from embedding space to original feature space
        self.output_map = nn.Linear(
            in_features=embedding_dim,
            out_features=in_shape,
            bias=True,
        )

        # XXX: This line in the official code is weird, as there is already
        # dropout in the MultiheadAttention layer. They also don't mention
        # any additional dropout between the attention block and TCN in the
        # paper, so we are removing this for now.
        # self.drop = nn.Dropout(0.3)

    def forward(self, X):
        # Dimension: (batch_size, F2, Tw)
        X = self.dimshuffle(X)
        # Dimension: (batch_size, Tw, F2)

        # ----- Layer norm -----
        out = self.ln(X)

        # ----- Embedding -----
        # (this is done manually here as it seems pytorch multihead attention
        # is contrained to have square weight matrices)
        key = self.key_map(out)
        query = self.query_map(out)
        value = self.value_map(out)

        # ----- Attention -----
        # Dimension: (batch_size, Tw, Dh)
        out, _ = self.mha(key, query, value)
        # Dimension: (batch_size, Tw, Dh)

        # ----- Getting back to input space -----
        out = self.output_map(out)
        # Dimension: (batch_size, Tw, F2)

        # XXX In the paper fig. 1, it is drawn that layer normalization is
        # performed before the skip connection, while it is done afterwards
        # in the official code. Here we follow the code.

        # ----- Skip connection -----
        out = X + out

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
        nn.init.kaiming_uniform_(self.conv1.conv.weight)

        self.bn1 = nn.BatchNorm1d(n_filters)

        self.drop1 = nn.Dropout(dropout)

        self.conv2 = CausalConv1d(
            in_channels=n_filters,
            out_channels=n_filters,
            kernel_size=kernel_size,
            dilation=dilation,
        )
        nn.init.kaiming_uniform_(self.conv2.conv.weight)

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


class CausalConv1d(nn.Module):
    """Causal 1-dimensional convolution

    Code modified from [1]_ and [2]_.

    Parameters
    ----------
    in_channels : int
        Input channels.
    out_channels : int
        Output channels (number of filters).
    kernel_size : int
        Kernel size.
    dilation : int, optional
        Dilation (number of elements to skip within kernel multiplication).
        Default to 1.
    **kwargs :
        Other keyword arguments to pass to torch.nn.Conv1d, except for
        `padding`!!

    References
    ----------
    .. [1] https://discuss.pytorch.org/t/causal-convolution/3456/4
    .. [2] https://gist.github.com/paultsw/7a9d6e3ce7b70e9e2c61bc9287addefc
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        dilation=1,
        **kwargs,
    ):
        super().__init__()
        assert "padding" not in kwargs, (
            "The padding parameter is controlled internally by "
            f"{type(self).__name__} class. You should not try to override this"
            " parameter."
        )

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dilation = dilation

        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=(kernel_size - 1) * dilation,
            **kwargs
        )

    def forward(self, X):
        out = self.conv(X)
        return out[..., :-self.conv.padding[0]]


class MaxNormLinear(nn.Module):
    """Linear layer with MaxNorm constraing on weights.

    Equivalent of Keras tf.keras.Dense(..., kernel_constraint=max_norm())
    [1, 2]_. Implemented as advised in [3]_.

    Parameters
    ----------
    in_features: int
        Size of each input sample.
    out_features: int
        Size of each output sample.
    bias: bool, optional
        If set to ``False``, the layer will not learn an additive bias.
        Default: ``True``.

    References
    ----------
    .. [1] https://keras.io/api/layers/core_layers/dense/#dense-class
    .. [2] https://www.tensorflow.org/api_docs/python/tf/keras/constraints/MaxNorm
    .. [3] https://discuss.pytorch.org/t/how-to-correctly-implement-in-place-max-norm-constraint/96769
    """
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        max_norm_val=2,
        eps=1e-5,
    ):
        super().__init__()
        self._max_norm_val = max_norm_val
        self._eps = eps
        self.linear = nn.Linear(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
        )

    def forward(self, X):
        self._max_norm(self.linear.weight)
        return self.linear(X)

    def _max_norm(self, w):
        with torch.no_grad():
            norm = w.norm(2, dim=0, keepdim=True).clamp(
                min=self._max_norm_val / 2
            )
            desired = torch.clamp(norm, max=self._max_norm_val)
            w *= (desired / (self._eps + norm))
