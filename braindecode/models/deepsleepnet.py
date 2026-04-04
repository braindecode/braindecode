# Authors: Théo Gnassounou <theo.gnassounou@inria.fr>
#          Sarthak Tayal <sarthaktayal2@gmail.com>
#
# License: BSD (3-clause)
import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

from braindecode.models.base import EEGModuleMixin


class DeepSleepNet(EEGModuleMixin, nn.Module):
    r"""DeepSleepNet from Supratak et al (2017) [Supratak2017]_.

    :bdg-success:`Convolution` :bdg-secondary:`Recurrent`

    .. figure:: https://raw.githubusercontent.com/akaraspt/deepsleepnet/master/img/deepsleepnet.png
        :align: center
        :alt: DeepSleepNet Architecture
        :width: 700px

    DeepSleepNet is a deep learning model for automatic sleep stage scoring
    based on raw single-channel EEG. It consists of two main parts:

    1. **Representation learning** — two CNNs with different filter sizes
       extract time-invariant features from each 30-s EEG epoch.
    2. **Sequence residual learning** — bidirectional LSTMs learn temporal
       information such as stage transition rules, combined with a residual
       shortcut from the CNN features.

    .. rubric:: Representation Learning

    Two parallel CNN paths process the raw input simultaneously:

    - **Small-filter path** — first conv uses filter length ≈ Fs/2 and
      stride ≈ Fs/16, capturing *when* characteristic transients occur
      (temporal precision).
    - **Large-filter path** — first conv uses filter length ≈ 4·Fs and
      stride ≈ Fs/2, capturing *which* frequency components dominate
      (frequency precision).

    Each path consists of four convolutional layers (1-D convolution →
    :class:`~torch.nn.BatchNorm2d` → activation, configurable via the
    per-path activation settings) and two :class:`~torch.nn.MaxPool2d`
    layers with :class:`~torch.nn.Dropout` after the first pooling.
    Outputs from both paths are **concatenated** to form the epoch
    embedding.

    .. rubric:: Sequence Residual Learning

    Two layers of bidirectional LSTMs encode temporal dependencies across
    epochs. A **residual shortcut** (fully connected →
    :class:`~torch.nn.BatchNorm1d` → :class:`~torch.nn.ReLU`) projects
    the CNN features to the BiLSTM output dimension and is **added** to
    the BiLSTM output, improving gradient flow and preserving salient
    CNN evidence.

    .. rubric:: Implementation Differences

    .. note::

       **Peephole connections.** The original implementation uses
       TensorFlow ``LSTMCell`` with ``use_peepholes=True``, which allows
       gates to inspect the cell state. :class:`torch.nn.LSTM` does not
       support peepholes; this implementation uses standard LSTM gates.

       **Sequence length.** The original model processes **sequences of
       epochs** through the BiLSTM to capture cross-epoch transition rules.
       This implementation processes **single epochs** (sequence length 1),
       so the BiLSTM acts as a nonlinear feature transform with a residual
       connection. To leverage multi-epoch context, batch consecutive
       epochs as a sequence externally.

       **Activation.** The original uses :class:`~torch.nn.ReLU` for both
       CNN paths. This implementation defaults to :class:`~torch.nn.ELU`
       for the large-filter path (``activation_large``), which can be
       overridden.

    .. rubric:: Training (from the paper)

    - **Two-step procedure.** (i) Pre-train the CNN part on a
      class-balanced training set using oversampling; (ii) fine-tune the
      whole network with sequential batches using a lower learning rate
      for the CNNs and a higher one for the sequence residual part.
    - **Dropout** with probability 0.5 is used throughout the model.
    - **L2 weight decay** (λ = 10⁻³) is applied only to the first
      convolutional layers of both CNN paths.
    - **Gradient clipping** rescales gradients when their global norm
      exceeds a threshold.
    - **State handling.** BiLSTM states are reinitialized per subject so
      that temporal context does not leak across recordings.

    Parameters
    ----------
    activation_large : type[nn.Module], default=nn.ELU
        Activation class for the large-filter CNN path.
    activation_small : type[nn.Module], default=nn.ReLU
        Activation class for the small-filter CNN path.
    return_feats : bool, default=False
        If True, return features before the final linear layer.
    drop_prob : float, default=0.5
        Dropout probability applied throughout the network.
    bilstm_hidden_size : int, default=512
        Hidden size of the BiLSTM. The residual FC output dimension is
        ``2 * bilstm_hidden_size`` to match the concatenated directions.
    bilstm_num_layers : int, default=2
        Number of stacked BiLSTM layers.
    small_n_filters_1 : int, default=64
        First-conv output channels for the small-filter path.
    small_n_filters_2 : int, default=128
        Deep-conv (conv2--conv4) output channels for the small-filter path.
    small_first_kernel_size : int, default=50
        First-conv kernel size for the small path (paper: Fs/2).
    small_first_stride : int, default=6
        First-conv stride for the small path (paper: Fs/16).
    small_first_padding : int, default=22
        First-conv padding for the small path.
    small_pool1_kernel_size : int, default=8
        First max-pool kernel for the small path.
    small_pool1_stride : int, default=8
        First max-pool stride for the small path.
    small_pool1_padding : int, default=2
        First max-pool padding for the small path.
    small_deep_kernel_size : int, default=8
        Deep-conv kernel size for the small path.
    small_pool2_kernel_size : int, default=4
        Second max-pool kernel for the small path.
    small_pool2_stride : int, default=4
        Second max-pool stride for the small path.
    small_pool2_padding : int, default=1
        Second max-pool padding for the small path.
    large_n_filters_1 : int, default=64
        First-conv output channels for the large-filter path.
    large_n_filters_2 : int, default=128
        Deep-conv (conv2--conv4) output channels for the large-filter path.
    large_first_kernel_size : int, default=400
        First-conv kernel size for the large path (paper: 4*Fs).
    large_first_stride : int, default=50
        First-conv stride for the large path (paper: Fs/2).
    large_first_padding : int, default=175
        First-conv padding for the large path.
    large_pool1_kernel_size : int, default=4
        First max-pool kernel for the large path.
    large_pool1_stride : int, default=4
        First max-pool stride for the large path.
    large_pool1_padding : int, default=0
        First max-pool padding for the large path.
    large_deep_kernel_size : int, default=6
        Deep-conv kernel size for the large path.
    large_pool2_kernel_size : int, default=2
        Second max-pool kernel for the large path.
    large_pool2_stride : int, default=2
        Second max-pool stride for the large path.
    large_pool2_padding : int, default=1
        Second max-pool padding for the large path.

    References
    ----------
    .. [Supratak2017] Supratak, A., Dong, H., Wu, C., & Guo, Y. (2017).
       DeepSleepNet: A model for automatic sleep stage scoring based
       on raw single-channel EEG. IEEE Transactions on Neural Systems
       and Rehabilitation Engineering, 25(11), 1998-2008.
    """

    def __init__(
        self,
        n_outputs=5,
        return_feats=False,
        n_chans=None,
        chs_info=None,
        n_times=None,
        input_window_seconds=None,
        sfreq=None,
        activation_large: type[nn.Module] = nn.ELU,
        activation_small: type[nn.Module] = nn.ReLU,
        drop_prob: float = 0.5,
        bilstm_hidden_size: int = 512,
        bilstm_num_layers: int = 2,
        # Small CNN parameters
        small_n_filters_1: int = 64,
        small_n_filters_2: int = 128,
        small_first_kernel_size: int = 50,
        small_first_stride: int = 6,
        small_first_padding: int = 22,
        small_pool1_kernel_size: int = 8,
        small_pool1_stride: int = 8,
        small_pool1_padding: int = 2,
        small_deep_kernel_size: int = 8,
        small_pool2_kernel_size: int = 4,
        small_pool2_stride: int = 4,
        small_pool2_padding: int = 1,
        # Large CNN parameters
        large_n_filters_1: int = 64,
        large_n_filters_2: int = 128,
        large_first_kernel_size: int = 400,
        large_first_stride: int = 50,
        large_first_padding: int = 175,
        large_pool1_kernel_size: int = 4,
        large_pool1_stride: int = 4,
        large_pool1_padding: int = 0,
        large_deep_kernel_size: int = 6,
        large_pool2_kernel_size: int = 2,
        large_pool2_stride: int = 2,
        large_pool2_padding: int = 1,
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
        self.cnn1 = _CNNPath(
            n_filters_1=small_n_filters_1,
            n_filters_2=small_n_filters_2,
            first_kernel_size=small_first_kernel_size,
            first_stride=small_first_stride,
            first_padding=small_first_padding,
            pool1_kernel_size=small_pool1_kernel_size,
            pool1_stride=small_pool1_stride,
            pool1_padding=small_pool1_padding,
            deep_kernel_size=small_deep_kernel_size,
            pool2_kernel_size=small_pool2_kernel_size,
            pool2_stride=small_pool2_stride,
            pool2_padding=small_pool2_padding,
            activation=activation_small,
            drop_prob=drop_prob,
        )
        self.cnn2 = _CNNPath(
            n_filters_1=large_n_filters_1,
            n_filters_2=large_n_filters_2,
            first_kernel_size=large_first_kernel_size,
            first_stride=large_first_stride,
            first_padding=large_first_padding,
            pool1_kernel_size=large_pool1_kernel_size,
            pool1_stride=large_pool1_stride,
            pool1_padding=large_pool1_padding,
            deep_kernel_size=large_deep_kernel_size,
            pool2_kernel_size=large_pool2_kernel_size,
            pool2_stride=large_pool2_stride,
            pool2_padding=large_pool2_padding,
            activation=activation_large,
            drop_prob=drop_prob,
        )
        self.dropout = nn.Dropout(drop_prob)

        feat_size = _compute_feat_size(
            self.n_chans,
            self.n_times,
            small_n_filters_2,
            (small_first_kernel_size, small_first_stride, small_first_padding),
            (small_pool1_kernel_size, small_pool1_stride, small_pool1_padding),
            (small_pool2_kernel_size, small_pool2_stride, small_pool2_padding),
            large_n_filters_2,
            (large_first_kernel_size, large_first_stride, large_first_padding),
            (large_pool1_kernel_size, large_pool1_stride, large_pool1_padding),
            (large_pool2_kernel_size, large_pool2_stride, large_pool2_padding),
        )

        fc_out_features = bilstm_hidden_size * 2
        self.bilstm = nn.LSTM(
            input_size=feat_size,
            hidden_size=bilstm_hidden_size,
            num_layers=bilstm_num_layers,
            batch_first=True,
            dropout=drop_prob if bilstm_num_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.residual_shortcut = nn.Sequential(
            nn.Linear(feat_size, fc_out_features, bias=False),
            nn.BatchNorm1d(num_features=fc_out_features),
            nn.ReLU(),
        )

        self.flatten_cnn = Rearrange(
            "batch filters nchans ntimes -> batch (filters nchans ntimes)"
        )
        self.add_seq_dim = Rearrange("batch features -> batch 1 features")
        self.remove_seq_dim = Rearrange("batch 1 features -> batch features")

        self.features_extractor = nn.Identity()
        self.len_last_layer = fc_out_features
        self.final_layer = (
            nn.Identity()
            if return_feats
            else nn.Linear(fc_out_features, self.n_outputs)
        )

    def forward(self, x):
        if x.ndim == 3:
            x = x.unsqueeze(1)

        x1 = self.flatten_cnn(self.cnn1(x))
        x2 = self.flatten_cnn(self.cnn2(x))

        x = self.dropout(torch.cat((x1, x2), dim=1))
        residual = self.residual_shortcut(x)
        x, _ = self.bilstm(self.add_seq_dim(x))
        x = self.remove_seq_dim(x)
        x = self.dropout(x + residual)

        return self.final_layer(self.features_extractor(x))


def _conv_out(width, kernel_stride_padding):
    """Compute output width after a conv/pool layer.

    Uses the standard formula::

        out = (width + 2 * padding - kernel_size) // stride + 1

    Parameters
    ----------
    width : int
        Input temporal width.
    kernel_stride_padding : tuple of (int, int, int)
        ``(kernel_size, stride, padding)`` of the layer.
    """
    kernel_size, stride, padding = kernel_stride_padding
    return (width + 2 * padding - kernel_size) // stride + 1


def _compute_feat_size(
    n_chans,
    n_times,
    s_filt,
    s_conv1,
    s_pool1,
    s_pool2,
    l_filt,
    l_conv1,
    l_pool1,
    l_pool2,
):
    """Compute concatenated feature size of both CNN paths.

    Each conv/pool arg is a ``(kernel, stride, padding)`` tuple.
    Conv2-4 use same-padding so they don't change the width.
    """
    sw = _conv_out(_conv_out(_conv_out(n_times, s_conv1), s_pool1), s_pool2)
    lw = _conv_out(_conv_out(_conv_out(n_times, l_conv1), l_pool1), l_pool2)
    if sw <= 0 or lw <= 0:
        raise ValueError(
            f"n_times={n_times} is too small for the configured conv/pool "
            f"parameters (small path width={sw}, large path width={lw}). "
            f"Increase n_times or adjust kernel sizes."
        )
    return n_chans * (s_filt * sw + l_filt * lw)


class _CNNPath(nn.Module):
    """Single CNN path: conv1 → pool1 → dropout → conv2 → conv3 → conv4 → pool2.

    Used twice in DeepSleepNet with different hyperparameters: once for the
    small-filter (temporal) path and once for the large-filter (frequency) path.

    Parameters
    ----------
    n_filters_1 : int
        Output channels of the first convolution.
    n_filters_2 : int
        Output channels of the deeper convolutions (conv2--conv4).
    first_kernel_size : int
        Temporal kernel size of the first convolution.
    first_stride : int
        Stride of the first convolution.
    first_padding : int
        Padding of the first convolution.
    pool1_kernel_size : int
        Kernel size of the first max-pooling.
    pool1_stride : int
        Stride of the first max-pooling.
    pool1_padding : int
        Padding of the first max-pooling.
    deep_kernel_size : int
        Temporal kernel size shared by conv2, conv3, conv4.
    pool2_kernel_size : int
        Kernel size of the second max-pooling.
    pool2_stride : int
        Stride of the second max-pooling.
    pool2_padding : int
        Padding of the second max-pooling.
    activation : type[nn.Module]
        Activation function class.
    drop_prob : float
        Dropout probability.
    """

    def __init__(
        self,
        n_filters_1: int = 64,
        n_filters_2: int = 128,
        first_kernel_size: int = 50,
        first_stride: int = 6,
        first_padding: int = 22,
        pool1_kernel_size: int = 8,
        pool1_stride: int = 8,
        pool1_padding: int = 2,
        deep_kernel_size: int = 8,
        pool2_kernel_size: int = 4,
        pool2_stride: int = 4,
        pool2_padding: int = 1,
        activation: type[nn.Module] = nn.ReLU,
        drop_prob: float = 0.5,
    ):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=n_filters_1,
                kernel_size=(1, first_kernel_size),
                stride=(1, first_stride),
                padding=(0, first_padding),
                bias=False,
            ),
            nn.BatchNorm2d(num_features=n_filters_1),
            activation(),
        )
        self.pool1 = nn.MaxPool2d(
            kernel_size=(1, pool1_kernel_size),
            stride=(1, pool1_stride),
            padding=(0, pool1_padding),
        )
        self.dropout = nn.Dropout(p=drop_prob)
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=n_filters_1,
                out_channels=n_filters_2,
                kernel_size=(1, deep_kernel_size),
                stride=1,
                padding="same",
                bias=False,
            ),
            nn.BatchNorm2d(num_features=n_filters_2),
            activation(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=n_filters_2,
                out_channels=n_filters_2,
                kernel_size=(1, deep_kernel_size),
                stride=1,
                padding="same",
                bias=False,
            ),
            nn.BatchNorm2d(num_features=n_filters_2),
            activation(),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=n_filters_2,
                out_channels=n_filters_2,
                kernel_size=(1, deep_kernel_size),
                stride=1,
                padding="same",
                bias=False,
            ),
            nn.BatchNorm2d(num_features=n_filters_2),
            activation(),
        )
        self.pool2 = nn.MaxPool2d(
            kernel_size=(1, pool2_kernel_size),
            stride=(1, pool2_stride),
            padding=(0, pool2_padding),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.dropout(self.pool1(x))
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool2(x)
        return x
