# Authors: Théo Gnassounou <theo.gnassounou@inria.fr>
#          Sarthak Tayal <sarthaktayal2@gmail.com>
#
# License: BSD (3-clause)
import torch
import torch.nn as nn

from braindecode.models.base import EEGModuleMixin


class DeepSleepNet(EEGModuleMixin, nn.Module):
    r"""DeepSleepNet from Supratak et al (2017) [Supratak2017]_.

    :bdg-success:`Convolution` :bdg-secondary:`Recurrent`

    .. figure:: https://raw.githubusercontent.com/akaraspt/deepsleepnet/master/img/deepsleepnet.png
        :align: center
        :alt: DeepSleepNet Architecture
        :width: 700px

    .. rubric:: Architectural Overview

    DeepSleepNet couples **dual-path convolution neural network representation learning** with
    **sequence residual learning** via bidirectional LSTMs.

    The network have:

    - (i) learns complementary, time-frequency features from each
      30-s epoch using **two parallel CNNs** (small vs. large first-layer filters), then
    - (ii) models **temporal dependencies across epochs** using **two-layer BiLSTMs**
      with a **residual shortcut** from the CNN features, and finally
    - (iii) outputs per-epoch sleep stages. This design encodes both
      epoch-local patterns and longer-range transition rules used by human scorers.

    In term of implementation:

    - (i) :class:`_RepresentationLearning` two CNNs extract epoch-wise features
      (small-filter path for temporal precision; large-filter path for frequency precision);
    - (ii) :class:`_SequenceResidualLearning` stacked BiLSTMs with peepholes + residual shortcut
      inject temporal context while preserving CNN evidence;
    - (iii) :class:`_Classifier` linear readout (softmax) for the five sleep stages.

    .. rubric:: Macro Components

    - :class:`_RepresentationLearning` **(dual-path CNN → epoch feature)**

        - *Operations.*
        - **Small-filter CNN** 4 times:

            - :class:`~torch.nn.Conv1d`
            - :class:`~torch.nn.BatchNorm1d`
            - :class:`~torch.nn.ReLU`
            - :class:`~torch.nn.MaxPool1d` after.

        - First conv uses **filter length ≈ Fs/2** and **stride ≈ Fs/16** to emphasize *timing* of graphoelements.

    - **Large-filter CNN**:

        - Same stack but first conv uses **filter length ≈ 4·Fs** and
        - **stride ≈ Fs/2** to emphasize *frequency* content.

    - Outputs from both paths are **concatenated** into the epoch embedding ``a_t``.

    - *Rationale.*
      Two first-layer scales provide a **learned, dual-scale filter bank** that trades
      temporal vs. frequency precision without hand-crafted features.

    - :class:`_SequenceResidualLearning` (:class:`~torch.nn.BiLSTM` **context + residual fusion)**

        - *Operations.*
        - **Two-layer BiLSTM** with **peephole connections** processes the sequence of epoch embeddings
          ``{a_t}`` forward and backward; hidden states from both directions are **concatenated**.
        - A **shortcut MLP** (fully connected + :class:`~torch.nn.BatchNorm1d` + :class:`~torch.nn.ReLU`) projects ``a_t`` to the BiLSTM output
          dimension and is **added** (residual) to the :class:`~torch.nn.BiLSTM` output at each time step.
        - *Role.* Encodes **stage-transition rules** and smooths predictions over time while preserving
          salient CNN features via the residual path.

    - :class:`_Classifier` **(epoch-wise prediction)**

        - *Operations.*
        - :class:`~torch.nn.Linear` to produce per-epoch class probabilities.

    Original training uses two-step optimization: CNN pretraining on class-balanced data,
    then end-to-end fine-tuning with sequential batches.

    .. rubric:: Convolutional Details

    - **Temporal (where time-domain patterns are learned).**

      Both CNN paths use **1-D temporal convolutions**. The *small-filter* path (first kernel ≈ Fs/2,
      stride ≈ Fs/16) captures *when* characteristic transients occur; the *large-filter* path
      (first kernel ≈ 4·Fs, stride ≈ Fs/2) captures *which* frequency components dominate over the
      epoch. Deeper layers use **small kernels** to refine features with fewer parameters, interleaved
      with **max pooling** for downsampling.

    - **Spatial (how channels are processed).**

      The original model operates on **single-channel** raw EEG; convolutions therefore mix only
      along time (no spatial convolution across electrodes).

    - **Spectral (how frequency information emerges).**

      No explicit Fourier/wavelet transform is used. The **large-filter path** serves as a
      *frequency-sensitive* analyzer, while the **small-filter path** remains *time-sensitive*,
      together functioning as a **two-band learned filter bank** at the first layer.

    .. rubric:: Attention / Sequential Modules

    - **Type.** **Bidirectional LSTM** (two layers) with **peephole connections**; forward and
      backward streams are independent and concatenated.
    - **Shapes.** For a sequence of ``N`` epochs, the CNN produces ``{a_t} ∈ R^{D}``;
      BiLSTM outputs ``h_t ∈ R^{2H}``; the shortcut MLP maps ``a_t → R^{2H}`` to enable
      **element-wise residual addition**.
    - **Role.** Models **long-range temporal dependencies** (e.g., persisting N2 without visible
      K-complex/spindles), stabilizing per-epoch predictions.


    .. rubric:: Additional Mechanisms

    - **Residual shortcut over sequence encoder.** Adds projected CNN features to BiLSTM outputs,
      improving gradient flow and retaining discriminative content from representation learning.
    - **Two-step training.**

        - (i) **Pretrain** the CNN paths with class-balanced sampling;
        - (ii) **fine-tune** the full network with sequential batches, using **lower LR** for CNNs and **higher LR** for the
          sequence encoder.

    - **State handling.** BiLSTM states are **reinitialized per subject** so that temporal context
      does not leak across recordings.


    .. rubric:: Usage and Configuration

    - **Epoch pipeline.** Use **two parallel CNNs** with the first conv sized to **Fs/2** (small path)
      and **4·Fs** (large path), with strides **Fs/16** and **Fs/2**, respectively; stack three more
      conv blocks with small kernels, plus **max pooling** in each path. Concatenate path outputs
      to form epoch embeddings.
    - **Sequence encoder.** Apply **two-layer BiLSTM (peepholes)** over the sequence of embeddings;
      add a **projection MLP** on the CNN features and **sum** with BiLSTM outputs (residual).
      Finish with :class:`~torch.nn.Linear` per epoch.
    - **Reference implementation.** See the official repository for a faithful implementation and
      training scripts.

    Parameters
    ----------
    activation_large: nn.Module, default=nn.ELU
        Activation function class to apply. Should be a PyTorch activation
        module class like ``nn.ReLU`` or ``nn.ELU``. Default is ``nn.ELU``.
    activation_small: nn.Module, default=nn.ReLU
        Activation function class to apply. Should be a PyTorch activation
        module class like ``nn.ReLU`` or ``nn.ELU``. Default is ``nn.ReLU``.
    return_feats : bool
        If True, return the features, i.e. the output of the feature extractor
        (before the final linear layer). If False, pass the features through
        the final linear layer.
    drop_prob : float, default=0.5
        The dropout rate for regularization. Values should be between 0 and 1.
    bilstm_hidden_size : int, default=512
        Hidden size of the bidirectional LSTM. The residual fully connected
        layer output dimension is ``2 * bilstm_hidden_size`` (to match the
        BiLSTM concatenated forward/backward outputs).
    bilstm_num_layers : int, default=2
        Number of stacked BiLSTM layers.
    small_n_filters_1 : int, default=64
        Number of output filters in the first convolution of the small-filter
        CNN path.
    small_n_filters_2 : int, default=128
        Number of output filters in the deeper convolutions (conv2–conv4) of
        the small-filter CNN path.
    small_first_kernel_size : int, default=50
        Temporal kernel size of the first convolution in the small-filter path
        (paper value ≈ Fs/2).
    small_first_stride : int, default=6
        Stride of the first convolution in the small-filter path
        (paper value ≈ Fs/16).
    small_first_padding : int, default=22
        Padding of the first convolution in the small-filter path.
    small_pool1_kernel_size : int, default=8
        Kernel size of the first max-pooling in the small-filter path.
    small_pool1_stride : int, default=8
        Stride of the first max-pooling in the small-filter path.
    small_pool1_padding : int, default=2
        Padding of the first max-pooling in the small-filter path.
    small_deep_kernel_size : int, default=8
        Temporal kernel size of the deeper convolutions (conv2–conv4) in the
        small-filter path.
    small_pool2_kernel_size : int, default=4
        Kernel size of the second max-pooling in the small-filter path.
    small_pool2_stride : int, default=4
        Stride of the second max-pooling in the small-filter path.
    small_pool2_padding : int, default=1
        Padding of the second max-pooling in the small-filter path.
    large_n_filters_1 : int, default=64
        Number of output filters in the first convolution of the large-filter
        CNN path.
    large_n_filters_2 : int, default=128
        Number of output filters in the deeper convolutions (conv2–conv4) of
        the large-filter CNN path.
    large_first_kernel_size : int, default=400
        Temporal kernel size of the first convolution in the large-filter path
        (paper value ≈ 4·Fs).
    large_first_stride : int, default=50
        Stride of the first convolution in the large-filter path
        (paper value ≈ Fs/2).
    large_first_padding : int, default=175
        Padding of the first convolution in the large-filter path.
    large_pool1_kernel_size : int, default=4
        Kernel size of the first max-pooling in the large-filter path.
    large_pool1_stride : int, default=4
        Stride of the first max-pooling in the large-filter path.
    large_pool1_padding : int, default=0
        Padding of the first max-pooling in the large-filter path.
    large_deep_kernel_size : int, default=6
        Temporal kernel size of the deeper convolutions (conv2–conv4) in the
        large-filter path.
    large_pool2_kernel_size : int, default=2
        Kernel size of the second max-pooling in the large-filter path.
    large_pool2_stride : int, default=2
        Stride of the second max-pooling in the large-filter path.
    large_pool2_padding : int, default=1
        Padding of the second max-pooling in the large-filter path.

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
        self.cnn1 = _SmallCNN(
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
        self.cnn2 = _LargeCNN(
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
            small_n_filters_2=small_n_filters_2,
            small_conv1_k=small_first_kernel_size,
            small_conv1_s=small_first_stride,
            small_conv1_p=small_first_padding,
            small_pool1_k=small_pool1_kernel_size,
            small_pool1_s=small_pool1_stride,
            small_pool1_p=small_pool1_padding,
            small_pool2_k=small_pool2_kernel_size,
            small_pool2_s=small_pool2_stride,
            small_pool2_p=small_pool2_padding,
            large_n_filters_2=large_n_filters_2,
            large_conv1_k=large_first_kernel_size,
            large_conv1_s=large_first_stride,
            large_conv1_p=large_first_padding,
            large_pool1_k=large_pool1_kernel_size,
            large_pool1_s=large_pool1_stride,
            large_pool1_p=large_pool1_padding,
            large_pool2_k=large_pool2_kernel_size,
            large_pool2_s=large_pool2_stride,
            large_pool2_p=large_pool2_padding,
        )

        fc_out_features = bilstm_hidden_size * 2
        self.bilstm = _BiLSTM(
            input_size=feat_size,
            hidden_size=bilstm_hidden_size,
            num_layers=bilstm_num_layers,
            drop_prob=drop_prob,
        )
        self.fc = nn.Sequential(
            nn.Linear(feat_size, fc_out_features, bias=False),
            nn.BatchNorm1d(num_features=fc_out_features),
        )

        self.features_extractor = nn.Identity()
        self.len_last_layer = fc_out_features
        self.return_feats = return_feats

        # TODO: Add new way to handle return_features == True
        if not return_feats:
            self.final_layer = nn.Linear(fc_out_features, self.n_outputs)
        else:
            self.final_layer = nn.Identity()

    def forward(self, x):
        """Forward pass.

        Parameters
        ----------
        x: torch.Tensor
            Batch of EEG windows of shape (batch_size, n_channels, n_times).
        """

        if x.ndim == 3:
            x = x.unsqueeze(1)

        x1 = self.cnn1(x)
        x1 = x1.flatten(start_dim=1)

        x2 = self.cnn2(x)
        x2 = x2.flatten(start_dim=1)

        x = torch.cat((x1, x2), dim=1)
        x = self.dropout(x)
        temp = x.clone()
        temp = self.fc(temp)
        x = x.unsqueeze(1)
        x = self.bilstm(x)
        x = x.squeeze()
        x = torch.add(x, temp)
        x = self.dropout(x)

        feats = self.features_extractor(x)

        if self.return_feats:
            return feats
        else:
            return self.final_layer(feats)


def _compute_path_width(
    n_times,
    conv1_k,
    conv1_s,
    conv1_p,
    pool1_k,
    pool1_s,
    pool1_p,
    pool2_k,
    pool2_s,
    pool2_p,
):
    """Compute output temporal width of a CNN path.

    Uses the standard formula: out = (in + 2*padding - kernel) // stride + 1
    for conv1, pool1, and pool2. Conv2–conv4 use ``padding='same'`` and
    therefore preserve the temporal dimension.
    """
    w = (n_times + 2 * conv1_p - conv1_k) // conv1_s + 1  # conv1
    w = (w + 2 * pool1_p - pool1_k) // pool1_s + 1  # pool1
    # conv2, conv3, conv4: same padding
    w = (w + 2 * pool2_p - pool2_k) // pool2_s + 1  # pool2
    return w


def _compute_feat_size(
    n_chans,
    n_times,
    *,
    small_n_filters_2,
    small_conv1_k,
    small_conv1_s,
    small_conv1_p,
    small_pool1_k,
    small_pool1_s,
    small_pool1_p,
    small_pool2_k,
    small_pool2_s,
    small_pool2_p,
    large_n_filters_2,
    large_conv1_k,
    large_conv1_s,
    large_conv1_p,
    large_pool1_k,
    large_pool1_s,
    large_pool1_p,
    large_pool2_k,
    large_pool2_s,
    large_pool2_p,
):
    """Analytically compute concatenated CNN feature size."""
    sw = _compute_path_width(
        n_times,
        small_conv1_k,
        small_conv1_s,
        small_conv1_p,
        small_pool1_k,
        small_pool1_s,
        small_pool1_p,
        small_pool2_k,
        small_pool2_s,
        small_pool2_p,
    )
    lw = _compute_path_width(
        n_times,
        large_conv1_k,
        large_conv1_s,
        large_conv1_p,
        large_pool1_k,
        large_pool1_s,
        large_pool1_p,
        large_pool2_k,
        large_pool2_s,
        large_pool2_p,
    )
    return small_n_filters_2 * n_chans * sw + large_n_filters_2 * n_chans * lw


class _SmallCNN(nn.Module):
    r"""
    Smaller filter sizes to learn temporal information.

    Parameters
    ----------
    n_filters_1 : int
        Output channels of the first convolution.
    n_filters_2 : int
        Output channels of the deeper convolutions (conv2–conv4).
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


class _LargeCNN(nn.Module):
    r"""
    Larger filter sizes to learn frequency information.

    Parameters
    ----------
    n_filters_1 : int
        Output channels of the first convolution.
    n_filters_2 : int
        Output channels of the deeper convolutions (conv2–conv4).
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
        first_kernel_size: int = 400,
        first_stride: int = 50,
        first_padding: int = 175,
        pool1_kernel_size: int = 4,
        pool1_stride: int = 4,
        pool1_padding: int = 0,
        deep_kernel_size: int = 6,
        pool2_kernel_size: int = 2,
        pool2_stride: int = 2,
        pool2_padding: int = 1,
        activation: type[nn.Module] = nn.ELU,
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


class _BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, drop_prob=0.5):
        super(_BiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=drop_prob if num_layers > 1 else 0.0,
            bidirectional=True,
        )

    def forward(self, x):
        # set initial hidden and cell states
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)

        # forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        return out
