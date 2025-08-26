# Authors: Théo Gnassounou <theo.gnassounou@inria.fr>
#
# License: BSD (3-clause)
import torch
import torch.nn as nn

from braindecode.models.base import EEGModuleMixin


class DeepSleepNet(EEGModuleMixin, nn.Module):
    """DeepSleepNet from Supratak et al. (2017) [Supratak2017]_.

    :bdg-success:`Convolution` :bdg-info:`Recurrent`

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
       First conv uses **filter length ≈ Fs/2** and **stride ≈ Fs/16** to emphasize *timing* of graphoelements.
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
        activation_large: nn.Module = nn.ELU,
        activation_small: nn.Module = nn.ReLU,
        drop_prob: float = 0.5,
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
        self.cnn1 = _SmallCNN(activation=activation_small, drop_prob=drop_prob)
        self.cnn2 = _LargeCNN(activation=activation_large, drop_prob=drop_prob)
        self.dropout = nn.Dropout(0.5)
        self.bilstm = _BiLSTM(input_size=3072, hidden_size=512, num_layers=2)
        self.fc = nn.Sequential(
            nn.Linear(3072, 1024, bias=False), nn.BatchNorm1d(num_features=1024)
        )

        self.features_extractor = nn.Identity()
        self.len_last_layer = 1024
        self.return_feats = return_feats

        # TODO: Add new way to handle return_features == True
        if not return_feats:
            self.final_layer = nn.Linear(1024, self.n_outputs)
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


class _SmallCNN(nn.Module):
    """
    Smaller filter sizes to learn temporal information.

    Parameters
    ----------
    activation: nn.Module, default=nn.ReLU
        Activation function class to apply. Should be a PyTorch activation
        module class like ``nn.ReLU`` or ``nn.ELU``. Default is ``nn.ReLU``.
    drop_prob : float, default=0.5
        The dropout rate for regularization. Values should be between 0 and 1.
    """

    def __init__(self, activation: nn.Module = nn.ReLU, drop_prob: float = 0.5):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=64,
                kernel_size=(1, 50),
                stride=(1, 6),
                padding=(0, 22),
                bias=False,
            ),
            nn.BatchNorm2d(num_features=64),
            activation(),
        )
        self.pool1 = nn.MaxPool2d(kernel_size=(1, 8), stride=(1, 8), padding=(0, 2))
        self.dropout = nn.Dropout(p=drop_prob)
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=(1, 8),
                stride=1,
                padding="same",
                bias=False,
            ),
            nn.BatchNorm2d(num_features=128),
            activation(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=128,
                out_channels=128,
                kernel_size=(1, 8),
                stride=1,
                padding="same",
                bias=False,
            ),
            nn.BatchNorm2d(num_features=128),
            activation(),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=128,
                out_channels=128,
                kernel_size=(1, 8),
                stride=1,
                padding="same",
                bias=False,
            ),
            nn.BatchNorm2d(num_features=128),
            activation(),
        )
        self.pool2 = nn.MaxPool2d(kernel_size=(1, 4), stride=(1, 4), padding=(0, 1))

    def forward(self, x):
        x = self.conv1(x)
        x = self.dropout(self.pool1(x))
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool2(x)
        return x


class _LargeCNN(nn.Module):
    """
    Larger filter sizes to learn frequency information.

    Parameters
    ----------
    activation: nn.Module, default=nn.ELU
        Activation function class to apply. Should be a PyTorch activation
        module class like ``nn.ReLU`` or ``nn.ELU``. Default is ``nn.ELU``.

    """

    def __init__(self, activation: nn.Module = nn.ELU, drop_prob: float = 0.5):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=64,
                kernel_size=(1, 400),
                stride=(1, 50),
                padding=(0, 175),
                bias=False,
            ),
            nn.BatchNorm2d(num_features=64),
            activation(),
        )
        self.pool1 = nn.MaxPool2d(kernel_size=(1, 4), stride=(1, 4))
        self.dropout = nn.Dropout(p=drop_prob)
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=(1, 6),
                stride=1,
                padding="same",
                bias=False,
            ),
            nn.BatchNorm2d(num_features=128),
            activation(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=128,
                out_channels=128,
                kernel_size=(1, 6),
                stride=1,
                padding="same",
                bias=False,
            ),
            nn.BatchNorm2d(num_features=128),
            activation(),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=128,
                out_channels=128,
                kernel_size=(1, 6),
                stride=1,
                padding="same",
                bias=False,
            ),
            nn.BatchNorm2d(num_features=128),
            activation(),
        )
        self.pool2 = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2), padding=(0, 1))

    def forward(self, x):
        x = self.conv1(x)
        x = self.dropout(self.pool1(x))
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool2(x)
        return x


class _BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(_BiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=0.5,
            bidirectional=True,
        )

    def forward(self, x):
        # set initial hidden and cell states
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)

        # forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        return out
