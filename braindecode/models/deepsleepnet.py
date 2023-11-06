# Authors: Théo Gnassounou <theo.gnassounou@inria.fr>
#
# License: BSD (3-clause)
import torch
import torch.nn as nn

from .base import EEGModuleMixin, deprecated_args


class _SmallCNN(nn.Module):  # smaller filter sizes to learn temporal information
    def __init__(self):
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
            nn.ReLU(),
        )
        self.pool1 = nn.MaxPool2d(kernel_size=(1, 8), stride=(1, 8), padding=(0, 2))
        self.dropout = nn.Dropout(p=0.5)
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
            nn.ReLU(),
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
            nn.ReLU(),
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
            nn.ReLU(),
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


class _LargeCNN(nn.Module):  # larger filter sizes to learn frequency information
    def __init__(self):
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
            nn.ReLU(),
        )
        self.pool1 = nn.MaxPool2d(kernel_size=(1, 4), stride=(1, 4))
        self.dropout = nn.Dropout(p=0.5)
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
            nn.ReLU(),
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
            nn.ReLU(),
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
            nn.ReLU(),
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
        h0 = torch.zeros(
            self.num_layers * 2, x.size(0), self.hidden_size
        ).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)

        # forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        return out


class DeepSleepNet(EEGModuleMixin, nn.Module):
    """Sleep staging architecture from Supratak et al 2017.

    Convolutional neural network and bidirectional-Long Short-Term
    for single channels sleep staging described in [Supratak2017]_.

    Parameters
    ----------
    return_feats : bool
        If True, return the features, i.e. the output of the feature extractor
        (before the final linear layer). If False, pass the features through
        the final linear layer.
    n_classes :
        Alias for n_outputs.

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
            n_classes=None,
    ):
        n_outputs, = deprecated_args(
            self,
            ('n_classes', 'n_outputs', n_classes, n_outputs),
        )
        super().__init__(
            n_outputs=n_outputs,
            n_chans=n_chans,
            chs_info=chs_info,
            n_times=n_times,
            input_window_seconds=input_window_seconds,
            sfreq=sfreq,
        )
        del n_outputs, n_chans, chs_info, n_times, input_window_seconds, sfreq
        del n_classes
        self.cnn1 = _SmallCNN()
        self.cnn2 = _LargeCNN()
        self.dropout = nn.Dropout(0.5)
        self.bilstm = _BiLSTM(input_size=3072, hidden_size=512, num_layers=2)
        self.fc = nn.Sequential(nn.Linear(3072, 1024, bias=False),
                                nn.BatchNorm1d(num_features=1024))

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
