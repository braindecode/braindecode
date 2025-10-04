# Authors: Divyesh Narayanan <divyesh.narayanan@gmail.com>
#
# License: BSD (3-clause)

import torch
from torch import nn

from braindecode.models.base import EEGModuleMixin


class SleepStagerBlanco2020(EEGModuleMixin, nn.Module):
    """Sleep staging architecture from Blanco et al. (2020) from [Blanco2020]_

    :bdg-success:`Convolution`

    .. figure:: https://media.springernature.com/full/springer-static/image/art%3A10.1007%2Fs00500-019-04174-1/MediaObjects/500_2019_4174_Fig2_HTML.png
        :align: center
        :alt: SleepStagerBlanco2020 Architecture

    Convolutional neural network for sleep staging described in [Blanco2020]_.
    A series of seven convolutional layers with kernel sizes running down from 7 to 3,
    in an attempt to extract more general features at the beginning, while more specific
    and complex features were extracted in the final stages.

    Parameters
    ----------
    n_conv_chans : int
        Number of convolutional channels. Set to 20 in [Blanco2020]_.
    n_groups : int
        Number of groups for the convolution. Set to 2 in [Blanco2020]_ for 2 Channel EEG.
        controls the connections between inputs and outputs. n_channels and n_conv_chans must be
        divisible by n_groups.
    drop_prob : float
        Dropout rate before the output dense layer.
    apply_batch_norm : bool
        If True, apply batch normalization after both temporal convolutional
        layers.
    return_feats : bool
        If True, return the features, i.e. the output of the feature extractor
        (before the final linear layer). If False, pass the features through
        the final linear layer.
    n_channels : int
        Alias for `n_chans`.
    n_classes : int
        Alias for `n_outputs`.
    input_size_s : float
        Alias for `input_window_seconds`.
    activation: nn.Module, default=nn.ReLU
        Activation function class to apply. Should be a PyTorch activation
        module class like ``nn.ReLU`` or ``nn.ELU``. Default is ``nn.ReLU``.

    References
    ----------
    .. [Blanco2020] Fernandez-Blanco, E., Rivero, D. & Pazos, A. Convolutional
        neural networks for sleep stage scoring on a two-channel EEG signal.
        Soft Comput 24, 4067–4079 (2020). https://doi.org/10.1007/s00500-019-04174-1
    """

    def __init__(
        self,
        n_chans=None,
        sfreq=None,
        n_conv_chans=20,
        input_window_seconds=None,
        n_outputs=5,
        n_groups=2,
        max_pool_size=2,
        drop_prob=0.5,
        apply_batch_norm=False,
        return_feats=False,
        activation: nn.Module = nn.ReLU,
        chs_info=None,
        n_times=None,
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

        self.mapping = {
            "fc.1.weight": "final_layer.1.weight",
            "fc.1.bias": "final_layer.1.bias",
        }

        batch_norm = nn.BatchNorm2d if apply_batch_norm else nn.Identity

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(self.n_chans, n_conv_chans, (1, 7), groups=n_groups, padding=0),
            batch_norm(n_conv_chans),
            activation(),
            nn.MaxPool2d((1, max_pool_size)),
            nn.Conv2d(
                n_conv_chans, n_conv_chans, (1, 7), groups=n_conv_chans, padding=0
            ),
            batch_norm(n_conv_chans),
            activation(),
            nn.MaxPool2d((1, max_pool_size)),
            nn.Conv2d(
                n_conv_chans, n_conv_chans, (1, 5), groups=n_conv_chans, padding=0
            ),
            batch_norm(n_conv_chans),
            activation(),
            nn.MaxPool2d((1, max_pool_size)),
            nn.Conv2d(
                n_conv_chans, n_conv_chans, (1, 5), groups=n_conv_chans, padding=0
            ),
            batch_norm(n_conv_chans),
            activation(),
            nn.MaxPool2d((1, max_pool_size)),
            nn.Conv2d(
                n_conv_chans, n_conv_chans, (1, 5), groups=n_conv_chans, padding=0
            ),
            batch_norm(n_conv_chans),
            activation(),
            nn.MaxPool2d((1, max_pool_size)),
            nn.Conv2d(
                n_conv_chans, n_conv_chans, (1, 3), groups=n_conv_chans, padding=0
            ),
            batch_norm(n_conv_chans),
            activation(),
            nn.MaxPool2d((1, max_pool_size)),
            nn.Conv2d(
                n_conv_chans, n_conv_chans, (1, 3), groups=n_conv_chans, padding=0
            ),
            batch_norm(n_conv_chans),
            activation(),
            nn.MaxPool2d((1, max_pool_size)),
        )

        self.len_last_layer = self._len_last_layer(self.n_chans, self.n_times)
        self.return_feats = return_feats

        # TODO: Add new way to handle return_features == True
        if not return_feats:
            self.final_layer = nn.Sequential(
                nn.Dropout(drop_prob),
                nn.Linear(self.len_last_layer, self.n_outputs),
                nn.Identity(),
            )

    def _len_last_layer(self, n_channels, input_size):
        self.feature_extractor.eval()
        with torch.no_grad():
            out = self.feature_extractor(
                torch.Tensor(1, n_channels, 1, input_size)
            )  # batch_size,n_channels,height,width
        self.feature_extractor.train()
        return len(out.flatten())

    def forward(self, x):
        """Forward pass.
        Parameters
        ----------
        x: torch.Tensor
            Batch of EEG windows of shape (batch_size, n_channels, n_times).
        """
        if x.ndim == 3:
            x = x.unsqueeze(2)

        feats = self.feature_extractor(x).flatten(start_dim=1)
        if self.return_feats:
            return feats
        else:
            return self.final_layer(feats)
