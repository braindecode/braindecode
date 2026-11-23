# Authors: Divyesh Narayanan <divyesh.narayanan@gmail.com>
#
# License: BSD (3-clause)

import math

from torch import nn

from braindecode.models.base import EEGModuleMixin, deprecated_init_flags


@deprecated_init_flags(
    identity_head=("return_feats",),
    return_feats="features",
)
class SleepStagerBlanco2020(EEGModuleMixin, nn.Module):
    r"""Sleep staging architecture from Blanco et al (2020) from [Blanco2020]_

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
    max_pool_size : int, default=2
        Kernel and stride of the max pooling layers.
    n_groups : int
        Number of groups for the convolution. Set to 2 in [Blanco2020]_ for 2 Channel EEG.
        Controls the connections between inputs and outputs. ``n_conv_chans``
        must be divisible by ``n_groups``; if ``n_chans`` is not, the first
        convolution uses ``gcd(n_chans, n_groups)`` groups.
    drop_prob : float
        Dropout rate before the output dense layer.
    apply_batch_norm : bool
        If True, apply batch normalization after both temporal convolutional
        layers.
    return_feats : bool
        Deprecated, removed in 2.3: call ``model(x, return_features=True)`` instead.
        If True, return the features, i.e. the output of the feature extractor
        (before the final linear layer). If False, pass the features through
        the final linear layer.
    activation : nn.Module, default=nn.ReLU
        Activation function class to apply. Should be a PyTorch activation
        module class like ``nn.ReLU`` or ``nn.ELU``. Default is ``nn.ReLU``.

    References
    ----------
    .. [Blanco2020] Fernandez-Blanco, E., Rivero, D. & Pazos, A. Convolutional
        neural networks for sleep stage scoring on a two-channel EEG signal.
        Soft Comput 24, 4067–4079 (2020). https://doi.org/10.1007/s00500-019-04174-1
    """

    _time_axis = "fixed"  # fixed: final_layer.1.weight
    _channel_axis = "fixed"  # fixed: feature_extractor.0.weight

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
        activation: type[nn.Module] = nn.ReLU,
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
        n_groups = math.gcd(self.n_chans, n_groups)

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

        self.len_last_layer = self._len_last_layer(
            n_conv_chans, self.n_times, max_pool_size
        )

        self.final_layer = nn.Sequential(
            nn.Dropout(drop_prob),
            nn.Linear(self.len_last_layer, self.n_outputs),
            nn.Identity(),
        )

    def _len_last_layer(
        self, n_conv_chans: int, input_size: int, max_pool_size: int
    ) -> int:
        n_time = input_size
        for kernel_size in (7, 7, 5, 5, 5, 3, 3):
            n_time = (n_time - kernel_size + 1) // max_pool_size
            if n_time < 1:
                raise ValueError(
                    "The convolutions and poolings leave no time sample: too small "
                    f"`n_times`/`input_window_seconds` for the input {self.input_shape}."
                )
        return n_conv_chans * n_time

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
        return self.final_layer(feats)
