# Authors: Theo Gnassounou <theo.gnassounou@inria.fr>
#          Omar Chehab <l-emir-omar.chehab@inria.fr>
#
# License: BSD (3-clause)

import numpy as np
import torch
from torch import nn


def _crop_tensors_to_match(x1, x2, axis=-1):
    '''Crops two tensors to their lowest-common-dimension along an axis.'''
    dim_cropped = min(x1.shape[axis], x2.shape[axis])

    x1_cropped = torch.index_select(
        x1, dim=axis,
        index=torch.arange(dim_cropped).to(device=x1.device)
    )
    x2_cropped = torch.index_select(
        x2, dim=axis,
        index=torch.arange(dim_cropped).to(device=x1.device)
    )
    return x1_cropped, x2_cropped


class _EncoderBlock(nn.Module):
    '''Encoding block for a timeseries x of shape (B, C, T).'''
    def __init__(self,
                 in_channels=2,
                 out_channels=2,
                 kernel_size=9,
                 downsample=2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.downsample = downsample
        padding = (kernel_size - 1) // 2   # chosen to preserve dimension

        self.block_prepool = nn.Sequential(
                nn.Conv1d(in_channels=in_channels,
                          out_channels=out_channels,
                          kernel_size=kernel_size,
                          padding=padding),
                nn.ELU(),
                nn.BatchNorm1d(num_features=out_channels),
            )

    def forward(self, x):
        x = self.block_prepool(x)
        residual = x
        if x.shape[-1] % 2:
            x = nn.ConstantPad1d(padding=1, value=0)(x)
        x = nn.MaxPool1d(kernel_size=self.downsample, stride=self.downsample)(x)
        return x, residual


class _DecoderBlock(nn.Module):
    '''Decoding block for a timeseries x of shape (B, C, T).'''
    def __init__(self,
                 in_channels=2,
                 out_channels=2,
                 kernel_size=9,
                 upsample=2,
                 with_skip_connection=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.upsample = upsample
        self.with_skip_connection = with_skip_connection
        padding = (kernel_size - 1) // 2   # chosen to preserve dimension

        self.block_preskip = nn.Sequential(
                    nn.Upsample(scale_factor=upsample),
                    nn.Conv1d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              padding=padding),
                    nn.ELU(),
                    nn.BatchNorm1d(num_features=out_channels),
                )
        self.block_postskip = nn.Sequential(
                    nn.Conv1d(in_channels=(
                            2 * out_channels if with_skip_connection else out_channels),
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              padding=padding),  # to preserve dimension (check)
                    nn.ELU(),
                    nn.BatchNorm1d(num_features=out_channels),
                )

    def forward(self, x, residual):
        x = self.block_preskip(x)
        if self.with_skip_connection:
            x, residual = crop_tensors_to_match(x, residual, axis=-1)  # in case of mismatch
            x = torch.cat([x, residual], axis=1)  # (B, 2 * C, T)
        x = self.block_postskip(x)
        return x


class USleep(nn.Module):
    """Sleep staging architecture from Perslev et al 2021.

    U-Net (autoencoder with skip connections) feature-extractor for sleep staging described in [1]_.

    For the encoder ('down'):
        -- the temporal dimension shrinks (via maxpooling in the time-domain)
        -- the spatial dimension expands (via more conv1d filters in the time-domain)
    For the decoder ('up'):
        -- the temporal dimension expands (via upsampling in the time-domain)
        -- the spatial dimension shrinks (via fewer conv1d filters in the time-domain)
    Both do so at exponential rates.

    Parameters
    ----------
    n_channels : int
        Number of EEG or EOG channels. Set to 2 in [1]_ (1 EEG, 1 EOG).
    sfreq : float
        EEG sampling frequency. Set to 128 in [1]_.
    depth : int
        Number of encoding (resp. decoding) blocks in the U-Net.
        Set to 12 in [1]_ with sfreq=128.
        Here we set it to 10 with sfreq=100.
    time_conv_size_s : float
        Size of filters in temporal convolution layers, in seconds.
        Set to 0.070 in [1]_ (9 samples at sfreq=128).
    max_pool_size_s : float
        Max pooling size, in seconds. Set to 0.016 in [1]_ (2 samples at
        sfreq=128).
    n_time_filters : int
        Number of channels (i.e. of temporal filters) of the output.
        Set to 5 in [1]_.
    complexity_factor : float
        Multiplicative factor for number of channels at each layer of the U-Net.
        Set to sqrt(2) in [1]_.
    input_size_s : float
        Size of the input, in seconds. Set to 30.
    n_classes : int
        Number of classes. Set to 5.
    apply_batch_norm : bool
        If True, apply batch normalization after temporal convolutional
        layers.

    References
    ----------
    .. [1] Perslev, M., Darkner, S., Kempfner, L. et al.
           U-Sleep: resilient high-frequency sleep staging. npj Digit. Med. 4, 72 (2021).
           https://github.com/perslev/U-Time/blob/master/utime/models/usleep.py
    """
    def __init__(self,
                 n_channels=2,  # XXX change n_channels -> in_chans
                 sfreq=100,
                 complexity_factor=np.sqrt(2),
                 with_skip_connection=True,
                 n_classes=5,
                 input_size_s=30,
                 ):
        super().__init__()

        self.n_channels = n_channels

        # Harcoded (otherwise dims can break)
        depth = 10
        time_conv_size = 9  # 0.09s at sfreq = 100 Hz
        max_pool_size = 2   # 0.02s at sfreq = 100 Hz
        n_time_filters = 5

        # Convert between units: seconds to time-points (at sfreq)
        input_size = np.ceil(input_size_s * sfreq).astype(int)

        # Define geometric sequence of channels
        channels = (
            n_time_filters * complexity_factor * np.sqrt(2) ** np.arange(0, depth + 1)
        )  # len = depth + 1
        channels = np.ceil(channels).astype(int).tolist()
        channels = [n_channels] + channels  # len = depth + 2
        self.channels = channels

        # Instantiate encoder
        encoder = []
        for idx in range(depth):
            encoder += [
                EncoderBlock(in_channels=channels[idx],
                             out_channels=channels[idx + 1],
                             kernel_size=time_conv_size,
                             downsample=max_pool_size)
            ]
        self.encoder = nn.Sequential(*encoder)

        # Instantiate bottom (channels increase, temporal dim stays the same)
        self.bottom = nn.Sequential(
                    nn.Conv1d(in_channels=channels[idx + 1],
                              out_channels=channels[idx + 2],
                              kernel_size=time_conv_size,
                              padding=(time_conv_size - 1) // 2),  # preserves dimension
                    nn.ELU(),
                    nn.BatchNorm1d(num_features=channels[idx + 2]),
                )

        # Instantiate decoder
        decoder = []
        channels_reverse = channels[::-1]
        for idx in range(depth):
            decoder += [
                DecoderBlock(in_channels=channels_reverse[idx],
                             out_channels=channels_reverse[idx + 1],
                             kernel_size=time_conv_size,
                             upsample=max_pool_size,
                             with_skip_connection=with_skip_connection)
            ]
        self.decoder = nn.Sequential(*decoder)

        # Instantiate classifier
        # self.clf = nn.Sequential(
        #     nn.Dropout(0.5),
        #     nn.Linear(channels[1] * input_size, n_classes)
        # )

        # The temporal dimension remains unchanged
        # (except through the AvgPooling which collapses it to 1)
        # The spatial dimension is preserved from the end of the UNet, and is mapped to n_classes
        self.clf = nn.Sequential(
            nn.Conv1d(
                in_channels=channels[1],
                out_channels=channels[1],
                kernel_size=1,
                stride=1,
                padding=0,
            ),                         # output is (B, C, 1, S * T)
            nn.Tanh(),
            nn.AvgPool1d(input_size),  # output is (B, C, S)
            nn.Conv1d(
                in_channels=channels[1],
                out_channels=n_classes,
                kernel_size=1,
                stride=1,
                padding=0,
            ),                         # output is (B, n_classes, S)
            nn.ELU(),
            nn.Conv1d(
                in_channels=n_classes,
                out_channels=n_classes,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.Softmax(dim=1),  # output is (B, n_classes, S)
        )

    def forward(self, x):
        '''If input x has shape (B, S, C, T), return y_pred of shape (B, n_classes, S).
        If input x has shape (B, C, T), return y_pred of shape (B, n_classes).
        '''
        # reshape input
        if len(x.shape) == 4:  # input x has shape (B, S, C, T)
            x = x.permute(0, 2, 1, 3)  # (B, C, S, T)
            x = x.flatten(start_dim=2)  # (B, C, S * T)

        # encoder
        residuals = []
        for down in self.encoder:
            x, res = down(x)
            residuals.append(res)

        # bottom
        x = self.bottom(x)

        # decoder
        residuals = residuals[::-1]  # flip order
        for up, res in zip(self.decoder, residuals):
            x = up(x, res)

        # classifier
        y_pred = self.clf(x)        # (B, n_classes, seq_length)

        # y_pred = self.clf(x.flatten(start_dim=1))        # (B, n_classes)
        if y_pred.shape[-1] == 1:  # seq_length of 1
            y_pred = y_pred[:, :, 0]

        return y_pred
