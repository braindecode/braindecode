# Authors: Theo Gnassounou <theo.gnassounou@inria.fr>
#          Omar Chehab <l-emir-omar.chehab@inria.fr>
#
# License: BSD (3-clause)

# TODO: add crop function, add classifier

import numpy as np
import torch
from torch import nn
from torch.nn import init
from torch.nn.functional import elu, upsample
from torch.nn.modules.batchnorm import BatchNorm1d


class EncoderBlock(nn.Module):
    '''Encoding block for a timeseries x of shape (B, C, T).'''
    def __init__(self,
                 in_channels=2,
                 out_channels=2,
                 kernel_size=9,
                 downsample=2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size  # choose odd nb
        self.downsample = downsample
        padding = (kernel_size - 1) // 2   # chosen to preserve dimension
        # assert kernel_size % 2 == 1, 'Choose kernel_size to be an odd number.'

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
        x = nn.MaxPool1d(kernel_size=self.downsample, stride=self.downsample)(x)
        return x, residual


class DecoderBlock(nn.Module):
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
        self.kernel_size = kernel_size  # choose odd nb
        self.upsample = upsample
        self.with_skip_connection = with_skip_connection
        padding = (kernel_size - 1) // 2   # chosen to preserve dimension
        # assert kernel_size % 2 == 1, 'Choose kernel_size to be an odd number.'

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
                    nn.Conv1d(in_channels=2 * out_channels if with_skip_connection else out_channels, 
                              out_channels=out_channels, 
                              kernel_size=kernel_size, 
                              padding=padding),  # to preserve dimension (check)
                    nn.ELU(),
                    nn.BatchNorm1d(num_features=out_channels),
                )

    def forward(self, x, residual):
        x = self.block_preskip(x)
        if self.with_skip_connection:
            x = torch.cat([x, residual], axis=1) # (B, 2 * C, T)
        x = self.block_postskip(x)
        return x


class USleep(nn.Module):
    """
    Sleep staging architecture from [1]_.

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
        Set to 12 in [1]_.
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
    n_classes : int
        Number of classes.
    apply_batch_norm : bool
        If True, apply batch normalization after temporal convolutional
        layers.

    References
    ----------
    .. [1] Perslev, M., Darkner, S., Kempfner, L. et al. 
           U-Sleep: resilient high-frequency sleep staging. npj Digit. Med. 4, 72 (2021). 
    """
    def __init__(self, 
                 n_channels=2,
                 sfreq=128,
                 depth=3,  # default should be 12
                 time_conv_size_s=0.070,
                 max_pool_size_s=0.016,
                 n_time_filters=5,
                 complexity_factor=np.sqrt(2),
                 with_skip_connection=True,
                 apply_batch_norm=True
                 ):
        super().__init__()

        self.n_channels = n_channels

        # Convert between units: seconds to time-points (at sfreq)
        time_conv_size = np.ceil(time_conv_size_s * sfreq).astype(int)
        max_pool_size = np.ceil(max_pool_size_s * sfreq).astype(int)

        # Define geometric sequence of channels
        channels = n_time_filters * complexity_factor * np.sqrt(2) ** np.arange(0, depth + 1)  # len = depth + 1
        channels = channels.astype(int).tolist()
        channels = [n_channels] + channels  # len = depth + 2

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

        # Instantiate bottom (channels increase, time dim stays the same)
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


    def forward(self, x):
        '''Input x has shape (B, C, T).'''
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
        
        return x

    # self.clf = # 


# Example: U-Net

batch_size, n_channels, n_times = 64, 2, 3000
x = torch.Tensor(batch_size, n_channels, n_times)
model = USleep(depth=1)
y = model(x)



# Example: mirror Encoder / Decoder pair (understand dims)

batch_size, n_channels, n_times = 64, 2, 3000
encoder = EncoderBlock(in_channels=2, out_channels=4, downsample=2)
decoder = DecoderBlock(in_channels=8, out_channels=4, upsample=2)

x = torch.Tensor(batch_size, n_channels, n_times) 
# print(x.shape)         # (64, 2, 3000)
z, residual = encoder(x)
# print(z.shape)         # (64, 4, 1500)
# print(residual.shape)  # (64, 4, 3000)
z_new = torch.cat([z, z], axis=1)
# print(z_new.shape)     # (64, 8, 1500)
x_hat = decoder(z_new, residual)  