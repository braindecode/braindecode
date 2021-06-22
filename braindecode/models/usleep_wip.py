# Authors: Theo Gnassounou <theo.gnassounou@inria.fr>
#          Omar Chehab <l-emir-omar.chehab@inria.fr>
#
# License: BSD (3-clause)

import numpy as np
import torch
from torch import nn
from torch.nn import init
from torch.nn.functional import elu, upsample
from torch.nn.modules.batchnorm import BatchNorm1d


# TODO: check extra params

class EncoderBlock(nn.Module):
    '''Encoding block for a timeseries x of shape (B, C, T).
    With each new block (depth):
        -- the temporal dimension shrinks (via maxpooling in the time-domain)
        -- the spatial dimension expands (via more conv1d filters in the time-domain).
    both do so at exponential rates.
    '''
    def __init__(self,
                 depth=3,
                 in_channels=2,
                 out_channels=2,
                 kernel_size=9,
                 downsample=2,
                 complexity_factor=2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size  # choose odd nb
        self.complexity_factor = complexity_factor
        self.downsample = downsample
        padding = (kernel_size - 1) // 2   # chosen to preserve dimension
        assert kernel_size % 2 == 1, 'Choose kernel_size to be an odd number.'

        self.encoder_block = nn.Sequential(
                nn.Conv1d(in_channels=in_channels, 
                            out_channels=out_channels, 
                            kernel_size=kernel_size, 
                            padding=padding),
                nn.ELU(),
                nn.BatchNorm1d(num_features=out_channels),
            )

    def forward(self, x):
        x = self.encoder_block(x)
        residual = x
        x = nn.MaxPool1d(kernel_size=self.downsample)(x)
        return x, residual


class DecoderBlock(nn.Module):
    '''Decoding block for a timeseries x of shape (B, C, T).
    With each new block (depth):
        -- the temporal dimension expands (via upsampling in the time-domain)
        -- the spatial dimension is preserved
    the first does so at an exponential rate.
    '''
    def __init__(self,
                 depth=3,
                 in_channels=2,
                 out_channels=2,
                 kernel_size=9,
                 upsample=2,
                 complexity_factor=2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size  # choose odd nb
        self.complexity_factor = complexity_factor
        self.upsample = upsample
        padding = (kernel_size - 1) // 2   # chosen to preserve dimension
        assert kernel_size % 2 == 1, 'Choose kernel_size to be an odd number.'

        self.decoder_block_preskip = nn.Sequential(
                    nn.Upsample(scale_factor=upsample),
                    nn.Conv1d(in_channels=in_channels, 
                              out_channels=out_channels, 
                              kernel_size=kernel_size, 
                              padding=padding),
                    nn.ELU(),
                    nn.BatchNorm1d(num_features=out_channels),
                )
        self.decoder_block_postskip = nn.Sequential(
                    nn.Conv1d(in_channels=2 * out_channels, 
                              out_channels=out_channels, 
                              kernel_size=kernel_size, 
                              stride=1, 
                              padding=padding),  # to preserve dimension (check)
                    nn.ELU(),
                    nn.BatchNorm1d(num_features=out_channels),
                )

    def forward(self, x, residual):
        x = self.decoder_block_preskip(x)
        x = torch.cat([x, residual], axis=1) # (B, 2 * C, T)
        x = self.decoder_block_postskip(x)
        return x



# Small testing script
batch_size, n_channels, n_times = 64, 2, 3000
x = torch.Tensor(batch_size, n_channels, n_times)
encoder = EncoderBlock(in_channels=2, out_channels=4)
decoder = DecoderBlock(in_channels=4, out_channels=2)
z, residual = encoder(x)
x_hat = decoder(z, residual)







class USleep(nn.Module):

    def __init__(self, 
                 n_classes=5,
                 depth=3,
                 dilation=1,
                 dense_classifier_activation="tanh",
                 kernel_size=9,
                 transition_window=1,
                 filters_init=5,
                 complexity_factor=2):
        '''TODO: remove redundant arguments.'''
        super().__init__()

        # Set attributes
        padding = (kernel_size - 1) // 2   # to preserve dimension (check)
    
        # Instantiate encoder : input has shape (B, C, T)
        encoder = []
        filters = filters_init
        for _ in range(depth):
            # update nb of input / output channels
            in_channels = 2 if _ == 0 else out_channels
            out_channels = int(filters * complexity_factor)

            # add encoder block (down)
            encoder += [
                nn.Sequential(
                    nn.Conv1d(in_channels=in_channels, 
                              out_channels=out_channels, 
                              kernel_size=kernel_size, 
                              stride=1, 
                              padding=padding),
                    nn.ELU(),
                    nn.BatchNorm1d(num_features=out_channels),
                )
            ]
            
            # update nb of filters
            filters = int(filters * np.sqrt(2))
        self.encoder = nn.Sequential(*encoder)

        # Instantiate bottom
        in_channels = out_channels
        out_channels = int(filters * complexity_factor)
        self.bottom = nn.Sequential(
                    nn.Conv1d(in_channels=in_channels, 
                              out_channels=out_channels, 
                              kernel_size=kernel_size, 
                              stride=1, 
                              padding=padding),
                    nn.ELU(),
                    nn.BatchNorm1d(num_features=out_channels),
                )

        # Instantiate decoder
        decoder_preskip = []
        decoder_postskip = []

        for _ in range(depth):

            in_channels = out_channels

            # add decoder blocks (up)
            decoder_preskip += [
                nn.Sequential(
                    nn.Upsample(scale_factor=2),
                    nn.Conv1d(in_channels=in_channels, 
                              out_channels=out_channels, 
                              kernel_size=kernel_size, 
                              stride=1, 
                              padding=padding),
                    nn.ELU(),
                    nn.BatchNorm1d(num_features=out_channels),
                )
            ]
            
            # we will concatenate channels via a skip connection, so they multiply by 2
            in_channels *= 2

            # add encoder block (down)
            decoder_postskip += [
                nn.Sequential(
                    nn.Conv1d(in_channels=in_channels, 
                              out_channels=out_channels, 
                              kernel_size=kernel_size, 
                              stride=1, 
                              padding=padding),  # to preserve dimension (check)
                    nn.ELU(),
                    nn.BatchNorm1d(num_features=out_channels),
                )
            ]

        self.decoder_preskip = nn.Sequential(*decoder_preskip)
        self.decoder_postskip = nn.Sequential(*decoder_postskip)


    def forward(self, x):
        '''Input x has shape (B, C, T).'''
        
        # encoder
        residuals = []
        for down in self.encoder:
            x = down(x)
            residuals.append(x)
            x = nn.MaxPool1d(kernel_size=2)(x)

        # decoder
        residuals = residuals[::-1]  # in order of up layers
        for (idx, (up_preskip, up_postskip)) in enumerate(zip(self.decoder_preskip, self.decoder_postskip)):
            x = up_preskip(x)
            x = torch.cat([x, residuals[idx]], axis=1) # (B, 2 * C, T)
            x = up_postskip(x)
        
        return x

    # self.clf = # 


# Small testing script
batch_size, n_channels, n_times = 1024, 2, 3000
x = torch.Tensor(batch_size, n_channels, n_times)
model = USleep()