# Authors: Bruno Aristimunha <b.aristimunha@gmail.com>
#
# License: BSD (3-clause)


from torch import nn
from einops.layers.torch import Rearrange


from .base import EEGModuleMixin


class EEGNeX(EEGModuleMixin, nn.Sequential):
    """EEGNeX model from XXXX.

    Parameters
    ----------

    Notes
    -----
    This implementation is not guaranteed to be correct, has not been checked
    by original authors, only reimplemented from the paper description and
    source code in tensorflow [EEGNexCode]_.

    References
    ----------
    .. [EEGNeX] Chen, X., Teng, X., Chen, H., Pan, Y., & Geyer, P. (2024).
       Toward reliable signals decoding for electroencephalogram: A benchmark
       study to EEGNeX. Biomedical Signal Processing and Control, 87, 105475.
    .. [EEGNexCode] Chen, X., Teng, X., Chen, H., Pan, Y., & Geyer, P. (2024).
       Toward reliable signals decoding for electroencephalogram: A benchmark
       study to EEGNeX. https://github.com/chenxiachan/EEGNeX
    """

    def __init__(
        self,
        # Signal related parameters
        n_chans=None,
        n_outputs=None,
        n_times=None,
        chs_info=None,
        input_window_seconds=None,
        sfreq=None,
        # Module related parameters
        activation=nn.ELU,
        drop_rate=0.5,
        conv_temp_out_channels=8,
        kernel_length=32,
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
        input_height = 12
        num_features = 1
        self.dimshuffle = Rearrange("batch ch t 1 -> batch 1 ch t")

        self.conv_temporal = nn.Conv2d(
            in_channels=1,
            out_channels=conv_temp_out_channels,
            bias=False,
            kernel_size=(1, kernel_length),
            padding=(0, kernel_length // 2 - 1),
        )

        self.ln1 = nn.GroupNorm(num_groups=1, num_channels=conv_temp_out_channels)
        self.activation = activation

        self.conv_spatial = nn.Conv2d(
            in_channels=conv_temp_out_channels,
            out_channels=kernel_length,
            kernel_size=(1, kernel_length),
            bias=False,
            padding=(0, kernel_length // 2 - 1),
        )
        self.ln2 = nn.GroupNorm(num_groups=1, num_channels=kernel_length)
        self.elu1 = nn.ELU()

        self.depthwise_conv = nn.Conv2d(
            in_channels=kernel_length,
            out_channels=kernel_length * 2,  # depth_multiplier=2
            kernel_size=(input_height, 1),
            groups=kernel_length,  # Depthwise convolution
            bias=False,
            padding=0,
        )

        self.ln3 = nn.GroupNorm(num_groups=1, num_channels=64)
        self.elu3 = nn.ELU()
        self.avgpool = nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4))
        self.dropout1 = nn.Dropout(drop_rate)

        # Second Convolutional Block
        self.conv3 = nn.Conv2d(
            in_channels=64,
            out_channels=32,
            kernel_size=(1, 16),
            bias=False,
            padding=(0, 15),
            dilation=(1, 2),
        )
        self.ln4 = nn.GroupNorm(num_groups=1, num_channels=32)
        self.elu4 = nn.ELU()

        self.conv4 = nn.Conv2d(
            in_channels=32,
            out_channels=8,
            kernel_size=(1, 16),
            bias=False,
            padding=(0, 30),
            dilation=(1, 4),
        )
        self.ln5 = nn.GroupNorm(num_groups=1, num_channels=8)
        self.elu5 = nn.ELU()
        self.dropout2 = nn.Dropout(self.drop_rate)

        self.fc = nn.Linear(num_features, self.n_classes)
