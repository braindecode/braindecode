# Authors: Bruno Aristimunha <b.aristimunha@gmail.com>
#
# License: BSD (3-clause)


from torch import nn


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
        activation=nn.ELU,
        # Module related parameters
        drop_rate=0.5,
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

        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=8,
            bias=False,
            kernel_size=(1, 32),
            padding=(0, 15),
        )

        self.ln1 = nn.GroupNorm(num_groups=1, num_channels=8)
        self.activation = activation

        self.conv2 = nn.Conv2d(
            in_channels=8,
            out_channels=32,
            kernel_size=(1, 32),
            bias=False,
            padding=(0, 15),
        )
        self.ln2 = nn.GroupNorm(num_groups=1, num_channels=32)
        self.elu1 = nn.ELU()

        self.depthwise_conv = nn.Conv2d(
            in_channels=32,
            out_channels=32 * 2,  # depth_multiplier=2
            kernel_size=(input_height, 1),
            groups=32,  # Depthwise convolution
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
