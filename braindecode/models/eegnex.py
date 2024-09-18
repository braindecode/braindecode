import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

from braindecode.models.base import EEGModuleMixin


class EEGNeX(EEGModuleMixin, nn.Module):
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
        # Model parameters
        activation=nn.ELU,
        depth_multiplier=2,
        filter_1=8,
        filter_2=32,
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

        self.filter_1 = filter_1
        self.filter_2 = filter_2
        self.drop_rate = drop_rate
        self.activation = activation

        kernel_size_block_4 = (1, 16)
        dilation_block_4 = (1, 2)
        padding_block_4 = self.calc_padding(kernel_size_block_4, dilation_block_4)

        kernel_size_block_5 = (1, 16)
        dilation_block_5 = (1, 4)
        padding_block_5 = self.calc_padding(kernel_size_block_5, dilation_block_5)

        # from the table 3 from the paper
        self.in_features = self.filter_1 * (self.n_times // self.filter_2)

        self.dimshuffle = Rearrange("batch ch t -> batch 1 ch t")

        # Following paper nomenclature
        self.block_1 = nn.Sequential(
            self.dimshuffle,
            nn.Conv2d(
                in_channels=1,
                out_channels=self.filter_1,
                kernel_size=(1, 64),
                padding="same",
                bias=False,
            ),
            nn.BatchNorm2d(num_features=self.filter_1),
            self.activation(),
        )

        self.block_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=self.filter_1,
                out_channels=self.filter_2,
                kernel_size=(1, 64),
                padding="same",
                bias=False,
            ),
            nn.BatchNorm2d(num_features=self.filter_2),
        )

        self.block_3 = nn.Sequential(
            nn.Conv2d(
                in_channels=self.filter_2,
                out_channels=self.filter_2 * depth_multiplier,
                kernel_size=(self.n_chans, 1),
                groups=self.filter_2,
                padding=(0, 0),
                bias=False,
            ),
            nn.BatchNorm2d(num_features=64),
            nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4), padding=(0, 1)),
            nn.Dropout(p=self.drop_rate),
        )

        self.block_4 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=self.filter_2,
                kernel_size=kernel_size_block_4,
                dilation=dilation_block_4,
                padding=padding_block_4,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=32),
        )

        self.block_5 = nn.Sequential(
            nn.Conv2d(
                in_channels=self.filter_2,
                out_channels=self.filter_1,
                kernel_size=kernel_size_block_5,
                dilation=dilation_block_5,
                padding=padding_block_5,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=self.filter_1),
            nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8), padding=(0, 1)),
            nn.Dropout(p=self.drop_rate),
            nn.Flatten(),
        )

        self.final_layer = nn.Linear(
            in_features=self.in_features, out_features=self.n_outputs
        )

    def forward(self, x):
        # x shape: (batch_size, n_features, n_timesteps)
        print("initial:", x.shape)
        x = self.block_1(x)
        print("block 1:", x.shape)
        x = self.block_2(x)
        print("block 2", x.shape)
        x = self.block_3(x)
        print("block 3", x.shape)
        x = self.block_4(x)
        print("block 4", x.shape)
        x = self.block_5(x)
        print("block 5", x.shape)
        x = self.final_layer(x)

        return x

    @staticmethod
    def calc_padding(kernel_size: tuple, dilation: tuple):
        """
            Util function to calculate padding, because the native version of
            Pytorch doesn't handle well the dilatation using the same

        Parameters
        ----------
        kernel_size: list
            tuple with kernel_size in block 4 and block 5.
        dilation: list
            tuple with kernel_size in block 4 and block 5.

        Returns
        -------
        padding_height, padding_width
        """
        # Calculate padding
        padding_height = ((kernel_size[0] - 1) * dilation[0]) // 2
        padding_width = ((kernel_size[1] - 1) * dilation[1]) // 2
        return padding_height, padding_width


# Test the implementation
if __name__ == "__main__":
    n_features = 22
    n_timesteps = 1000
    n_outputs = 2  # Example number of output classes

    # Create a test input tensor
    X = torch.zeros(1, n_features, n_timesteps)

    # Initialize the model
    model = EEGNeX(n_chans=n_features, n_times=n_timesteps, n_outputs=n_outputs)

    # Get the model output
    out = model(X)

    # print("Model output shape:", out.shape)
    # print("Model output:", out)
