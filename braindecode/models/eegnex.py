# Authors: Bruno Aristimunha <b.aristimunha@gmail.com>
#
# License: BSD (3-clause)
from __future__ import annotations

import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

from braindecode.models.base import EEGModuleMixin


class EEGNeX(EEGModuleMixin, nn.Module):
    """EEGNeX model from Chen et al. (2024) [eegnex]_.

    .. figure:: https://braindecode.org/dev/_images/model/eegnex.jpg
        :align: center
        :alt: EEGNeX Architecture

    Parameters
    ----------
    activation : nn.Module, optional
        Activation function to use. Default is `nn.ELU`.
    depth_multiplier : int, optional
        Depth multiplier for the depthwise convolution. Default is 2.
    filter_1 : int, optional
        Number of filters in the first convolutional layer. Default is 8.
    filter_2 : int, optional
        Number of filters in the second convolutional layer. Default is 32.
    drop_prob: float, optional
        Dropout rate. Default is 0.5.
    kernel_block_4 : tuple[int, int], optional
        Kernel size for block 4. Default is (1, 16).
    dilation_block_4 : tuple[int, int], optional
        Dilation rate for block 4. Default is (1, 2).
    avg_pool_block4 : tuple[int, int], optional
        Pooling size for block 4. Default is (1, 4).
    kernel_block_5 : tuple[int, int], optional
        Kernel size for block 5. Default is (1, 16).
    dilation_block_5 : tuple[int, int], optional
        Dilation rate for block 5. Default is (1, 4).
    avg_pool_block5 : tuple[int, int], optional
        Pooling size for block 5. Default is (1, 8).

    Notes
    -----
    This implementation is not guaranteed to be correct, has not been checked
    by original authors, only reimplemented from the paper description and
    source code in tensorflow [EEGNexCode]_.

    References
    ----------
    .. [eegnex] Chen, X., Teng, X., Chen, H., Pan, Y., & Geyer, P. (2024).
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
        activation: nn.Module = nn.ELU,
        depth_multiplier: int = 2,
        filter_1: int = 8,
        filter_2: int = 32,
        drop_prob: float = 0.5,
        kernel_block_4: tuple[int, int] = (1, 16),
        dilation_block_4: tuple[int, int] = (1, 2),
        avg_pool_block4: tuple[int, int] = (1, 4),
        kernel_block_5: tuple[int, int] = (1, 16),
        dilation_block_5: tuple[int, int] = (1, 4),
        avg_pool_block5: tuple[int, int] = (1, 8),
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

        self.depth_multiplier = depth_multiplier
        self.filter_1 = filter_1
        self.filter_2 = filter_2
        self.filter_3 = self.filter_2 * self.depth_multiplier
        self.drop_prob = drop_prob
        self.activation = activation

        self.kernel_block_4 = kernel_block_4
        self.dilation_block_4 = dilation_block_4
        self.padding_block_4 = self.calc_padding(
            self.kernel_block_4, self.dilation_block_4
        )
        self.avg_pool_block4 = avg_pool_block4

        self.kernel_block_5 = kernel_block_5
        self.dilation_block_5 = dilation_block_5
        self.padding_block_5 = self.calc_padding(
            self.kernel_block_5, self.dilation_block_5
        )
        self.avg_pool_block5 = avg_pool_block5

        # final layers output
        self.in_features = self.filter_1 * (self.n_times // self.filter_2)

        # Following paper nomenclature
        self.block_1 = nn.Sequential(
            Rearrange("batch ch time -> batch 1 ch time"),
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
                out_channels=self.filter_3,
                kernel_size=(self.n_chans, 1),
                groups=self.filter_2,
                padding=(0, 0),
                bias=False,
            ),
            nn.BatchNorm2d(num_features=self.filter_3),
            nn.AvgPool2d(
                kernel_size=self.avg_pool_block4,
                stride=self.avg_pool_block4,
                padding=(0, 1),
            ),
            nn.Dropout(p=self.drop_prob),
        )

        self.block_4 = nn.Sequential(
            nn.Conv2d(
                in_channels=self.filter_3,
                out_channels=self.filter_2,
                kernel_size=self.kernel_block_4,
                dilation=self.dilation_block_4,
                padding=self.padding_block_4,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=self.filter_2),
        )

        self.block_5 = nn.Sequential(
            nn.Conv2d(
                in_channels=self.filter_2,
                out_channels=self.filter_1,
                kernel_size=self.kernel_block_5,
                dilation=self.dilation_block_5,
                padding=self.padding_block_5,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=self.filter_1),
            nn.AvgPool2d(
                kernel_size=self.avg_pool_block5,
                stride=self.avg_pool_block5,
                padding=(0, 1),
            ),
            nn.Dropout(p=self.drop_prob),
            nn.Flatten(),
        )

        self.final_layer = nn.Linear(
            in_features=self.in_features, out_features=self.n_outputs
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the EEGNeX model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, n_chans, n_times).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, n_outputs).
        """
        # x shape: (batch_size, n_chans, n_times)
        x = self.block_1(x)
        # (batch_size, n_filter, n_chans, n_times)
        x = self.block_2(x)
        # (batch_size, n_filter*4, n_chans, n_times)
        x = self.block_3(x)
        # (batch_size, 1, n_filter*8, n_times//4)
        x = self.block_4(x)
        # (batch_size, 1, n_filter*8, n_times//4)
        x = self.block_5(x)
        # (batch_size, n_filter*(n_times//32))
        x = self.final_layer(x)

        return x

    @staticmethod
    def calc_padding(
        kernel_size: tuple[int, int], dilation: tuple[int, int]
    ) -> tuple[int, int]:
        """
        Calculate padding size for 'same' convolution with dilation.

        Parameters
        ----------
        kernel_size : tuple
            Tuple containing the kernel size (height, width).
        dilation : tuple
            Tuple containing the dilation rate (height, width).

        Returns
        -------
        tuple
            Padding sizes (padding_height, padding_width).
        """
        # Calculate padding
        padding_height = ((kernel_size[0] - 1) * dilation[0]) // 2
        padding_width = ((kernel_size[1] - 1) * dilation[1]) // 2
        return padding_height, padding_width
