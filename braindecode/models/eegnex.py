# Authors: Bruno Aristimunha <b.aristimunha@gmail.com>
#
# License: BSD (3-clause)
from __future__ import annotations

import math

import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

from braindecode.models.base import EEGModuleMixin
from braindecode.modules import Conv2dWithConstraint, LinearWithConstraint


class EEGNeX(EEGModuleMixin, nn.Module):
    """EEGNeX model from Chen et al. (2024) [eegnex]_.

    .. figure:: https://braindecode.org/dev/_static/model/eegnex.jpg
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
        kernel_block_1_2: int = 64,
        kernel_block_4: int = 16,
        dilation_block_4: int = 2,
        avg_pool_block4: int = 4,
        kernel_block_5: int = 16,
        dilation_block_5: int = 4,
        avg_pool_block5: int = 8,
        max_norm_conv: float = 1.0,
        max_norm_linear: float = 0.25,
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
        self.kernel_block_1_2 = (1, kernel_block_1_2)
        self.kernel_block_4 = (1, kernel_block_4)
        self.dilation_block_4 = (1, dilation_block_4)
        self.avg_pool_block4 = (1, avg_pool_block4)
        self.kernel_block_5 = (1, kernel_block_5)
        self.dilation_block_5 = (1, dilation_block_5)
        self.avg_pool_block5 = (1, avg_pool_block5)

        # final layers output
        self.in_features = self._calculate_output_length()

        # Following paper nomenclature
        self.block_1 = nn.Sequential(
            Rearrange("batch ch time -> batch 1 ch time"),
            nn.Conv2d(
                in_channels=1,
                out_channels=self.filter_1,
                kernel_size=self.kernel_block_1_2,
                padding="same",
                bias=False,
            ),
            nn.BatchNorm2d(num_features=self.filter_1),
        )

        self.block_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=self.filter_1,
                out_channels=self.filter_2,
                kernel_size=self.kernel_block_1_2,
                padding="same",
                bias=False,
            ),
            nn.BatchNorm2d(num_features=self.filter_2),
        )

        self.block_3 = nn.Sequential(
            Conv2dWithConstraint(
                in_channels=self.filter_2,
                out_channels=self.filter_3,
                max_norm=max_norm_conv,
                kernel_size=(self.n_chans, 1),
                groups=self.filter_2,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=self.filter_3),
            self.activation(),
            nn.AvgPool2d(
                kernel_size=self.avg_pool_block4,
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
                padding="same",
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
                padding="same",
                bias=False,
            ),
            nn.BatchNorm2d(num_features=self.filter_1),
            self.activation(),
            nn.AvgPool2d(
                kernel_size=self.avg_pool_block5,
                padding=(0, 1),
            ),
            nn.Dropout(p=self.drop_prob),
            nn.Flatten(),
        )

        self.final_layer = LinearWithConstraint(
            in_features=self.in_features,
            out_features=self.n_outputs,
            max_norm=max_norm_linear,
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

    def _calculate_output_length(self) -> int:
        # Pooling kernel sizes for the time dimension
        p4 = self.avg_pool_block4[1]
        p5 = self.avg_pool_block5[1]

        # Padding for the time dimension (assumed from padding=(0, 1))
        pad4 = 1
        pad5 = 1

        # Stride is assumed to be equal to kernel size (p4 and p5)

        # Calculate time dimension after block 3 pooling
        # Formula: floor((L_in + 2*padding - kernel_size) / stride) + 1
        T3 = math.floor((self.n_times + 2 * pad4 - p4) / p4) + 1

        # Calculate time dimension after block 5 pooling
        T5 = math.floor((T3 + 2 * pad5 - p5) / p5) + 1

        # Calculate final flattened features (channels * 1 * time_dim)
        # The spatial dimension is reduced to 1 after block 3's depthwise conv.
        final_in_features = (
            self.filter_1 * T5
        )  # filter_1 is the number of channels before flatten
        return final_in_features
