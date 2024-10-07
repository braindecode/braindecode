"""FB-EEGNet: A fusion neural network across multi-stimulus for SSVEP target detection.
Authors: Yao Huiming <yaohuiming789@gmail.com>
         Bruno Aristimunha <b.aristimunha@gmail.com> (braindecode adaptation)
Yao, H., Liu, K., Deng, X., Tang, X., & Yu, H. (2022). FB-EEGNet: A fusion
neural network across multi-stimulus for SSVEP target detection.
Journal of Neuroscience Methods, 379, 109674.
"""

from __future__ import annotations

import torch
from torch import nn

from braindecode.models import EEGNetv4
from braindecode.models.base import EEGModuleMixin
from braindecode.models.modules import FilterBankLayer


class FBEEGNet(EEGModuleMixin, nn.Module):
    """FBEEGNet from Yao H et al (2022) [fbeegnet]_.

        .. figure:: https://ars.els-cdn.com/content/image/1-s2.0-S0165027022002011-gr3_lrg.jpg
           :align: center
           :alt: FBEEGNet Architecture
        Overview of the Interactive FBEEGNet architecture.


    FBEEGNet is EEGNet combined with Filterbank Layers.


    Parameters
    ----------
    activation : Callable[..., nn.Module], optional
        Activation function, by default nn.ELU.
    drop_prob : float, optional
        Dropout probability, by default 0.5.
    verbose : bool, default=False
        Verbose to control the filtering layer
    filter_parameters : dict, default={}
        Additional parameters for the filter bank layer.


    Notes
    -----
    This implementation is not guaranteed to be correct, has not been checked
    by original authors, only reimplemented from the paper description and
    Tensorflow source code [fbeegnetcode]_.


    References
    ----------
    .. [fbeegnet] Yao, H., Liu, K., Deng, X., Tang, X., & Yu, H. (2022).
        FB-EEGNet: A fusion neural network across multi-stimulus for SSVEP
        target detection. Journal of Neuroscience Methods, 379, 109674.
    .. [fbeegnetcode] Yao, H., Liu, K., Deng, X., Tang, X., & Yu, H. (2022).
        FB-EEGNet: A fusion neural network across multi-stimulus for SSVEP
        target detection.
        https://github.com/YHM404/FB-EEGNet

    """

    def __init__(
        self,
        # Braindecode parameters
        n_chans=None,
        n_outputs=None,
        n_times=None,
        chs_info=None,
        input_window_seconds=None,
        sfreq=None,
        # Model related parameters
        n_filters_time: int = 8,
        depth_multiplier: int = 2,
        kernel_size: int = 64,
        third_kernel_size=(8, 4),
        pool_mode: str = "mean",
        band_filters=[(8, 55), (16, 55), (24, 55)],
        activation: nn.Module = nn.ELU,
        drop_prob: float = 0.5,
        verbose: bool = False,
        filter_parameters: dict = {},
    ):
        super().__init__(
            n_chans=n_chans,
            n_outputs=n_outputs,
            chs_info=chs_info,
            n_times=n_times,
            input_window_seconds=input_window_seconds,
            sfreq=sfreq,
        )
        del n_outputs, n_chans, chs_info, n_times, input_window_seconds, sfreq

        # Parameters
        self.band_filters = band_filters
        self.activation = activation
        self.drop_prob = drop_prob
        self.n_filters_time = n_filters_time
        self.depth_multiplier = depth_multiplier
        self.kernel_size = kernel_size
        self.third_kernel_size = third_kernel_size
        self.verbose = verbose
        self.filter_parameters = filter_parameters
        self.pool_mode = pool_mode

        # Model layers
        self.filter_bank = FilterBankLayer(
            n_chans=self.n_chans,
            sfreq=self.sfreq,
            band_filters=self.band_filters,
            verbose=verbose,
            **filter_parameters,
        )
        # Getting the final number of n_bands
        n_bands = self.filter_bank.n_bands

        self.models = nn.ModuleList(
            [
                EEGNetv4(
                    n_chans=self.n_chans,
                    n_outputs=self.n_outputs,
                    n_times=self.n_times,
                    sfreq=self.sfreq,
                    pool_mode=self.pool_mode,
                    F1=self.n_filters_time,
                    D=self.depth_multiplier,
                    F2=int(self.n_filters_time * self.depth_multiplier),
                    kernel_length=kernel_size,
                    third_kernel_size=self.third_kernel_size,
                    drop_prob=self.drop_prob,
                    activation=self.activation,
                )
                for _ in range(n_bands)
            ]
        )
        self.final_layer = nn.Linear(self.n_outputs * n_bands, self.n_outputs)

        nn.init.normal_(self.final_layer.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.final_layer.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the FB_EEGNet.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, channels, time).

        Returns
        -------
        torch.Tensor
            Output logits of shape (batch_size, num_classes).
        """
        x = self.filter_bank(x)
        features = []
        for idx_band, model in enumerate(self.models):
            features.append(model(x[::, idx_band, ::, ::]))
        features = torch.cat(features, dim=1)
        out = self.final_layer(features)
        return out
