"""
EEG Simple Convolutional Neural Network from Yassine El Ouahidi et al. (2023).

Each EEG channels is convoled individually.

"""
# Authors: Yassine El Ouahidi <eloua.yas@gmail.com>
#
# License: BSD-3

import torch

from .modules import Resample
from .base import EEGModuleMixin


class EEGSimpleConv(EEGModuleMixin, torch.nn.Module):
    """EEGSimpleConv from [Yassine2023]_.

    Can you increase the description here Yassine?

    The paper and original code with more details about the methodological
    choices are available at the [Yassine2023]_ and [Yassine2023Code]_.

    Notes
    -----
    The authors recommend using the default parameters for MI decoding.
    Please refer to the original paper and code for more details.

    An intesive ablation study is available in the paper to understand the
    impact of each parameter on the model performance.

    .. versionadded:: 0.9

    Parameters
    ----------
    fm: int
        Number of Feature Maps at the first Convolution, width of the model.
    n_convs: int
        Number of blocks of convolutions (2 convolutions per block), depth of the model.
    resampling: int
        Resampling Frequency.
    kernel_size: int
        Size of the convolutions kernels.

    References
    ----------
    .. [Yassine2023] Yassine El Ouahidi, V. Gripon, B. Pasdeloup, G. Bouallegue
    N. Farrugia, G. Lioi, 2023. A Strong and Simple Deep Learning Baseline for
    BCI Motor Imagery Decoding. Arxiv preprint. arxiv.org/abs/2309.07159
    .. [Yassine2023] Yassine El Ouahidi, V. Gripon, B. Pasdeloup, G. Bouallegue
    N. Farrugia, G. Lioi, 2023. A Strong and Simple Deep Learning Baseline for
    BCI Motor Imagery Decoding. GitHub repository.
    https://github.com/elouayas/EEGSimpleConv.

    """

    def __init__(
        self,
        # Base arguments
        n_outputs=None,
        n_chans=None,
        sfreq=None,
        # Model specific arguments
        fm=128,
        n_convs=2,
        resampling=80,
        kernel_size=8,
        # Other ways to initialize the model
        chs_info=None,
        n_times=None,
        input_window_seconds=None,
    ):
        super().__init__(
            n_outputs=n_outputs,
            n_chans=n_chans,
            chs_info=chs_info,
            n_times=n_times,
            input_window_seconds=input_window_seconds,
            sfreq=sfreq,
        )
        self.rs = (
            Resample(orig_freq=sfreq, new_freq=resampling)
            if sfreq != resampling
            else torch.nn.Identity()
        )

        self.conv = torch.nn.Conv1d(
            n_chans, fm, kernel_size=kernel_size, padding=kernel_size // 2, bias=False
        )
        self.bn = torch.nn.BatchNorm1d(fm)
        self.blocks = []
        newfm = fm
        oldfm = fm
        for i in range(n_convs):
            if i > 0:
                newfm = int(1.414 * newfm)
            self.blocks.append(
                torch.nn.Sequential(
                    (
                        torch.nn.Conv1d(
                            oldfm,
                            newfm,
                            kernel_size=kernel_size,
                            padding=kernel_size // 2,
                            bias=False,
                        )
                    ),
                    (torch.nn.BatchNorm1d(newfm)),
                    (torch.nn.MaxPool1d(2) if i > 0 - 1 else torch.nn.MaxPool1d(1)),
                    (torch.nn.ReLU()),
                    (
                        torch.nn.Conv1d(
                            newfm,
                            newfm,
                            kernel_size=kernel_size,
                            padding=kernel_size // 2,
                            bias=False,
                        )
                    ),
                    (torch.nn.BatchNorm1d(newfm)),
                    (torch.nn.ReLU()),
                )
            )
            oldfm = newfm
        self.blocks = torch.nn.ModuleList(self.blocks)
        self.final_layer = torch.nn.Linear(oldfm, n_outputs)

    def forward(self, x):
        """
        Forward pass of the model.

        Parameters
        ----------
        x: PyTorch Tensor
            Input tensor of shape (batch_size, n_channels, n_times)

        Returns
        -------
        PyTorch Tensor
            Output tensor of shape (batch_size, n_outputs)
        """
        x_rs = self.rs(x.contiguous())
        y = torch.relu(self.bn(self.conv(x_rs)))
        for seq in self.blocks:
            y = seq(y)
        y = y.mean(dim=2)
        return self.final_layer(y)
