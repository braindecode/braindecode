"""
EEG-SimpleConv is a 1D Convolutional Neural Network from Yassine El Ouahidi et al. (2023).
Originally designed for Motor Imagery decoding, from EEG signals.
The model is composed of a first 1D convolutional layer followed by a series of blocks of two 1D convolutional layers.

Each block is followed by a max pooling layer and a ReLU activation function.

"""
# Authors: Yassine El Ouahidi <eloua.yas@gmail.com>
#
# License: BSD-3

import torch

from .modules import Resample
from .base import EEGModuleMixin


class EEGSimpleConv(EEGModuleMixin, torch.nn.Module):
    """EEGSimpleConv from [Yassine2023]_.

    EEGSimpleConv is a 1D Convolutional Neural Network originally designed
    for decoding motor imagery from EEG signals. The model aims to have a
    very simple and straightforward architecture that allows a low latency,
    while still achieving very competitive performance.
    
    EEG-SimpleConv starts with a 1D convolutional layer, where each EEG channel
    enters a separate 1D convolutional channel. This is followed by a series of
    blocks of two 1D convolutional layers. Between the two convolutional layers
    of each block is a max pooling layer, which downsamples the data by a factor
    of 2. Each convolution is followed by a batch normalisation layer and a ReLU
    activation function. Finally, a global average pooling (in the time domain)
    is performed to obtain a single value per feature map, which is then fed
    into a linear layer to obtain the final classification prediction output.
    

    The paper and original code with more details about the methodological
    choices are available at the [Yassine2023]_ and [Yassine2023Code]_.
    
    The input shape should be three-dimensional matrix representing the EEG
    signals.

    `(batch_size, n_channels, n_timesteps)`.

    Notes
    -----
    The authors recommend using the default parameters for MI decoding.
    Please refer to the original paper and code for more details.
    
    Recommended range for the choice of the hyperparameters, regarding the
    evaluation paradigm. 
    
    |    Parameter    | Within-Subject | Cross-Subject |
    |-----------------|----------------|---------------|
    | fm              | [64-144]       |   [64-144]    |
    | n_convs         |    1           |   [2-4]       |
    | resampling_freq | [70-100]       |   [50-80]     |
    | kernel_size     | [12-17]        |   [5-8]       |


    An intensive ablation study is included in the paper to understand the
    of each parameter on the model performance.

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
    .. [Yassine2023Code] Yassine El Ouahidi, V. Gripon, B. Pasdeloup, G. Bouallegue
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
        resampling_freq=80,
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
        del n_outputs, n_chans, chs_info, n_times, sfreq, input_window_seconds

        self.rs = (
            Resample(orig_freq=self.sfreq, new_freq=resampling_freq)
            if self.sfreq != resampling_freq
            else torch.nn.Identity()
        )

        self.conv = torch.nn.Conv1d(
            self.n_chans, fm, kernel_size=kernel_size,
            padding=kernel_size // 2, bias=False
        )
        self.bn = torch.nn.BatchNorm1d(fm)
        self.blocks = []
        newfm = fm
        oldfm = fm
        for i in range(n_convs):
            if i > 0:
                newfm = int(1.414 * newfm) # 1.414 = sqrt(2) in order to maitain the flops constant with the deacreasing depth
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
        self.final_layer = torch.nn.Linear(oldfm, self.n_outputs)

    def forward(self, x,return_feature=False):
        """
        Forward pass of the model.

        Parameters
        ----------
        x: PyTorch Tensor
            Input tensor of shape (batch_size, n_channels, n_times)
        return_feature: bool
            If True, the model returns the features before the final layer

        Returns
        -------
        PyTorch Tensor
            Output tensor of shape (batch_size, n_outputs)
        """
        x_rs = self.rs(x.contiguous())
        feat = torch.relu(self.bn(self.conv(x_rs)))
        for seq in self.blocks:
            feat = seq(feat)
        feat = feat.mean(dim=2)
        return self.final_layer(feat) if not return_feature else feat
