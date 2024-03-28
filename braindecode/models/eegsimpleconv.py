# Authors: Yassine El Ouahidi <eloua.yas@gmail.com>
#
# License: BSD-3

import torch
from torch import nn
from . import Resample
from .base import EEGModuleMixin



class EEGSimpleConv(EEGModuleMixin, torch.nn.Module):
    '''
    EEGSimpleConv. Model described in [1]. Original code from: https://github.com/elouayas/EEGSimpleConv
    
    Parameters
    ----------
    fm (int)         : Number of Feature Maps at the first Convolution, width of the model.
    n_convs (int)    : Number of blocks of convolutions (2 convolutions per block), depth of the model.
    resampling (int) : Resampling Frequency.
    kernel_size (int): Size of the convolutions kernels .
    Guidelines on how to set those parameters are available in the appendix of the paper.
    It is advised to read it before using the model. Default parameter should be effective for MI decoding.

    References
    ----------
    .. [1] Yassine El Ouahidi, Vincent Gripon, Bastien Pasdeloup, Ghaith Bouallegue, Nicolas Farrugia, Giulia Lioi (2023).
    A Strong and Simple Deep Learning Baseline for BCI Motor Imagery Decoding.
    Arxiv preprint, under review for a journal publication.
    Online: https://arxiv.org/abs/2309.07159
    '''
    def __init__(self,n_outputs=None,n_chans=None,sfreq = None,        # Base arguments
                 fm = 128,n_convs = 2,resampling = 80,kernel_size = 8, # Model specific arguments
                chs_info=None,n_times=None,input_window_seconds=None   # Not used base arguments
                 ):
        super().__init__(
            n_outputs=n_outputs,
            n_chans=n_chans,
            chs_info=chs_info,
            n_times=n_times,
            input_window_seconds=input_window_seconds,
            sfreq=sfreq,
        )
        self.rs = Resample(orig_freq=sfreq,new_freq=resampling) if sfreq!=resampling else torch.nn.Identity()

        self.conv = torch.nn.Conv1d(n_chans, fm, kernel_size = kernel_size, padding = kernel_size // 2, bias = False)
        self.bn = torch.nn.BatchNorm1d(fm)
        self.blocks = []
        newfm = fm
        oldfm = fm
        for i in range(n_convs):
            if i > 0:
                newfm = int(1.414 * newfm)
            self.blocks.append(torch.nn.Sequential(
                (torch.nn.Conv1d(oldfm, newfm, kernel_size = kernel_size, padding = kernel_size // 2, bias = False)),
                (torch.nn.BatchNorm1d(newfm)),
                (torch.nn.MaxPool1d(2) if i > 0 - 1 else torch.nn.MaxPool1d(1)),
                (torch.nn.ReLU()),
                (torch.nn.Conv1d(newfm, newfm, kernel_size = kernel_size, padding = kernel_size // 2, bias = False)),
                (torch.nn.BatchNorm1d(newfm)),
                (torch.nn.ReLU())
            ))
            oldfm = newfm
        self.blocks = torch.nn.ModuleList(self.blocks)
        self.final_layer = torch.nn.Linear(oldfm, n_outputs)

    def forward(self, x): # x = Batch x Channels x Time
        x_rs = self.rs(x.contiguous())
        y = torch.relu(self.bn(self.conv(x_rs)))
        for seq in self.blocks:
            y = seq(y)
        y = y.mean(dim = 2)
        return self.final_layer(y)
