# Authors: Robin Tibor Schirrmeister <robintibor@gmail.com>
#          Tonio Ball
#
# License: BSD-3

import numpy as np

import torch
from torch import nn
from torch.nn import init
from torch.nn.functional import elu

from .functions import transpose_time_to_spat, squeeze_final_output
from .modules import Expression, AvgPool2dWithConv, Ensure4d


class EEGResNet(nn.Sequential):
    """Residual Network for EEG.

    XXX missing reference

    Parameters
    ----------
    in_chans : int
        XXX

    """
    def __init__(self,
                 in_chans,
                 n_classes,
                 input_window_samples,
                 final_pool_length,
                 n_first_filters,
                 n_layers_per_block=2,
                 first_filter_length=3,
                 nonlinearity=elu,
                 split_first_layer=True,
                 batch_norm_alpha=0.1,
                 batch_norm_epsilon=1e-4,
                 conv_weight_init_fn=lambda w: init.kaiming_normal_(w, a=0)):
        super().__init__()
        self.in_chans = in_chans
        self.n_classes = n_classes
        self.input_window_samples = input_window_samples
        if final_pool_length == 'auto':
            assert input_window_samples is not None
        assert first_filter_length % 2 == 1
        self.final_pool_length = final_pool_length
        self.n_first_filters = n_first_filters
        self.n_layers_per_block = n_layers_per_block
        self.first_filter_length = first_filter_length
        self.nonlinearity = nonlinearity
        self.split_first_layer = split_first_layer
        self.batch_norm_alpha = batch_norm_alpha
        self.batch_norm_epsilon = batch_norm_epsilon
        self.conv_weight_init_fn = conv_weight_init_fn

        self.add_module("ensuredims", Ensure4d())
        if self.split_first_layer:
            self.add_module('dimshuffle', Expression(transpose_time_to_spat))
            self.add_module('conv_time', nn.Conv2d(1, self.n_first_filters,
                                                   (self.first_filter_length, 1),
                                                   stride=1,
                                                   padding=(self.first_filter_length // 2, 0)))
            self.add_module('conv_spat',
                            nn.Conv2d(self.n_first_filters, self.n_first_filters,
                                      (1, self.in_chans),
                                      stride=(1, 1),
                                      bias=False))
        else:
            self.add_module('conv_time',
                            nn.Conv2d(self.in_chans, self.n_first_filters,
                                      (self.first_filter_length, 1),
                                      stride=(1, 1),
                                      padding=(self.first_filter_length // 2, 0),
                                      bias=False,))
        n_filters_conv = self.n_first_filters
        self.add_module('bnorm',
                        nn.BatchNorm2d(n_filters_conv,
                                       momentum=self.batch_norm_alpha,
                                       affine=True,
                                       eps=1e-5),)
        self.add_module('conv_nonlin', Expression(self.nonlinearity))
        cur_dilation = np.array([1, 1])
        n_cur_filters = n_filters_conv
        i_block = 1
        for i_layer in range(self.n_layers_per_block):
            self.add_module('res_{:d}_{:d}'.format(i_block, i_layer),
                            _ResidualBlock(n_cur_filters, n_cur_filters,
                                           dilation=cur_dilation))
        i_block += 1
        cur_dilation[0] *= 2
        n_out_filters = int(2 * n_cur_filters)
        self.add_module('res_{:d}_{:d}'.format(i_block, 0),
                        _ResidualBlock(n_cur_filters, n_out_filters,
                                       dilation=cur_dilation,))
        n_cur_filters = n_out_filters
        for i_layer in range(1, self.n_layers_per_block):
            self.add_module('res_{:d}_{:d}'.format(i_block, i_layer),
                            _ResidualBlock(n_cur_filters, n_cur_filters,
                                           dilation=cur_dilation))

        i_block += 1
        cur_dilation[0] *= 2
        n_out_filters = int(1.5 * n_cur_filters)
        self.add_module('res_{:d}_{:d}'.format(i_block, 0),
                        _ResidualBlock(n_cur_filters, n_out_filters,
                                       dilation=cur_dilation,))
        n_cur_filters = n_out_filters
        for i_layer in range(1, self.n_layers_per_block):
            self.add_module('res_{:d}_{:d}'.format(i_block, i_layer),
                            _ResidualBlock(n_cur_filters, n_cur_filters,
                                           dilation=cur_dilation))

        i_block += 1
        cur_dilation[0] *= 2
        self.add_module('res_{:d}_{:d}'.format(i_block, 0),
                        _ResidualBlock(n_cur_filters, n_cur_filters,
                                       dilation=cur_dilation,))
        for i_layer in range(1, self.n_layers_per_block):
            self.add_module('res_{:d}_{:d}'.format(i_block, i_layer),
                            _ResidualBlock(n_cur_filters, n_cur_filters,
                                           dilation=cur_dilation))

        i_block += 1
        cur_dilation[0] *= 2
        self.add_module('res_{:d}_{:d}'.format(i_block, 0),
                        _ResidualBlock(n_cur_filters, n_cur_filters,
                                       dilation=cur_dilation,))
        for i_layer in range(1, self.n_layers_per_block):
            self.add_module('res_{:d}_{:d}'.format(i_block, i_layer),
                            _ResidualBlock(n_cur_filters, n_cur_filters,
                                           dilation=cur_dilation))

        i_block += 1
        cur_dilation[0] *= 2
        self.add_module('res_{:d}_{:d}'.format(i_block, 0),
                        _ResidualBlock(n_cur_filters, n_cur_filters,
                                       dilation=cur_dilation,))
        for i_layer in range(1, self.n_layers_per_block):
            self.add_module('res_{:d}_{:d}'.format(i_block, i_layer),
                            _ResidualBlock(n_cur_filters, n_cur_filters,
                                           dilation=cur_dilation))
        i_block += 1
        cur_dilation[0] *= 2
        self.add_module('res_{:d}_{:d}'.format(i_block, 0),
                        _ResidualBlock(n_cur_filters, n_cur_filters,
                                       dilation=cur_dilation,))
        for i_layer in range(1, self.n_layers_per_block):
            self.add_module('res_{:d}_{:d}'.format(i_block, i_layer),
                            _ResidualBlock(n_cur_filters, n_cur_filters,
                                           dilation=cur_dilation))

        self.eval()
        if self.final_pool_length == 'auto':
            self.add_module('mean_pool', nn.AdaptiveAvgPool2d((1, 1)))
        else:
            pool_dilation = int(cur_dilation[0]), int(cur_dilation[1])
            self.add_module('mean_pool', AvgPool2dWithConv(
                (self.final_pool_length, 1), (1, 1),
                dilation=pool_dilation))
        self.add_module('conv_classifier',
                        nn.Conv2d(n_cur_filters, self.n_classes,
                                  (1, 1), bias=True))
        self.add_module('softmax', nn.LogSoftmax(dim=1))
        self.add_module('squeeze', Expression(squeeze_final_output))

        # Initialize all weights
        self.apply(lambda module: _weights_init(module, self.conv_weight_init_fn))

        # Start in eval mode
        self.eval()


def _weights_init(module, conv_weight_init_fn):
    """
    initialize weights
    """
    classname = module.__class__.__name__
    if 'Conv' in classname and classname != "AvgPool2dWithConv":
        conv_weight_init_fn(module.weight)
        if module.bias is not None:
            init.constant_(module.bias, 0)
    elif 'BatchNorm' in classname:
        init.constant_(module.weight, 1)
        init.constant_(module.bias, 0)


class _ResidualBlock(nn.Module):
    """
    create a residual learning building block with two stacked 3x3 convlayers as in paper
    """
    def __init__(self, in_filters,
                 out_num_filters,
                 dilation,
                 filter_time_length=3,
                 nonlinearity=elu,
                 batch_norm_alpha=0.1, batch_norm_epsilon=1e-4):
        super(_ResidualBlock, self).__init__()
        time_padding = int((filter_time_length - 1) * dilation[0])
        assert time_padding % 2 == 0
        time_padding = int(time_padding // 2)
        dilation = (int(dilation[0]), int(dilation[1]))
        assert (out_num_filters - in_filters) % 2 == 0, (
            "Need even number of extra channels in order to be able to "
            "pad correctly")
        self.n_pad_chans = out_num_filters - in_filters

        self.conv_1 = nn.Conv2d(
            in_filters, out_num_filters, (filter_time_length, 1), stride=(1, 1),
            dilation=dilation,
            padding=(time_padding, 0))
        self.bn1 = nn.BatchNorm2d(
            out_num_filters, momentum=batch_norm_alpha, affine=True,
            eps=batch_norm_epsilon)
        self.conv_2 = nn.Conv2d(
            out_num_filters, out_num_filters, (filter_time_length, 1), stride=(1, 1),
            dilation=dilation,
            padding=(time_padding, 0))
        self.bn2 = nn.BatchNorm2d(
            out_num_filters, momentum=batch_norm_alpha,
            affine=True, eps=batch_norm_epsilon)
        # also see https://mail.google.com/mail/u/0/#search/ilya+joos/1576137dd34c3127
        # for resnet options as ilya used them
        self.nonlinearity = nonlinearity

    def forward(self, x):
        stack_1 = self.nonlinearity(self.bn1(self.conv_1(x)))
        stack_2 = self.bn2(self.conv_2(stack_1))  # next nonlin after sum
        if self.n_pad_chans != 0:
            zeros_for_padding = torch.autograd.Variable(
                torch.zeros(x.size()[0], self.n_pad_chans // 2, x.size()[2], x.size()[3]))
            if x.is_cuda:
                zeros_for_padding = zeros_for_padding.cuda()
            x = torch.cat((zeros_for_padding, x, zeros_for_padding), dim=1)
        out = self.nonlinearity(x + stack_2)
        return out
