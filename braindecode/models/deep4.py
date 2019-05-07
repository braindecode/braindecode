import numpy as np
from torch import nn
from torch.nn import init
from torch.nn.functional import elu

from braindecode.models.base import BaseModel
from braindecode.torch_ext.modules import Expression, AvgPool2dWithConv
from braindecode.torch_ext.functions import identity
from braindecode.torch_ext.util import np_to_var


class Deep4Net(BaseModel):
    """
    Deep ConvNet model from [1]_.

    References
    ----------

    .. [1] Schirrmeister, R. T., Springenberg, J. T., Fiederer, L. D. J., 
       Glasstetter, M., Eggensperger, K., Tangermann, M., Hutter, F. & Ball, T. (2017).
       Deep learning with convolutional neural networks for EEG decoding and
       visualization.
       Human Brain Mapping , Aug. 2017. Online: http://dx.doi.org/10.1002/hbm.23730
    """

    def __init__(
        self,
        in_chans,
        n_classes,
        input_time_length,
        final_conv_length,
        n_filters_time=25,
        n_filters_spat=25,
        filter_time_length=10,
        pool_time_length=3,
        pool_time_stride=3,
        n_filters_2=50,
        filter_length_2=10,
        n_filters_3=100,
        filter_length_3=10,
        n_filters_4=200,
        filter_length_4=10,
        first_nonlin=elu,
        first_pool_mode="max",
        first_pool_nonlin=identity,
        later_nonlin=elu,
        later_pool_mode="max",
        later_pool_nonlin=identity,
        drop_prob=0.5,
        double_time_convs=False,
        split_first_layer=True,
        batch_norm=True,
        batch_norm_alpha=0.1,
        stride_before_pool=False,
    ):
        if final_conv_length == "auto":
            assert input_time_length is not None

        self.__dict__.update(locals())
        del self.self

    def create_network(self):
        if self.stride_before_pool:
            conv_stride = self.pool_time_stride
            pool_stride = 1
        else:
            conv_stride = 1
            pool_stride = self.pool_time_stride
        pool_class_dict = dict(max=nn.MaxPool2d, mean=AvgPool2dWithConv)
        first_pool_class = pool_class_dict[self.first_pool_mode]
        later_pool_class = pool_class_dict[self.later_pool_mode]
        model = nn.Sequential()
        if self.split_first_layer:
            model.add_module("dimshuffle", Expression(_transpose_time_to_spat))
            model.add_module(
                "conv_time",
                nn.Conv2d(
                    1,
                    self.n_filters_time,
                    (self.filter_time_length, 1),
                    stride=1,
                ),
            )
            model.add_module(
                "conv_spat",
                nn.Conv2d(
                    self.n_filters_time,
                    self.n_filters_spat,
                    (1, self.in_chans),
                    stride=(conv_stride, 1),
                    bias=not self.batch_norm,
                ),
            )
            n_filters_conv = self.n_filters_spat
        else:
            model.add_module(
                "conv_time",
                nn.Conv2d(
                    self.in_chans,
                    self.n_filters_time,
                    (self.filter_time_length, 1),
                    stride=(conv_stride, 1),
                    bias=not self.batch_norm,
                ),
            )
            n_filters_conv = self.n_filters_time
        if self.batch_norm:
            model.add_module(
                "bnorm",
                nn.BatchNorm2d(
                    n_filters_conv,
                    momentum=self.batch_norm_alpha,
                    affine=True,
                    eps=1e-5,
                ),
            )
        model.add_module("conv_nonlin", Expression(self.first_nonlin))
        model.add_module(
            "pool",
            first_pool_class(
                kernel_size=(self.pool_time_length, 1), stride=(pool_stride, 1)
            ),
        )
        model.add_module("pool_nonlin", Expression(self.first_pool_nonlin))

        def add_conv_pool_block(
            model, n_filters_before, n_filters, filter_length, block_nr
        ):
            suffix = "_{:d}".format(block_nr)
            model.add_module("drop" + suffix, nn.Dropout(p=self.drop_prob))
            model.add_module(
                "conv" + suffix,
                nn.Conv2d(
                    n_filters_before,
                    n_filters,
                    (filter_length, 1),
                    stride=(conv_stride, 1),
                    bias=not self.batch_norm,
                ),
            )
            if self.batch_norm:
                model.add_module(
                    "bnorm" + suffix,
                    nn.BatchNorm2d(
                        n_filters,
                        momentum=self.batch_norm_alpha,
                        affine=True,
                        eps=1e-5,
                    ),
                )
            model.add_module("nonlin" + suffix, Expression(self.later_nonlin))

            model.add_module(
                "pool" + suffix,
                later_pool_class(
                    kernel_size=(self.pool_time_length, 1),
                    stride=(pool_stride, 1),
                ),
            )
            model.add_module(
                "pool_nonlin" + suffix, Expression(self.later_pool_nonlin)
            )

        add_conv_pool_block(
            model, n_filters_conv, self.n_filters_2, self.filter_length_2, 2
        )
        add_conv_pool_block(
            model, self.n_filters_2, self.n_filters_3, self.filter_length_3, 3
        )
        add_conv_pool_block(
            model, self.n_filters_3, self.n_filters_4, self.filter_length_4, 4
        )

        # model.add_module('drop_classifier', nn.Dropout(p=self.drop_prob))
        model.eval()
        if self.final_conv_length == "auto":
            out = model(
                np_to_var(
                    np.ones(
                        (1, self.in_chans, self.input_time_length, 1),
                        dtype=np.float32,
                    )
                )
            )
            n_out_time = out.cpu().data.numpy().shape[2]
            self.final_conv_length = n_out_time
        model.add_module(
            "conv_classifier",
            nn.Conv2d(
                self.n_filters_4,
                self.n_classes,
                (self.final_conv_length, 1),
                bias=True,
            ),
        )
        model.add_module("softmax", nn.LogSoftmax(dim=1))
        model.add_module("squeeze", Expression(_squeeze_final_output))

        # Initialization, xavier is same as in our paper...
        # was default from lasagne
        init.xavier_uniform_(model.conv_time.weight, gain=1)
        # maybe no bias in case of no split layer and batch norm
        if self.split_first_layer or (not self.batch_norm):
            init.constant_(model.conv_time.bias, 0)
        if self.split_first_layer:
            init.xavier_uniform_(model.conv_spat.weight, gain=1)
            if not self.batch_norm:
                init.constant_(model.conv_spat.bias, 0)
        if self.batch_norm:
            init.constant_(model.bnorm.weight, 1)
            init.constant_(model.bnorm.bias, 0)
        param_dict = dict(list(model.named_parameters()))
        for block_nr in range(2, 5):
            conv_weight = param_dict["conv_{:d}.weight".format(block_nr)]
            init.xavier_uniform_(conv_weight, gain=1)
            if not self.batch_norm:
                conv_bias = param_dict["conv_{:d}.bias".format(block_nr)]
                init.constant_(conv_bias, 0)
            else:
                bnorm_weight = param_dict["bnorm_{:d}.weight".format(block_nr)]
                bnorm_bias = param_dict["bnorm_{:d}.bias".format(block_nr)]
                init.constant_(bnorm_weight, 1)
                init.constant_(bnorm_bias, 0)

        init.xavier_uniform_(model.conv_classifier.weight, gain=1)
        init.constant_(model.conv_classifier.bias, 0)

        # Start in eval mode
        model.eval()
        return model


# remove empty dim at end and potentially remove empty time dim
# do not just use squeeze as we never want to remove first dim
def _squeeze_final_output(x):
    assert x.size()[3] == 1
    x = x[:, :, :, 0]
    if x.size()[2] == 1:
        x = x[:, :, 0]
    return x


def _transpose_time_to_spat(x):
    return x.permute(0, 3, 2, 1)
