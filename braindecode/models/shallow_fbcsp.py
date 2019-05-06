import numpy as np
from torch import nn
from torch.nn import init

from braindecode.models.base import BaseModel
from braindecode.torch_ext.modules import Expression
from braindecode.torch_ext.functions import safe_log, square
from braindecode.torch_ext.util import np_to_var


class ShallowFBCSPNet(BaseModel):
    """
    Shallow ConvNet model from [2]_.

    References
    ----------

    .. [2] Schirrmeister, R. T., Springenberg, J. T., Fiederer, L. D. J.,
       Glasstetter, M., Eggensperger, K., Tangermann, M., Hutter, F. & Ball, T. (2017).
       Deep learning with convolutional neural networks for EEG decoding and
       visualization.
       Human Brain Mapping , Aug. 2017. Online: http://dx.doi.org/10.1002/hbm.23730
    """

    def __init__(
        self,
        in_chans,
        n_classes,
        input_time_length=None,
        n_filters_time=40,
        filter_time_length=25,
        n_filters_spat=40,
        pool_time_length=75,
        pool_time_stride=15,
        final_conv_length=30,
        conv_nonlin=square,
        pool_mode="mean",
        pool_nonlin=safe_log,
        split_first_layer=True,
        batch_norm=True,
        batch_norm_alpha=0.1,
        drop_prob=0.5,
    ):
        if final_conv_length == "auto":
            assert input_time_length is not None
        self.__dict__.update(locals())
        del self.self

    def create_network(self):
        pool_class = dict(max=nn.MaxPool2d, mean=nn.AvgPool2d)[self.pool_mode]
        model = nn.Sequential()
        if self.split_first_layer:
            model.add_module("dimshuffle", Expression(_transpose_time_to_spat))
            model.add_module(
                "conv_time",
                nn.Conv2d(
                    1, self.n_filters_time, (self.filter_time_length, 1), stride=1
                ),
            )
            model.add_module(
                "conv_spat",
                nn.Conv2d(
                    self.n_filters_time,
                    self.n_filters_spat,
                    (1, self.in_chans),
                    stride=1,
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
                    stride=1,
                    bias=not self.batch_norm,
                ),
            )
            n_filters_conv = self.n_filters_time
        if self.batch_norm:
            model.add_module(
                "bnorm",
                nn.BatchNorm2d(
                    n_filters_conv, momentum=self.batch_norm_alpha, affine=True
                ),
            )
        model.add_module("conv_nonlin", Expression(self.conv_nonlin))
        model.add_module(
            "pool",
            pool_class(
                kernel_size=(self.pool_time_length, 1),
                stride=(self.pool_time_stride, 1),
            ),
        )
        model.add_module("pool_nonlin", Expression(self.pool_nonlin))
        model.add_module("drop", nn.Dropout(p=self.drop_prob))
        model.eval()
        if self.final_conv_length == "auto":
            out = model(
                np_to_var(
                    np.ones(
                        (1, self.in_chans, self.input_time_length, 1), dtype=np.float32
                    )
                )
            )
            n_out_time = out.cpu().data.numpy().shape[2]
            self.final_conv_length = n_out_time
        model.add_module(
            "conv_classifier",
            nn.Conv2d(
                n_filters_conv, self.n_classes, (self.final_conv_length, 1), bias=True
            ),
        )
        model.add_module("softmax", nn.LogSoftmax(dim=1))
        model.add_module("squeeze", Expression(_squeeze_final_output))

        # Initialization, xavier is same as in paper...
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
        init.xavier_uniform_(model.conv_classifier.weight, gain=1)
        init.constant_(model.conv_classifier.bias, 0)

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
