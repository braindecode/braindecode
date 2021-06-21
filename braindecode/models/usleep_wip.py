# Authors: Theo Gnassounou <theo.gnassounou@inria.fr>
#          Omar Chehab <l-emir-omar.chehab@inria.fr>
#
# License: BSD (3-clause)

import numpy as np
import torch
from torch import nn
from torch.nn import init
from torch.nn.functional import elu


class USleep(nn.Module):

    def __init__(self, 
                n_classes=5,
                batch_shape=128,
                depth=12,
                dilation=1,
                activation="elu",
                dense_classifier_activation="tanh",
                kernel_size=9,
                transition_window=1,
                #  padding="same",
                filters=5,
                complexity_factor=2,
                l2_reg=None,
                data_per_prediction=None):
        '''TODO: remove redundant arguments.'''
        super().__init__()


        # Step 1: write self.attribute = default


        # Step 2: instantiate encoder
        self.unet = # to be defined
        self.clf = # 


        for idx in range(depth):


        # residual_connections = []
        for i in range(depth):
            conv = Conv2D(int(filters*complexity_factor), (kernel_size, 1),
                          activation=activation, padding=padding,
                          kernel_regularizer=regularizer,
                          bias_regularizer=regularizer,
                          dilation_rate=dilation,
                          name=l_name + "_conv1")(in_)
            bn = BatchNormalization(name=l_name + "_BN1")(conv)
            s = bn.get_shape()[1]
            if s % 2:
                bn = ZeroPadding2D(padding=[[1, 0], [0, 0]],
                                   name=l_name + "_padding")(bn)
            in_ = MaxPooling2D(pool_size=(2, 1), name=l_name + "_pool")(bn)

            # add bn layer to list for residual conn.
            residual_connections.append(bn)
            filters = int(filters * np.sqrt(2))




        return 0




    """
    def __init__(self,
                 n_classes,
                 batch_shape,
                 depth=12,
                 dilation=1,
                 activation="elu",
                 dense_classifier_activation="tanh",
                 kernel_size=9,
                 transition_window=1,
                 padding="same",
                 init_filters=5,
                 complexity_factor=2,
                 l2_reg=None,
                 data_per_prediction=None,
                 logger=None,
                 build=True,
                 **kwargs):






# Test model on a single input
sfreq = 100  # unit: Hz
time_window = 30  # unit: s
batch_size, n_times, n_channels = 1024, time_window * sfreq, 2 

x = torch.randn(batch_size, n_channels, n_times)
usleep = USleep()
y = usleep(x)
print("x has shape: ", x.shape)
print("y has shape: ", y.shape)






