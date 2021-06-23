"""
Implementation of UTime as described in:
Mathias Perslev, Michael Hejselbak Jensen, Sune Darkner, Poul Jørgen Jennum
and Christian Igel. U-Time: A Fully Convolutional Network for Time Series
Segmentation Applied to Sleep Staging. Advances in Neural Information
Processing Systems (NeurIPS 2019)
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
from tensorflow.keras.layers import (
    Input,
    BatchNormalization,
    Cropping2D,
    Concatenate,
    MaxPooling2D,
    Dense,
    UpSampling2D,
    ZeroPadding2D,
    Lambda,
    Conv2D,
    AveragePooling2D,
    DepthwiseConv2D,
)


class USleep(Model):
    """
    OBS: Uses 2D operations internally with a 'dummy' axis, so that a batch
        of shape [bs, d, c] is processed as [bs, d, 1, c]. These operations
        are (on our systems, at least) currently significantly faster than
        their 1D counterparts in tf.keras.
    See also original U-net paper at http://arxiv.org/abs/1505.04597
    """

    def __init__(
        self,
        n_classes,
        batch_shape,
        depth=3,
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
        **kwargs
    ):
        """
        n_classes (int):
            The number of classes to model, gives the number of filters in the
            final 1x1 conv layer.
        batch_shape (list): Giving the shape of one one batch of data,
                            potentially omitting the zeroth axis (the batch
                            size dim)
        depth (int):
            Number of conv blocks in encoding layer (number of 2x2 max pools)
            Note: each block doubles the filter count while halving the spatial
            dimensions of the features.
        dilation (int):
            TODO
        activation (string):
            Activation function for convolution layers
        dense_classifier_activation (string):
            TODO
        kernel_size (int):
            Kernel size for convolution layers
        transition_window (int):
            TODO
        padding (string):
            Padding type ('same' or 'valid')
        complexity_factor (int/float):
            Use int(N * sqrt(complexity_factor)) number of filters in each
            convolution layer instead of default N.
        l2_reg (float in [0, 1])
            L2 regularization on conv weights
        data_per_prediction (int):
            TODO
        logger (mpunet.logging.Logger | ScreenLogger):
            MutliViewUNet.Logger object, logging to files or screen.
        build (bool):
            TODO
        """
        super(USleep, self).__init__()

        # Set logger or standard print wrapper
        # self.logger = logger or ScreenLogger()

        # Set various attributes
        assert len(batch_shape) == 4
        self.n_periods = batch_shape[1]
        self.input_dims = batch_shape[2]
        self.n_channels = batch_shape[3]
        self.n_classes = int(n_classes)
        self.dilation = int(dilation)
        self.cf = np.sqrt(complexity_factor)
        self.init_filters = init_filters
        self.kernel_size = int(kernel_size)
        self.transition_window = transition_window
        self.activation = activation
        self.l2_reg = l2_reg
        self.depth = depth
        self.n_crops = 0
        self.padding = padding.lower()
        if self.padding != "same":
            raise ValueError("Currently, must use 'same' padding.")

        self.dense_classifier_activation = dense_classifier_activation
        self.data_per_prediction = data_per_prediction or self.input_dims
        if not isinstance(self.data_per_prediction, (int, np.integer)):
            raise TypeError("data_per_prediction must be an integer value")
        if self.input_dims % self.data_per_prediction:
            raise ValueError(
                "'input_dims' ({}) must be evenly divisible by "
                "'data_per_prediction' ({})".format(
                    self.input_dims, self.data_per_prediction
                )
            )

        if build:
            # Build model and init base keras Model class
            super().__init__(*self.init_model())

            # Compute receptive field
            ind = [x.__class__.__name__ for x in self.layers].index("UpSampling2D")
            # self.receptive_field = compute_receptive_fields(self.layers[:ind])[-1][-1]

            # Log the model definition
            # self.log()
        else:
            self.receptive_field = [None]

    @staticmethod
    def create_encoder(
        in_,
        depth,
        filters,
        kernel_size,
        activation,
        dilation,
        padding,
        complexity_factor,
        regularizer=None,
        name="encoder",
        name_prefix="",
    ):
        name = "{}{}".format(name_prefix, name)
        residual_connections = []
        for i in range(depth):
            l_name = name + "_L%i" % i
            print(complexity_factor)
            conv = Conv2D(
                int(filters * complexity_factor),
                (kernel_size, 1),
                activation=activation,
                padding=padding,
                kernel_regularizer=regularizer,
                bias_regularizer=regularizer,
                dilation_rate=dilation,
                name=l_name + "_conv1",
            )(in_)
            print("conv shape:", conv.shape)
            bn = BatchNormalization(name=l_name + "_BN1")(conv)
            s = bn.get_shape()[1]
            if s % 2:
                bn = ZeroPadding2D(padding=[[1, 0], [0, 0]], name=l_name + "_padding")(
                    bn
                )
            in_ = MaxPooling2D(pool_size=(2, 1), name=l_name + "_pool")(bn)

            # add bn layer to list for residual conn.
            residual_connections.append(bn)
            filters = int(filters * np.sqrt(2))

        # Bottom
        name = "{}bottom".format(name_prefix)
        conv = Conv2D(
            int(filters * complexity_factor),
            (kernel_size, 1),
            activation=activation,
            padding=padding,
            kernel_regularizer=regularizer,
            bias_regularizer=regularizer,
            dilation_rate=1,
            name=name + "_conv1",
        )(in_)
        encoded = BatchNormalization(name=name + "_BN1")(conv)
        return encoded, residual_connections, filters

    def create_upsample(
        self,
        in_,
        res_conns,
        depth,
        filters,
        kernel_size,
        activation,
        dilation,  # NOT USED
        padding,
        complexity_factor,
        regularizer=None,
        name="upsample",
        name_prefix="",
    ):
        name = "{}{}".format(name_prefix, name)
        residual_connections = res_conns[::-1]
        for i in range(depth):
            filters = int(np.ceil(filters / np.sqrt(2)))
            l_name = name + "_L%i" % i

            # Up-sampling block
            up = UpSampling2D(size=(2, 1), name=l_name + "_up")(in_)
            conv = Conv2D(
                int(filters * complexity_factor),
                (2, 1),
                activation=activation,
                padding=padding,
                kernel_regularizer=regularizer,
                bias_regularizer=regularizer,
                name=l_name + "_conv1",
            )(up)
            bn = BatchNormalization(name=l_name + "_BN1")(conv)

            # Crop and concatenate
            res_con = residual_connections[i]
            cropped_bn = self.crop_nodes_to_match(bn, res_con)
            merge = Concatenate(axis=-1, name=l_name + "_concat")([res_con, cropped_bn])

            conv = Conv2D(
                int(filters * complexity_factor),
                (kernel_size, 1),
                activation=activation,
                padding=padding,
                kernel_regularizer=regularizer,
                bias_regularizer=regularizer,
                name=l_name + "_conv2",
            )(merge)
            in_ = BatchNormalization(name=l_name + "_BN2")(conv)

        return in_

    def create_dense_modeling(
        self,
        in_,
        in_reshaped,
        filters,
        dense_classifier_activation,
        regularizer,
        complexity_factor,
        name_prefix="",
        **kwargs
    ):
        cls = Conv2D(
            filters=int(filters * complexity_factor),
            kernel_size=(1, 1),
            kernel_regularizer=regularizer,
            bias_regularizer=regularizer,
            activation=dense_classifier_activation,
            name="{}dense_classifier_out".format(name_prefix),
        )(in_)
        s = (self.n_periods * self.input_dims) - cls.get_shape().as_list()[1]
        out = self.crop_nodes_to_match(
            node1=ZeroPadding2D(padding=[[s // 2, s // 2 + s % 2], [0, 0]])(cls),
            node2=in_reshaped,
        )
        return out

    @staticmethod
    def create_seq_modeling(
        in_,
        input_dims,
        data_per_period,
        n_periods,
        n_classes,
        transition_window,
        activation,
        regularizer=None,
        name_prefix="",
    ):
        cls = AveragePooling2D(
            (data_per_period, 1), name="{}average_pool".format(name_prefix)
        )(in_)
        out = Conv2D(
            filters=n_classes,
            kernel_size=(transition_window, 1),
            activation=activation,
            kernel_regularizer=regularizer,
            bias_regularizer=regularizer,
            padding="same",
            name="{}sequence_conv_out_1".format(name_prefix),
        )(cls)
        out = Conv2D(
            filters=n_classes,
            kernel_size=(transition_window, 1),
            activation="softmax",
            kernel_regularizer=regularizer,
            bias_regularizer=regularizer,
            padding="same",
            name="{}sequence_conv_out_2".format(name_prefix),
        )(out)
        s = [-1, n_periods, input_dims // data_per_period, n_classes]
        if s[2] == 1:
            s.pop(2)  # Squeeze the dim
        out = Lambda(
            lambda x: tf.reshape(x, s),
            name="{}sequence_classification_reshaped".format(name_prefix),
        )(out)
        return out

    def init_model(self, inputs=None, name_prefix=""):
        """
        Build the UNet model with the specified input image shape.
        """
        if inputs is None:
            inputs = Input(shape=[self.n_periods, self.input_dims, self.n_channels])
        reshaped = [-1, self.n_periods * self.input_dims, 1, self.n_channels]
        in_reshaped = Lambda(lambda x: tf.reshape(x, reshaped))(inputs)

        # Apply regularization if not None or 0
        regularizer = regularizers.l2(self.l2_reg) if self.l2_reg else None

        # Get activation func from tf or tfa
        activation = self.activation

        settings = {
            "depth": self.depth,
            "filters": self.init_filters,
            "kernel_size": self.kernel_size,
            "activation": activation,
            "dilation": self.dilation,
            "padding": self.padding,
            "regularizer": regularizer,
            "name_prefix": name_prefix,
            "complexity_factor": self.cf,
        }

        """
        Encoding path
        """
        enc, residual_cons, filters = self.create_encoder(in_=in_reshaped, **settings)

        """
        Decoding path
        """
        settings["filters"] = filters
        up = self.create_upsample(enc, residual_cons, **settings)

        """
        Dense class modeling layers
        """
        # cls = self.create_dense_modeling(in_=up,
        #                                 in_reshaped=in_reshaped,
        #                                 filters=self.n_classes,
        #                                 dense_classifier_activation=self.dense_classifier_activation,
        #                                 regularizer=regularizer,
        #                                 complexity_factor=self.cf,
        #                                 name_prefix=name_prefix)

        # """
        # Sequence modeling
        # """
        # out = self.create_seq_modeling(in_=cls,
        #                             input_dims=self.input_dims,
        #                             data_per_period=self.data_per_prediction,
        #                             n_periods=self.n_periods,
        #                             n_classes=self.n_classes,
        #                             transition_window=self.transition_window,
        #                             activation=self.activation,
        #                             regularizer=regularizer,
        #                             name_prefix=name_prefix)

        return [inputs], [up]

    def crop_nodes_to_match(self, node1, node2):
        """
        If necessary, applies Cropping2D layer to node1 to match shape of node2
        """
        s1 = np.array(node1.get_shape().as_list())[1:-2]
        s2 = np.array(node2.get_shape().as_list())[1:-2]

        if np.any(s1 != s2):
            self.n_crops += 1
            c = (s1 - s2).astype(np.int)
            cr = np.array([c // 2, c // 2]).flatten()
            cr[self.n_crops % 2] += c % 2
            cropped_node1 = Cropping2D([list(cr), [0, 0]])(node1)
        else:
            cropped_node1 = node1
        return cropped_node1


# testing

batch_size, n_channels, n_times = 1024, 2, 3000

np.random.seed(0)
x = np.random.random((batch_size, n_times, 1, n_channels))
x = tf.convert_to_tensor(x)

model = USleep(5, batch_shape=(batch_size, n_times, 1, n_channels))
