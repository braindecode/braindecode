import math

import tensorflow as tf
import keras
from keras import layers
from keras import constraints


class SincFilter(layers.Layer):
    """SincFilter."""

    def __init__(
        self,
        low_freqs,
        kernel_size,
        sample_rate,
        bandwidth=4,
        min_freq=1.0,
        padding="SAME",
    ):
        super().__init__(name="sinc_filter_layer")
        if kernel_size % 2 == 0:
            raise ValueError("Kernel size must be odd.")

        self.num_filters = len(low_freqs)
        self.kernel_size = kernel_size
        self.sample_rate = sample_rate
        self.min_freq = min_freq
        self.padding = padding
        self.ones = tf.ones((1, 1, 1, self.num_filters))
        window = tf.signal.hamming_window(kernel_size, periodic=False)
        # `self.window` has shape: [kernel_size // 2, 1].
        self.window = tf.expand_dims(window[: kernel_size // 2], axis=-1)
        # `self.n_pi` has shape: [kernel_size // 2, 1].
        self.n_pi = tf.range(-(kernel_size // 2), 0, dtype=tf.float32) / sample_rate
        self.n_pi *= 2 * math.pi
        self.n_pi = tf.expand_dims(self.n_pi, axis=-1)
        # `bandwidths` has shape: [1, num_filters].
        bandwidths = tf.ones((1, self.num_filters)) * bandwidth
        self.bandwidths = tf.Variable(bandwidths, name="bandwidths")
        # `low_freqs` has shape: [1, num_filters].
        self.low_freqs = tf.Variable([low_freqs], name="low_freqs", dtype=tf.float32)

    def build_sinc_filters(self):
        # `low_freqs` has shape: [1, num_filters].
        low_freqs = self.min_freq + tf.math.abs(self.low_freqs)
        # `high_freqs` has shape: [1, num_filters].
        high_freqs = tf.clip_by_value(
            low_freqs + tf.math.abs(self.bandwidths),
            self.min_freq,
            self.sample_rate / 2.0,
        )
        bandwidths = high_freqs - low_freqs

        low = self.n_pi * low_freqs  # size [kernel_size // 2, num_filters].
        high = self.n_pi * high_freqs  # size [kernel_size // 2, num_filters].

        # `filters_left` has shape: [kernel_size // 2, num_filters].
        filters_left = (tf.math.sin(high) - tf.math.sin(low)) / (self.n_pi / 2.0)
        filters_left *= self.window
        filters_left /= 2.0 * bandwidths

        # `filters_left` has shape: [1, kernel_size // 2, 1, num_filters].
        filters_left = filters_left[tf.newaxis, :, tf.newaxis, :]
        # filters_left = tf.ensure_shape(filters_left,
        #     shape=(1, self.kernel_size // 2, 1, self.num_filters))
        filters_right = tf.experimental.numpy.flip(filters_left, axis=1)

        # `filters` has shape: [1, kernel_size, 1, num_filters].
        filters = tf.concat([filters_left, self.ones, filters_right], axis=1)
        filters = filters / tf.math.reduce_std(filters)
        return filters

    def call(self, inputs):
        filters = self.build_sinc_filters()
        # `inputs` has shape: [num_epochs, num_channels, num_samples, 1].
        # `filtered` has shape: [num_epochs, num_channels, num_samples, num_filters]
        filtered = tf.nn.convolution(inputs, filters, padding=self.padding)
        return filtered


class SincShallowNet(keras.Model):
    """Implementation of Sinc-ShallowNet [1] adapted to work with EEG sampled at
    128 Hz, for that, average pooling filter size and strides were divided by 2.

    [1] Borra, D. et. al. (2020) Interpretable and lightweight convolutional
    neural network for EEG decoding: Application to movement execution and
    imagination.
    """

    def __init__(
        self,
        num_classes,
        num_channels,
        num_temp_filters,
        temp_filter_size,
        sample_rate,
        num_spatial_filters_x_temp,
    ):
        super().__init__()
        self.block_1 = keras.Sequential(
            [
                build_sinc_layer(
                    num_temp_filters,
                    temp_filter_size,
                    sample_rate,
                    first_freq=5,
                    freq_stride=1,
                    padding="VALID",
                ),
                layers.BatchNormalization(name="block_1_batchnorm"),
                layers.DepthwiseConv2D(
                    kernel_size=(num_channels, 1),
                    depth_multiplier=num_spatial_filters_x_temp,
                    use_bias=False,
                    name="spatial_filter",
                ),
            ],
            name="block_1",
        )

        self.block_2 = keras.Sequential(
            [
                layers.BatchNormalization(name="block_2_batchnorm"),
                layers.ELU(),
                layers.AveragePooling2D(pool_size=(1, 55), strides=(1, 12)),  # 128 Hz
                # layers.AveragePooling2D(pool_size=(1, 109), strides=(1, 23)), # 250 Hz
                layers.Dropout(0.5),
            ],
            name="block_2",
        )

        self.block_3 = keras.Sequential(
            [layers.Flatten(), layers.Dense(num_classes, name="dense")], name="block_3"
        )

    def call(self, epochs):
        x = self.block_1(epochs)
        x = self.block_2(x)
        logits = self.block_3(x)
        return logits


def build_sinc_layer(
    num_filters=8,
    filter_size=33,
    sample_rate=128,
    first_freq=6,
    freq_stride=4,
    bandwidth=4,
    padding="SAME",
):
    low_freqs = [first_freq]
    for _ in range(num_filters - 1):
        low_freqs.append(low_freqs[-1] + freq_stride)

    return SincFilter(low_freqs, filter_size, sample_rate, bandwidth, padding=padding)


if __name__ == "__main__":
    # Define example parameters
    num_classes = 2
    num_channels = 32
    num_temp_filters = 8
    temp_filter_size = 33
    sample_rate = 128
    num_spatial_filters_x_temp = 2

    # Instantiate the model
    model = SincShallowNet(
        num_classes=num_classes,
        num_channels=num_channels,
        num_temp_filters=num_temp_filters,
        temp_filter_size=temp_filter_size,
        sample_rate=sample_rate,
        num_spatial_filters_x_temp=num_spatial_filters_x_temp,
    )

    # Build the model by calling it on a dummy input
    batch_size = 1
    num_samples = 256  # Example number of samples; adjust as needed
    dummy_input = tf.zeros((batch_size, num_channels, num_samples, 1), dtype=tf.float32)

    # Forward pass
    output = model(dummy_input)

    # Print the output shape to verify
    print("Output shape:", output.shape)

    # Optionally, print the output tensor
    print("Output tensor:", output.numpy())
