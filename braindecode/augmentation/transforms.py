# Authors: Cédric Rommel <cpe.rommel@gmail.com>
#
# License: BSD (3-clause)

from numbers import Real

import numpy as np
from sklearn.utils import check_random_state
import torch

from braindecode.augmentation.base import Transform
from braindecode.augmentation.functionals import time_reverse, sign_flip,\
    downsample_shift_from_arrays, fft_surrogate, channel_dropout,\
    channel_shuffle, add_gaussian_noise, permute_channels, random_time_mask,\
    identity, random_bandstop, freq_shift, random_rotation,\
    get_standard_10_20_positions


class TimeReverse(Transform):
    """ Flip the time axis of each sample with a given probability

    Parameters
    ----------
    probability : float | callable
        Either a float between 0 and 1 defining the uniform probability of
        applying the operation, or a `callable` (modelling a non-uniform
        probability law for example) taking no input and returning a boolean.
    magnitude : object, optional
        Always ignored, exists for compatibility.
    mag_range : object, optional
        Always ignored, exists for compatibility.
    random_state: int | numpy.random.Generator, optional
        Seed to be used to instantiate numpy random number generator instance.
        Used to decide whether or not to transform given the probability
        argument. Defaults to None.
    """

    def __init__(
        self,
        probability,
        magnitude=None,
        mag_range=None,
        random_state=None
    ):
        super().__init__(
            operation=time_reverse,
            probability=probability,
            random_state=random_state
        )


class SignFlip(Transform):
    """ Flip the sign axis of each feature sample with a given probability

    Parameters
    ----------
    probability : float | callable
        Either a float between 0 and 1 defining the uniform probability of
        applying the operation, or a `callable` (modelling a non-uniform
        probability law for example) taking no input and returning a boolean.
    magnitude : object, optional
        Always ignored, exists for compatibility.
    mag_range : object, optional
        Always ignored, exists for compatibility.
    random_state: int | numpy.random.Generator, optional
        Seed to be used to instantiate numpy random number generator instance.
        Used to decide whether or not to transform given the probability
        argument. Defaults to None.
    """

    def __init__(
        self,
        probability,
        magnitude=None,
        mag_range=None,
        random_state=None
    ):
        super().__init__(
            operation=sign_flip,
            probability=probability,
            random_state=random_state
        )


class DownsamplingShift(Transform):
    """ Downsamples and offsets features matrix with a given probability

    Augmentation proposed in [1]_

    DEPRECATED

    Parameters
    ----------
    probability : object, optional
        Always ignored, exists for compatibility.
    magnitude : object, optional
        Always ignored, exists for compatibility.
    mag_range : object, optional
        Always ignored, exists for compatibility.
    factor: int, optional
        Factor by which X should be downsampled. For example, if factor is 2,
        only every other column of X will be kept. Defaults to 2.
    offset: int, optional
        Offset (in number of columns) to be used to time-shift the data.
        Offset needs to be less than factor.
        When downsampling by a factor N, you have N different offsets possible.
        If not value is passed to ofset, it is randomly relected between 0 and
        factor-1.
    random_state: int | numpy.random.Generator, optional
        Seed to be used to instantiate numpy random number generator instance.
        Used to decide whether or not to transform given the probability
        argument and to sample the offset when omitted. Defaults to None.

    References
    ----------
    .. [1] Frydenlund, A., & Rudzicz, F. (2015). Emotional Affect Estimation
        Using Video and EEG Data in Deep Neural Networks. Proceedings of
        Canadian Conference on Artificial Intelligence 273-280.

    """

    def __init__(
        self,
        probability=None,
        magnitude=None,
        mag_range=None,
        factor=2,
        offset=None,
        random_state=None
    ):
        assert isinstance(factor, int), "factor has to be an int"
        kwargs = {'random_state': random_state}
        if offset is None:
            self.rng = check_random_state(random_state)
            offset = self.rng.choice(np.arange(factor))
            kwargs = {'random_state': self.rng}
        else:
            assert isinstance(offset, int), "offset has to be an int"
            assert offset < factor, (
                "offset needs to be less than factor. When down sampling"
                " by a factor N, you have N different offsets possible."
            )

        super().__init__(
            operation=downsample_shift_from_arrays,
            probability=1.0,  # always applied as it changes the shape of X
            factor=factor,
            offset=offset,
            **kwargs
        )


class FTSurrogate(Transform):
    """ FT surrogate augmentation of a single EEG channel, as proposed in [1]_

    Parameters
    ----------
    probability: float | callable
        Either a float between 0 and 1 defining the uniform probability of
        applying the operation, or a `callable` (modelling a non-uniform
        probability law for example) taking no input and returning a boolean.
    magnitude : object, optional
        Float between 0 and 1 setting the range over which the phase
        pertubation is uniformly sampled: [0, `magnitude` * 2 * `pi`]. Defaults
        to 1.
    mag_range : object, optional
        Always ignored, exists for compatibility.
    random_state: int | numpy.random.Generator, optional
        Seed to be used to instantiate numpy random number generator instance.
        Used to decide whether or not to transform given the probability
        argument. Defaults to None.

    References
    ----------
    .. [1] Schwabedal, J. T., Snyder, J. C., Cakmak, A., Nemati, S., &
       Clifford, G. D. (2018). Addressing Class Imbalance in Classification
       Problems of Noisy Signals by using Fourier Transform Surrogates. arXiv
       preprint arXiv:1806.08675.
    """

    def __init__(
        self,
        probability,
        magnitude=1,
        mag_range=None,
        random_state=None
    ):
        if magnitude is None:
            magnitude = 1
        super().__init__(
            operation=fft_surrogate,
            probability=probability,
            magnitude=magnitude,
            random_state=random_state
        )


class MissingChannels(Transform):
    """ Randomly set channels to flat signal

    Part of the CMSAugment policy proposed in [1]_

    Parameters
    ----------
    probability: float | callable
        Either a float between 0 and 1 defining the
        uniform probability of applying the operation, or a `callable`
        (modelling a non-uniform probability law for example) taking no input
        and returning a boolean.
    magnitude: float | None, optional
        Float between 0 and 1 setting the fraction of channels to zero-out.
        If ommited, a random number of channels is (uniformly) sampled.
        Defaults to None.
    mag_range : object, optional
        Always ignored, exists for compatibility.
    random_state: int | numpy.random.Generator, optional
        Seed to be used to instantiate numpy random number generator instance.
        Used to decide whether or not to transform given the probability
        argument and to sample channels to erase. Defaults to None.

    References
    ----------
    .. [1] Saeed, A., Grangier, D., Pietquin, O., & Zeghidour, N. (2020).
       Learning from Heterogeneous EEG Signals with Differentiable Channel
       Reordering. arXiv preprint arXiv:2010.13694.
    """

    def __init__(
        self,
        probability,
        magnitude=0.2,
        mag_range=None,
        random_state=None
    ):
        super().__init__(
            operation=channel_dropout,
            probability=probability,
            magnitude=magnitude,
            random_state=random_state
        )


class ShuffleChannels(Transform):
    """ Randomly shuffle channels in EEG data matrix

    Part of the CMSAugment policy proposed in [1]_

    Parameters
    ----------
    probability: float | callable
        Either a float between 0 and 1 defining the
        uniform probability of applying the operation, or a `callable`
        (modelling a non-uniform probability law for example) taking no input
        and returning a boolean.
    magnitude: float | None, optional
        Float between 0 and 1 setting the fraction of channels to permute.
        If ommited, a random number of channels is (uniformly) sampled.
        Defaults to None.
    mag_range : object, optional
        Always ignored, exists for compatibility.
    random_state: int | numpy.random.Generator, optional
        Seed to be used to instantiate numpy random number generator instance.
        Used to decide whether or not to transform given the probability
        argument. Defaults to None.

    References
    ----------
    .. [1] Saeed, A., Grangier, D., Pietquin, O., & Zeghidour, N. (2020).
       Learning from Heterogeneous EEG Signals with Differentiable Channel
       Reordering. arXiv preprint arXiv:2010.13694.
    """

    def __init__(
        self,
        probability,
        magnitude=0.2,
        mag_range=None,
        random_state=None
    ):
        super().__init__(
            operation=channel_shuffle,
            probability=probability,
            magnitude=magnitude,
            random_state=random_state
        )


class GaussianNoise(Transform):
    """Randomly add white noise to all channels

    Suggested e.g. in [1]_, [2]_ and [3]_

    Parameters
    ----------
    probability : float | callable
        Either a float between 0 and 1 defining the
        uniform probability of applying the operation, or a `callable`
        (modelling a non-uniform probability law for example) taking no input
        and returning a boolean.
    magnitude : float | None, optional
        Float between 0 and 1 encoding the standard deviation to use for the
        additive noise:
        ```
        std = magnitude * mag_range[1] + (1 - magnitude) * mag_range[0]
        ```
        Defaults to None (ignored).
    mag_range : tuple of two floats | None, optional
        Std range when set using the magnitude (see `magnitude`).
        If omitted, the range (0, 0.2) will be used.
    std : float, optional
        Standard deviation to use for the additive noise. Will be ignored if
        magnitude is not set to None. Defaults to 0.1.
    random_state: int | numpy.random.Generator, optional
        Seed to be used to instantiate numpy random number generator instance.
        Defaults to None.

    References
    ----------
    .. [1] Wang, F., Zhong, S. H., Peng, J., Jiang, J., & Liu, Y. (2018). Data
       augmentation for eeg-based emotion recognition with deep convolutional
       neural networks. In International Conference on Multimedia Modeling
       (pp. 82-93).
    .. [2] Cheng, J. Y., Goh, H., Dogrusoz, K., Tuzel, O., & Azemi, E. (2020).
       Subject-aware contrastive learning for biosignals. arXiv preprint
       arXiv:2007.04871.
    .. [3] Mohsenvand, M. N., Izadi, M. R., & Maes, P. (2020). Contrastive
       Representation Learning for Electroencephalogram Classification. In
       Machine Learning for Health (pp. 238-253). PMLR.

    """

    def __init__(
        self,
        probability,
        magnitude=None,
        mag_range=(0, 0.2),
        std=0.1,
        random_state=None
    ):
        # override std value when a magnitude is passed
        if magnitude is not None:
            min_val, max_val = mag_range
            self.std = magnitude * max_val + (1 - magnitude) * min_val
        else:
            self.std = std

        super().__init__(
            operation=add_gaussian_noise,
            probability=probability,
            magnitude=magnitude,
            mag_range=mag_range,
            std=self.std,
            random_state=random_state,
        )


class ChannelSymmetry(Transform):
    """Permute EEG channels inverting left and right-side sensors

    Suggested e.g. in [1]_

    Parameters
    ----------
    probability : float | callable
        Either a float between 0 and 1 defining the
        uniform probability of applying the operation, or a `callable`
        (modelling a non-uniform probability law for example) taking no input
        and returning a boolean.
    ordered_ch_names : list
        Ordered list of strings containing the names (in 10-20
        nomenclature) of the EEG channels that will be transformed. The
        first name should correspond the data in the first row of X, the
        second name in the second row and so on.
    magnitude : object, optional
        Always ignored, exists for compatibility.
    mag_range : object, optional
        Always ignored, exists for compatibility.
    random_state: int | numpy.random.Generator, optional
        Seed to be used to instantiate numpy random number generator instance.
        Used to decide whether or not to transform given the probability
        argument. Defaults to None.

    References
    ----------
    .. [1] Deiss, O., Biswal, S., Jin, J., Sun, H., Westover, M. B., & Sun, J.
       (2018). HAMLET: interpretable human and machine co-learning technique.
       arXiv preprint arXiv:1803.09702.

    """

    def __init__(
        self,
        probability,
        ordered_ch_names,
        magnitude=None,
        mag_range=None,
        random_state=None
    ):
        assert (
            isinstance(ordered_ch_names, list) and
            all(isinstance(ch, str) for ch in ordered_ch_names)
        ), "ordered_ch_names should be a list of str."

        permutation = list()
        for idx, ch_name in enumerate(ordered_ch_names):
            new_position = idx
            # Find digits in channel name (assuming 10-20 system)
            d = ''.join(list(filter(str.isdigit, ch_name)))
            if len(d) > 0:
                d = int(d)
                if d % 2 == 0:  # pair/right electrodes
                    sym = d - 1
                else:  # odd/left electrodes
                    sym = d + 1
                new_channel = ch_name.replace(str(d), str(sym))
                if new_channel in ordered_ch_names:
                    new_position = ordered_ch_names.index(new_channel)
            permutation.append(new_position)

        super().__init__(
            operation=permute_channels,
            probability=probability,
            permutation=permutation,
            random_state=random_state,
        )


class TimeMask(Transform):
    """Replace part of all channels by zeros

    Suggested e.g. in [1]_ and [2]_
    Similar to the time variant of SpecAugment for speech signals [3]_

    Parameters
    ----------
    probability : float | callable
        Either a float between 0 and 1 defining the
        uniform probability of applying the operation, or a `callable`
        (modelling a non-uniform probability law for example) taking no input
        and returning a boolean.
    magnitude : float | None, optional
        Float between 0 and 1 encoding the number of consecutive samples within
        `mag_range` to set to 0:
        ```
        mask_len_samples = int(round(magnitude * mag_range[1] +
            (1 - magnitude) * mag_range[0]))
        ```
        Defaults to None (ignored).
    mag_range : tuple of two floats | None, optional
        Range of possible values for `mask_len_samples` settable using the
        magnitude (see `magnitude`). If omitted, the range (0, 100) samples
        will be used.
    mask_len_samples : int, optional
        Number of consecutive samples to zero out. Will be ignored if
        magnitude is not set to None. Defaults to 100.
    random_state: int | numpy.random.Generator, optional
        Seed to be used to instantiate numpy random number generator instance.
        Defaults to None.

    References
    ----------
    .. [1] Cheng, J. Y., Goh, H., Dogrusoz, K., Tuzel, O., & Azemi, E. (2020).
       Subject-aware contrastive learning for biosignals. arXiv preprint
       arXiv:2007.04871.
    .. [2] Mohsenvand, M. N., Izadi, M. R., & Maes, P. (2020). Contrastive
       Representation Learning for Electroencephalogram Classification. In
       Machine Learning for Health (pp. 238-253). PMLR.
    .. [3] Park, D.S., Chan, W., Zhang, Y., Chiu, C., Zoph, B., Cubuk, E.D.,
       Le, Q.V. (2019) SpecAugment: A Simple Data Augmentation Method for
       Automatic Speech Recognition. Proc. Interspeech 2019, 2613-2617

    """

    def __init__(
        self,
        probability,
        magnitude=None,
        mag_range=(0, 100),
        mask_len_samples=100,
        random_state=None
    ):
        assert (
            isinstance(mask_len_samples, int) and
            mask_len_samples > 0
        ), "mask_len_samples has to be a positive integer"

        # override mask_len_samples value when a magnitude is passed
        if magnitude is not None:
            min_val, max_val = mag_range
            self.mask_len_samples = int(round(magnitude * max_val +
                                              (1 - magnitude) * min_val))
        else:
            self.mask_len_samples = mask_len_samples

        # Handle case of mask of length 0 (possible through magnitude)
        if self.mask_len_samples > 0:
            operation = random_time_mask
        else:
            operation = identity

        super().__init__(
            probability=probability,
            operation=operation,
            magnitude=magnitude,
            mag_range=mag_range,
            mask_len_samples=self.mask_len_samples,
            random_state=random_state,
        )


class BandstopFilter(Transform):
    """Applies a stopband filter with desired bandwidth at a randomly selected
    frequency position between 0 and `max_freq`.

    Suggested e.g. in [1]_ and [2]_
    Similar to the frequency variant of SpecAugment for speech signals [3]_

    Parameters
    ----------
    probability : float | callable
        Either a float between 0 and 1 defining the
        uniform probability of applying the operation, or a `callable`
        (modelling a non-uniform probability law for example) taking no input
        and returning a boolean.
    magnitude : float | None, optional
        Float between 0 and 1 encoding the `bandwidth` parameter:
        ```
        bandwidth = magnitude * mag_range[1] + (1 - magnitude) * mag_range[0]
        ```
        Defaults to None (ignored).
    mag_range : tuple of two floats | None, optional
        Range of possible values for `bandwidth` settable using the magnitude
        (see `magnitude`). If omitted, the range (0, 2 Hz) will be used.
    bandwidth : float, optional
        Bandwidth of the filter, i.e. distance between the low and high cut
        frequencies. Will be ignored if magnitude is not set to None. Defaults
        to 1Hz.
    sfreq : float, optional
        Sampling frequency of the signals to be filtered. Defaults to 100 Hz.
    max_freq : float | None, optional
        Maximal admissible frequency. The low cut frequency will be sampled so
        that the corresponding high cut frequency + transition are below
        `max_freq`. If omitted or `None`, will default to the Nyquist frequency
        (`sfreq / 2`).
    random_state: int | numpy.random.Generator, optional
        Seed to be used to instantiate numpy random number generator instance.
        Defaults to None.

    References
    ----------
    .. [1] Cheng, J. Y., Goh, H., Dogrusoz, K., Tuzel, O., & Azemi, E. (2020).
       Subject-aware contrastive learning for biosignals. arXiv preprint
       arXiv:2007.04871.
    .. [2] Mohsenvand, M. N., Izadi, M. R., & Maes, P. (2020). Contrastive
       Representation Learning for Electroencephalogram Classification. In
       Machine Learning for Health (pp. 238-253). PMLR.
    .. [3] Park, D.S., Chan, W., Zhang, Y., Chiu, C., Zoph, B., Cubuk, E.D.,
       Le, Q.V. (2019) SpecAugment: A Simple Data Augmentation Method for
       Automatic Speech Recognition. Proc. Interspeech 2019, 2613-2617
    """

    def __init__(
        self,
        probability,
        magnitude=None,
        mag_range=(0, 2),
        bandwidth=1,
        sfreq=100,
        max_freq=50,
        random_state=None
    ):
        assert isinstance(bandwidth, Real) and bandwidth >= 0,\
            "bandwidth should be a non-negative float."
        assert isinstance(sfreq, Real) and sfreq > 0,\
            "sfreq should be a positive float."
        assert isinstance(max_freq, Real) and max_freq > 0,\
            "max_freq should be a positive float."
        nyq = sfreq / 2
        if max_freq is None or max_freq > nyq:
            max_freq = nyq
        assert bandwidth < max_freq,\
            f"`bandwidth` needs to be smaller than `max_freq`={max_freq}"

        # override bandwidth value when a magnitude is passed
        self.sfreq = sfreq
        self.max_freq = max_freq
        if magnitude is not None:
            min_val, max_val = mag_range
            self.bandwidth = magnitude * max_val + (1 - magnitude) * min_val
        else:
            self.bandwidth = bandwidth

        # Handle case of band of length 0
        if self.bandwidth == 0:
            operation = identity
        else:
            operation = random_bandstop

        super().__init__(
            probability=probability,
            operation=operation,
            magnitude=magnitude,
            mag_range=mag_range,
            bandwidth=self.bandwidth,
            max_freq=self.max_freq,
            sfreq=self.sfreq,
            random_state=random_state,
        )


class FrequencyShift(Transform):
    """Add a random shift in the frequency domain to all channels.

    Parameters
    ----------
    probability : float | callable
        Either a float between 0 and 1 defining the
        uniform probability of applying the operation, or a `callable`
        (modelling a non-uniform probability law for example) taking no input
        and returning a boolean.
    magnitude : float | None, optional
        Float between 0 and 1 encoding the `max_shift` parameter:
        ```
        max_shift = magnitude * mag_range[1] + (1 - magnitude) * mag_range[0]
        ```
        Defaults to None (ignored).
    mag_range : tuple of two floats | None, optional
        Range of possible values for `max_shift` settable using the magnitude
        (see `magnitude`). If omitted the range (0 Hz, 5 Hz) will be used.
    max_shift : float, optional
        Random frequency shifts will be samples uniformly in the interval
        `[0, max_shift]`. Defaults to 2 Hz.
    sfreq : float, optional
        Sampling frequency of the signals to be transformed. Default to 100 Hz.
    random_state: int | numpy.random.Generator, optional
        Seed to be used to instantiate numpy random number generator instance.
        Defaults to None.
    """

    def __init__(
        self,
        probability,
        magnitude=None,
        mag_range=(0, 5),
        max_shift=0.5,
        sfreq=100,
        random_state=None
    ):
        assert isinstance(max_shift, Real) and max_shift >= 0,\
            "max_shift should be a non-negative float."
        assert isinstance(sfreq, Real) and sfreq > 0,\
            "sfreq should be a positive float."
        self.sfreq = sfreq

        # override max_shift value when a magnitude is passed
        if magnitude is not None:
            min_val, max_val = mag_range
            self.max_shift = magnitude * max_val + (1 - magnitude) * min_val
        else:
            self.max_shift = max_shift

        # Handle case of no shift
        if self.max_shift == 0:
            operation = identity
        else:
            operation = freq_shift

        super().__init__(
            probability=probability,
            operation=operation,
            magnitude=magnitude,
            mag_range=mag_range,
            max_shift=self.max_shift,
            sfreq=self.sfreq,
            random_state=random_state,
        )


class RandomSensorsRotation(Transform):
    """Interpolates EEG signals over sensors rotated around the desired axis
    with an angle sampled uniformly between 0 and `max_degree`.

    Suggested in [1]_

    Parameters
    ----------
    probability : float | callable
        Either a float between 0 and 1 defining the
        uniform probability of applying the operation, or a `callable`
        (modelling a non-uniform probability law for example) taking no input
        and returning a boolean.
    sensors_positions_matrix : numpy.ndarray
        Matrix giving the positions of each sensor in a 3D cartesian coordiante
        systemsof. Should have shape (3, n_channels), where n_channels is the
        number of channels. Standard 10-20 positions can be obtained from
        `mne` through:
        ```
        ten_twenty_montage = mne.channels.make_standard_montage(
            'standard_1020'
        ).get_positions()['ch_pos']
        ```
    magnitude : float | None, optional
        Float between 0 and 1 encoding the `max_degree` parameter:
        ```
        max_degree = magnitude * mag_range[1] + (1 - magnitude) * mag_range[0]
        ```
        Defaults to None (ignored).
    mag_range : tuple of two floats | None, optional
        Range of possible values for `max_degree` settable using the magnitude
        (see `magnitude`). If omitted, the range (0, 30 degrees) will be used.
    axis : 'x' | 'y' | 'z', optional
        Axis around which to rotate. Defaults to 'z'.
    max_degree : float, optional
        Maximum rotation. Rotation angles will be sampled between `-max_degree`
        and `max_degree`. Defaults to 15 degrees.
    spherical_splines : bool, optional
        Whether to use spherical splines for the interpolation or not. When
        `False`, standard scipy.interpolate.Rbf (with quadratic kernel) will be
        used (as in the original paper). Defaults to True.
    random_state: int | numpy.random.Generator, optional
        Seed to be used to instantiate numpy random number generator instance.
        Defaults to None.

    References
    ----------
    .. [1] Krell, M. M., & Kim, S. K. (2017). Rotational data augmentation for
       electroencephalographic data. In 2017 39th Annual International
       Conference of the IEEE Engineering in Medicine and Biology Society
       (EMBC) (pp. 471-474).
    """

    def __init__(
        self,
        probability,
        sensors_positions_matrix,
        magnitude=None,
        mag_range=(0, 30),
        axis='z',
        max_degrees=15,
        spherical_splines=True,
        random_state=None
    ):
        if isinstance(sensors_positions_matrix, (np.ndarray, list)):
            sensors_positions_matrix = torch.as_tensor(
                sensors_positions_matrix
            )
        assert isinstance(sensors_positions_matrix, torch.Tensor),\
            "sensors_positions should be an Tensor"
        assert isinstance(max_degrees, Real) and max_degrees >= 0,\
            "max_degrees should be non-negative float."
        assert isinstance(axis, str) and axis in ['x', 'y', 'z'],\
            "axis can be either x, y or z."
        assert sensors_positions_matrix.shape[0] == 3,\
            "sensors_positions_matrix shape should be 3 x n_channels."
        assert isinstance(spherical_splines, bool),\
            "spherical_splines should be a boolean"
        self.sensors_positions_matrix = sensors_positions_matrix
        self.axis = axis
        self.spherical_splines = spherical_splines

        # override max_degrees value when a magnitude is passed
        if magnitude is not None:
            min_val, max_val = mag_range
            self.max_degrees = magnitude * max_val + (1 - magnitude) * min_val
        else:
            self.max_degrees = max_degrees

        # Handle case of 0 degrees
        if self.max_degrees == 0:
            operation = identity
        else:
            operation = random_rotation

        super().__init__(
            probability=probability,
            operation=operation,
            magnitude=magnitude,
            mag_range=mag_range,
            axis=self.axis,
            max_degrees=self.max_degrees,
            sensors_positions_matrix=self.sensors_positions_matrix,
            spherical_splines=self.spherical_splines,
            random_state=random_state
        )


class RandomZRotation(RandomSensorsRotation):
    """Interpolates EEG signals over sensors rotated around the Z axis
    with an angle sampled uniformly between 0 and `max_degree`.

    Suggested in [1]_

    Parameters
    ----------
    probability : float | callable
        Either a float between 0 and 1 defining the
        uniform probability of applying the operation, or a `callable`
        (modelling a non-uniform probability law for example) taking no input
        and returning a boolean.
    ordered_ch_names : list
        List of strings representing the channels of the montage considered.
        Has to be in standard 10-20 style. The order has to be consistent with
        the order of channels in the input matrices that will be fed to the
        transform. This channel will be used to compute approximate sensors
        positions from a standard 10-20 montage.
    magnitude : float | None, optional
        Float between 0 and 1 encoding the `max_degree` parameter:
        ```
        max_degree = magnitude * mag_range[1] + (1 - magnitude) * mag_range[0]
        ```
        Defaults to None (ignored).
    mag_range : tuple of two floats | None, optional
        Range of possible values for `max_degree` settable using the magnitude
        (see `magnitude`). If omitted, the range (0, 30 degrees) will be used.
    axis : 'x' | 'y' | 'z', optional
        Axis around which to rotate. Defaults to 'z'.
    max_degree : float, optional
        Maximum rotation. Rotation angles will be sampled between `-max_degree`
        and `max_degree`. Defaults to 15 degrees.
    spherical_splines : bool, optional
        Whether to use spherical splines for the interpolation or not. When
        `False`, standard scipy.interpolate.Rbf (with quadratic kernel) will be
        used (as in the original paper). Defaults to True.
    random_state: int | numpy.random.Generator, optional
        Seed to be used to instantiate numpy random number generator instance.
        Defaults to None.

    References
    ----------
    .. [1] Krell, M. M., & Kim, S. K. (2017). Rotational data augmentation for
       electroencephalographic data. In 2017 39th Annual International
       Conference of the IEEE Engineering in Medicine and Biology Society
       (EMBC) (pp. 471-474).
    """

    def __init__(
        self,
        probability,
        ordered_ch_names,
        magnitude=None,
        mag_range=(0, 30),
        max_degrees=15,
        random_state=None
    ):
        sensors_positions_matrix = torch.as_tensor(
            get_standard_10_20_positions(ordered_ch_names=ordered_ch_names)
        )
        super().__init__(
            probability=probability,
            sensors_positions_matrix=sensors_positions_matrix,
            magnitude=magnitude,
            mag_range=mag_range,
            axis='z',
            max_degrees=max_degrees,
            random_state=random_state
        )


class RandomYRotation(RandomSensorsRotation):
    """Interpolates EEG signals over sensors rotated around the Z axis
    with an angle sampled uniformly between 0 and `max_degree`.

    Suggested in [1]_

    Parameters
    ----------
    probability : float | callable
        Either a float between 0 and 1 defining the
        uniform probability of applying the operation, or a `callable`
        (modelling a non-uniform probability law for example) taking no input
        and returning a boolean.
    ordered_ch_names : list
        List of strings representing the channels of the montage considered.
        Has to be in standard 10-20 style. The order has to be consistent with
        the order of channels in the input matrices that will be fed to the
        transform. This channel will be used to compute approximate sensors
        positions from a standard 10-20 montage.
    magnitude : float | None, optional
        Float between 0 and 1 encoding the `max_degree` parameter:
        ```
        max_degree = magnitude * mag_range[1] + (1 - magnitude) * mag_range[0]
        ```
        Defaults to None (ignored).
    mag_range : tuple of two floats | None, optional
        Range of possible values for `max_degree` settable using the magnitude
        (see `magnitude`). If omitted, the range (0, 30 degrees) will be used.
    axis : 'x' | 'y' | 'z', optional
        Axis around which to rotate. Defaults to 'z'.
    max_degree : float, optional
        Maximum rotation. Rotation angles will be sampled between `-max_degree`
        and `max_degree`. Defaults to 15 degrees.
    spherical_splines : bool, optional
        Whether to use spherical splines for the interpolation or not. When
        `False`, standard scipy.interpolate.Rbf (with quadratic kernel) will be
        used (as in the original paper). Defaults to True.
    random_state: int | numpy.random.Generator, optional
        Seed to be used to instantiate numpy random number generator instance.
        Defaults to None.

    References
    ----------
    .. [1] Krell, M. M., & Kim, S. K. (2017). Rotational data augmentation for
       electroencephalographic data. In 2017 39th Annual International
       Conference of the IEEE Engineering in Medicine and Biology Society
       (EMBC) (pp. 471-474).
    """

    def __init__(
        self,
        probability,
        ordered_ch_names,
        magnitude=None,
        mag_range=(0, 30),
        max_degrees=15,
        random_state=None
    ):
        sensors_positions_matrix = torch.as_tensor(
            get_standard_10_20_positions(ordered_ch_names=ordered_ch_names)
        )
        super().__init__(
            probability=probability,
            sensors_positions_matrix=sensors_positions_matrix,
            magnitude=magnitude,
            mag_range=mag_range,
            axis='y',
            max_degrees=max_degrees,
            random_state=random_state
        )


class RandomXRotation(RandomSensorsRotation):
    """Interpolates EEG signals over sensors rotated around the Z axis
    with an angle sampled uniformly between 0 and `max_degree`.

    Suggested in [1]_

    Parameters
    ----------
    probability : float | callable
        Either a float between 0 and 1 defining the
        uniform probability of applying the operation, or a `callable`
        (modelling a non-uniform probability law for example) taking no input
        and returning a boolean.
    ordered_ch_names : list
        List of strings representing the channels of the montage considered.
        Has to be in standard 10-20 style. The order has to be consistent with
        the order of channels in the input matrices that will be fed to the
        transform. This channel will be used to compute approximate sensors
        positions from a standard 10-20 montage.
    magnitude : float | None, optional
        Float between 0 and 1 encoding the `max_degree` parameter:
        ```
        max_degree = magnitude * mag_range[1] + (1 - magnitude) * mag_range[0]
        ```
        Defaults to None (ignored).
    mag_range : tuple of two floats | None, optional
        Range of possible values for `max_degree` settable using the magnitude
        (see `magnitude`). If omitted, the range (0, 30 degrees) will be used.
    axis : 'x' | 'y' | 'z', optional
        Axis around which to rotate. Defaults to 'z'.
    max_degree : float, optional
        Maximum rotation. Rotation angles will be sampled between `-max_degree`
        and `max_degree`. Defaults to 15 degrees.
    spherical_splines : bool, optional
        Whether to use spherical splines for the interpolation or not. When
        `False`, standard scipy.interpolate.Rbf (with quadratic kernel) will be
        used (as in the original paper). Defaults to True.
    random_state: int | numpy.random.Generator, optional
        Seed to be used to instantiate numpy random number generator instance.
        Defaults to None.

    References
    ----------
    .. [1] Krell, M. M., & Kim, S. K. (2017). Rotational data augmentation for
       electroencephalographic data. In 2017 39th Annual International
       Conference of the IEEE Engineering in Medicine and Biology Society
       (EMBC) (pp. 471-474).
    """

    def __init__(
        self,
        probability,
        ordered_ch_names,
        magnitude=None,
        mag_range=(0, 30),
        max_degrees=15,
        random_state=None
    ):
        sensors_positions_matrix = torch.as_tensor(
            get_standard_10_20_positions(ordered_ch_names=ordered_ch_names)
        )
        super().__init__(
            probability=probability,
            sensors_positions_matrix=sensors_positions_matrix,
            magnitude=magnitude,
            mag_range=mag_range,
            axis='x',
            max_degrees=max_degrees,
            random_state=random_state
        )
