# Authors: Cédric Rommel <cedric.rommel@inria.fr>
#          Alexandre Gramfort <alexandre.gramfort@inria.fr>
#
# License: BSD (3-clause)

import warnings
from numbers import Real

import numpy as np
import torch
from mne.channels import make_standard_montage

from .base import Transform
from .functional import bandstop_filter
from .functional import channels_dropout
from .functional import channels_permute
from .functional import channels_shuffle
from .functional import frequency_shift
from .functional import ft_surrogate
from .functional import gaussian_noise
from .functional import mixup
from .functional import sensors_rotation
from .functional import sign_flip
from .functional import smooth_time_mask
from .functional import time_reverse


class TimeReverse(Transform):
    """Flip the time axis of each input with a given probability.

    Parameters
    ----------
    probability : float
        Float setting the probability of applying the operation.
    random_state: int | numpy.random.Generator, optional
        Seed to be used to instantiate numpy random number generator instance.
        Used to decide whether or not to transform given the probability
        argument. Defaults to None.
    """
    operation = staticmethod(time_reverse)

    def __init__(
        self,
        probability,
        random_state=None
    ):
        super().__init__(
            probability=probability,
            random_state=random_state
        )


class SignFlip(Transform):
    """Flip the sign axis of each input with a given probability.

    Parameters
    ----------
    probability : float
        Float setting the probability of applying the operation.
    random_state: int | numpy.random.Generator, optional
        Seed to be used to instantiate numpy random number generator instance.
        Used to decide whether or not to transform given the probability
        argument. Defaults to None.
    """
    operation = staticmethod(sign_flip)

    def __init__(
        self,
        probability,
        random_state=None
    ):
        super().__init__(
            probability=probability,
            random_state=random_state
        )


class FTSurrogate(Transform):
    """FT surrogate augmentation of a single EEG channel, as proposed in [1]_.

    Parameters
    ----------
    probability: float
        Float setting the probability of applying the operation.
    phase_noise_magnitude : float | torch.Tensor, optional
        Float between 0 and 1 setting the range over which the phase
        pertubation is uniformly sampled:
        ``[0, phase_noise_magnitude * 2 * pi]``. Defaults to 1.
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
    operation = staticmethod(ft_surrogate)

    def __init__(
        self,
        probability,
        phase_noise_magnitude=1,
        random_state=None
    ):
        super().__init__(
            probability=probability,
            random_state=random_state
        )
        assert isinstance(phase_noise_magnitude, (float, int, torch.Tensor)),\
            "phase_noise_magnitude should be a float."
        assert 0 <= phase_noise_magnitude <= 1,\
            "phase_noise_magnitude should be between 0 and 1."
        self.phase_noise_magnitude = phase_noise_magnitude

    def get_params(self, *batch):
        """Return transform parameters.

        Parameters
        ----------
        X : tensor.Tensor
            The data.
        y : tensor.Tensor
            The labels.

        Returns
        -------
        params : dict
            Contains:

            * phase_noise_magnitude : float
                The magnitude of the transformation.
            * random_state : numpy.random.Generator
                The generator to use.
        """
        return {
            "phase_noise_magnitude": self.phase_noise_magnitude,
            "random_state": self.rng,
        }


class ChannelsDropout(Transform):
    """Randomly set channels to flat signal.

    Part of the CMSAugment policy proposed in [1]_

    Parameters
    ----------
    probability: float
        Float setting the probability of applying the operation.
    proba_drop: float | None, optional
        Float between 0 and 1 setting the probability of dropping each channel.
        Defaults to 0.2.
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
    operation = staticmethod(channels_dropout)

    def __init__(
        self,
        probability,
        p_drop=0.2,
        random_state=None
    ):
        super().__init__(
            probability=probability,
            random_state=random_state
        )
        self.p_drop = p_drop

    def get_params(self, *batch):
        """Return transform parameters.

        Parameters
        ----------
        X : tensor.Tensor
            The data.
        y : tensor.Tensor
            The labels.

        Returns
        -------
        params : dict
            Contains

            * p_drop : float
                Float between 0 and 1 setting the probability of dropping each
                channel.
            * random_state : numpy.random.Generator
                The generator to use.
        """
        return {
            "p_drop": self.p_drop,
            "random_state": self.rng,
        }


class ChannelsShuffle(Transform):
    """Randomly shuffle channels in EEG data matrix.

    Part of the CMSAugment policy proposed in [1]_

    Parameters
    ----------
    probability: float
        Float setting the probability of applying the operation.
    p_shuffle: float | None, optional
        Float between 0 and 1 setting the probability of including the channel
        in the set of permuted channels. Defaults to 0.2.
    random_state: int | numpy.random.Generator, optional
        Seed to be used to instantiate numpy random number generator instance.
        Used to decide whether or not to transform given the probability
        argument, to sample which channels to shuffle and to carry the shuffle.
        Defaults to None.

    References
    ----------
    .. [1] Saeed, A., Grangier, D., Pietquin, O., & Zeghidour, N. (2020).
       Learning from Heterogeneous EEG Signals with Differentiable Channel
       Reordering. arXiv preprint arXiv:2010.13694.
    """
    operation = staticmethod(channels_shuffle)

    def __init__(
        self,
        probability,
        p_shuffle=0.2,
        random_state=None
    ):
        super().__init__(
            probability=probability,
            random_state=random_state
        )
        self.p_shuffle = p_shuffle

    def get_params(self, *batch):
        """Return transform parameters.

        Parameters
        ----------
        X : tensor.Tensor
            The data.
        y : tensor.Tensor
            The labels.

        Returns
        -------
        params : dict
            Contains

            * p_shuffle : float
                Float between 0 and 1 setting the probability of including the
                channel in the set of permuted channels.
            * random_state : numpy.random.Generator
                The generator to use.
        """
        return {
            "p_shuffle": self.p_shuffle,
            "random_state": self.rng,
        }


class GaussianNoise(Transform):
    """Randomly add white noise to all channels.

    Suggested e.g. in [1]_, [2]_ and [3]_

    Parameters
    ----------
    probability : float
        Float setting the probability of applying the operation.
    std : float, optional
        Standard deviation to use for the additive noise. Defaults to 0.1.
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
    operation = staticmethod(gaussian_noise)

    def __init__(
        self,
        probability,
        std=0.1,
        random_state=None
    ):
        super().__init__(
            probability=probability,
            random_state=random_state,
        )
        self.std = std

    def get_params(self, *batch):
        """Return transform parameters.

        Parameters
        ----------
        X : tensor.Tensor
            The data.
        y : tensor.Tensor
            The labels.

        Returns
        -------
        params : dict
            Contains

            * std : float
                Standard deviation to use for the additive noise.
            * random_state : numpy.random.Generator
                The generator to use.
        """
        return {
            "std": self.std,
            "random_state": self.rng,
        }


class ChannelsSymmetry(Transform):
    """Permute EEG channels inverting left and right-side sensors.

    Suggested e.g. in [1]_

    Parameters
    ----------
    probability : float
        Float setting the probability of applying the operation.
    ordered_ch_names : list
        Ordered list of strings containing the names (in 10-20
        nomenclature) of the EEG channels that will be transformed. The
        first name should correspond the data in the first row of X, the
        second name in the second row and so on.
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
    operation = staticmethod(channels_permute)

    def __init__(
        self,
        probability,
        ordered_ch_names,
        random_state=None
    ):
        super().__init__(
            probability=probability,
            random_state=random_state,
        )
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
        self.permutation = permutation

    def get_params(self, *batch):
        """Return transform parameters.

        Parameters
        ----------
        X : tensor.Tensor
            The data.
        y : tensor.Tensor
            The labels.

        Returns
        -------
        params : dict
            Contains

            * permutation : float
                List of integers defining the new channels order.
        """
        return {"permutation": self.permutation}


class SmoothTimeMask(Transform):
    """Smoothly replace a randomly chosen contiguous part of all channels by
    zeros.

    Suggested e.g. in [1]_ and [2]_

    Parameters
    ----------
    probability : float
        Float setting the probability of applying the operation.
    mask_len_samples : int | torch.Tensor, optional
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
    """
    operation = staticmethod(smooth_time_mask)

    def __init__(
        self,
        probability,
        mask_len_samples=100,
        random_state=None
    ):
        super().__init__(
            probability=probability,
            random_state=random_state,
        )

        assert (
            isinstance(mask_len_samples, (int, torch.Tensor)) and
            mask_len_samples > 0
        ), "mask_len_samples has to be a positive integer"
        self.mask_len_samples = mask_len_samples

    def get_params(self, *batch):
        """Return transform parameters.

        Parameters
        ----------
        X : tensor.Tensor
            The data.
        y : tensor.Tensor
            The labels.

        Returns
        -------
        params : dict
            Contains two elements:

            * mask_start_per_sample : torch.tensor
                Tensor of integers containing the position (in last dimension)
                where to start masking the signal. Should have the same size as
                the first dimension of X (i.e. one start position per example
                in the batch).
            * mask_len_samples : int
                Number of consecutive samples to zero out.
        """
        if len(batch) == 0:
            return super().get_params(*batch)
        X = batch[0]

        seq_length = torch.as_tensor(X.shape[-1], device=X.device)
        mask_len_samples = self.mask_len_samples
        if isinstance(mask_len_samples, torch.Tensor):
            mask_len_samples = mask_len_samples.to(X.device)
        mask_start = torch.as_tensor(self.rng.uniform(
            low=0, high=1, size=X.shape[0],
        ), device=X.device) * (seq_length - mask_len_samples)
        return {
            "mask_start_per_sample": mask_start,
            "mask_len_samples": mask_len_samples,
        }


class BandstopFilter(Transform):
    """Apply a band-stop filter with desired bandwidth at a randomly selected
    frequency position between 0 and ``max_freq``.

    Suggested e.g. in [1]_ and [2]_

    Parameters
    ----------
    probability : float
        Float setting the probability of applying the operation.
    bandwidth : float
        Bandwidth of the filter, i.e. distance between the low and high cut
        frequencies.
    sfreq : float, optional
        Sampling frequency of the signals to be filtered. Defaults to 100 Hz.
    max_freq : float | None, optional
        Maximal admissible frequency. The low cut frequency will be sampled so
        that the corresponding high cut frequency + transition (=1Hz) are below
        ``max_freq``. If omitted or `None`, will default to the Nyquist
        frequency (``sfreq / 2``).
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
    """
    operation = staticmethod(bandstop_filter)

    def __init__(
        self,
        probability,
        sfreq,
        bandwidth=1,
        max_freq=None,
        random_state=None
    ):
        super().__init__(
            probability=probability,
            random_state=random_state,
        )
        assert isinstance(bandwidth, Real) and bandwidth >= 0,\
            "bandwidth should be a non-negative float."
        assert isinstance(sfreq, Real) and sfreq > 0,\
            "sfreq should be a positive float."
        if max_freq is not None:
            assert isinstance(max_freq, Real) and max_freq > 0,\
                "max_freq should be a positive float."
        nyq = sfreq / 2
        if max_freq is None or max_freq > nyq:
            max_freq = nyq
            warnings.warn(
                "You either passed None or a frequency greater than the"
                f" Nyquist frequency ({nyq} Hz)."
                f" Falling back to max_freq = {nyq}."
            )
        assert bandwidth < max_freq,\
            f"`bandwidth` needs to be smaller than max_freq={max_freq}"

        # override bandwidth value when a magnitude is passed
        self.sfreq = sfreq
        self.max_freq = max_freq
        self.bandwidth = bandwidth

    def get_params(self, *batch):
        """Return transform parameters.

        Parameters
        ----------
        X : tensor.Tensor
            The data.
        y : tensor.Tensor
            The labels.

        Returns
        -------
        params : dict
            Contains

            * sfreq : float
                Sampling frequency of the signals to be filtered.
            * bandwidth : float
                Bandwidth of the filter, i.e. distance between the low and high
                cut frequencies.
            * freqs_to_notch : array-like | None
                Array of floats of size ``(batch_size,)`` containing the center
                of the frequency band to filter out for each sample in the
                batch. Frequencies should be greater than
                ``bandwidth/2 + transition`` and lower than
                ``sfreq/2 - bandwidth/2 - transition`` (where
                ``transition = 1 Hz``).
        """
        if len(batch) == 0:
            return super().get_params(*batch)
        X = batch[0]

        # Prevents transitions from going below 0 and above max_freq
        notched_freqs = self.rng.uniform(
            low=1 + 2 * self.bandwidth,
            high=self.max_freq - 1 - 2 * self.bandwidth,
            size=X.shape[0]
        )
        return {
            "sfreq": self.sfreq,
            "bandwidth": self.bandwidth,
            "freqs_to_notch": notched_freqs,
        }


class FrequencyShift(Transform):
    """Add a random shift in the frequency domain to all channels.

    Note that here, the shift is the same for all channels of a single example.

    Parameters
    ----------
    probability : float
        Float setting the probability of applying the operation.
    sfreq : float
        Sampling frequency of the signals to be transformed.
    max_delta_freq : float | torch.Tensor, optional
        Maximum shift in Hz that can be sampled (in absolute value).
        Defaults to 2 (shift sampled between -2 and 2 Hz).
    random_state: int | numpy.random.Generator, optional
        Seed to be used to instantiate numpy random number generator instance.
        Defaults to None.
    """
    operation = staticmethod(frequency_shift)

    def __init__(
        self,
        probability,
        sfreq,
        max_delta_freq=2,
        random_state=None
    ):
        super().__init__(
            probability=probability,
            random_state=random_state,
        )
        assert isinstance(sfreq, Real) and sfreq > 0,\
            "sfreq should be a positive float."
        self.sfreq = sfreq

        self.max_delta_freq = max_delta_freq

    def get_params(self, *batch):
        """Return transform parameters.

        Parameters
        ----------
        X : tensor.Tensor
            The data.
        y : tensor.Tensor
            The labels.

        Returns
        -------
        params : dict
            Contains

            * delta_freq : float
                The amplitude of the frequency shift (in Hz).
            * sfreq : float
                Sampling frequency of the signals to be transformed.
        """
        if len(batch) == 0:
            return super().get_params(*batch)
        X = batch[0]

        u = torch.as_tensor(
            self.rng.uniform(size=X.shape[0]),
            device=X.device
        )
        max_delta_freq = self.max_delta_freq
        if isinstance(max_delta_freq, torch.Tensor):
            max_delta_freq = max_delta_freq.to(X.device)
        delta_freq = u * 2 * max_delta_freq - max_delta_freq
        return {
            "delta_freq": delta_freq,
            "sfreq": self.sfreq,
        }


def _get_standard_10_20_positions(raw_or_epoch=None, ordered_ch_names=None):
    """Returns standard 10-20 sensors position matrix (for instantiating
    SensorsRotation for example).

    Parameters
    ----------
    raw_or_epoch : mne.io.Raw | mne.Epoch, optional
        Example of raw or epoch to retrive ordered channels list from. Need to
        be named as in 10-20. By default None.
    ordered_ch_names : list, optional
        List of strings representing the channels of the montage considered.
        The order has to be consistent with the order of channels in the input
        matrices that will be fed to `SensorsRotation` transform. By
        default None.
    """
    assert raw_or_epoch is not None or ordered_ch_names is not None,\
        "At least one of raw_or_epoch and ordered_ch_names is needed."
    if ordered_ch_names is None:
        ordered_ch_names = raw_or_epoch.info['ch_names']
    ten_twenty_montage = make_standard_montage('standard_1020')
    positions_dict = ten_twenty_montage.get_positions()['ch_pos']
    positions_subdict = {
        k: positions_dict[k] for k in ordered_ch_names if k in positions_dict
    }
    return np.stack(list(positions_subdict.values())).T


class SensorsRotation(Transform):
    """Interpolates EEG signals over sensors rotated around the desired axis
    with an angle sampled uniformly between ``-max_degree`` and ``max_degree``.

    Suggested in [1]_

    Parameters
    ----------
    probability : float
        Float setting the probability of applying the operation.
    sensors_positions_matrix : numpy.ndarray
        Matrix giving the positions of each sensor in a 3D cartesian coordinate
        system. Should have shape (3, n_channels), where n_channels is the
        number of channels. Standard 10-20 positions can be obtained from
        `mne` through::

         >>> ten_twenty_montage = mne.channels.make_standard_montage(
         ...    'standard_1020'
         ... ).get_positions()['ch_pos']

    axis : 'x' | 'y' | 'z', optional
        Axis around which to rotate. Defaults to 'z'.
    max_degree : float, optional
        Maximum rotation. Rotation angles will be sampled between
        ``-max_degree`` and ``max_degree``. Defaults to 15 degrees.
    spherical_splines : bool, optional
        Whether to use spherical splines for the interpolation or not. When
        ``False``, standard scipy.interpolate.Rbf (with quadratic kernel) will
        be used (as in the original paper). Defaults to True.
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
    operation = staticmethod(sensors_rotation)

    def __init__(
        self,
        probability,
        sensors_positions_matrix,
        axis='z',
        max_degrees=15,
        spherical_splines=True,
        random_state=None
    ):
        super().__init__(
            probability=probability,
            random_state=random_state
        )
        if isinstance(sensors_positions_matrix, (np.ndarray, list)):
            sensors_positions_matrix = torch.as_tensor(
                sensors_positions_matrix
            )
        assert isinstance(sensors_positions_matrix, torch.Tensor),\
            "sensors_positions should be an Tensor"
        assert (
            isinstance(max_degrees, (Real, torch.Tensor)) and
            max_degrees >= 0
        ), "max_degrees should be non-negative float."
        assert isinstance(axis, str) and axis in ['x', 'y', 'z'],\
            "axis can be either x, y or z."
        assert sensors_positions_matrix.shape[0] == 3,\
            "sensors_positions_matrix shape should be 3 x n_channels."
        assert isinstance(spherical_splines, bool),\
            "spherical_splines should be a boolean"
        self.sensors_positions_matrix = sensors_positions_matrix
        self.axis = axis
        self.spherical_splines = spherical_splines
        self.max_degrees = max_degrees

    def get_params(self, *batch):
        """Return transform parameters.

        Parameters
        ----------
        X : tensor.Tensor
            The data.
        y : tensor.Tensor
            The labels.

        Returns
        -------
        params : dict
            Contains four elements:

            * sensors_positions_matrix : numpy.ndarray
                Matrix giving the positions of each sensor in a 3D cartesian
                coordinate system. Should have shape (3, n_channels), where
                n_channels is the number of channels.
            * axis : 'x' | 'y' | 'z'
                Axis around which to rotate.
            * angles : array-like
                Array of float of shape ``(batch_size,)`` containing the
                rotation angles (in degrees) for each element of the input
                batch, sampled uniformly between ``-max_degrees``and
                ``max_degrees``.
            * spherical_splines : bool
                Whether to use spherical splines for the interpolation or not.
                When ``False``, standard scipy.interpolate.Rbf (with quadratic
                kernel) will be used (as in the original paper).
        """
        if len(batch) == 0:
            return super().get_params(*batch)
        X = batch[0]

        u = self.rng.uniform(
            low=0,
            high=1,
            size=X.shape[0]
        )
        max_degrees = self.max_degrees
        if isinstance(max_degrees, torch.Tensor):
            max_degrees = max_degrees.to(X.device)
        random_angles = torch.as_tensor(
            u, device=X.device) * 2 * max_degrees - max_degrees
        return {
            "sensors_positions_matrix": self.sensors_positions_matrix,
            "axis": self.axis,
            "angles": random_angles,
            "spherical_splines": self.spherical_splines
        }


class SensorsZRotation(SensorsRotation):
    """Interpolates EEG signals over sensors rotated around the Z axis
    with an angle sampled uniformly between ``-max_degree`` and ``max_degree``.

    Suggested in [1]_

    Parameters
    ----------
    probability : float
        Float setting the probability of applying the operation.
    ordered_ch_names : list
        List of strings representing the channels of the montage considered.
        Has to be in standard 10-20 style. The order has to be consistent with
        the order of channels in the input matrices that will be fed to the
        transform. This channel will be used to compute approximate sensors
        positions from a standard 10-20 montage.
    max_degree : float, optional
        Maximum rotation. Rotation angles will be sampled between
        ``-max_degree`` and ``max_degree``. Defaults to 15 degrees.
    spherical_splines : bool, optional
        Whether to use spherical splines for the interpolation or not. When
        ``False``, standard scipy.interpolate.Rbf (with quadratic kernel) will
        be used (as in the original paper). Defaults to True.
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
        max_degrees=15,
        spherical_splines=True,
        random_state=None
    ):
        sensors_positions_matrix = torch.as_tensor(
            _get_standard_10_20_positions(ordered_ch_names=ordered_ch_names)
        )
        super().__init__(
            probability=probability,
            sensors_positions_matrix=sensors_positions_matrix,
            axis='z',
            max_degrees=max_degrees,
            spherical_splines=spherical_splines,
            random_state=random_state
        )


class SensorsYRotation(SensorsRotation):
    """Interpolates EEG signals over sensors rotated around the Y axis
    with an angle sampled uniformly between ``-max_degree`` and ``max_degree``.

    Suggested in [1]_

    Parameters
    ----------
    probability : float
        Float setting the probability of applying the operation.
    ordered_ch_names : list
        List of strings representing the channels of the montage considered.
        Has to be in standard 10-20 style. The order has to be consistent with
        the order of channels in the input matrices that will be fed to the
        transform. This channel will be used to compute approximate sensors
        positions from a standard 10-20 montage.
    max_degree : float, optional
        Maximum rotation. Rotation angles will be sampled between
        ``-max_degree`` and ``max_degree``. Defaults to 15 degrees.
    spherical_splines : bool, optional
        Whether to use spherical splines for the interpolation or not. When
        ``False``, standard scipy.interpolate.Rbf (with quadratic kernel) will
        be used (as in the original paper). Defaults to True.
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
        max_degrees=15,
        spherical_splines=True,
        random_state=None
    ):
        sensors_positions_matrix = torch.as_tensor(
            _get_standard_10_20_positions(ordered_ch_names=ordered_ch_names)
        )
        super().__init__(
            probability=probability,
            sensors_positions_matrix=sensors_positions_matrix,
            axis='y',
            max_degrees=max_degrees,
            spherical_splines=spherical_splines,
            random_state=random_state
        )


class SensorsXRotation(SensorsRotation):
    """Interpolates EEG signals over sensors rotated around the X axis
    with an angle sampled uniformly between ``-max_degree`` and ``max_degree``.

    Suggested in [1]_

    Parameters
    ----------
    probability : float
        Float setting the probability of applying the operation.
    ordered_ch_names : list
        List of strings representing the channels of the montage considered.
        Has to be in standard 10-20 style. The order has to be consistent with
        the order of channels in the input matrices that will be fed to the
        transform. This channel will be used to compute approximate sensors
        positions from a standard 10-20 montage.
    max_degree : float, optional
        Maximum rotation. Rotation angles will be sampled between
        ``-max_degree`` and ``max_degree``. Defaults to 15 degrees.
    spherical_splines : bool, optional
        Whether to use spherical splines for the interpolation or not. When
        ``False``, standard scipy.interpolate.Rbf (with quadratic kernel) will
        be used (as in the original paper). Defaults to True.
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
        max_degrees=15,
        spherical_splines=True,
        random_state=None
    ):
        sensors_positions_matrix = torch.as_tensor(
            _get_standard_10_20_positions(ordered_ch_names=ordered_ch_names)
        )
        super().__init__(
            probability=probability,
            sensors_positions_matrix=sensors_positions_matrix,
            axis='x',
            max_degrees=max_degrees,
            spherical_splines=spherical_splines,
            random_state=random_state
        )


class Mixup(Transform):
    """Implements Iterator for Mixup for EEG data. See [1]_.
    Implementation based on [2]_.

    Parameters
    ----------
    alpha: float
        Mixup hyperparameter.
    beta_per_sample: bool (default=False)
        By default, one mixing coefficient per batch is drawn from a beta
        distribution. If True, one mixing coefficient per sample is drawn.
    random_state: int | numpy.random.Generator, optional
        Seed to be used to instantiate numpy random number generator instance.
        Defaults to None.

    References
    ----------
    .. [1] Hongyi Zhang, Moustapha Cisse, Yann N. Dauphin, David Lopez-Paz
       (2018). mixup: Beyond Empirical Risk Minimization. In 2018
       International Conference on Learning Representations (ICLR)
       Online: https://arxiv.org/abs/1710.09412
    .. [2] https://github.com/facebookresearch/mixup-cifar10/blob/master/train.py
    """
    operation = staticmethod(mixup)

    def __init__(
        self,
        alpha,
        beta_per_sample=False,
        random_state=None
    ):
        super().__init__(
            probability=1.0,  # Mixup has to be applied to whole batches
            random_state=random_state
        )
        self.alpha = alpha
        self.beta_per_sample = beta_per_sample

    def get_params(self, *batch):
        """Return transform parameters.

        Parameters
        ----------
        X : tensor.Tensor
            The data.
        y : tensor.Tensor
            The labels.

        Returns
        -------
        params: dict
            Contains the values sampled uniformly between 0 and 1 setting the
            linear interpolation between examples (lam) and the shuffled
            indices of examples that are mixed into original examples
            (idx_perm).
        """
        X = batch[0]
        device = X.device
        batch_size, _, _ = X.shape

        if self.alpha > 0:
            if self.beta_per_sample:
                lam = torch.as_tensor(
                    self.rng.beta(self.alpha, self.alpha, batch_size)
                ).to(device)
            else:
                lam = torch.ones(batch_size).to(device)
                lam *= self.rng.beta(self.alpha, self.alpha)
        else:
            lam = torch.ones(batch_size).to(device)

        idx_perm = torch.as_tensor(self.rng.permutation(batch_size,))

        return {
            "lam": lam,
            "idx_perm": idx_perm,
        }
