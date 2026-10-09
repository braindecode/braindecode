# Authors: Cédric Rommel <cedric.rommel@inria.fr>
#          Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Gustavo Rodrigues <gustavenrique01@gmail.com>
#          Bruna Lopes <brunajaflopes@gmail.com>
#          Sarthak Tayal <sarthaktayal2@gmail.com>
#
# License: BSD (3-clause)

import warnings
from numbers import Real
from typing import NamedTuple, Optional

import numpy as np
import torch
from mne.channels import make_standard_montage

from braindecode.util import resolve_montage_name

from .base import Transform
from .functional import (
    amplitude_scale,
    band_rotation,
    bandstop_filter,
    channels_dropout,
    channels_permute,
    channels_rereference,
    channels_shuffle,
    frequency_shift,
    ft_surrogate,
    gaussian_noise,
    mask_encoding,
    mixup,
    segmentation_reconstruction,
    sensors_rotation,
    sign_flip,
    smooth_time_mask,
    time_reverse,
)


class TimeReverse(Transform):
    """Flip the time axis of each input with a given probability.

    Parameters
    ----------
    probability : float
        Float setting the probability of applying the operation.
    random_state : int | numpy.random.Generator, optional
        Seed to be used to instantiate numpy random number generator instance.
        Used to decide whether or not to transform given the probability
        argument. Defaults to None.
    """

    operation = staticmethod(time_reverse)  # type: ignore[assignment]

    def __init__(
        self,
        probability,
        random_state=None,
    ):
        super().__init__(
            probability=probability,
            random_state=random_state,
        )


class SignFlip(Transform):
    """Flip the sign axis of each input with a given probability.

    Parameters
    ----------
    probability : float
        Float setting the probability of applying the operation.
    random_state : int | numpy.random.Generator, optional
        Seed to be used to instantiate numpy random number generator instance.
        Used to decide whether or not to transform given the probability
        argument. Defaults to None.
    """

    operation = staticmethod(sign_flip)  # type: ignore[assignment]

    def __init__(self, probability, random_state=None):
        super().__init__(probability=probability, random_state=random_state)


class FTSurrogate(Transform):
    """FT surrogate augmentation of a single EEG channel, as proposed in [1]_.

    Parameters
    ----------
    probability : float
        Float setting the probability of applying the operation.
    phase_noise_magnitude : float | torch.Tensor, optional
        Float between 0 and 1 setting the range over which the phase
        perturbation is uniformly sampled:
        ``[0, phase_noise_magnitude * 2 * pi]``. Defaults to 1.
    channel_indep : bool, optional
        Whether to sample phase perturbations independently for each channel or
        not. It is advised to set it to False when spatial information is
        important for the task, like in BCI. Default False.
    random_state : int | numpy.random.Generator, optional
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

    operation = staticmethod(ft_surrogate)  # type: ignore[assignment]

    def __init__(
        self,
        probability,
        phase_noise_magnitude=1,
        channel_indep=False,
        random_state=None,
    ):
        super().__init__(probability=probability, random_state=random_state)
        assert isinstance(phase_noise_magnitude, (float, int, torch.Tensor)), (
            "phase_noise_magnitude should be a float."
        )
        assert 0 <= phase_noise_magnitude <= 1, (
            "phase_noise_magnitude should be between 0 and 1."
        )
        assert isinstance(channel_indep, bool), (
            "channel_indep is expected to be a boolean"
        )
        self.phase_noise_magnitude = phase_noise_magnitude
        self.channel_indep = channel_indep

    def get_augmentation_params(self, *batch):
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
            "channel_indep": self.channel_indep,
            "random_state": self.rng,
        }


class ChannelsDropout(Transform):
    """Randomly set channels to flat signal.

    Part of the CMSAugment policy proposed in [1]_

    Parameters
    ----------
    probability : float
        Float setting the probability of applying the operation.
    p_drop : float | None, optional
        Float between 0 and 1 setting the probability of dropping each channel.
        Defaults to 0.2.
    random_state : int | numpy.random.RandomState, optional
        Seed to be used to instantiate numpy random number generator instance.
        Used to decide whether or not to transform given the probability
        argument and to sample channels to erase. Defaults to None.

    References
    ----------
    .. [1] Saeed, A., Grangier, D., Pietquin, O., & Zeghidour, N. (2020).
       Learning from Heterogeneous EEG Signals with Differentiable Channel
       Reordering. arXiv preprint arXiv:2010.13694.
    """

    operation = staticmethod(channels_dropout)  # type: ignore[assignment]

    def __init__(self, probability, p_drop=0.2, random_state=None):
        super().__init__(probability=probability, random_state=random_state)
        self.p_drop = p_drop

    def get_augmentation_params(self, *batch):
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
    probability : float
        Float setting the probability of applying the operation.
    p_shuffle : float | None, optional
        Float between 0 and 1 setting the probability of including the channel
        in the set of permuted channels. Defaults to 0.2.
    random_state : int | numpy.random.Generator, optional
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

    operation = staticmethod(channels_shuffle)  # type: ignore[assignment]

    def __init__(self, probability, p_shuffle=0.2, random_state=None):
        super().__init__(probability=probability, random_state=random_state)
        self.p_shuffle = p_shuffle

    def get_augmentation_params(self, *batch):
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
    random_state : int | numpy.random.Generator, optional
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

    operation = staticmethod(gaussian_noise)  # type: ignore[assignment]

    def __init__(self, probability, std=0.1, random_state=None):
        super().__init__(
            probability=probability,
            random_state=random_state,
        )
        self.std = std

    def get_augmentation_params(self, *batch):
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
    random_state : int | numpy.random.Generator, optional
        Seed to be used to instantiate numpy random number generator instance.
        Used to decide whether or not to transform given the probability
        argument. Defaults to None.

    References
    ----------
    .. [1] Deiss, O., Biswal, S., Jin, J., Sun, H., Westover, M. B., & Sun, J.
       (2018). HAMLET: interpretable human and machine co-learning technique.
       arXiv preprint arXiv:1803.09702.
    """

    operation = staticmethod(channels_permute)  # type: ignore[assignment]

    def __init__(self, probability, ordered_ch_names, random_state=None):
        super().__init__(
            probability=probability,
            random_state=random_state,
        )
        assert isinstance(ordered_ch_names, list) and all(
            isinstance(ch, str) for ch in ordered_ch_names
        ), "ordered_ch_names should be a list of str."

        permutation = list()
        for idx, ch_name in enumerate(ordered_ch_names):
            new_position = idx
            # Find digits in channel name (assuming 10-20 system)
            d = "".join(list(filter(str.isdigit, ch_name)))
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

    def get_augmentation_params(self, *batch):
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
    """Smoothly replace a randomly chosen contiguous part of all channels by.

    zeros.

    Suggested e.g. in [1]_ and [2]_

    Parameters
    ----------
    probability : float
        Float setting the probability of applying the operation.
    mask_len_samples : int | torch.Tensor, optional
        Number of consecutive samples to zero out. Will be ignored if
        magnitude is not set to None. Defaults to 100.
    random_state : int | numpy.random.Generator, optional
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

    operation = staticmethod(smooth_time_mask)  # type: ignore[assignment]

    def __init__(self, probability, mask_len_samples=100, random_state=None):
        super().__init__(
            probability=probability,
            random_state=random_state,
        )

        assert (
            isinstance(mask_len_samples, (int, torch.Tensor)) and mask_len_samples > 0
        ), "mask_len_samples has to be a positive integer"
        self.mask_len_samples = mask_len_samples

    def get_augmentation_params(self, *batch):
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
            return super().get_augmentation_params(*batch)
        X = batch[0]

        seq_length = torch.as_tensor(X.shape[-1], device=X.device)
        mask_len_samples = self.mask_len_samples
        if isinstance(mask_len_samples, torch.Tensor):
            mask_len_samples = mask_len_samples.to(X.device)
        mask_start = torch.as_tensor(
            self.rng.uniform(
                low=0,
                high=1,
                size=X.shape[0],
            ),
            device=X.device,
        ) * (seq_length - mask_len_samples)
        return {
            "mask_start_per_sample": mask_start,
            "mask_len_samples": mask_len_samples,
        }


class BandstopFilter(Transform):
    """Apply a band-stop filter with desired bandwidth at a randomly selected.

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
    random_state : int | numpy.random.Generator, optional
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

    operation = staticmethod(bandstop_filter)  # type: ignore[assignment]

    def __init__(
        self, probability, sfreq, bandwidth=1, max_freq=None, random_state=None
    ):
        super().__init__(
            probability=probability,
            random_state=random_state,
        )
        assert isinstance(bandwidth, Real) and bandwidth >= 0, (
            "bandwidth should be a non-negative float."
        )
        assert isinstance(sfreq, Real) and sfreq > 0, (
            "sfreq should be a positive float."
        )
        if max_freq is not None:
            assert isinstance(max_freq, Real) and max_freq > 0, (
                "max_freq should be a positive float."
            )
        nyq = sfreq / 2
        if max_freq is None or max_freq > nyq:
            max_freq = nyq
            warnings.warn(
                "You either passed None or a frequency greater than the"
                f" Nyquist frequency ({nyq} Hz)."
                f" Falling back to max_freq = {nyq}."
            )
        assert bandwidth < max_freq - 2, (
            f"`bandwidth` needs to be smaller than max_freq - 2={max_freq - 2} "
            f"to allow valid notch frequency sampling with 1 Hz transition bands."
        )

        # override bandwidth value when a magnitude is passed
        self.sfreq = sfreq
        self.max_freq = max_freq
        self.bandwidth = bandwidth

    def get_augmentation_params(self, *batch):
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
            return super().get_augmentation_params(*batch)
        X = batch[0]

        # Prevents transitions from going below 0 and above max_freq
        notched_freqs = self.rng.uniform(
            low=1 + self.bandwidth / 2,
            high=self.max_freq - 1 - self.bandwidth / 2,
            size=X.shape[0],
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
    random_state : int | numpy.random.Generator, optional
        Seed to be used to instantiate numpy random number generator instance.
        Defaults to None.
    """

    operation = staticmethod(frequency_shift)  # type: ignore[assignment]

    def __init__(self, probability, sfreq, max_delta_freq=2, random_state=None):
        super().__init__(
            probability=probability,
            random_state=random_state,
        )
        assert isinstance(sfreq, Real) and sfreq > 0, (
            "sfreq should be a positive float."
        )
        self.sfreq = sfreq

        self.max_delta_freq = max_delta_freq

    def get_augmentation_params(self, *batch):
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
            return super().get_augmentation_params(*batch)
        X = batch[0]

        u = torch.as_tensor(self.rng.uniform(size=X.shape[0]), device=X.device)
        max_delta_freq = self.max_delta_freq
        if isinstance(max_delta_freq, torch.Tensor):
            max_delta_freq = max_delta_freq.to(X.device)
        delta_freq = u * 2 * max_delta_freq - max_delta_freq
        return {
            "delta_freq": delta_freq,
            "sfreq": self.sfreq,
        }


def _get_standard_10_20_positions(raw_or_epoch=None, ordered_ch_names=None):
    """Returns standard 10-20 sensors position matrix (for instantiating.

    SensorsRotation for example).

    Parameters
    ----------
    raw_or_epoch : mne.io.Raw | mne.Epoch, optional
        Example of raw or epoch to retrieve ordered channels list from. Need to
        be named as in 10-20. By default None.
    ordered_ch_names : list, optional
        List of strings representing the channels of the montage considered.
        The order has to be consistent with the order of channels in the input
        matrices that will be fed to `SensorsRotation` transform. By
        default None.
    """
    assert raw_or_epoch is not None or ordered_ch_names is not None, (
        "At least one of raw_or_epoch and ordered_ch_names is needed."
    )
    if ordered_ch_names is None:
        ordered_ch_names = raw_or_epoch.info["ch_names"]
    ten_twenty_montage = make_standard_montage(resolve_montage_name("standard_1020"))
    positions_dict = ten_twenty_montage.get_positions()["ch_pos"]
    positions_subdict = {
        k: positions_dict[k] for k in ordered_ch_names if k in positions_dict
    }
    return np.stack(list(positions_subdict.values())).T


class SensorsRotation(Transform):
    """Interpolates EEG signals over sensors rotated around the desired axis.

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
         ...    'standard_1020'  # 'colin27_1020' on MNE >= 1.13
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
    random_state : int | numpy.random.Generator, optional
        Seed to be used to instantiate numpy random number generator instance.
        Defaults to None.

    References
    ----------
    .. [1] Krell, M. M., & Kim, S. K. (2017). Rotational data augmentation for
       electroencephalographic data. In 2017 39th Annual International
       Conference of the IEEE Engineering in Medicine and Biology Society
       (EMBC) (pp. 471-474).
    """

    operation = staticmethod(sensors_rotation)  # type: ignore[assignment]

    def __init__(
        self,
        probability,
        sensors_positions_matrix,
        axis="z",
        max_degrees=15,
        spherical_splines=True,
        random_state=None,
    ):
        super().__init__(probability=probability, random_state=random_state)
        if isinstance(sensors_positions_matrix, (np.ndarray, list)):
            sensors_positions_matrix = torch.as_tensor(sensors_positions_matrix)
        assert isinstance(sensors_positions_matrix, torch.Tensor), (
            "sensors_positions should be an Tensor"
        )
        assert isinstance(max_degrees, (Real, torch.Tensor)) and max_degrees >= 0, (
            "max_degrees should be non-negative float."
        )
        assert isinstance(axis, str) and axis in [
            "x",
            "y",
            "z",
        ], "axis can be either x, y or z."
        assert sensors_positions_matrix.shape[0] == 3, (
            "sensors_positions_matrix shape should be 3 x n_channels."
        )
        assert isinstance(spherical_splines, bool), (
            "spherical_splines should be a boolean"
        )
        self.sensors_positions_matrix = sensors_positions_matrix
        self.axis = axis
        self.spherical_splines = spherical_splines
        self.max_degrees = max_degrees

    def get_augmentation_params(self, *batch):
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
            return super().get_augmentation_params(*batch)
        X = batch[0]

        u = self.rng.uniform(low=0, high=1, size=X.shape[0])
        max_degrees = self.max_degrees
        if isinstance(max_degrees, torch.Tensor):
            max_degrees = max_degrees.to(X.device)
        random_angles = (
            torch.as_tensor(u, device=X.device) * 2 * max_degrees - max_degrees
        )
        return {
            "sensors_positions_matrix": self.sensors_positions_matrix,
            "axis": self.axis,
            "angles": random_angles,
            "spherical_splines": self.spherical_splines,
        }


class SensorsZRotation(SensorsRotation):
    """Interpolates EEG signals over sensors rotated around the Z axis.

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
    random_state : int | numpy.random.Generator, optional
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
        random_state=None,
    ):
        sensors_positions_matrix = torch.as_tensor(
            _get_standard_10_20_positions(ordered_ch_names=ordered_ch_names)
        )
        super().__init__(
            probability=probability,
            sensors_positions_matrix=sensors_positions_matrix,
            axis="z",
            max_degrees=max_degrees,
            spherical_splines=spherical_splines,
            random_state=random_state,
        )


class SensorsYRotation(SensorsRotation):
    """Interpolates EEG signals over sensors rotated around the Y axis.

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
    random_state : int | numpy.random.Generator, optional
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
        random_state=None,
    ):
        sensors_positions_matrix = torch.as_tensor(
            _get_standard_10_20_positions(ordered_ch_names=ordered_ch_names)
        )
        super().__init__(
            probability=probability,
            sensors_positions_matrix=sensors_positions_matrix,
            axis="y",
            max_degrees=max_degrees,
            spherical_splines=spherical_splines,
            random_state=random_state,
        )


class SensorsXRotation(SensorsRotation):
    """Interpolates EEG signals over sensors rotated around the X axis.

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
    random_state : int | numpy.random.Generator, optional
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
        random_state=None,
    ):
        sensors_positions_matrix = torch.as_tensor(
            _get_standard_10_20_positions(ordered_ch_names=ordered_ch_names)
        )
        super().__init__(
            probability=probability,
            sensors_positions_matrix=sensors_positions_matrix,
            axis="x",
            max_degrees=max_degrees,
            spherical_splines=spherical_splines,
            random_state=random_state,
        )


class Mixup(Transform):
    """Implements Iterator for Mixup for EEG data.

    See [1]_.
    Implementation based on [2]_.

    Parameters
    ----------
    alpha : float
        Mixup hyperparameter.
    beta_per_sample : bool (default=False)
        By default, one mixing coefficient per batch is drawn from a beta
        distribution. If True, one mixing coefficient per sample is drawn.
    random_state : int | numpy.random.Generator, optional
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

    operation = staticmethod(mixup)  # type: ignore[assignment]

    def __init__(self, alpha, beta_per_sample=False, random_state=None):
        super().__init__(
            probability=1.0,  # Mixup has to be applied to whole batches
            random_state=random_state,
        )
        self.alpha = alpha
        self.beta_per_sample = beta_per_sample

    def get_augmentation_params(self, *batch):
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
            Contains the values sampled uniformly between 0 and 1 setting the
            linear interpolation between examples (lam) and the shuffled
            indices of examples that are mixed into original examples
            (idx_perm).
        """
        X = batch[0]
        device = X.device
        batch_size, _, _ = X.shape

        # lam follows the dtype of X, numpy draws float64 and that would leak
        # into the mixed signal and into the loss returned by mixup_criterion
        if self.alpha > 0:
            if self.beta_per_sample:
                lam = torch.as_tensor(
                    self.rng.beta(self.alpha, self.alpha, batch_size)
                ).to(device=device, dtype=X.dtype)
            else:
                lam = torch.ones(batch_size, dtype=X.dtype).to(device)
                lam *= self.rng.beta(self.alpha, self.alpha)
        else:
            lam = torch.ones(batch_size, dtype=X.dtype).to(device)

        idx_perm = torch.as_tensor(
            self.rng.permutation(
                batch_size,
            )
        )

        return {
            "lam": lam,
            "idx_perm": idx_perm,
        }


class SegmentationReconstruction(Transform):
    """Segmentation Reconstruction from Lotte (2015) [Lotte2015]_.

    Applies a segmentation-reconstruction transform to the input data, as
    proposed in [Lotte2015]_. It segments each trial in the batch and randomly mix
    it to generate new synthetic trials by label, preserving the original
    order of the segments in time domain.

    Parameters
    ----------
    probability : float
        Float setting the probability of applying the operation.
    random_state : int | numpy.random.Generator, optional
        Seed to be used to instantiate numpy random number generator instance.
        Used to decide whether to transform given the probability
        argument and to sample the segments mixing. Defaults to None.
    n_segments : int, optional
        Number of segments to use in the batch. If None, X will be
        automatically segmented, getting the last element in a list
        of factors of the number of samples's square root. Defaults to None.

    References
    ----------
    .. [Lotte2015] Lotte, F. (2015). Signal processing approaches to minimize
        or suppress calibration time in oscillatory activity-based brain–computer
        interfaces. Proceedings of the IEEE, 103(6), 871-890.
    """

    operation = staticmethod(segmentation_reconstruction)  # type: ignore[assignment]

    def __init__(
        self,
        probability,
        n_segments=None,
        random_state=None,
    ):
        super().__init__(
            probability=probability,
            random_state=random_state,
        )
        self.n_segments = n_segments

    def get_augmentation_params(self, *batch):
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
            Contains the number of segments to split the signal into.
        """
        X, y = batch[0], batch[1]

        if y is not None:
            if not isinstance(X, torch.Tensor) or not isinstance(y, torch.Tensor):
                raise ValueError("X and y must be torch tensors.")

            if X.shape[0] != y.shape[0]:
                raise ValueError("Number of samples in X and y must be the same.")

        if self.n_segments is None:
            self.n_segments = int(X.shape[2])
            n_segments_list = []
            for i in range(1, int(self.n_segments**0.5) + 1):
                if self.n_segments % i == 0:
                    n_segments_list.append(i)
            self.n_segments = n_segments_list[-1]

        elif not (
            isinstance(self.n_segments, (int, float))
            and 1 <= self.n_segments <= X.shape[2]
        ):
            raise ValueError(
                f"Number of segments must be a positive integer less than "
                f"(or equal) the window size. Got {self.n_segments}"
            )

        if y is None:
            data_classes = [(np.nan, X)]

        else:
            classes = torch.unique(y)

            data_classes = [(i, X[y == i]) for i in classes]

        rand_indices = dict()
        for label, X_class in data_classes:
            n_trials = X_class.shape[0]
            rand_indices[label] = self.rng.randint(
                0, n_trials, (n_trials, self.n_segments)
            )

        idx_shuffle = self.rng.permutation(X.shape[0])

        return {
            "n_segments": self.n_segments,
            "data_classes": data_classes,
            "rand_indices": rand_indices,
            "idx_shuffle": idx_shuffle,
        }


class MaskEncoding(Transform):
    """MaskEncoding from [1]_.

    Replaces randomly chosen contiguous part (or parts) of all channels by
    zeros (if more than one segment, it may overlap).

    Implementation based on [1]_

    Parameters
    ----------
    probability : float
        Float setting the probability of applying the operation.
    max_mask_ratio : float, optional
        Signal ratio to zero out. Defaults to 0.1.
    n_segments : int, optional
        Number of segments to zero out in each example.
        Defaults to 1.
    random_state : int | numpy.random.Generator, optional
        Seed to be used to instantiate numpy random number generator instance.
        Defaults to None.

    References
    ----------
    .. [1] Ding, Wenlong, et al. "A Novel Data Augmentation Approach
        Using Mask Encoding for Deep Learning-Based Asynchronous SSVEP-BCI."
        IEEE Transactions on Neural Systems and Rehabilitation Engineering
        32 (2024): 875-886.
    """

    operation = staticmethod(mask_encoding)  # type: ignore[assignment]

    def __init__(
        self,
        probability,
        max_mask_ratio=0.1,
        n_segments=1,
        random_state=None,
    ):
        super().__init__(
            probability=probability,
            random_state=random_state,
        )
        assert isinstance(n_segments, int) and n_segments > 0, (
            "n_segments should be a positive integer."
        )
        assert isinstance(max_mask_ratio, (int, float)) and 0 <= max_mask_ratio <= 1, (
            "mask_ratio should be a float between 0 and 1."
        )

        self.mask_ratio = max_mask_ratio
        self.n_segments = n_segments

    def get_augmentation_params(self, *batch):
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
            Contains ...
        """
        if len(batch) == 0:
            return super().get_augmentation_params(*batch)
        X = batch[0]

        batch_size, _, n_times = X.shape

        segment_length = int((n_times * self.mask_ratio) / self.n_segments)

        assert segment_length >= 1, (
            "n_segments should be a positive integer not higher than (max_mask_ratio * window size)."
        )

        time_start = self.rng.randint(
            0, n_times - segment_length, (batch_size, self.n_segments)
        )
        time_start = torch.from_numpy(time_start)

        return {
            "time_start": time_start,
            "segment_length": segment_length,
            "n_segments": self.n_segments,
        }


class ChannelsReref(Transform):
    """Randomly re-reference channels in EEG data matrix.

    Part of the augmentations proposed in [1]_

    Parameters
    ----------
    probability : float
        Float setting the probability of applying the operation.
    random_state : int | numpy.random.Generator, optional
        Seed to be used to instantiate numpy random number generator instance.
        Used to decide whether or not to transform given the probability
        argument, to sample which channels to shuffle and to carry the shuffle.
        Defaults to None.

    References
    ----------
    .. [1] Mohsenvand, M.N., Izadi, M.R. &amp; Maes, P.. (2020). Contrastive
        Representation Learning for Electroencephalogram Classification. Proceedings
        of the Machine Learning for Health NeurIPS Workshop, in Proceedings of Machine
        Learning Research 136:238-253 Available from https://proceedings.mlr.press/v136/mohsenvand20a.html.
    """

    operation = staticmethod(channels_rereference)  # type: ignore[assignment]

    def __init__(self, probability, random_state=None):
        super().__init__(probability=probability, random_state=random_state)

    def get_augmentation_params(self, *batch):
        """Return transform parameters."""
        return {
            "random_state": self.rng,
        }


class AmplitudeScale(Transform):
    """Rescale amplitude based on a random sampled scaling value.

    Part of the augmentations proposed in [1]_

    Parameters
    ----------
    probability : float
        Float setting the probability of applying the operation.
    random_state : int | numpy.random.Generator, optional
        Seed to be used to instantiate numpy random number generator instance.
        Used to decide whether or not to transform given the probability
        argument, to sample which channels to shuffle and to carry the shuffle.
        Defaults to None.

    References
    ----------
    .. [1] Mohsenvand, M.N., Izadi, M.R. &amp; Maes, P.. (2020). Contrastive
        Representation Learning for Electroencephalogram Classification. Proceedings
        of the Machine Learning for Health NeurIPS Workshop, in Proceedings of Machine
        Learning Research 136:238-253 Available from https://proceedings.mlr.press/v136/mohsenvand20a.html.
    """

    operation = staticmethod(amplitude_scale)  # type: ignore[assignment]

    def __init__(self, probability, interval=(0.5, 2), random_state=None):
        super().__init__(probability=probability, random_state=random_state)
        self.scale = interval

    def get_augmentation_params(self, *batch):
        """Return transform parameters."""
        return {"random_state": self.rng, "scale": self.scale}


class BandRotation(Transform):
    """Per-band electrode rotation + inter-band temporal jitter.

    Models small wristband rotation between sessions and relative timing
    noise between two arms.  Introduced in [Sivakumar2024]_ for the
    emg2qwerty surface-EMG keystroke decoding task: the channel axis is
    laid out as ``(B, num_bands * electrodes_per_band, T)`` with bands
    contiguous, each band gets a uniform circular roll along the channel
    axis, and when ``num_bands >= 2``, band 1 also gets a sample-level
    temporal shift.  The same offset / shift is applied to every sample
    in a transformed sub-batch (one set of parameters per call).

    Parameters
    ----------
    probability : float
        Float setting the probability of applying the operation.
    num_bands : int, optional
        Number of electrode bands (e.g. ``2`` for left + right wristband).
        Must be ``>= 1``.  Defaults to 2.
    electrodes_per_band : int, optional
        Electrodes per band (e.g. ``16``).  Must be ``>= 1``.  Defaults
        to 16.
    band_offsets : tuple of int, optional
        Per-band roll values to sample from uniformly.  ``(-1, 0, 1)``
        covers ±1-electrode misalignment.  Must be non-empty.  Defaults
        to ``(-1, 0, 1)``.
    max_temporal_jitter : int, optional
        Max ±-sample temporal shift applied to band 1.  Defaults to 0
        (jitter disabled).  Must be ``>= 0``.  The emg2qwerty paper uses
        120 samples (60 ms at 2 kHz).
    circular_jitter : bool, optional
        If True (default, paper-faithful) the jitter is a circular roll;
        if False the gap left by the shift is zero-padded.  See
        :func:`band_rotation`.
    random_state : int | numpy.random.RandomState, optional
        Seed for the rotation / jitter sampler.  Defaults to None.

    References
    ----------
    .. [Sivakumar2024] Sivakumar, V., Seely, J., Du, A., Bittner, S. R.,
       Berenzweig, A., Bolarinwa, A., Gramfort, A., & Mandel, M. I. (2024).
       "emg2qwerty: A Large Dataset with Baselines for Touch Typing using
       Surface Electromyography." *NeurIPS Datasets and Benchmarks Track*.
    """

    operation = staticmethod(band_rotation)  # type: ignore[assignment]

    def __init__(
        self,
        probability,
        num_bands=2,
        electrodes_per_band=16,
        band_offsets=(-1, 0, 1),
        max_temporal_jitter=0,
        circular_jitter=True,
        random_state=None,
    ):
        super().__init__(probability=probability, random_state=random_state)
        # Up-front parameter validation; the underlying ``band_rotation``
        # also re-checks at call time, but raising here surfaces config
        # mistakes when the Transform is built rather than on the first
        # batch.
        if num_bands < 1:
            raise ValueError(f"num_bands must be >= 1, got {num_bands}")
        if electrodes_per_band < 1:
            raise ValueError(
                f"electrodes_per_band must be >= 1, got {electrodes_per_band}"
            )
        band_offsets = tuple(band_offsets)
        if not band_offsets:
            raise ValueError("band_offsets must be non-empty")
        if not all(isinstance(o, (int, np.integer)) for o in band_offsets):
            raise ValueError(
                f"band_offsets must contain integers, got {band_offsets!r}"
            )
        if max_temporal_jitter < 0:
            raise ValueError(
                f"max_temporal_jitter must be >= 0, got {max_temporal_jitter}"
            )
        self.num_bands = num_bands
        self.electrodes_per_band = electrodes_per_band
        self.band_offsets = band_offsets
        self.max_temporal_jitter = max_temporal_jitter
        self.circular_jitter = circular_jitter

    def get_augmentation_params(self, *batch):
        return {
            "num_bands": self.num_bands,
            "electrodes_per_band": self.electrodes_per_band,
            "band_offsets": self.band_offsets,
            "max_temporal_jitter": self.max_temporal_jitter,
            "circular_jitter": self.circular_jitter,
            "random_state": self.rng,
        }


class _PoolEntry(NamedTuple):
    """One :class:`TrivialAugment` pool entry (``variants=None``: built per call)."""

    name: str
    strengths: list
    variants: Optional[list]


class TrivialAugment(Transform):
    """Apply one randomly sampled label-preserving augmentation per example.

    For every example in the batch, independently samples one transform from
    a pool of label-preserving augmentations and one strength from a discrete
    grid of ``num_bins`` levels between the per-transform bounds, then applies
    the sampled transform to that example only. As the image version of
    TrivialAugment [1]_, no policy search is involved: the pool and the
    strength grids are fixed, which keeps the method tuning-free.

    The default pool only contains transforms that are well-defined for any
    EEG batch, without assumptions on the signal's unit, sampling frequency
    or montage:

    * :class:`TimeReverse` — no strength parameter.
    * :class:`SignFlip` — no strength parameter.
    * :class:`FTSurrogate` — phase noise magnitude, 0.2 to 1.
    * :class:`ChannelsDropout` — channel drop probability, 0.05 to 0.4.
    * :class:`ChannelsShuffle` — shuffle probability, 0.1 to 0.75.
    * :class:`SmoothTimeMask` — mask length as a fraction of the window
      length, 0.05 to 0.25.
    * :class:`GaussianNoise` — noise standard deviation as a fraction of the
      sub-batch standard deviation, 0.02 to 0.3, so the op is usable whether
      signals are stored in volts or microvolts.
    * :class:`AmplitudeScale` — symmetric scale bound ``s`` (interval
      ``(1/s, s)``), 1.25 to 2.5.

    Three more transforms are label-preserving but need extra information
    about the recording, so they are only added to the pool when that
    information is provided: :class:`BandstopFilter` (bandwidth 0.5 to 4 Hz,
    capped 2.5 Hz below Nyquist) and :class:`FrequencyShift` (maximum shift
    0.5 to 2 Hz) when ``sfreq`` is given, and :class:`SensorsRotation` around
    the z axis (5 to 25 degrees) when ``sensors_positions_matrix`` is given.

    Transforms that are not label-preserving (:class:`Mixup`,
    :class:`SegmentationReconstruction`, :class:`MaskEncoding`) or that need
    a specific channel layout or nomenclature (:class:`ChannelsSymmetry`,
    :class:`BandRotation`) stay out of the pool but can be added through
    ``custom_ops``.

    Parameters
    ----------
    probability : float, optional
        Float setting the probability of augmenting each example. Examples
        that are not augmented stay unchanged. Defaults to 1.0.
    sfreq : float | None, optional
        Sampling frequency of the signals, in Hz. When given, adds
        :class:`BandstopFilter` and :class:`FrequencyShift` to the pool.
        Defaults to None.
    sensors_positions_matrix : array-like | None, optional
        Matrix of shape ``(3, n_channels)`` giving the 3D positions of the
        sensors (see :class:`SensorsRotation`). When given, adds
        :class:`SensorsRotation` to the pool. Defaults to None.
    num_bins : int, optional
        Number of discrete strength levels sampled for every transform.
        Defaults to 10.
    custom_ops : list | None, optional
        Extra entries appended to the pool, each a tuple
        ``(name, make, bounds)``: ``name`` is a label reported by
        ``TrivialAugment.op_names``, ``make(strength)`` returns a callable
        applied as ``apply(X_sub, y_sub) -> (X_sub, y_sub)``, and ``bounds``
        is either ``None`` (magnitude-free) or a ``(lo, hi)`` pair of floats.
        Defaults to None.
    random_state : int | numpy.random.RandomState, optional
        Seed controlling both the per-example sampling of transforms and
        strengths and the randomness of the pooled transforms themselves.
        Defaults to None.

    Attributes
    ----------
    op_names : list of str
        Names of the transforms in the pool, in sampling order.

    Notes
    -----
    Strength bins are uniformly spaced between the per-transform bounds
    (inclusive), and both the transform and the bin are sampled uniformly,
    as in the original TrivialAugment [1]_. Examples sharing the same
    (transform, strength) pair are transformed together in one call for
    efficiency; every pooled transform still samples its own randomness
    (e.g. the mask position, the exact scale or rotation angle)
    independently per example.

    References
    ----------
    .. [1] Müller, S. G., & Hutter, F. (2021). TrivialAugment:
       Tuning-free Yet State-of-the-Art Data Augmentation. Proceedings of
       the IEEE/CVF International Conference on Computer Vision (ICCV),
       pp. 774-782.
    """

    def __init__(
        self,
        probability=1.0,
        sfreq=None,
        sensors_positions_matrix=None,
        num_bins=10,
        custom_ops=None,
        random_state=None,
    ):
        super().__init__(probability=probability, random_state=random_state)
        if not isinstance(num_bins, (int, np.integer)) or num_bins < 1:
            raise ValueError(f"num_bins must be a positive integer, got {num_bins}")
        self.num_bins = int(num_bins)
        # single generator shared by the per-example sampler and every pooled
        # transform, so one seed makes the whole augmentation reproducible
        rng = self.rng

        def _bins(lo, hi):
            return np.linspace(lo, hi, self.num_bins).tolist()

        pool = []
        static_transforms = []

        def _add_static(name, factory, lo=None, hi=None):
            strengths = [None] if lo is None else _bins(lo, hi)
            variants = [factory(s) for s in strengths]
            static_transforms.extend(variants)
            pool.append(_PoolEntry(name, strengths, variants))

        _add_static("TimeReverse", lambda s: TimeReverse(1.0, random_state=rng))
        _add_static("SignFlip", lambda s: SignFlip(1.0, random_state=rng))
        _add_static(
            "FTSurrogate",
            lambda s: FTSurrogate(1.0, phase_noise_magnitude=s, random_state=rng),
            0.2,
            1.0,
        )
        _add_static(
            "ChannelsDropout",
            lambda s: ChannelsDropout(1.0, p_drop=s, random_state=rng),
            0.05,
            0.4,
        )
        _add_static(
            "ChannelsShuffle",
            lambda s: ChannelsShuffle(1.0, p_shuffle=s, random_state=rng),
            0.1,
            0.75,
        )
        # strength relative to the sub-batch (window length, signal scale):
        # built per call in _apply_dynamic, which also keeps this picklable
        pool.append(_PoolEntry("SmoothTimeMask", _bins(0.05, 0.25), None))
        pool.append(_PoolEntry("GaussianNoise", _bins(0.02, 0.3), None))
        _add_static(
            "AmplitudeScale",
            lambda s: AmplitudeScale(1.0, interval=(1.0 / s, s), random_state=rng),
            1.25,
            2.5,
        )

        if sfreq is not None:
            if not isinstance(sfreq, Real) or sfreq <= 0:
                raise ValueError(f"sfreq should be a positive float, got {sfreq}")
            nyquist = sfreq / 2.0
            # BandstopFilter needs bandwidth < nyquist - 2; keep a margin
            min_bandwidth, max_bandwidth = 0.5, min(4.0, nyquist - 2.5)
            if max_bandwidth <= min_bandwidth:
                raise ValueError(
                    f"sfreq={sfreq} is too small to sample valid band-stop "
                    f"bandwidths (needs at least {2 * (min_bandwidth + 2.5)} Hz)."
                )
            _add_static(
                "BandstopFilter",
                lambda s: BandstopFilter(
                    1.0, sfreq=sfreq, bandwidth=s, max_freq=nyquist, random_state=rng
                ),
                min_bandwidth,
                max_bandwidth,
            )
            _add_static(
                "FrequencyShift",
                lambda s: FrequencyShift(
                    1.0, sfreq=sfreq, max_delta_freq=s, random_state=rng
                ),
                0.5,
                2.0,
            )

        if sensors_positions_matrix is not None:
            _add_static(
                "SensorsRotation",
                lambda s: SensorsRotation(
                    1.0,
                    sensors_positions_matrix=sensors_positions_matrix,
                    max_degrees=s,
                    random_state=rng,
                ),
                5.0,
                25.0,
            )

        if custom_ops is not None:
            if isinstance(custom_ops, tuple):
                custom_ops = [custom_ops]
            for entry in custom_ops:
                pool.append(self._check_custom_op(entry))

        self._pool = pool
        # register the pre-instantiated transforms as submodules (parameters,
        # pickle, ``.to()``); callables from custom_ops are kept as-is
        self._static_transforms = torch.nn.ModuleList(static_transforms)
        self.op_names = [entry.name for entry in pool]

    def _check_custom_op(self, entry):
        """Validate one ``custom_ops`` entry and turn it into a pool entry."""
        try:
            name, make, bounds = entry
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"custom_ops entries must be (name, make, bounds) tuples, got {entry!r}"
            ) from e
        if not isinstance(name, str) or not callable(make):
            raise ValueError(
                f"custom_ops entries need a str name and a callable make, got {entry!r}"
            )
        if bounds is None:
            strengths = [None]
        else:
            try:
                lo, hi = bounds
            except (TypeError, ValueError) as e:
                raise ValueError(
                    f"custom_ops bounds must be None or a (lo, hi) pair, got {bounds!r}"
                ) from e
            if not (
                isinstance(lo, Real) and isinstance(hi, Real) and float(lo) < float(hi)
            ):
                raise ValueError(f"custom_ops bounds need lo < hi, got {bounds!r}")
            strengths = np.linspace(lo, hi, self.num_bins).tolist()
        return _PoolEntry(name, strengths, [make(s) for s in strengths])

    def _apply_dynamic(self, name, strength, X_sub, y_sub):
        """Apply a built-in transform whose strength depends on the sub-batch."""
        if name == "SmoothTimeMask":
            mask_len_samples = max(1, int(round(strength * X_sub.shape[-1])))
            transform = SmoothTimeMask(
                1.0, mask_len_samples=mask_len_samples, random_state=self.rng
            )
        else:  # GaussianNoise, scale-relative for unit independence
            std = strength * X_sub.std().item()
            transform = GaussianNoise(1.0, std=std, random_state=self.rng)
        return transform(X_sub, y_sub)

    def operation(self, X, y):
        """Apply one sampled transform and strength to each example of X."""
        n_strengths = np.array([len(entry.strengths) for entry in self._pool])
        op_ids = self.rng.randint(0, len(self._pool), size=X.shape[0])
        bin_ids = self.rng.randint(0, self.num_bins, size=X.shape[0])
        # magnitude-free transforms collapse all examples onto bin 0
        bin_ids[n_strengths[op_ids] == 1] = 0
        keys = op_ids * self.num_bins + bin_ids

        out_X, out_y = X.clone(), y.clone()
        for key in np.unique(keys):
            group = torch.from_numpy(np.flatnonzero(keys == key)).to(X.device)
            op_i, bin_j = divmod(int(key), self.num_bins)
            entry = self._pool[op_i]
            if entry.variants is not None:
                tr_X, tr_y = entry.variants[bin_j](X[group], y[group])
            else:
                tr_X, tr_y = self._apply_dynamic(
                    entry.name, entry.strengths[bin_j], X[group], y[group]
                )
            out_X[group] = tr_X.to(out_X.dtype)
            out_y[group] = tr_y
        return out_X, out_y
