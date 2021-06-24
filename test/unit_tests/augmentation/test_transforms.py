# Authors: Cédric Rommel <cpe.rommel@gmail.com>
#
# License: BSD (3-clause)

import pytest
import numpy as np
from scipy.fft import fft, fftfreq, fftshift
from scipy.signal import find_peaks, welch
from sklearn.utils import check_random_state
import torch

from braindecode.augmentation.transforms import (
    TimeReverse, SignFlip, FTSurrogate, MissingChannels, ShuffleChannels,
    GaussianNoise, ChannelSymmetry, TimeMask, BandstopFilter, FrequencyShift,
    RandomZRotation, RandomYRotation, RandomXRotation, Mixup
)
from braindecode.augmentation.functional import (
    _freq_shift, sensors_rotation, get_standard_10_20_positions
)
from test.unit_tests.augmentation.test_base import common_tranform_assertions


@pytest.fixture
def time_aranged_batch(batch_size=5):
    """Generates a batch of size 1, where the feature matrix has 64 repeated
    rows of integers aranged between 0 and 49.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    X = torch.stack(
        [torch.stack([torch.arange(50, device=device)] * 64)] * batch_size
    ).float()
    return X, torch.zeros(batch_size)


@pytest.fixture
def ch_aranged_batch(time_aranged_batch):
    """Generates a batch of size 1, where the feature matrix has 50 repeated
    columns of integers aranged between 0 and 63.
    """
    X, y = time_aranged_batch
    return X.transpose(1, 2), y


def test_flip_transform(time_aranged_batch):
    X, y = time_aranged_batch
    flip_transform = TimeReverse(1.0)

    device = X.device.type
    expected_tensor = np.stack([np.arange(50)] * 64)
    expected_tensor = torch.as_tensor(
        expected_tensor[:, ::-1].copy(), device=device
    ).repeat(X.shape[0], 1, 1).float()

    common_tranform_assertions(
        time_aranged_batch,
        flip_transform(*time_aranged_batch),
        expected_tensor
    )


def test_sign_transform(time_aranged_batch):
    X, y = time_aranged_batch
    sign_flip_transform = SignFlip(1.0)

    device = X.device.type
    expected_tensor = np.stack([-np.arange(50)] * 64)
    expected_tensor = torch.as_tensor(
        expected_tensor.copy(), device=device
    ).repeat(X.shape[0], 1, 1).float()

    common_tranform_assertions(
        time_aranged_batch,
        sign_flip_transform(*time_aranged_batch),
        expected_tensor
    )


@pytest.mark.parametrize("even,magnitude", [
    (False, 1,),
    (True, 1),
    (True, 0.5),
])
def test_ft_surrogate_transforms(
    random_batch,
    even,
    magnitude,
):
    if even:
        X, y = random_batch
        random_batch = X.repeat(1, 1, 2), y
    transform = FTSurrogate(
        probability=1,
        magnitude=magnitude,
    )
    common_tranform_assertions(random_batch, transform(*random_batch))


def ones_and_zeros_batch(zeros_ratio=0., shape=None, batch_size=100):
    """Generates a batch of size one, where the feature matrix (of size 66x50)
    contains rows full of zeros first, then rows full of ones.

    Parameters
    ----------
    zeros_ratio : float, optional
        Ratio of rows to be set to 0. Must be between 0 and 1. By default 0.
    """
    assert isinstance(zeros_ratio, float)
    assert zeros_ratio <= 1 and zeros_ratio >= 0
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if shape is None:
        X = torch.ones(batch_size, 66, 50, device=device)
    else:
        X = torch.ones(batch_size, *shape, device=device)
    nb_zero_rows = int(round(X.shape[1] * zeros_ratio))
    X[:, :nb_zero_rows, :] *= 0
    return X, torch.zeros(batch_size)


@pytest.mark.parametrize("proba_drop", [0.25, 0.5])
def test_missing_channels_transform(rng_seed, proba_drop):
    ones_batch = ones_and_zeros_batch()
    X, y = ones_batch
    transform = MissingChannels(
        1, proba_drop=proba_drop, random_state=rng_seed
    )
    new_batch = transform(*ones_batch)
    tr_X, _ = new_batch
    common_tranform_assertions(ones_batch, new_batch)
    zeros_mask = np.all(tr_X.cpu().numpy() <= 1e-3, axis=-1)
    average_nb_of_zero_rows = np.mean(np.sum(zeros_mask.astype(int), axis=-1))
    expected_nb_zero_rows = transform.proba_drop * X.shape[-2]
    # test that the expected number of channels was set to zero
    assert np.abs(average_nb_of_zero_rows - expected_nb_zero_rows) <= 1
    # test that channels are conserved (same across it)
    assert all([torch.equal(tr_X[0, :, 0], tr_X[0, :, i])
                for i in range(tr_X.shape[2])])


@pytest.mark.parametrize("proba_shuffle", [0.25, 0.5])
def test_shuffle_channels(rng_seed, ch_aranged_batch, proba_shuffle):
    X, y = ch_aranged_batch
    transform = ShuffleChannels(
        1, proba_shuffle=proba_shuffle, random_state=rng_seed
    )
    new_batch = transform(*ch_aranged_batch)
    tr_X, _ = new_batch
    common_tranform_assertions(ch_aranged_batch, new_batch)
    # test that rows (channels) are conserved
    assert all([torch.equal(tr_X[0, :, 0], tr_X[0, :, i])
                for i in range(tr_X.shape[2])])
    # test that rows (channels) have been shuffled
    assert not torch.equal(tr_X[0, :, :], X)
    # test that number of shuffled channels is correct
    batch_size, n_channels, _ = tr_X.shape
    n_shuffled_channels = np.sum(
        [
            not torch.equal(tr_X[k, i, :], X[k, i, :])
            for i in range(n_channels)
            for k in range(batch_size)
        ]
    )
    theor_n_shuffled_channels = int(
        round(proba_shuffle * n_channels * batch_size)
    )
    # Check we are within 5% of asymptotic number of shuffled channels
    assert (
        theor_n_shuffled_channels - n_shuffled_channels
    )/theor_n_shuffled_channels < 0.05


def test_gaussian_noise(rng_seed):
    ones_batch = ones_and_zeros_batch(shape=(1000, 1000))
    X, y = ones_batch
    std = 2.0
    transform = GaussianNoise(
        1,
        std=std,
        random_state=rng_seed
    )
    new_batch = transform(*ones_batch)
    tr_X, _ = new_batch
    common_tranform_assertions(ones_batch, new_batch)

    # check that the values of X changed, but the rows and cols means are
    # unchanged (within Gaussian confidence interval)
    assert not torch.equal(tr_X, X)
    assert torch.mean(
        (torch.abs(torch.mean(tr_X, 1) - 1.0) < 1.96 * std).float()
    ) > 0.95
    assert torch.mean(
        (torch.abs(torch.mean(tr_X, 2) - 1.0) < 1.96 * std).float()
    ) > 0.95


def test_channel_symmetry():
    batch_size = 5
    seq_len = 64
    X = torch.stack([torch.stack([torch.arange(21)] * seq_len).T] * batch_size)

    ch_names = [
        'Fp1', 'Fpz', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
        'T3', 'C3', 'Cz', 'C4', 'T4',
        'T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'Oz', 'O2'
    ]
    transform = ChannelSymmetry(1, ch_names)

    expected_perm = [
        2, 1, 0, 7, 6, 5, 4, 3, 12, 11, 10, 9, 8, 17, 16, 15, 14,
        13, 20, 19, 18
    ]
    expected_tensor = X[:, expected_perm, :]

    ordered_batch = (X, torch.zeros(batch_size))

    common_tranform_assertions(
        ordered_batch,
        transform(*ordered_batch),
        expected_tensor
    )


@pytest.mark.parametrize("mask_len_samples,fail", [
    (0.5, True),
    (5, False),
    (10, False),
])
def test_time_mask(rng_seed, random_batch, mask_len_samples, fail):
    if fail:
        # Check TimeMask with max length smaller than 1 cannot be instantiated
        with pytest.raises(AssertionError):
            transform = TimeMask(
                1.0,
                mask_len_samples=mask_len_samples,
                random_state=rng_seed
            )
    else:
        transform = TimeMask(
            1.0,
            mask_len_samples=mask_len_samples,
            random_state=rng_seed
        )
        ones_batch = ones_and_zeros_batch()
        transformed_batch = transform(*ones_batch)
        common_tranform_assertions(ones_batch, transformed_batch)

        # Check that masks are the same for all channels
        transformed_X = transformed_batch[0]
        for sample in transformed_X:
            first_channel_zeros = sample[0, :].detach().cpu().numpy() <= 1e-2
            for i in range(1, sample.shape[0]):
                assert all(
                    val <= 1e-2 for val in sample[i, first_channel_zeros]
                )
        # check that the number of zeros in the masked matrix is +- equal to
        # the mask length
        assert np.abs(np.sum(first_channel_zeros) - mask_len_samples) <= 1


@pytest.mark.parametrize("bandwidth,fail", [
    (2, False),
    (55, True),
    (1, False),
    (0., False),
])
def test_bandstop_filter(rng_seed, random_batch, bandwidth, fail):
    sfreq = 100
    if fail:
        # Check Bandstopfilter with bandwdth higher than max_freq cannot be
        # instantiated
        with pytest.raises(AssertionError):
            transform = BandstopFilter(
                1.0,
                bandwidth=bandwidth,
                sfreq=sfreq,
                random_state=rng_seed
            )
    else:
        transform = BandstopFilter(
            1.0,
            bandwidth=bandwidth,
            sfreq=sfreq,
            random_state=rng_seed
        )

        transformed_batch = transform(*random_batch)
        common_tranform_assertions(random_batch, transformed_batch)

        if transform.bandwidth > 0:
            # Transform white noise
            duration_s = 1000
            time = np.arange(0, duration_s, 1 / sfreq)
            rng = check_random_state(rng_seed)
            white_noise = rng.normal(size=time.shape[0])
            batch_size = 5
            n_channels = 2
            X = torch.as_tensor(
                [np.stack([white_noise] * n_channels)] * batch_size
            )
            transformed_noise, _ = transform(X, torch.zeros(batch_size))
            transformed_noise = transformed_noise[0][0].detach().numpy()

            # Check that the filtered bandwidth is close to filter's bandwidth
            freq, psd = welch(white_noise, fs=sfreq)
            freq, transformed_psd = welch(transformed_noise, fs=sfreq)

            # For this we say that the filtered bandwidth is where the
            # transformed psd is below 10% of the min of the psd of the
            # original signal (after removing boundary glitches)
            filtered_bandwidth = (
                np.sum(transformed_psd < 0.1 * psd[1:-1].min()) * freq.max()
            ) / psd.size
            # We expect the observed bandwidth to be smaller than the double
            # of the filter's one, and greater than half of it
            assert filtered_bandwidth < 2 * transform.bandwidth
            assert filtered_bandwidth > 0.5 * transform.bandwidth


def _get_frequency_peaks(time, signal, sfreq, min_peak_height=100):
    sp = fftshift(fft(signal))
    freq = fftshift(fftfreq(time.shape[-1], 1 / sfreq))
    peaks_idx, _ = find_peaks(sp, height=min_peak_height)
    return np.array(list(set(np.abs(freq[peaks_idx]))))


@ pytest.fixture
def make_sinusoid():
    def _make_sinusoid(sfreq, freq, duration_s, batch_size=1):
        time = torch.arange(0, duration_s, 1 / sfreq)
        sinusoid = torch.cos(2 * np.pi * freq * time)
        return time, torch.stack([torch.stack([sinusoid] * 2)] * batch_size)
    return _make_sinusoid


@ pytest.mark.parametrize("shift", [1, 2, 5])
def test_freq_shift_funcion(make_sinusoid, shift):
    sfreq = 100
    _, sinusoid_epoch = make_sinusoid(sfreq=sfreq, freq=20, duration_s=30)
    transformed_sinusoid = _freq_shift(
        sinusoid_epoch, sfreq, shift)[0, 0, :]
    sinusoid = sinusoid_epoch[0, 0, :]
    _, psd_orig = welch(sinusoid, sfreq, nperseg=1024)
    f, psd_shifted = welch(transformed_sinusoid, sfreq, nperseg=1024)
    shift_samples = int(shift * len(f) / f.max())

    rolled_psd = np.roll(psd_orig, shift_samples)[shift_samples:-shift_samples]
    diff = np.abs(psd_shifted[shift_samples:-shift_samples] - rolled_psd)
    assert np.max(diff) / np.max(psd_orig) < 0.4


@ pytest.mark.parametrize("max_shift", [0., 1., 2])
def test_frequency_shift_transform(
    rng_seed, random_batch, make_sinusoid, max_shift,
):
    sfreq = 100
    transform = FrequencyShift(
        probability=1.0,
        sfreq=sfreq,
        delta_freq_range=(-max_shift, max_shift),
        random_state=rng_seed
    )

    transformed_batch = transform(*random_batch)
    common_tranform_assertions(random_batch, transformed_batch)

    # Transform a pure sinusoid with known frequency...
    freq = 5
    time, sinusoid_batch = make_sinusoid(
        sfreq=sfreq, freq=freq, duration_s=5, batch_size=100)
    transformed_sinusoid_batch, _ = transform(sinusoid_batch, torch.zeros(100))
    transformed_sinusoid_batch = transformed_sinusoid_batch[:, 0, :]
    transformed_sinusoid_batch = transformed_sinusoid_batch.detach().numpy()

    # Check that frequencies are shifted
    shifted_frequencies = np.hstack([
        _get_frequency_peaks(time, transformed_sinusoid, sfreq)
        for transformed_sinusoid in transformed_sinusoid_batch
    ])
    effective_freq_shifts = shifted_frequencies - freq
    if max_shift > 0:  # Unless the allowed shift is 0...
        assert np.abs(effective_freq_shifts).std() > 0

    # ... and that shifts are within desired range
    assert np.abs(effective_freq_shifts).max() <= max_shift


def test_rotate_signals():
    channels = ['C4', 'C3']
    batch_size = 5
    positions_matrix = torch.as_tensor(
        get_standard_10_20_positions(ordered_ch_names=channels),
        dtype=torch.float
    )
    signal_length = 300
    zero_one_X = torch.stack([
        torch.zeros(signal_length),
        torch.ones(signal_length)
    ]).repeat(batch_size, 1, 1)
    angles = [180] * batch_size
    transformed_X, _ = sensors_rotation(
        zero_one_X, torch.zeros(batch_size), positions_matrix, 'z', angles,
        spherical_splines=True
    )
    expected_X = torch.stack([
        torch.ones(signal_length),
        torch.zeros(signal_length)
    ]).repeat(batch_size, 1, 1)
    assert torch.all(torch.abs(expected_X - transformed_X) < 0.02)


@ pytest.mark.parametrize("rotation,max_degrees,fail", [
    (RandomXRotation, 15, False),
    (RandomYRotation, 15, False),
    (RandomZRotation, 15, False),
    (RandomZRotation, -15, True),
])
def test_random_rotations(
    rng_seed,
    random_batch,
    rotation,
    max_degrees,
    fail
):
    channels = ['O2', 'C4', 'C3', 'F4', 'F3', 'O1']
    if fail:
        # Check Bandstopfilter with bandwdth higher than max_freq cannot be
        # instantiated
        with pytest.raises(AssertionError):
            transform = rotation(
                1.0,
                channels,
                max_degrees=max_degrees,
                random_state=rng_seed,
            )
    else:
        X, y = random_batch
        X = X[:, :6, :]
        cropped_random_batch = X, y
        transform = rotation(
            1.0,
            channels,
            max_degrees=max_degrees,
            random_state=rng_seed,
        )
        transformed_batch = transform(*cropped_random_batch)
        common_tranform_assertions(cropped_random_batch, transformed_batch)


@ pytest.mark.parametrize("alpha,beta_per_sample", [
    (0.5, False),
    (0.5, True),
    (-.1, True)
])
def test_mixup(rng_seed, random_batch, alpha, beta_per_sample):
    transform = Mixup(
        alpha=alpha,
        beta_per_sample=beta_per_sample,
        random_state=rng_seed
    )
    batch_size = random_batch[0].shape[0]
    random_batch = (random_batch[0], torch.arange(batch_size))
    X, y = random_batch
    transformed_batch = transform(*random_batch)

    X_t, y_t = transformed_batch
    idx, idx_perm, lam = y_t

    # y_t[0] should equal y
    assert torch.equal(idx, y)
    # basic mixup
    for i in range(batch_size):
        mixed = lam[i] * X[i] \
            + (1 - lam[i]) * X[idx_perm[i]]
        assert torch.equal(X_t[i], mixed)
    # all lam should be equal
    if not beta_per_sample:
        assert torch.equal(lam, torch.ones_like(lam) * lam[0])
    # no mixup
    if alpha < 0:
        assert torch.equal(lam, torch.ones_like(lam))
        assert torch.equal(X_t, X)
