from numbers import Real

import numpy as np
from scipy.interpolate import Rbf
from sklearn.utils import check_random_state
import torch
from torch.fft import fft, ifft
from torch.nn.functional import pad, one_hot
from mne.filter import notch_filter
from mne.channels import make_standard_montage


def identity(x, y, *args, **kwargs):
    return (x, y)


def time_reverse(X, y, *args, **kwargs):
    return torch.flip(X, [-1]), y


def sign_flip(X, y, *args, **kwargs):
    return -X, y


def downsample_shift_from_arrays(X, y, factor, offset, *args, **kwargs):
    return X[..., offset::factor], y


def _new_random_fft_phase_odd(n, device, random_state=None):
    rng = check_random_state(random_state)
    random_phase = torch.from_numpy(
        2j * np.pi * rng.random((n - 1) // 2)
    ).to(device)
    return torch.cat([
        torch.as_tensor([0.0], device=device),
        random_phase,
        -torch.flip(random_phase, [-1])
    ])


def _new_random_fft_phase_even(n, device, random_state=None):
    rng = check_random_state(random_state)
    random_phase = torch.from_numpy(
        2j * np.pi * rng.random(n // 2 - 1)
    ).to(device)
    return torch.cat([
        torch.as_tensor([0.0], device=device),
        random_phase,
        torch.as_tensor([0.0], device=device),
        -torch.flip(random_phase, [-1])
    ])


_new_random_fft_phase = {
    0: _new_random_fft_phase_even,
    1: _new_random_fft_phase_odd
}


def _fft_surrogate(x=None, f=None, eps=1, random_state=None):
    """ FT surrogate augmentation of a single EEG channel, as proposed in [1]_

    Function copied from https://github.com/cliffordlab/sleep-convolutions-tf
    and modified.

    MIT License

    Copyright (c) 2018 Clifford Lab

    Permission is hereby granted, free of charge, to any person obtaining a
    copy of this software and associated documentation files (the "Software"),
    to deal in the Software without restriction, including without limitation
    the rights to use, copy, modify, merge, publish, distribute, sublicense,
    and/or sell copies of the Software, and to permit persons to whom the
    Software is furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in
    all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
    DEALINGS IN THE SOFTWARE.

    Parameters
    ----------
    x: torch.tensor, optional
        Single EEG channel signal in time space. Should not be passed if f is
        given. Defaults to None.
    f: torch.tensor, optional
        Fourier spectrum of a single EEG channel signal. Should not be passed
        if x is given. Defaults to None.
    eps: float, optional
        Float between 0 and 1 setting the range over which the phase
        pertubation is uniformly sampled: [0, `eps` * 2 * `pi`]. Defaults to 1.
    random_state: int | numpy.random.Generator, optional
        By default None.

    References
    ----------
    .. [1] Schwabedal, J. T., Snyder, J. C., Cakmak, A., Nemati, S., &
       Clifford, G. D. (2018). Addressing Class Imbalance in Classification
       Problems of Noisy Signals by using Fourier Transform Surrogates. arXiv
       preprint arXiv:1806.08675.
    """
    assert isinstance(
        eps,
        (Real, torch.FloatTensor, torch.cuda.FloatTensor)
    ) and 0 <= eps <= 1, f"eps must be a float beween 0 and 1. Got {eps}."
    if f is None:
        assert x is not None, 'Neither x nor f provided.'
        f = fft(x.double(), dim=-1)
        device = x.device
    else:
        device = f.device
    n = f.shape[-1]
    random_phase = _new_random_fft_phase[n % 2](
        n,
        device=device,
        random_state=random_state
    )
    f_shifted = f * torch.exp(eps * random_phase)
    shifted = ifft(f_shifted, dim=-1)
    return shifted.real.float()


def fft_surrogate(X, y, magnitude, random_state, *args, **kwargs):
    transformed_X = _fft_surrogate(
        x=X,
        eps=magnitude,
        random_state=random_state
    )
    return transformed_X, y


def _pick_channels_randomly(X, magnitude, random_state):
    rng = check_random_state(random_state)
    batch_size, n_channels, _ = X.shape
    # allows to use the same RNG
    unif_samples = torch.as_tensor(
        rng.uniform(0, 1, size=(batch_size, n_channels)),
        dtype=torch.float,
        device=X.device,
    )
    # equivalent to a 0s and 1s mask, but allows to backprop through
    # could also be done using torch.distributions.RelaxedBernoulli
    return torch.sigmoid(1000*(unif_samples - magnitude)).to(X.device)


def channel_dropout(X, y, magnitude, random_state, *args, **kwargs):
    mask = _pick_channels_randomly(X, magnitude, random_state)
    return X * mask.unsqueeze(-1), y


def _make_permutation_matrix(X, mask, random_state):
    rng = check_random_state(random_state)
    batch_size, n_channels, _ = X.shape
    hard_mask = mask.round()
    batch_permutations = torch.empty(
        batch_size, n_channels, n_channels, device=X.device
    )
    for b, mask in enumerate(hard_mask):
        channels_to_shuffle = torch.arange(n_channels)
        channels_to_shuffle = channels_to_shuffle[mask.bool()]
        channels_permutation = np.arange(n_channels)
        channels_permutation[channels_to_shuffle] = rng.permutation(
            channels_to_shuffle
        )
        channels_permutation = torch.as_tensor(
            channels_permutation, dtype=torch.int64, device=X.device
        )
        batch_permutations[b, ...] = one_hot(channels_permutation)
    return batch_permutations


def channel_shuffle(X, y, magnitude, random_state, *args, **kwargs):
    mask = _pick_channels_randomly(X, 1-magnitude, random_state)
    batch_permutations = _make_permutation_matrix(X, mask, random_state)
    return torch.matmul(batch_permutations, X), y


def add_gaussian_noise(X, y, std, random_state, *args, **kwargs):
    # XXX: Maybe have rng passed as argument here
    rng = check_random_state(random_state)
    noise = torch.from_numpy(
        rng.normal(
            loc=np.zeros(X.shape),
            scale=1
        )
    ).float().to(X.device) * std
    transformed_X = X + noise
    return transformed_X, y


def permute_channels(X, y, permutation, *args, **kwargs):
    return X[..., permutation, :], y


def _sample_mask_start(X, mask_len_samples, random_state):
    rng = check_random_state(random_state)
    seq_length = torch.as_tensor(X.shape[-1], device=X.device)
    mask_start = torch.as_tensor(rng.uniform(
        low=0, high=1, size=X.shape[0],
    ), device=X.device) * (seq_length - mask_len_samples)
    return mask_start


def _mask_time(X, mask_start_per_sample, mask_len_samples):
    mask = torch.ones_like(X)
    for i, start in enumerate(mask_start_per_sample):
        mask[i, :, start:start + mask_len_samples] = 0
    return X * mask


def _relaxed_mask_time(X, mask_start_per_sample, mask_len_samples):
    batch_size, n_channels, seq_len = X.shape
    t = torch.arange(seq_len, device=X.device).float()
    t = t.repeat(batch_size, n_channels, 1)
    mask_start_per_sample = mask_start_per_sample.view(-1, 1, 1)
    s = 1000 * 10**-np.log10(seq_len)
    mask = 2 - (
        torch.sigmoid(s * (t - mask_start_per_sample)) +
        torch.sigmoid(s * -(t - mask_start_per_sample - mask_len_samples))
    ).float().to(X.device)
    return X * mask


def random_time_mask(X, y, mask_len_samples, random_state, *args, **kwargs):
    mask_start = _sample_mask_start(X, mask_len_samples, random_state)
    return _relaxed_mask_time(X, mask_start, mask_len_samples), y


def random_bandstop(X, y, bandwidth, max_freq, sfreq, random_state, *args,
                    **kwargs):
    rng = check_random_state(random_state)
    transformed_X = X.clone()
    # Prevents transitions from going below 0 and above max_freq
    notched_freqs = rng.uniform(
        low=1 + 2 * bandwidth,
        high=max_freq - 1 - 2 * bandwidth,
        size=X.shape[0]
    )

    # I just worry that this might be to complex for gradient descent and
    # it would be a shame to make a straight-through here... A new version
    # using torch convolution might be necessary...
    for c, (sample, notched_freq) in enumerate(
            zip(transformed_X, notched_freqs)):
        sample = sample.cpu().numpy().astype(np.float64)
        transformed_X[c] = torch.as_tensor(notch_filter(
            sample,
            Fs=sfreq,
            freqs=notched_freq,
            method='fir',
            notch_widths=bandwidth,
            verbose=False
        ))
    return transformed_X, y


def hilbert_transform(x):
    if torch.is_complex(x):
        raise ValueError("x must be real.")

    N = x.shape[-1]
    f = fft(x, N, dim=-1)
    h = torch.zeros_like(f)
    if N % 2 == 0:
        h[..., 0] = h[..., N // 2] = 1
        h[..., 1:N // 2] = 2
    else:
        h[..., 0] = 1
        h[..., 1:(N + 1) // 2] = 2

    return ifft(f * h, dim=-1)


def nextpow2(n):
    """Return the first integer N such that 2**N >= abs(n)"""
    return int(np.ceil(np.log2(np.abs(n))))


def _freq_shift(x, fs, f_shift):
    """
    Shift the specified signal by the specified frequency.

    See https://gist.github.com/lebedov/4428122
    """
    # Pad the signal with zeros to prevent the FFT invoked by the transform
    # from slowing down the computation:
    n_channels, N_orig = x.shape[-2:]
    N_padded = 2 ** nextpow2(N_orig)
    t = torch.arange(N_padded, device=x.device) / fs
    padded = pad(x, (0, N_padded - N_orig))
    analytical = hilbert_transform(padded)
    if isinstance(f_shift, (float, int, np.ndarray, list)):
        f_shift = torch.as_tensor(f_shift).float()
    reshaped_f_shift = f_shift.repeat(
        N_padded, n_channels, 1).T
    shifted = analytical * torch.exp(2j * np.pi * reshaped_f_shift * t)
    return shifted[..., :N_orig].real.float()


def freq_shift(X, y, max_shift, sfreq, random_state, *args, **kwargs):
    rng = check_random_state(random_state)
    delta_freq = torch.as_tensor(
        rng.uniform(size=X.shape[0]), device=X.device) * max_shift
    transformed_X = _freq_shift(
        x=X,
        fs=sfreq,
        f_shift=delta_freq,
    )
    return transformed_X, y


def _torch_normalize_vectors(rr):
    """Normalize surface vertices."""
    new_rr = rr.clone()
    size = torch.linalg.norm(rr, axis=1)
    mask = (size > 0)
    new_rr[mask] = rr[mask] / size[mask].unsqueeze(-1)
    return new_rr


def _torch_legval(x, c, tensor=True):
    """
    Evaluate a Legendre series at points x.
    If `c` is of length `n + 1`, this function returns the value:
    .. math:: p(x) = c_0 * L_0(x) + c_1 * L_1(x) + ... + c_n * L_n(x)
    The parameter `x` is converted to an array only if it is a tuple or a
    list, otherwise it is treated as a scalar. In either case, either `x`
    or its elements must support multiplication and addition both with
    themselves and with the elements of `c`.
    If `c` is a 1-D array, then `p(x)` will have the same shape as `x`.  If
    `c` is multidimensional, then the shape of the result depends on the
    value of `tensor`. If `tensor` is true the shape will be c.shape[1:] +
    x.shape. If `tensor` is false the shape will be c.shape[1:]. Note that
    scalars have shape (,).
    Trailing zeros in the coefficients will be used in the evaluation, so
    they should be avoided if efficiency is a concern.
    Parameters
    ----------
    x : array_like, compatible object
        If `x` is a list or tuple, it is converted to an ndarray, otherwise
        it is left unchanged and treated as a scalar. In either case, `x`
        or its elements must support addition and multiplication with
        with themselves and with the elements of `c`.
    c : array_like
        Array of coefficients ordered so that the coefficients for terms of
        degree n are contained in c[n]. If `c` is multidimensional the
        remaining indices enumerate multiple polynomials. In the two
        dimensional case the coefficients may be thought of as stored in
        the columns of `c`.
    tensor : boolean, optional
        If True, the shape of the coefficient array is extended with ones
        on the right, one for each dimension of `x`. Scalars have dimension 0
        for this action. The result is that every column of coefficients in
        `c` is evaluated for every element of `x`. If False, `x` is broadcast
        over the columns of `c` for the evaluation.  This keyword is useful
        when `c` is multidimensional. The default value is True.
        .. versionadded:: 1.7.0
    Returns
    -------
    values : ndarray, algebra_like
        The shape of the return value is described above.
    See Also
    --------
    legval2d, leggrid2d, legval3d, leggrid3d
    Notes
    -----
    The evaluation uses Clenshaw recursion, aka synthetic division.
    Examples
    --------
    """
    c = torch.as_tensor(c)
    c = c.double()
    if isinstance(x, (tuple, list)):
        x = torch.as_tensor(x)
    if isinstance(x, torch.Tensor) and tensor:
        c = c.view(c.shape + (1,)*x.ndim)

    c = c.to(x.device)

    if len(c) == 1:
        c0 = c[0]
        c1 = 0
    elif len(c) == 2:
        c0 = c[0]
        c1 = c[1]
    else:
        nd = len(c)
        c0 = c[-2]
        c1 = c[-1]
        for i in range(3, len(c) + 1):
            tmp = c0
            nd = nd - 1
            c0 = c[-i] - (c1*(nd - 1))/nd
            c1 = tmp + (c1*x*(2*nd - 1))/nd
    return c0 + c1*x


def _torch_calc_g(cosang, stiffness=4, n_legendre_terms=50):
    """Calculate spherical spline g function between points on a sphere.
    Parameters
    ----------
    cosang : array-like of float, shape(n_channels, n_channels)
        cosine of angles between pairs of points on a spherical surface. This
        is equivalent to the dot product of unit vectors.
    stiffness : float
        stiffness of the spline.
    n_legendre_terms : int
        number of Legendre terms to evaluate.
    Returns
    -------
    G : np.ndrarray of float, shape(n_channels, n_channels)
        The G matrix.
    """
    factors = [(2 * n + 1) / (n ** stiffness * (n + 1) ** stiffness *
                              4 * np.pi)
               for n in range(1, n_legendre_terms + 1)]
    return _torch_legval(cosang, [0] + factors)


def _torch_make_interpolation_matrix(pos_from, pos_to, alpha=1e-5):
    """Compute interpolation matrix based on spherical splines.
    Implementation based on [1]
    Parameters
    ----------
    pos_from : np.ndarray of float, shape(n_good_sensors, 3)
        The positions to interpoloate from.
    pos_to : np.ndarray of float, shape(n_bad_sensors, 3)
        The positions to interpoloate.
    alpha : float
        Regularization parameter. Defaults to 1e-5.
    Returns
    -------
    interpolation : np.ndarray of float, shape(len(pos_from), len(pos_to))
        The interpolation matrix that maps good signals to the location
        of bad signals.
    References
    ----------
    [1] Perrin, F., Pernier, J., Bertrand, O. and Echallier, JF. (1989).
        Spherical splines for scalp potential and current density mapping.
        Electroencephalography Clinical Neurophysiology, Feb; 72(2):184-7.
    """
    pos_from = pos_from.clone()
    pos_to = pos_to.clone()
    n_from = pos_from.shape[0]
    n_to = pos_to.shape[0]

    # normalize sensor positions to sphere
    pos_from = _torch_normalize_vectors(pos_from)
    pos_to = _torch_normalize_vectors(pos_to)

    # cosine angles between source positions
    cosang_from = torch.matmul(pos_from, pos_from.T)
    cosang_to_from = torch.matmul(pos_to, pos_from.T)
    G_from = _torch_calc_g(cosang_from)
    G_to_from = _torch_calc_g(cosang_to_from)
    assert G_from.shape == (n_from, n_from)
    assert G_to_from.shape == (n_to, n_from)

    if alpha is not None:
        G_from.flatten()[::len(G_from) + 1] += alpha

    device = G_from.device
    C = torch.vstack([
            torch.hstack([G_from, torch.ones((n_from, 1), device=device)]),
            torch.hstack([
                torch.ones((1, n_from), device=device),
                torch.as_tensor([[0]], device=device)])
        ])
    C_inv = torch.linalg.pinv(C)

    interpolation = torch.hstack([
        G_to_from,
        torch.ones((n_to, 1), device=device)
    ]).matmul(C_inv[:, :-1])
    assert interpolation.shape == (n_to, n_from)
    return interpolation


def _rotate_signals(X, rotations, sensors_positions_matrix, spherical=True):
    sensors_positions_matrix = sensors_positions_matrix.to(X.device)
    rot_sensors_matrices = [
        rotation.matmul(sensors_positions_matrix) for rotation in rotations
    ]
    if spherical:
        interpolation_matrix = torch.stack(
            [torch.as_tensor(
                _torch_make_interpolation_matrix(
                    sensors_positions_matrix.T, rot_sensors_matrix.T
                ), device=X.device
            ).float() for rot_sensors_matrix in rot_sensors_matrices]
        )
        return torch.matmul(interpolation_matrix, X)
    else:
        transformed_X = X.clone()
        sensors_positions = list(sensors_positions_matrix)
        for s, rot_sensors_matrix in enumerate(rot_sensors_matrices):
            rot_sensors_positions = list(rot_sensors_matrix.T)
            for time in range(X.shape[-1]):
                interpolator_t = Rbf(*sensors_positions, X[s, :, time])
                transformed_X[s, :, time] = torch.from_numpy(
                    interpolator_t(*rot_sensors_positions)
                )
        return transformed_X


def make_rotation_matrix(axis, angle, degrees=True):
    assert axis in ['x', 'y', 'z'], "axis should be either x, y or z."

    if isinstance(angle, (float, int, np.ndarray, list)):
        angle = torch.as_tensor(angle)

    if degrees:
        angle = angle * np.pi / 180

    device = angle.device
    zero = torch.zeros(1, device=device)
    rot = torch.stack([
        torch.as_tensor([1, 0, 0], device=device),
        torch.hstack([zero, torch.cos(angle), -torch.sin(angle)]),
        torch.hstack([zero, torch.sin(angle), torch.cos(angle)]),
    ])
    if axis == "x":
        return rot
    elif axis == "y":
        rot = rot[[2, 0, 1], :]
        return rot[:, [2, 0, 1]]
    else:
        rot = rot[[1, 2, 0], :]
        return rot[:, [1, 2, 0]]


def random_rotation(X, y, axis, max_degrees, sensors_positions_matrix,
                    spherical_splines, random_state, *args, **kwargs):
    rng = check_random_state(random_state)
    random_angles = torch.as_tensor(rng.uniform(
        low=0,
        high=1,
        size=X.shape[0]
    ), device=X.device) * 2 * max_degrees - max_degrees
    rots = [
        make_rotation_matrix(axis, random_angle, degrees=True)
        for random_angle in random_angles
    ]
    rotated_X = _rotate_signals(
        X, rots, sensors_positions_matrix, spherical_splines
    )
    return rotated_X, y


def get_standard_10_20_positions(raw_or_epoch=None, ordered_ch_names=None):
    """Returns standard 10-20 sensors position matrix (for instantiating
    RandomSensorsRotation for example).

    Parameters
    ----------
    raw_or_epoch : mne.io.Raw | mne.Epoch, optional
        Example of raw or epoch to retrive ordered channels list from. Need to
        be named as in 10-20. By default None.
    ordered_ch_names : list, optional
        List of strings representing the channels of the montage considered.
        The order has to be consistent with the order of channels in the input
        matrices that will be fed to `RandomSensorsRotation` transform. By
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


def mixup(X, y, alpha, beta_per_sample, random_state=None, magnitude=None):
    """Mixes two channels of EEG data. See [1]_.
    Implementation based on [2]_.

    Parameters
    ----------
    X:  torch.tensor
        EEG data in form 'batch_size', 'n_channels', 'n_times'
    y : torch.tensor
        Target of length 'batch_size'
    alpha : float
        mixup hyperparameter.
    beta_per_sample: bool (default=False)
        by default, one mixing coefficient per batch is drawn from an beta
        distribution. If True, one mixing coefficient per sample is drawn.
    Returns
    -------
    tuple
        'X', 'y'. Where 'X' is augmented and 'y' is a tuple  of length 3
        containing the labels of the two mixed channels and the mixing coefficient.

    References
    ----------
        [1] Hongyi Zhang, Moustapha Cisse, Yann N. Dauphin, David Lopez-Paz
        mixup: Beyond Empirical Risk Minimization
        Online: https://arxiv.org/abs/1710.09412
        [2] https://github.com/facebookresearch/mixup-cifar10/blob/master/train.py
    """
    rng = check_random_state(random_state)

    device = X.device
    batch_size, n_channels, n_times = X.shape

    if alpha > 0:
        if beta_per_sample:
            lam = torch.tensor(rng.beta(alpha, alpha, batch_size)).to(device)
        else:
            lam = torch.ones(batch_size).to(device)
            lam *= rng.beta(alpha, alpha)
    else:
        lam = torch.ones(batch_size).to(device)

    idx_perm = torch.tensor(rng.permutation(batch_size,))
    X_mix = torch.zeros((batch_size, n_channels, n_times)).to(device)
    y_a = torch.arange(batch_size).to(device)
    y_b = torch.arange(batch_size).to(device)

    for idx in range(batch_size):
        X_mix[idx] = lam[idx] * X[idx] \
            + (1 - lam[idx]) * X[idx_perm[idx]]
        y_a[idx] = y[idx]
        y_b[idx] = y[idx_perm[idx]]

    return X_mix, (y_a, y_b, lam)
