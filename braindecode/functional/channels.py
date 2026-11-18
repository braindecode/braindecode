# Authors: Bruno Aristimunha <b.aristimunha@gmail.com>
#
# License: BSD (3-clause)
r"""Channels as linear measurements of the scalp potential.

Every channel is a linear combination of electrode potentials ``v``: ``A`` is
``V(A)`` against the recording reference, ``A-B`` is ``V(A) - V(B)``
(:func:`derivation_matrix`). The inputs are ``y = A v`` and the channels a
model wants are ``t = B v``, so re-montaging, re-referencing and interpolation
are all one linear estimate of ``B v`` from ``A v`` (:func:`linear_channel_map`).
"""

from __future__ import annotations

import re
import warnings
from functools import lru_cache
from itertools import chain
from typing import NamedTuple

import mne
import numpy as np
import torch
from mne.io.constants import FIFF
from scipy.cluster.vq import kmeans2
from scipy.linalg import solve
from scipy.spatial.distance import cdist
from torch import Tensor

from braindecode.util import resolve_montage_name


def _channel_electrodes(name):
    """Return the two electrodes of a bipolar channel, or its literal name."""
    ends = name.split("-")
    return ends if len(ends) == 2 and all(ends) else [name]


def _derivation_rows(parts, electrodes):
    """Build signed rows, keeping the first spelling of each electrode."""
    names = {}
    for name in chain(electrodes, chain.from_iterable(parts)):
        names.setdefault(name.lower(), name)
    columns = {name: i for i, name in enumerate(names)}
    matrix = torch.zeros(len(parts), len(names), dtype=torch.float64)
    for row, ends in enumerate(parts):
        matrix[row, columns[ends[0].lower()]] += 1
        if len(ends) == 2:
            matrix[row, columns[ends[1].lower()]] -= 1
    return matrix, list(names.values())


def derivation_matrix(
    ch_names: list[str],
    electrodes: list[str] | None = None,
    reference: str | None = None,
) -> tuple[Tensor, list[str]]:
    """Each channel as a row over electrode potentials.

    Parameters
    ----------
    ch_names : list of str
        ``"A"`` is ``V(A)``, ``"A-B"`` is ``V(A) - V(B)``. Names are compared
        ignoring case; any other name is one electrode.
    electrodes : list of str, optional
        Columns (lower case). Defaults to the electrodes of ``ch_names`` in
        order of appearance; electrodes missing from the list are added.
    reference : str, optional
        Reference of the single-electrode channels: ``None`` (the recording's,
        left implicit), an electrode name (``"Cz"``: ``A`` is ``V(A) - V(Cz)``)
        or ``"average"`` (``A`` minus the mean of the single-electrode channels).

    Returns
    -------
    matrix : torch.Tensor
        ``(n_channels, n_electrodes)``, float64.
    electrodes : list of str
        The columns.
    """
    parts = [_channel_electrodes(name.lower()) for name in ch_names]
    if reference is not None and reference != "average":
        parts = [p if len(p) == 2 else [p[0], reference.lower()] for p in parts]
    matrix, electrodes = _derivation_rows(parts, (e.lower() for e in electrodes or []))
    if reference == "average":
        single = torch.tensor([len(p) == 1 for p in parts], dtype=torch.bool)
        matrix[single] -= matrix[single].mean(0)
    return matrix, electrodes


def linear_channel_map(
    A: Tensor,
    B: Tensor,
    cov: Tensor,
    noise: float = 0.0,
    flat: Tensor | None = None,
    match_gain: bool = False,
    atol: float = 1e-9,
) -> tuple[Tensor, Tensor]:
    r"""Best linear estimate of the channels ``B v`` from the channels ``A v``.

    The electrode potentials :math:`v` have prior covariance :math:`\Sigma`
    (``cov``), plus an unknown offset along each column of :math:`F`
    (``flat``: a flat prior, e.g. the unknown recording reference). The inputs
    are :math:`y = A v + \varepsilon` with :math:`\varepsilon \sim
    \mathcal{N}(0, \sigma^2 I)` (``noise``) and the targets are :math:`t = B v`.
    The best linear unbiased estimate :math:`\hat t = W y` solves

    .. math::

        \begin{bmatrix} A \Sigma A^\top + \sigma^2 I & A F \\ (A F)^\top & 0
        \end{bmatrix}
        \begin{bmatrix} W^\top \\ M \end{bmatrix} =
        \begin{bmatrix} A \Sigma B^\top \\ (B F)^\top \end{bmatrix},
        \qquad
        \operatorname{var}(t_k - \hat t_k) = \big[B \Sigma B^\top
        - W A \Sigma B^\top - (B F) M\big]_{kk},

    which is kriging [Matheron1963]_, universal kriging when :math:`F` is
    given [Matheron1969]_, and the posterior mean and variance of a Gaussian
    process [Rasmussen2006]_. Without ``flat`` it is
    :math:`W = B \Sigma A^\top (A \Sigma A^\top + \sigma^2 I)^{-1}`. The choice
    of :math:`\Sigma` gives the known methods:

    - :math:`\Sigma = 0` and :math:`F = I` (every potential unknown): the
      exact derivations of :func:`exact_derivation`, ``W = B A^+``, such as
      bipolar or Laplacian (source) derivations [Hjorth1975]_; a target is
      exact when :math:`B F` lies in the span of :math:`A F`.
    - the spherical-spline kernel (:func:`spherical_spline_kernel`), ``noise``
      the regularisation and :math:`F = \mathbf 1` (the unknown reference):
      spherical-spline interpolation [Perrin1989]_, which is spline smoothing on
      the sphere [Wahba1981]_ and the Gaussian-process estimate with that kernel
      [Kimeldorf1970]_.
    - :math:`\Sigma = L L^\top` for a lead field :math:`L`: the minimum-norm
      estimate [Hamalainen1994]_; with the targets taken against a point at
      infinity (zero potential) it is REST re-referencing [Yao2001]_.
    - a covariance fitted on dense recordings: the Wiener (linear MMSE) filter.

    Parameters
    ----------
    A : torch.Tensor
        ``(n_inputs, n_electrodes)`` input channels (:func:`derivation_matrix`).
    B : torch.Tensor
        ``(n_targets, n_electrodes)`` target channels.
    cov : torch.Tensor
        ``(n_electrodes, n_electrodes)`` prior covariance :math:`\Sigma`.
    noise : float
        Input noise variance :math:`\sigma^2` (the regularisation).
    flat : torch.Tensor, optional
        ``(n_electrodes, n_flat)`` directions :math:`F` with a flat prior.
    match_gain : bool
        Rescale the part of each row free of the flat constraint (so
        :math:`W A F = B F` still holds) so that the estimate has, as closely as
        that constraint allows, the prior variance of its
        target, :math:`[W A \Sigma A^\top W^\top]_{kk} = [B \Sigma B^\top]_{kk}`:
        the minimum-variance estimate shrinks the amplitude.
    atol : float
        Largest residual of :math:`B F` from the span of :math:`A F` for a
        target to be estimable (the flat part alone, whatever :math:`\Sigma`).

    Returns
    -------
    weight : torch.Tensor
        ``(n_targets, n_inputs)``, float64; rows of targets that cannot be
        estimated (``W A F != B F``) are 0.
    variance : torch.Tensor
        ``(n_targets,)`` posterior variance (of the estimate before
        ``match_gain``); ``inf`` where not estimable.

    References
    ----------
    .. [Matheron1963] Matheron, G. (1963). Principles of geostatistics.
       Economic Geology, 58(8), 1246-1266. https://doi.org/10.2113/gsecongeo.58.8.1246
    .. [Matheron1969] Matheron, G. (1969). Le krigeage universel. Cahiers du
       Centre de Morphologie Mathématique, 1. École des Mines de Paris.
       https://www.cg.minesparis.psl.eu/bibliotheque/public/MATHERON_Ouvrage_00131.pdf
    .. [Kimeldorf1970] Kimeldorf, G. S., & Wahba, G. (1970). A correspondence
       between Bayesian estimation on stochastic processes and smoothing by
       splines. The Annals of Mathematical Statistics, 41(2), 495-502.
       https://doi.org/10.1214/aoms/1177697089
    .. [Hjorth1975] Hjorth, B. (1975). An on-line transformation of EEG scalp
       potentials into orthogonal source derivations. Electroencephalography and
       Clinical Neurophysiology, 39(5), 526-530.
       https://doi.org/10.1016/0013-4694(75)90056-5
    .. [Wahba1981] Wahba, G. (1981). Spline interpolation and smoothing on the
       sphere. SIAM Journal on Scientific and Statistical Computing, 2(1), 5-16.
       https://doi.org/10.1137/0902002
    .. [Perrin1989] Perrin, F., Pernier, J., Bertrand, O., & Echallier, J. F.
       (1989). Spherical splines for scalp potential and current density
       mapping. Electroencephalography and Clinical Neurophysiology, 72(2),
       184-187. https://doi.org/10.1016/0013-4694(89)90180-6
    .. [Hamalainen1994] Hämäläinen, M. S., & Ilmoniemi, R. J. (1994).
       Interpreting magnetic fields of the brain: minimum norm estimates.
       Medical & Biological Engineering & Computing, 32(1), 35-42.
       https://doi.org/10.1007/BF02512476
    .. [Yao2001] Yao, D. (2001). A method to standardize a reference of scalp
       EEG recordings to a point at infinity. Physiological Measurement, 22(4),
       693-711. https://doi.org/10.1088/0967-3334/22/4/305
    .. [Rasmussen2006] Rasmussen, C. E., & Williams, C. K. I. (2006). Gaussian
       Processes for Machine Learning. MIT Press.
       https://doi.org/10.7551/mitpress/3206.001.0001
    """
    A, B, cov = (torch.as_tensor(t, dtype=torch.float64) for t in (A, B, cov))
    F = A.new_zeros(len(cov), 0) if flat is None else torch.as_tensor(flat).to(A)
    n, f = len(A), F.shape[1]
    AF, BF, K, KyT = A @ F, B @ F, A @ cov @ A.T, A @ cov @ B.T
    M = A.new_zeros(n + f, n + f)
    M[:n, :n] = K + noise * torch.eye(n, dtype=A.dtype, device=A.device)
    M[:n, n:], M[n:, :n] = AF, AF.T
    sol = torch.linalg.pinv(M) @ torch.cat([KyT, BF.T])  # SVD cutoff, as scipy/MNE
    W, mu = sol[:n].T, sol[n:]
    prior = (B @ cov * B).sum(1)
    var = prior - (W * KyT.T).sum(1) - (mu.T * BF).sum(1)
    W0 = BF @ torch.linalg.pinv(AF)  # the flat part: W0 A F = B F when estimable
    ok = (
        (W0 @ AF - BF).abs().amax(1) <= atol
        if f
        else torch.ones(len(B), dtype=torch.bool, device=A.device)
    )
    if match_gain:  # (W0 + s D) K (W0 + s D)^T = prior, D = W - W0, per row
        D = W - W0
        a, b, c = ((D @ K) * D).sum(1), ((D @ K) * W0).sum(1), ((W0 @ K) * W0).sum(1)
        root = (b**2 - a * (c - prior)).clamp_min(0).sqrt()
        W = W0 + torch.where(a > 0, (root - b) / a, 1.0)[:, None] * D
    return W * ok[:, None], torch.where(ok, var.clamp_min(0), torch.inf)


def _torch_legval(
    x: torch.Tensor, c: torch.Tensor, tensor: bool = True
) -> torch.Tensor:
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
    Code copied and modified from Numpy:
    https://github.com/numpy/numpy/blob/v1.20.0/numpy/polynomial/legendre.py#L835-L920

    Copyright (c) 2005-2021, NumPy Developers.
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are
    met:
        * Redistributions of source code must retain the above copyright
        notice, this list of conditions and the following disclaimer.
        * Redistributions in binary form must reproduce the above
        copyright notice, this list of conditions and the following
        disclaimer in the documentation and/or other materials provided
        with the distribution.
        * Neither the name of the NumPy Developers nor the names of any
        contributors may be used to endorse or promote products derived
        from this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
    "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
    LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
    A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
    OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
    LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
    DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
    """
    if isinstance(x, (tuple, list)):
        x = torch.as_tensor(x)
    dtype = torch.float32 if x.device.type == "mps" else torch.float64
    c = torch.as_tensor(c, device=x.device, dtype=dtype)
    if isinstance(x, torch.Tensor) and tensor:
        c = c.view(c.shape + (1,) * x.ndim)

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
            c0 = c[-i] - (c1 * (nd - 1)) / nd
            c1 = tmp + (c1 * x * (2 * nd - 1)) / nd
    return c0 + c1 * x


def _torch_calc_g(
    cosang: torch.Tensor, stiffness: float = 4, n_legendre_terms: int = 50
) -> torch.Tensor:
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
    G : torch.Tensor of shape (n_channels, n_channels)
        The G matrix.

    Notes
    -----
    Code copied and modified from MNE-Python:
    https://github.com/mne-tools/mne-python/blob/bdaa1d460201a3bc3cec95b67fc2b8d31a933652/mne/channels/interpolation.py#L35

    Copyright © 2011-2019, authors of MNE-Python
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:
        * Redistributions of source code must retain the above copyright
        notice, this list of conditions and the following disclaimer.
        * Redistributions in binary form must reproduce the above copyright
        notice, this list of conditions and the following disclaimer in the
        documentation and/or other materials provided with the distribution.
        * Neither the name of the copyright holder nor the names of its
        contributors may be used to endorse or promote products derived from
        this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
    AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
    IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
    ARE DISCLAIMED. IN NO EVENT SHALL COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
    FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
    DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
    SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
    CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
    LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
    OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
    DAMAGE.
    """
    factors = [
        (2 * n + 1) / (n**stiffness * (n + 1) ** stiffness * 4 * np.pi)
        for n in range(1, n_legendre_terms + 1)
    ]
    return _torch_legval(cosang, [0] + factors)


def spherical_spline_kernel(
    pos_a: Tensor, pos_b: Tensor, stiffness: int = 4, n_terms: int = 50
) -> Tensor:
    r"""Spherical-spline covariance :math:`g(\cos\gamma)` between two sets of positions.

    :math:`g(x) = \frac{1}{4\pi}\sum_{n=1}^{N} \frac{2n + 1}{(n (n + 1))^m}
    P_n(x)`, with :math:`P_n` the Legendre polynomials, :math:`\gamma` the
    angle between the positions seen from the origin, :math:`m` the
    ``stiffness`` and :math:`N` the number of terms (MNE's defaults: 4 and 50), as
    in [Perrin1989]_.
    Positions are taken from the origin: subtract the head centre first. Pass
    it, or any function with this signature, to
    :class:`~braindecode.modules.ChannelLayer` as the strategy.

    Parameters
    ----------
    pos_a, pos_b : torch.Tensor
        ``(n_a, 3)`` and ``(n_b, 3)`` positions.
    stiffness : int
        :math:`m`.
    n_terms : int
        :math:`N`.

    Returns
    -------
    torch.Tensor
        ``(n_a, n_b)``, float64 (float32 on MPS).

    References
    ----------
    .. [Perrin1989] Perrin, F., Pernier, J., Bertrand, O., & Echallier, J. F.
       (1989). Spherical splines for scalp potential and current density
       mapping. Electroencephalography and Clinical Neurophysiology, 72(2),
       184-187. https://doi.org/10.1016/0013-4694(89)90180-6
    """
    pos_a = torch.as_tensor(pos_a)
    dtype = torch.float32 if pos_a.device.type == "mps" else torch.float64
    a, b = (
        torch.nn.functional.normalize(torch.as_tensor(p, dtype=dtype), dim=1)
        for p in (pos_a, pos_b)
    )
    return _torch_calc_g((a @ b.T).clamp(-1, 1), stiffness, n_terms)


def exact_derivation(
    target: list[str], inputs: list[str], atol: float = 1e-9
) -> tuple[Tensor, Tensor]:
    """``W`` with ``W @ x`` the ``target`` channels, from the ``inputs`` ``x``.

    A target is exact when its row of :func:`derivation_matrix` is a
    combination of the input rows: ``F3-C3`` from ``C3-F3`` (weight ``-1``),
    ``F3-P3`` from ``F3-C3`` and ``C3-P3``, ``F3-C3`` from ``F3`` and ``C3``.
    This is :func:`linear_channel_map` with every potential unknown, so ``W``
    is the minimum-norm solution and duplicate measurements are averaged.
    No reference is guessed: ``F3-C3`` is not exact from ``F3-R1`` and
    ``C3-R2``, and ``F3`` is not exact from bipolar channels alone.

    Parameters
    ----------
    target, inputs : list of str
        Channel names, as in :func:`derivation_matrix`.
    atol : float
        Largest residual of an exact row.

    Returns
    -------
    weight : torch.Tensor
        ``(len(target), len(inputs))``, float64; rows of non-exact targets are 0.
    exact : torch.Tensor
        ``(len(target),)`` bool.
    """
    R, electrodes = derivation_matrix(inputs)
    T, electrodes = derivation_matrix(target, electrodes)
    R = torch.nn.functional.pad(R, (0, T.shape[1] - R.shape[1]))
    eye = torch.eye(T.shape[1], dtype=torch.float64)
    weight, var = linear_channel_map(R, T, 0 * eye, flat=eye, atol=atol)
    return weight.round(decimals=12) + 0.0, torch.isfinite(var)  # -1.0, not -0.99..


_ELECTRODES = ("eeg", "seeg", "ecog", "dbs")
_FIFF_KINDS = {FIFF.FIFFV_EEG_CH: "eeg", FIFF.FIFFV_SEEG_CH: "seeg"}
_FIFF_KINDS.update({FIFF.FIFFV_ECOG_CH: "ecog", FIFF.FIFFV_DBS_CH: "dbs"})
# Template head: mne.bem._fit_sphere of the dense standard_1005 in the head frame
# (MNE's default (0, 0, 0.04), R 0.09 sits 15 mm behind the electrodes' centre).
_SPHERE_R0, _SPHERE_RADIUS = (-0.0010, 0.0148, 0.0392), 0.0988
_MU_LAMBDA = (  # Berg-Scherg fit of make_sphere_model's 4 shells (COBYLA, 0.5 s);
    # depends on the relative radii and conductivities only, not on r0 or R
    np.array([0.9433448511080679, 0.663623934869853, 0.079878238156799]),
    np.array([0.4260455268056578, 2.0834380895598508, -0.05381815373454841]),
)


@lru_cache(maxsize=1)
def _standard_positions() -> dict[str, np.ndarray]:
    """Lower-case ``standard_1005`` name (T3 = T7 included) -> position in the
    head frame, as ``raw.set_montage`` gives it."""
    std = mne.channels.make_standard_montage(resolve_montage_name("standard_1005"))
    std.apply_trans(mne.channels.compute_native_head_t(std))
    return {n.lower(): p for n, p in std.get_positions()["ch_pos"].items()}


class _ChannelGeometry(NamedTuple):
    names: list[str]
    positions: np.ndarray
    standard_positions: np.ndarray
    kinds: list[str]
    has_location: np.ndarray


def _channel_geometry(chs_info: list[dict]) -> _ChannelGeometry:
    """Names, positions ``(C, 3)`` (given ``loc``, else ``standard_1005``, else NaN),
    standard positions of the names, channel types and which ``loc`` were given."""
    if not chs_info:
        raise ValueError("A montage must contain at least one channel.")
    names = [str(ch["ch_name"]) for ch in chs_info]
    types = [  # missing kind = EEG; FIFF kind ints as in raw.info
        k.lower() if isinstance(k, str) else _FIFF_KINDS.get(int(k), str(k))
        for k in (ch.get("kind") or ch.get("ch_type") or "eeg" for ch in chs_info)
    ]
    table, nan = _standard_positions(), np.full(3, np.nan)
    std = np.array(
        [
            table.get(n.lower(), nan) if t in _ELECTRODES else nan
            for n, t in zip(names, types)
        ],
        float,
    ).reshape(-1, 3)
    loc = np.array(
        [
            np.r_[ch.get("loc") if ch.get("loc") is not None else (), 0, 0, 0][:3]
            for ch in chs_info
        ],
        float,
    )
    given = np.isfinite(loc).all(1) & (np.abs(loc).max(1) >= 1e-8)
    pos = np.where(given[:, None], loc, std)
    return _ChannelGeometry(names, pos, std, types, given)


def resolve_channel_positions(chs_info, bank, montage=None, on_missing="error"):
    """Positions of ``chs_info`` in a name -> position ``bank`` (REVE's bank).

    Each channel takes, in order: ``"<montage>_<name>"`` when ``montage`` is
    given (``"biosemi128"``), its name, its name ignoring case and an ``EEG``
    prefix or ``-REF`` suffix, its legacy alias (``T3`` -> ``T7``), or for a
    bipolar ``A-B`` the midpoint of ``A`` and ``B``. Other channels follow
    ``on_missing``: ``"error"`` raises, ``"drop"`` leaves them out, and
    ``"nearest"`` takes the bank entry closest to the channel's ``loc`` (in the
    bank's frame), or drops it when it has no ``loc``. No position is invented.

    Parameters
    ----------
    chs_info : list of dict
        Channels in signal order, each with a ``ch_name`` and optional ``loc``.
    bank : mapping of str to array-like
        Channel names and their three-dimensional positions in a common frame.
    montage : str, optional
        Prefix for montage-specific names in the bank.
    on_missing : {"error", "drop", "nearest"}
        How to handle a name missing from the bank. For ``"nearest"``, the
        first three entries of ``loc`` must use the same frame and units as the bank.

    Returns
    -------
    positions : numpy.ndarray
        ``(n_kept, 3)`` positions.
    kept : list of int
        Indices of the channels of ``chs_info`` that got a position.
    """
    if on_missing not in ("error", "drop", "nearest"):
        raise ValueError(
            f"on_missing must be error, drop or nearest; got {on_missing!r}"
        )
    lower = {k.lower(): k for k in bank}

    def lookup(name):
        name = re.sub(r"^eeg\s+|-(ref|le|ar)$", "", name.strip(), flags=re.I)
        for key in (
            f"{montage}_{name}" if montage else None,
            name,
            _ALIASES.get(name.lower()),
        ):
            if key is not None and key.lower() in lower:
                return np.asarray(bank[lower[key.lower()]], float)
        ends = _channel_electrodes(name)
        if len(ends) == 2:
            a, b = lookup(ends[0]), lookup(ends[1])
            if a is not None and b is not None:
                return (a + b) / 2
        return None

    keys = list(bank)
    table = np.asarray([bank[k] for k in keys], float)
    positions, kept, missing = [], [], []
    for i, ch in enumerate(chs_info):
        pos = lookup(str(ch["ch_name"]))
        loc = np.asarray(
            ch.get("loc") if ch.get("loc") is not None else [np.nan] * 3, float
        )[:3]
        if (
            pos is None
            and on_missing == "nearest"
            and np.isfinite(loc).all()
            and loc.any()
        ):
            pos = table[np.linalg.norm(table - loc, axis=1).argmin()]
        if pos is None:
            missing.append(ch["ch_name"])
        else:
            positions.append(pos)
            kept.append(i)
    if missing and (on_missing == "error" or not kept):
        raise ValueError(
            f"No position in the bank for {missing}; pass montage=, on_missing="
            '"drop" or "nearest", or positions to forward.'
        )
    if missing:
        warnings.warn(
            f"Channels without a position are dropped: {missing}.", UserWarning
        )
    return np.asarray(positions), kept


# T5/T6 are P7/P8 in 10-10 (MNE's own alias table maps them to T9/T10 sites).
_ALIASES = {"t3": "T7", "t4": "T8", "t5": "P7", "t6": "P8"}


def _interp(method: str, src_pos, tgt_pos, reg: float) -> np.ndarray:
    """``(M, k)`` matrix of :meth:`mne.io.Raw.interpolate_to` (spline or MNE field)."""
    src = {f"S{i}": p for i, p in enumerate(src_pos)}
    info = mne.create_info(list(src), 100.0, "eeg")
    info.set_montage(mne.channels.make_dig_montage(src, coord_frame="head"))
    raw = mne.io.RawArray(np.eye(len(src)), info, verbose="ERROR")
    tgt = mne.channels.make_dig_montage(
        {f"T{i}": p for i, p in enumerate(tgt_pos)}, coord_frame="head"
    )
    with mne.utils.use_log_level("ERROR"):
        return raw.interpolate_to(tgt, method=method, reg=reg).get_data()


def _sphere_forward(positions: np.ndarray, sphere, src) -> np.ndarray:
    """Free-orientation lead field ``(C, 3 * n_grid)`` of the template sphere head."""
    positions = np.array(positions, float)
    rel = positions - np.asarray(sphere["r0"], float)
    positions[np.hypot(rel[:, 0], rel[:, 1]) < 1e-9, 0] += 1e-6  # z axis: singular
    names = [f"E{i}" for i in range(len(positions))]
    info = mne.create_info(names, 100.0, "eeg")
    info.set_montage(
        mne.channels.make_dig_montage(dict(zip(names, positions)), coord_frame="head")
    )
    with mne.utils.use_log_level("ERROR"):
        fwd = mne.make_forward_solution(
            info, trans=None, src=src, bem=sphere, eeg=True, meg=False
        )
    return fwd["sol"]["data"]


@lru_cache(maxsize=4)
def _sphere_head(n_parcels: int, grid_mm: float):
    """4-shell sphere fitted to standard_1005, volume grid, k-means parcels
    (seed 0) and their orientation (first right singular vector of the lead
    field on the dense standard_1005)."""
    with mne.utils.use_log_level("ERROR"):
        sphere = mne.make_sphere_model(r0=_SPHERE_R0, head_radius=_SPHERE_RADIUS)
        # Keep the calibrated parcel basis used by existing trainable source layers.
        sphere["mu"], sphere["lambda"] = (values.copy() for values in _MU_LAMBDA)
        sphere["nfit"] = len(sphere["mu"])
        src = mne.setup_volume_source_space(
            sphere=sphere, pos=grid_mm, mindist=5.0, exclude=20.0
        )
    rr = src[0]["rr"][src[0]["vertno"]]
    _, labels = kmeans2(rr, n_parcels, seed=0, minit="++")
    dense = np.stack(list(_standard_positions().values()))
    L = _sphere_forward(np.unique(dense.round(6), axis=0), sphere, src)
    L -= L.mean(0)
    orient = np.zeros((L.shape[1], n_parcels))
    for p in range(n_parcels):
        cols = (3 * np.flatnonzero(labels == p)[:, None] + np.arange(3)).ravel()
        if len(cols):
            orient[cols, p] = np.linalg.svd(L[:, cols], full_matrices=False)[2][0]
    return sphere, src, orient


def _source(src_pos, tgt_pos, n_parcels=64, lam=0.1, grid_mm=15.0):
    """Lead field at the targets minus the inputs' mean ``(M, P)`` and the
    minimum-norm inverse of the average-referenced inputs ``(P, k)``."""
    sphere, src, orient = _sphere_head(n_parcels, grid_mm)
    k = len(src_pos)
    L = _sphere_forward(np.vstack([src_pos, tgt_pos]), sphere, src) @ orient
    A = np.eye(k) - 1.0 / k
    Lo = A @ L[:k]
    G = Lo @ Lo.T
    S = solve(G + lam * np.trace(G) / k * np.eye(k), Lo, assume_a="sym").T @ A
    return L[k:] - L[:k].mean(0), S


def _head_origin(src_pos) -> np.ndarray:
    """Centre of the sphere fitted to ``src_pos``, as ``interpolate_to(origin="auto")``."""
    names = [f"S{i}" for i in range(len(src_pos))]
    info = mne.create_info(names, 100.0, "eeg")
    info.set_montage(
        mne.channels.make_dig_montage(dict(zip(names, src_pos)), coord_frame="head")
    )
    return mne.bem.fit_sphere_to_headshape(info, units="m", verbose="ERROR")[1]


def _kriging(cov, k, noise, match_gain=False, flat=True):
    """Rows ``k:`` of ``cov`` from rows ``:k`` (``linear_channel_map``): ``(W, var)``."""
    eye = torch.eye(len(cov), dtype=torch.float64)
    W, var = linear_channel_map(
        eye[:k],
        eye[k:],
        torch.as_tensor(cov),
        noise,
        eye.sum(1, keepdim=True) if flat else None,
        match_gain,
    )  # flat: the unknown recording reference
    return W.numpy(), var.numpy()


def _mixing(
    name,
    src_pos,
    tgt_pos,
    reg=None,
    p=2.0,
    noise=0.01,
    radius=None,
    cov=None,
    dense_pos=None,
    match_gain=False,
):
    """``(M, k)`` matrix filling ``M`` target positions from ``k`` input positions,
    and the targets' posterior variance (``None`` without a prior covariance)."""
    if name == "spline" or callable(name):  # positions from MNE's fitted head centre
        k = torch.as_tensor(np.vstack([src_pos, tgt_pos]) - _head_origin(src_pos))
        kernel = spherical_spline_kernel if name == "spline" else name
        cov = np.asarray(kernel(k, k), float)
        return _kriging(cov, len(src_pos), 1e-3 if reg is None else reg, match_gain)
    if name in ("zero", "latent"):  # latent: the attention fills these rows
        return np.zeros((len(tgt_pos), len(src_pos))), None
    if name == "wiener":  # C_to (C_oo + noise * tr / k * I)^-1
        if cov is None or not len(cov):
            raise ValueError("The 'wiener' strategy needs a covariance: call fit().")
        both = np.vstack([src_pos, tgt_pos])  # matched to the dense montage
        d = np.nan_to_num(cdist(both, dense_pos), nan=np.inf)
        if (d.min(1) > 0.015).any():
            raise ValueError(
                f"'wiener': positions {both[d.min(1) > 0.015].round(3).tolist()} have no electrode of the fitted dense montage within 15 mm."
            )
        jo, jt = np.split(d.argmin(1), [len(src_pos)])
        sigma2 = noise * np.trace(cov[np.ix_(jo, jo)]) / len(jo)
        j = np.r_[jo, jt]
        return _kriging(cov[np.ix_(j, j)], len(jo), sigma2, match_gain, flat=False)
    if name == "region":  # mean within radius (default 1.5 x median input spacing)
        if radius is None:
            nn_d = cdist(src_pos, src_pos) + np.diag(np.full(len(src_pos), np.inf))
            radius = 1.5 * np.median(nn_d.min(1))
        w = (cdist(tgt_pos, src_pos) <= radius).astype(float)
        return w / np.maximum(w.sum(1, keepdims=True), 1.0), None
    if name in ("nearest", "idw"):
        w = np.maximum(cdist(tgt_pos, src_pos), 1e-6) ** -p
        w = w == w.max(1, keepdims=True) if name == "nearest" else w
        return w / w.sum(1, keepdims=True), None
    if name == "field":  # MNE maps average-referenced data: keep the input reference
        W = _interp("MNE", src_pos, tgt_pos, 0.0 if reg is None else reg)
        return W + (1.0 - W.sum(1, keepdims=True)) / W.shape[1], None
    raise ValueError(f"Unknown spatial interpolation method {name!r}.")
