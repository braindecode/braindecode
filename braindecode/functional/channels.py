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

import numpy as np
import torch
from torch import Tensor


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
    parts = [
        ends if len(ends := n.lower().split("-")) == 2 and all(ends) else [n.lower()]
        for n in ch_names
    ]
    if reference is not None and reference != "average":
        parts = [p if len(p) == 2 else [p[0], reference.lower()] for p in parts]
    electrodes = list(dict.fromkeys([*(electrodes or []), *sum(parts, [])]))
    matrix = torch.zeros(len(ch_names), len(electrodes), dtype=torch.float64)
    for row, ends in enumerate(parts):
        matrix[row, electrodes.index(ends[0])] += 1.0
        if len(ends) == 2:
            matrix[row, electrodes.index(ends[1])] -= 1.0
    if reference == "average":
        single = torch.tensor([len(p) == 1 for p in parts])
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

    which is (universal) kriging [Matheron1963]_ and the posterior mean and
    variance of a Gaussian process [Rasmussen2006]_. Without ``flat`` it is
    :math:`W = B \Sigma A^\top (A \Sigma A^\top + \sigma^2 I)^{-1}`. The choice
    of :math:`\Sigma` gives the known methods:

    - :math:`\Sigma = 0` and :math:`F = I` (every potential unknown): the
      exact derivations of :func:`exact_derivation`, ``W = B A^+``; a target is
      exact when :math:`B F` lies in the span of :math:`A F` [Hjorth1975]_.
    - the spherical-spline kernel (:func:`spherical_spline_kernel`), ``noise``
      the regularisation and :math:`F = \mathbf 1` (the unknown reference):
      spherical-spline interpolation [Perrin1989]_, which is spline smoothing on
      the sphere [Wahba1981]_ and the Gaussian-process estimate with that kernel
      [Kimeldorf1970]_.
    - :math:`\Sigma = L L^\top` for a lead field :math:`L`: the minimum-norm
      estimate [Hamalainen1994]_; with :math:`B = L` taken against a point at
      infinity it is REST re-referencing [Yao2001]_.
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
        Rescale each row so that the estimate has the prior variance of its
        target, :math:`[W A \Sigma A^\top W^\top]_{kk} = [B \Sigma B^\top]_{kk}`:
        the minimum-variance estimate shrinks the amplitude.
    atol : float
        Largest residual of :math:`W A F = B F` for a target to be estimable.

    Returns
    -------
    weight : torch.Tensor
        ``(n_targets, n_inputs)``, float64; rows of targets that cannot be
        estimated (``W A F != B F``) are 0.
    variance : torch.Tensor
        ``(n_targets,)`` posterior variance; ``inf`` where not estimable.

    References
    ----------
    .. [Matheron1963] Matheron, G. (1963). Principles of geostatistics.
       Economic Geology, 58(8), 1246-1266. https://doi.org/10.2113/gsecongeo.58.8.1246
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
    F = torch.zeros(len(cov), 0, dtype=torch.float64) if flat is None else flat.double()
    n, f = len(A), F.shape[1]
    AF, BF, KyT = A @ F, B @ F, A @ cov @ B.T
    M = torch.zeros(n + f, n + f, dtype=torch.float64)
    M[:n, :n] = A @ cov @ A.T + noise * torch.eye(n, dtype=torch.float64)
    M[:n, n:], M[n:, :n] = AF, AF.T
    sol = torch.linalg.pinv(M) @ torch.cat([KyT, BF.T])  # SVD cutoff, as scipy/MNE
    W, mu = sol[:n].T.round(decimals=12) + 0.0, sol[n:]  # copies stay exactly +-1
    prior = (B @ cov * B).sum(1)
    var = prior - (W * KyT.T).sum(1) - (mu.T * BF).sum(1)
    ok = (W @ AF - BF).abs().amax(1) <= atol if f else torch.ones(len(B), dtype=bool)
    if match_gain:
        signal = (W @ A @ cov * (W @ A)).sum(1)
        W = (
            W
            * torch.where(signal > 0, (prior / signal).clamp_min(0).sqrt(), 1.0)[
                :, None
            ]
        )
    return W * ok[:, None], torch.where(ok, var.clamp_min(0), torch.inf)


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
        ``(n_a, n_b)``, float64.

    References
    ----------
    .. [Perrin1989] Perrin, F., Pernier, J., Bertrand, O., & Echallier, J. F.
       (1989). Spherical splines for scalp potential and current density
       mapping. Electroencephalography and Clinical Neurophysiology, 72(2),
       184-187. https://doi.org/10.1016/0013-4694(89)90180-6
    """
    a, b = (
        torch.nn.functional.normalize(torch.as_tensor(p, dtype=torch.float64), dim=1)
        for p in (pos_a, pos_b)
    )
    n = np.arange(1, n_terms + 1)
    coef = np.r_[0.0, (2 * n + 1) / ((n * (n + 1)) ** stiffness * 4 * np.pi)]
    return torch.as_tensor(
        np.polynomial.legendre.legval((a @ b.T).clamp(-1, 1).numpy(), coef)
    )


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
    return weight, torch.isfinite(var)
