# Authors: Bruno Aristimunha <b.aristimunha@gmail.com>
#
# License: BSD (3-clause)
"""Channels as linear combinations of electrode potentials.

A channel named ``A`` is ``V(A)`` against the recording's common reference and
a channel named ``A-B`` is ``V(A) - V(B)``. Writing every channel this way turns
re-montaging into linear algebra: a target channel is exact from the inputs
when its row is a combination of the input rows.
"""

from __future__ import annotations

import torch
from torch import Tensor


def derivation_matrix(
    ch_names: list[str], electrodes: list[str] | None = None
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

    Returns
    -------
    matrix : torch.Tensor
        ``(n_channels, n_electrodes)``, float64, entries ``+1``/``-1``.
    electrodes : list of str
        The columns.
    """
    parts = [
        ends if len(ends := n.lower().split("-")) == 2 and all(ends) else [n.lower()]
        for n in ch_names
    ]
    electrodes = list(
        dict.fromkeys([*(e.lower() for e in (electrodes or [])), *sum(parts, [])])
    )
    matrix = torch.zeros(len(ch_names), len(electrodes), dtype=torch.float64)
    for row, ends in enumerate(parts):
        matrix[row, electrodes.index(ends[0])] += 1.0
        if len(ends) == 2:
            matrix[row, electrodes.index(ends[1])] -= 1.0
    return matrix, electrodes


def exact_derivation(
    target: list[str], inputs: list[str], atol: float = 1e-9
) -> tuple[Tensor, Tensor]:
    """``W`` with ``W @ x`` the ``target`` channels, from the ``inputs`` ``x``.

    A target is exact when its row of :func:`derivation_matrix` is a
    combination of the input rows: ``F3-C3`` from ``C3-F3`` (weight ``-1``),
    ``F3-P3`` from ``F3-C3`` and ``C3-P3``, ``F3-C3`` from ``F3`` and ``C3``.
    ``W`` is the minimum-norm solution, so duplicate measurements are averaged.
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
    if not target:
        return torch.zeros((0, len(inputs)), dtype=torch.float64), torch.zeros(
            0, dtype=torch.bool
        )

    R, electrodes = derivation_matrix(inputs)
    T, electrodes = derivation_matrix(target, electrodes)
    R = torch.nn.functional.pad(R, (0, T.shape[1] - R.shape[1]))
    weight = (T @ torch.linalg.pinv(R)).round(decimals=12) + 0.0  # -1.0, not -0.99..
    exact = (weight @ R - T).abs().amax(1) <= atol
    return weight * exact[:, None], exact
