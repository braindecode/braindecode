# Authors: Bruno Aristimunha <b.aristimunha@gmail.com>
#
# License: BSD (3-clause)
"""Template head model for the ``source`` strategy (offline, no download)."""

from __future__ import annotations

from collections import OrderedDict
from functools import lru_cache

import numpy as np

from .resolve import _standard_1005


class SphereHead:
    """Lead fields of cortical parcels on an analytic sphere head.

    MNE 3-shell sphere (``r0=(0, 0, 0.04)``, radius 90 mm), volume grid of
    ``grid_mm`` spacing with free-orientation dipoles. Grid points are grouped
    into ``n_parcels`` parcels by k-means (``seed``); each parcel has one
    orientation pattern, the first right singular vector of its lead field on
    the dense ``standard_1005`` montage. Built once in about a second.

    Parameters
    ----------
    n_parcels : int
        Number of parcels (the source-space size).
    grid_mm : float
        Spacing of the volume source grid in millimetres.
    seed : int
        Seed of the k-means parcellation.
    """

    def __init__(self, n_parcels: int = 64, grid_mm: float = 15.0, seed: int = 0):
        import mne
        from scipy.cluster.vq import kmeans2

        self.n_parcels = n_parcels
        with mne.utils.use_log_level("ERROR"):
            self.sphere = mne.make_sphere_model(r0=(0.0, 0.0, 0.04), head_radius=0.09)
            self.src = mne.setup_volume_source_space(
                sphere=self.sphere, pos=grid_mm, mindist=5.0, exclude=20.0
            )
        rr = self.src[0]["rr"][self.src[0]["vertno"]]
        if len(rr) < n_parcels:
            raise ValueError(
                f"n_parcels={n_parcels} exceeds the {len(rr)} grid points of a "
                f"{grid_mm} mm grid; lower n_parcels or grid_mm."
            )
        _, labels = kmeans2(rr, n_parcels, seed=seed, minit="++")
        if np.bincount(labels, minlength=n_parcels).min() == 0:
            raise ValueError(
                f"k-means left an empty parcel; try another seed ({seed})."
            )
        self.centroids = np.stack([rr[labels == p].mean(0) for p in range(n_parcels)])

        dense = np.stack(list(_standard_1005().values()))
        dense = np.unique(dense.round(6), axis=0)  # aliases share positions
        L = self._free_leadfield(dense)
        L = L - L.mean(0, keepdims=True)
        # (3 * n_grid, n_parcels): each parcel's orientation pattern.
        self._orient = np.zeros((L.shape[1], n_parcels))
        for p in range(n_parcels):
            cols = (3 * np.flatnonzero(labels == p)[:, None] + np.arange(3)).ravel()
            self._orient[cols, p] = np.linalg.svd(L[:, cols], full_matrices=False)[2][0]
        # ponytail: a second cache below the tokenizer's map LRU on purpose; a
        # lead field costs an MNE forward solution, and source rebuilds it for
        # used + target positions of every montage (32 entries, a few MB).
        self._cache: OrderedDict[bytes, np.ndarray] = OrderedDict()

    def _free_leadfield(self, positions: np.ndarray) -> np.ndarray:
        import mne

        # MNE's sphere EEG formula is singular for an electrode exactly on the
        # z axis through the sphere centre (e.g. biosemi64 Cz at (0, 0, r)):
        # it returns NaN. A 1 um shift is far below any electrode precision.
        positions = np.array(positions, float)
        rel = positions - np.asarray(self.sphere["r0"], float)
        positions[np.hypot(rel[:, 0], rel[:, 1]) < 1e-9, 0] += 1e-6
        names = [f"E{i}" for i in range(len(positions))]
        try:
            info = mne.create_info(names, 100.0, "eeg")
            info.set_montage(
                mne.channels.make_dig_montage(
                    dict(zip(names, positions)), coord_frame="head"
                )
            )
            with mne.utils.use_log_level("ERROR"):
                fwd = mne.make_forward_solution(
                    info, trans=None, src=self.src, bem=self.sphere, eeg=True, meg=False
                )
        except (ValueError, RuntimeError) as exc:
            raise ValueError(
                f"MNE could not compute the lead field of the template sphere head "
                f"(3-shell, r0=(0, 0, 0.04), radius 90 mm) at {len(positions)} "
                f"electrode positions {np.round(positions, 4).tolist()[:4]}...: "
                f"{exc}"
            ) from exc
        return fwd["sol"]["data"]

    def leadfield(self, positions: np.ndarray) -> np.ndarray:
        """``(C, n_parcels)`` lead field, average-referenced over ``positions``."""
        positions = np.asarray(positions, float).reshape(-1, 3)
        key = np.round(positions * 1e4).tobytes()
        if key not in self._cache:
            L = self._free_leadfield(positions) @ self._orient
            self._cache[key] = L - L.mean(0, keepdims=True)
            if len(self._cache) > 32:
                self._cache.popitem(last=False)
        return self._cache[key]


@lru_cache(maxsize=4)
def get_sphere_head(
    n_parcels: int = 64, grid_mm: float = 15.0, seed: int = 0
) -> SphereHead:
    """Shared :class:`SphereHead` (built once per configuration)."""
    return SphereHead(n_parcels, grid_mm, seed)
