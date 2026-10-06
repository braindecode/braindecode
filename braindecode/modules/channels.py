# Authors: Bruno Aristimunha <b.aristimunha@gmail.com>
#
# License: BSD (3-clause)
"""Channel layer: any montage in, the channels a pretrained backbone consumes out.

Every strategy is a matrix ``W`` (K x C) from the C input channels to the K
target channels. Names are copied first (case-insensitive, then legacy names
such as T3 = T7); an input whose name MNE does not know is copied to a target
within 15 mm. The strategy fills the other targets that have a position from
the positioned inputs; a target without a position (BENDR's ``SCALE``) is zero.
A target named ``A-B`` is the derivation ``V(A) - V(B)`` (BIOT).
"""

from __future__ import annotations

import difflib
import warnings
from functools import lru_cache

import mne
import numpy as np
import torch
from scipy.cluster.vq import kmeans2
from scipy.linalg import solve
from scipy.spatial.distance import cdist
from torch import Tensor, nn

from braindecode.util import resolve_montage_name

STRATEGIES = ("exact", "zero", "nearest", "idw", "spline", "field", "source")
_MIN_POSITIONS = dict(zip(STRATEGIES, (0, 0, 1, 1, 4, 4, 4)))  # positioned inputs
_ELECTRODES = ("eeg", "seeg", "ecog", "dbs")


@lru_cache(maxsize=1)
def _standard_montage() -> mne.channels.DigMontage:
    """``standard_1005`` in its own coordinates (no fiducial transform)."""
    std = mne.channels.make_standard_montage(resolve_montage_name("standard_1005"))
    return mne.channels.make_dig_montage(
        std.get_positions()["ch_pos"], coord_frame="head"
    )


def _resolve(chs_info: list[dict]):
    """Names, positions ``(C, 3)`` (given ``loc``, else ``standard_1005``, else NaN),
    standard positions of the names, MNE channel types and which ``loc`` were given."""
    names = [str(ch["ch_name"]) for ch in chs_info]
    kinds = [ch.get("kind") or "eeg" for ch in chs_info]  # missing kind = EEG
    info = mne.create_info(
        names, 1.0, [k.lower() if isinstance(k, str) else "eeg" for k in kinds]
    )
    with info._unlock():
        for ch, kind in zip(info["chs"], kinds):
            if not isinstance(kind, str):
                ch["kind"] = int(kind)  # FIFF kind
    with mne.utils.use_log_level("ERROR"), warnings.catch_warnings():
        warnings.simplefilter("ignore")  # non-electrode channels get no position
        info.set_montage(_standard_montage(), match_case=False, on_missing="ignore")
    std = np.array([ch["loc"][:3] for ch in info["chs"]], float)
    loc = np.array(
        [
            np.r_[ch.get("loc") if ch.get("loc") is not None else (), 0, 0, 0][:3]
            for ch in chs_info
        ],
        float,
    )
    given = np.isfinite(loc).all(1) & (np.abs(loc).max(1) >= 1e-8)
    pos = np.where(given[:, None], loc, std)
    return names, pos, std, info.get_channel_types(), given


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
    """3-shell sphere, volume grid, k-means parcels (seed 0) and their orientation
    (first right singular vector of the lead field on the dense standard_1005)."""
    with mne.utils.use_log_level("ERROR"):
        sphere = mne.make_sphere_model(r0=(0.0, 0.0, 0.04), head_radius=0.09)
        src = mne.setup_volume_source_space(
            sphere=sphere, pos=grid_mm, mindist=5.0, exclude=20.0
        )
    rr = src[0]["rr"][src[0]["vertno"]]
    _, labels = kmeans2(rr, n_parcels, seed=0, minit="++")
    dense = np.stack(list(_standard_montage().get_positions()["ch_pos"].values()))
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


def _mixing(name: str, src_pos, tgt_pos, reg=None, p=2.0, **source_kw) -> np.ndarray:
    """``(M, k)`` matrix filling ``M`` target positions from ``k`` input positions."""
    if name == "zero":
        return np.zeros((len(tgt_pos), len(src_pos)))
    if name in ("nearest", "idw"):
        w = np.maximum(cdist(tgt_pos, src_pos), 1e-6) ** -p
        w = w == w.max(1, keepdims=True) if name == "nearest" else w
        return w / w.sum(1, keepdims=True)
    if name == "spline":
        return _interp("spline", src_pos, tgt_pos, 1e-3 if reg is None else reg)
    if name == "field":  # MNE maps average-referenced data: keep the input reference
        W = _interp("MNE", src_pos, tgt_pos, 0.0 if reg is None else reg)
        return W + (1.0 - W.sum(1, keepdims=True)) / W.shape[1]
    Lt, S = _source(src_pos, tgt_pos, **source_kw)
    return Lt @ S + 1.0 / len(src_pos)  # rows sum to 1: the input's own reference


class ChannelLayer(nn.Module):
    """Map any montage onto ``target`` with one matrix per montage.

    .. warning:: Experimental. Public API may change without a deprecation cycle.

    Parameters
    ----------
    target : list of dict
        Channels the backbone consumes (``ch_name``, optional ``loc``).
    strategy : str
        ``"exact"`` (a missing target is an error), ``"zero"``, ``"nearest"``,
        ``"idw"`` (``p``), ``"spline"`` (``reg=1e-3``), ``"field"`` (``reg=0``)
        or ``"source"`` (``n_parcels=64``, ``lam=0.1``, ``grid_mm=15``,
        ``trainable``: a zero-initialised parcel mixing on top of the physics).
    chs_info : list of dict, optional
        Montage used when :meth:`forward` gets none.
    drop_non_eeg : bool
        Ignore input channels that are not electrodes instead of raising.
    """

    def __init__(
        self,
        target,
        strategy,
        chs_info=None,
        *,
        drop_non_eeg=False,
        trainable=False,
        **kwargs,
    ):
        super().__init__()
        if strategy not in STRATEGIES:
            close = difflib.get_close_matches(strategy, STRATEGIES, n=3)
            raise ValueError(
                f"Unknown channel strategy {strategy!r}. Did you mean {close}? Available: {STRATEGIES}."
            )
        _, pos, _, _, given = _resolve(target)  # the backbone gets the positions
        self.target = [
            c if g or np.isnan(p).any() else {**c, "loc": np.r_[p, np.zeros(9)]}
            for c, p, g in zip(target, pos, given)
        ]
        self.strategy, self.chs_info = strategy, chs_info
        self.drop_non_eeg, self.kwargs, self._key = drop_non_eeg, kwargs, None
        self.trainable = trainable and strategy == "source"
        if self.trainable:
            n = kwargs.get("n_parcels", 64)
            self.parcel_mix = nn.Parameter(torch.zeros(n, n))
        if chs_info is not None:
            self._build(chs_info)  # surface montage errors at construction

    def _build(self, chs_info: list[dict]) -> None:
        key = repr([(c["ch_name"], c.get("kind"), c.get("loc")) for c in chs_info])
        if key == self._key:
            return
        names, pos, std, types, _ = _resolve(chs_info)
        bad = [n for n, t in zip(names, types) if t not in _ELECTRODES]
        if bad and not self.drop_non_eeg:
            raise ValueError(
                f"Channels {bad} are not EEG electrodes; remove them (e.g. raw.pick('eeg')) or pass drop_non_eeg=True."
            )
        keep = np.array([t in _ELECTRODES for t in types])
        if self.strategy == "source" and {"seeg", "ecog", "dbs"} & set(types):
            raise ValueError(
                "channel_strategy='source' uses a scalp-EEG sphere head; use a sensor strategy for intracranial channels."
            )
        tnames = [str(c["ch_name"]) for c in self.target]
        mono = list(dict.fromkeys(m for n in tnames for m in n.split("-")))
        given = {str(c["ch_name"]): c for c in self.target}
        _, tpos, tstd, *_ = _resolve([given.get(m, {"ch_name": m}) for m in mono])
        D = np.array(
            [
                [(m == n.split("-")[0]) - (m == n.split("-")[-1] != n) for m in mono]
                for n in tnames
            ],
            float,
        )
        # Copy: same name, else same standard position (T3 = T7), else unknown name within 15 mm.
        low = np.array([n.lower() for n in names])
        alias = np.nan_to_num(cdist(tstd, std), nan=np.inf) < 1e-6
        dist = np.nan_to_num(cdist(tpos, pos), nan=np.inf)
        near = np.where(np.isnan(std).any(1), dist, np.inf)
        W = np.zeros((len(mono), len(names)))
        for k, m in enumerate(mono):
            for hit in (low == m.lower(), alias[k], near[k] <= 0.015):
                if (hit & keep).any():
                    j = np.flatnonzero(hit & keep)
                    W[k, j[np.argmin(near[k][j])]] = 1.0
                    break
        observed = W.sum(1) > 0
        todo = ~observed & np.isfinite(tpos).all(1)
        if self.strategy == "exact" and todo.any():
            raise ValueError(
                f"Strategy 'exact': target channels {[m for m, t in zip(mono, todo) if t]} are not in the input montage {names}; use a reconstructing strategy (e.g. 'spline') or supply them."
            )
        use = keep & np.isfinite(pos).all(1)
        if todo.any() and use.sum() < _MIN_POSITIONS[self.strategy]:
            raise ValueError(
                f"Strategy {self.strategy!r} needs at least {_MIN_POSITIONS[self.strategy]} channels with a position; got {int(use.sum())} of {names}. Supply 'loc' or standard 10-05 names."
            )
        if todo.any():
            W[np.ix_(todo, use)] = _mixing(
                self.strategy, pos[use], tpos[todo], **self.kwargs
            )
            gain = np.abs(W[todo]).sum(1).max()
            if gain > 2:
                warnings.warn(
                    f"Strategy {self.strategy!r}: reconstructed row gain |w|_1 = {gain:.3g} > 2; the map amplifies noise. Supply more channels or a smoother strategy.",
                    UserWarning,
                    stacklevel=3,
                )
        if self.trainable:
            Lt, S = _source(pos[use], tpos[todo], **self.kwargs)
            S_all = np.zeros((len(S), len(names)))
            S_all[:, use] = S
            self.register_buffer(
                "lead", torch.tensor(D[:, todo] @ Lt).float(), persistent=False
            )
            self.register_buffer(
                "inverse", torch.tensor(S_all).float(), persistent=False
            )
        self.register_buffer("weight", torch.tensor(D @ W).float(), persistent=False)
        self.observed = torch.as_tensor((np.abs(D) @ ~observed) == 0)
        self._key = key

    def _load_from_state_dict(self, *args, **kwargs):
        self._key = None  # maps are rebuilt after a load
        super()._load_from_state_dict(*args, **kwargs)

    def forward(
        self, x: Tensor, chs_info: list[dict] | None = None
    ) -> tuple[Tensor, Tensor]:
        """Map ``x`` (B, C, T) recorded with ``chs_info``; return ``(x, observed)``."""
        chs = chs_info if chs_info is not None else self.chs_info
        if chs is None:
            raise ValueError(
                f"No montage: pass chs_info to the model (or to forward) so channel strategy {self.strategy!r} knows the input channels."
            )
        if x.shape[1] != len(chs):
            raise ValueError(
                f"Input has {x.shape[1]} channels but the montage has {len(chs)}; they must match."
            )
        self._build(chs)
        out = self.weight.to(x) @ x
        if self.trainable:
            mix = self.parcel_mix.to(x) @ (self.inverse.to(x) @ x)
            out = out + self.lead.to(x) @ mix
        return out, self.observed.to(x.device)
