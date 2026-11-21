# Authors: Bruno Aristimunha <b.aristimunha@gmail.com>
#
# License: BSD (3-clause)
"""Channel layer: any montage in, the channels a pretrained backbone consumes out.

Every strategy is a matrix ``W`` (K x C) from the C input channels to the K
target channels. Names are copied first (case-insensitive, then legacy names
such as T3 = T7); an input whose name MNE does not know is copied to a target
within 15 mm. A channel named ``A-B`` is ``V(A) - V(B)`` (BIOT), so a target
that is a combination of measured channels is computed exactly
(:func:`braindecode.functional.exact_derivation`): ``F3-C3`` from ``C3-F3``,
``F3-P3`` from ``F3-C3`` and ``C3-P3``, ``F3-C3`` from ``F3`` and ``C3``. The
strategy fills the other targets that have a position from the positioned
inputs; a target without a position (BENDR's ``SCALE``) is zero.
"""

from __future__ import annotations

import difflib
import re
import warnings
from collections import OrderedDict
from functools import lru_cache
from typing import cast

import mne
import numpy as np
import torch
from mne.io.constants import FIFF
from scipy.cluster.vq import kmeans2
from scipy.linalg import solve
from scipy.spatial.distance import cdist
from torch import Tensor, nn

from braindecode.functional import check_ch_mask, exact_derivation
from braindecode.functional.channels import channel_electrodes, derivation_rows
from braindecode.util import resolve_montage_name

STRATEGIES = (
    *("exact", "zero", "nearest", "idw", "spline", "field", "source"),
    *("wiener", "region", "latent"),
)
_MIN_POSITIONS = dict(zip(STRATEGIES, (0, 0, 1, 1, 4, 4, 4, 1, 2, 1)))  # positioned
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


def _resolve(chs_info: list[dict]):
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
    return names, pos, std, types, given


def resolve_channel_positions(chs_info, bank, montage=None, on_missing="error"):
    """Positions of ``chs_info`` in a name -> position ``bank`` (REVE's bank).

    Each channel takes, in order: ``"<montage>_<name>"`` when ``montage`` is
    given (``"biosemi128"``), its name, its name ignoring case and an ``EEG``
    prefix or ``-REF`` suffix, its legacy alias (``T3`` -> ``T7``), or for a
    bipolar ``A-B`` the midpoint of ``A`` and ``B``. A name the bank holds
    exactly takes that entry; one that matches only ignoring case takes the last
    entry that matches (``cz`` in ``{"CZ": a, "Cz": b}`` gives ``b``). Other
    channels follow ``on_missing``: ``"error"`` raises, ``"drop"`` leaves them
    out, and ``"nearest"`` takes the bank entry closest to the channel's ``loc``
    (in the bank's frame), or drops it when it has no ``loc``. No position is
    invented.

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
    lower = {k.lower(): k for k in bank}  # entries differing only in case: the last

    def lookup(name):
        name = re.sub(r"^eeg\s+|-(ref|le|ar)$", "", name.strip(), flags=re.I)
        for key in (
            f"{montage}_{name}" if montage else None,
            name,
            _ALIASES.get(name.lower()),
        ):
            if key is not None and (key in bank or key.lower() in lower):
                return np.asarray(
                    bank[key if key in bank else lower[key.lower()]], float
                )
        ends = name.split("-")
        if len(ends) == 2 and all(ends):
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
        # Deterministic: keep the calibrated fit, not MNE's COBYLA result.
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


def _mixing(
    name: str,
    src_pos,
    tgt_pos,
    reg=None,
    p=2.0,
    noise=0.01,
    radius=None,
    cov=None,
    dense_pos=None,
    **source_kw,
) -> np.ndarray:
    """``(M, k)`` matrix filling ``M`` target positions from ``k`` input positions."""
    if name in ("zero", "latent"):  # latent: the attention fills these rows
        return np.zeros((len(tgt_pos), len(src_pos)))
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
        Cxx = cov[np.ix_(jo, jo)]
        Cxx = Cxx + noise * np.trace(Cxx) / len(jo) * np.eye(len(jo))
        return solve(Cxx, cov[np.ix_(jo, jt)], assume_a="sym").T
    if name == "region":  # mean within radius (default 1.5 x median input spacing)
        if radius is None:
            nn_d = cdist(src_pos, src_pos) + np.diag(np.full(len(src_pos), np.inf))
            radius = 1.5 * np.median(nn_d.min(1))
        w = (cdist(tgt_pos, src_pos) <= radius).astype(float)
        return w / np.maximum(w.sum(1, keepdims=True), 1.0)
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


@lru_cache(maxsize=32)  # bounded: augmentations that drop channels at random vary it
def _montage_at(positions: bytes) -> list[dict]:
    """Channels ``#0, #1, ...`` at ``positions``, an ``(n, 3)`` float64 array as bytes.

    No montage uses these names, so each channel is matched by position only.
    """
    xyz = np.frombuffer(positions, dtype=np.float64).reshape(-1, 3)
    return [
        {"ch_name": f"#{i}", "kind": "eeg", "loc": np.r_[p, np.zeros(9)]}
        for i, p in enumerate(xyz)
    ]


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
        ``"wiener"`` (``noise=0.01``; call :meth:`fit` first), ``"region"``
        (mean within ``radius``, default 1.5 x the median input spacing) or
        ``"latent"`` (``d_model=64``, ``n_freqs=8``: learned cross-attention from
        each missing target's position over the inputs' positions and statistics).
    chs_info : list of dict, optional
        Montage used when :meth:`forward` gets none.
    drop_non_eeg : bool
        Ignore input channels that are not electrodes instead of raising.

    Notes
    -----
    ``forward`` also takes one montage per sample (``chs_info`` a list of
    montages, ``x`` padded to the largest and ``ch_mask`` the real channels),
    for the strategies without trained state (not ``latent`` or ``trainable``);
    ``observed`` is then ``(batch, n_targets)`` and the buffer ``weight`` is that
    of the last sample. A model with a ``channel_strategy`` builds these montages
    from the ``pos`` and ``ch_mask`` of
    :func:`~braindecode.datasets.pad_channels_collate` with :meth:`montages`.
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
        self.drop_non_eeg, self.kwargs = drop_non_eeg, kwargs
        # Per-montage maps: LRU of device tensors; the current one is the buffers.
        self._cache, self._key = OrderedDict(), None
        self.register_buffer("weight", torch.zeros(0, 0), persistent=False)
        parts = [channel_electrodes(str(ch["ch_name"])) for ch in self.target]
        named = {
            ends[0].lower(): ch
            for ends, ch in zip(parts, self.target)
            if len(ends) == 1
        }
        # A measured A-B takes precedence over reconstructing A and B separately.
        bipolar = {
            row: "-".join(ends) for row, ends in enumerate(parts) if len(ends) == 2
        }
        derivations, mono = derivation_rows(parts, bipolar.values())
        columns = {name.lower(): i for i, name in enumerate(mono)}
        self._whole = [
            columns[bipolar[row].lower()] if row in bipolar else -1
            for row in range(len(parts))
        ]
        _, tpos, tstd, *_ = _resolve(
            [named.get(m.lower(), {"ch_name": m}) for m in mono]
        )
        self._mono, self._tpos, self._tstd, self._D = (
            mono,
            tpos,
            tstd,
            derivations.numpy(),
        )
        self.trainable = trainable and strategy == "source"
        if self.trainable:
            n = kwargs.get("n_parcels", 64)
            self.parcel_mix = nn.Parameter(torch.zeros(n, n))
        if strategy == "wiener":  # fitted state, saved in the state_dict
            self.register_buffer("cov", torch.zeros(0, 0))
            self.register_buffer("dense_pos", torch.zeros(0, 3))
        if strategy == "latent":
            d, self.n_freqs = kwargs.pop("d_model", 64), kwargs.pop("n_freqs", 8)
            f = 6 * self.n_freqs  # queries: target positions; keys: input positions
            self.query, self.key_pos = (
                nn.Sequential(nn.Linear(f, d), nn.GELU(), nn.Linear(d, d))
                for _ in range(2)
            )
            self.key_stats = nn.Linear(2, d)  # + input statistics
        if chs_info is not None and strategy != "wiener":  # wiener: after fit()
            self._use(chs_info)  # surface montage errors at construction

    def _use(self, chs: list[dict]) -> None:
        """Select the cached map for the current channel metadata, or build one."""
        metadata = tuple(
            (
                str(c["ch_name"]),
                c.get("kind") or c.get("ch_type") or "eeg",
                np.asarray(c.get("loc"), dtype=np.float64).tobytes(),
            )
            for c in chs
        )
        key = (self.drop_non_eeg, metadata)
        if key != self._key:
            maps = self._cache.pop(key, None) or self._build(chs)
            self._cache[key] = maps  # most recent last
            # shortcut: keep 16 maps; revisit if mixed batches repeatedly rebuild evicted maps.
            if len(self._cache) > 16:
                self._cache.popitem(last=False)
            for name, t in maps.items():
                self.register_buffer(name, t, persistent=False)
            self._key = key

    @torch.inference_mode(False)  # maps built in an eval pass serve training
    def _build(self, chs_info: list[dict]) -> dict[str, Tensor]:
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
        mono, tpos, tstd = self._mono, self._tpos, self._tstd
        # Copy: same name, else same standard position (T3 = T7), else unknown name within 15 mm.
        low = np.array([n.lower() for n in names])
        alias = np.nan_to_num(cdist(tstd, std), nan=np.inf) < 1e-6
        dist = np.nan_to_num(cdist(tpos, pos), nan=np.inf)
        near = np.where(np.isnan(std).any(1), dist, np.inf)
        # Names before positions: B-A shares the midpoint of A-B but needs -1.
        kept = np.flatnonzero(keep)
        derived, exact = exact_derivation(mono, [names[j] for j in kept])
        W = np.zeros((len(mono), len(names)))
        for k, m in enumerate(mono):
            direct = (low == m.lower()) & keep
            if direct.any():
                j = np.flatnonzero(direct)
                W[k, j[np.argmin(near[k][j])]] = 1.0
                continue
            if exact[k]:  # a combination of measured channels
                W[k, kept] = derived[k].numpy()
                continue
            for hit in (alias[k], near[k] <= 0.015):
                if (hit & keep).any():
                    j = np.flatnonzero(hit & keep)
                    W[k, j[np.argmin(near[k][j])]] = 1.0
                    break
        observed = np.abs(W).sum(1) > 0
        D = self._D.copy()
        for row, column in enumerate(self._whole):  # a measured A-B target: copy it
            if column >= 0 and observed[column]:
                D[row] = 0
                D[row, column] = 1
        missing = ~observed & np.abs(D).any(0)
        todo = missing & np.isfinite(tpos).all(1)
        if self.strategy == "exact" and todo.any():
            raise ValueError(
                f"Strategy 'exact': target channels {[m for m, t in zip(mono, todo) if t]} are not in the input montage {names}; use a reconstructing strategy (e.g. 'spline') or supply them."
            )
        seen = np.abs(D) @ missing == 0
        if not seen.any() and (self.strategy == "zero" or not todo.any()):
            raise ValueError(
                f"No target channel of {[c['ch_name'] for c in self.target]} is in the input montage {names}, and strategy {self.strategy!r} reconstructs none: the model would see only zeros. Supply matching names or positions, or a reconstructing strategy (e.g. 'spline')."
            )
        if (missing & ~todo).any():
            warnings.warn(
                f"Target channels {[m for m, t in zip(mono, missing & ~todo) if t]} are not in the input montage and have no position to reconstruct them from; they are zero-filled.",
                UserWarning,
                stacklevel=4,
            )
        use = keep & np.isfinite(pos).all(1)
        if todo.any() and use.sum() < _MIN_POSITIONS[self.strategy]:
            raise ValueError(
                f"Strategy {self.strategy!r} needs at least {_MIN_POSITIONS[self.strategy]} channels with a position; got {int(use.sum())} of {names}. Supply 'loc' or standard 10-05 names."
            )
        fitted = {  # wiener: device-safe float64 copies
            k: getattr(self, k).detach().cpu().double().numpy()
            for k in ("cov", "dense_pos")
            if hasattr(self, k)
        }
        if todo.any():
            W[np.ix_(todo, use)] = _mixing(
                self.strategy, pos[use], tpos[todo], **self.kwargs, **fitted
            )
            gain = np.abs(W[todo]).sum(1).max()
            if gain > 2:
                warnings.warn(
                    f"Strategy {self.strategy!r}: reconstructed row gain |w|_1 = {gain:.3g} > 2; the map amplifies noise. Supply more channels or a smoother strategy.",
                    UserWarning,
                    stacklevel=4,
                )
        dev, dt = self.weight.device, self.weight.dtype  # built where the module is
        maps = dict(weight=D @ W)
        if self.trainable:  # nothing to reconstruct: zero correction, no inverse
            n = self.parcel_mix.shape[0]
            Lt, S = (
                _source(pos[use], tpos[todo], **self.kwargs)
                if todo.any()
                else (np.zeros((0, n)), np.zeros((n, use.sum())))
            )
            maps.update(lead=D[:, todo] @ Lt, inverse=np.zeros((len(S), len(names))))
            maps["inverse"][:, use] = S
        if self.strategy == "latent":  # Fourier features of positions (in dm)
            arg = np.nan_to_num(np.vstack([pos, tpos[todo]]))[..., None] * 10 * np.pi
            arg = arg * 2.0 ** np.arange(self.n_freqs)
            feat = np.concatenate([np.sin(arg), np.cos(arg)], -1).reshape(len(arg), -1)
            maps.update(src_feat=feat[: len(pos)], tgt_feat=feat[len(pos) :])
            maps.update(lat_D=D[:, todo], lat_w=use / max(use.sum(), 1))
            maps["lat_mask"] = torch.as_tensor(use, device=dev)
        maps = {
            k: v if torch.is_tensor(v) else torch.tensor(v, device=dev, dtype=dt)
            for k, v in maps.items()
        }
        if self.drop_non_eeg:
            maps["electrode_mask"] = torch.as_tensor(keep, device=dev)
        maps["observed"] = torch.as_tensor(seen, device=dev)
        return maps

    def _apply(self, fn, recurse=True):
        for maps in self._cache.values():  # cached maps follow .to() / .half()
            maps.update({k: fn(t) for k, t in maps.items()})
        super()._apply(fn, recurse)
        self._buffers.update(self._cache.get(self._key, {}))
        return self

    def fit(self, X, chs_info_dense: list[dict]) -> ChannelLayer:
        """Fit the spatial covariance of the ``wiener`` strategy on a dense recording.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_dense)
            Samples of the dense recording.
        chs_info_dense : list of dict
            Channels of ``X`` in its column order, in metres in the head frame (a
            standard electrode name supplies a missing position). Each montage is
            matched to them by position, within 15 mm.

        Returns
        -------
        ChannelLayer
            This layer; its covariance and positions are saved in the ``state_dict``.
        """
        pos = _resolve(chs_info_dense)[1]
        X = np.asarray(X, float)
        if self.strategy != "wiener" or X.shape[1:] != pos.shape[:1]:
            raise ValueError(
                f"fit() is for 'wiener' and needs X (n_samples, {len(pos)}); got {self.strategy!r}, X {X.shape}."
            )
        dev = self.get_buffer("cov").device
        self.cov = torch.tensor(np.cov(X.T), dtype=torch.float32, device=dev)
        self.dense_pos = torch.tensor(pos, dtype=torch.float32, device=dev)
        self._cache.clear()
        self._key = None
        return self

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        self._cache.clear()  # maps are rebuilt after a load
        self._key = None
        for k in ("cov", "dense_pos"):  # fitted buffers change size
            if hasattr(self, k) and prefix + k in state_dict:
                setattr(self, k, torch.empty_like(state_dict[prefix + k]))
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def montages(self, pos: Tensor, ch_mask: Tensor) -> list[list[dict]]:
        """One montage per sample from the positions of a padded batch.

        Parameters
        ----------
        pos : torch.Tensor of shape (batch, channels, 3)
            Head-frame positions of each sample's channels, as
            :func:`~braindecode.datasets.pad_channels_collate` pads them.
        ch_mask : torch.Tensor of shape (batch, channels)
            ``True`` for the real channels of each sample.

        Returns
        -------
        list of list of dict
            For each sample, its real channels named ``#0, #1, ...`` at their
            positions, for :meth:`forward`. A montage is built once per set of
            positions (the last 32 are kept), so the maps of :meth:`forward` stay
            cached per montage.
        """
        # MNE builds the maps on the host; one small copy per forward.
        positions = pos.detach().cpu().double().numpy()
        real = (
            cast(Tensor, check_ch_mask(ch_mask, pos)).cpu().numpy()
        )  # (batch, channels)
        return [_montage_at(p[m].tobytes()) for p, m in zip(positions, real)]

    def forward(
        self,
        x: Tensor,
        chs_info: list[dict] | list[list[dict]] | None = None,
        ch_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Map a batch onto the target channels.

        Parameters
        ----------
        x : torch.Tensor of shape (batch, channels, time)
            Signal recorded with ``chs_info``; with one montage per sample, padded
            to the largest montage.
        chs_info : list of dict or list of list of dict, optional
            Montage of ``x``, or one montage per sample. Defaults to the montage
            given at construction.
        ch_mask : torch.Tensor of shape (batch, channels), optional
            With one montage per sample, ``True`` for the real channels of each
            sample (default: the first ones, as
            :func:`~braindecode.datasets.pad_channels_collate` pads).

        Returns
        -------
        mapped : torch.Tensor of shape (batch, n_targets, time)
            ``x`` in the target channels.
        observed : torch.Tensor of bool
            Targets that are measured or exactly derived: ``(n_targets,)``, or
            ``(batch, n_targets)`` with one montage per sample.
        """
        chs = chs_info if chs_info is not None else self.chs_info
        if chs is None:
            raise ValueError(
                f"No montage: pass chs_info to the model (or to forward) so channel strategy {self.strategy!r} knows the input channels."
            )
        if chs and isinstance(chs[0], (list, tuple)):  # one montage per sample
            big = max(map(len, chs)) > x.shape[1]
            if self.trainable or self.strategy == "latent" or len(chs) != len(x) or big:
                raise ValueError(
                    f"One montage per sample needs a strategy without trained state and one montage of at most {x.shape[1]} channels for each of the {len(x)} samples; got {self.strategy!r} and {len(chs)} montages."
                )
            ch_mask = check_ch_mask(ch_mask, x)
            # real: the mask on the host, one read per batch instead of a sync per sample
            real = None if ch_mask is None else ch_mask.cpu().numpy()
            signals, observed = [], []
            for b, montage in enumerate(cast(list[list[dict]], chs)):
                signal = (  # x[b]: (C, T) -> the real channels (len(montage), T)
                    x[b, : len(montage)]
                    if real is None
                    else x[b, torch.as_tensor(np.flatnonzero(real[b]), device=x.device)]
                )
                if signal.shape[0] != len(montage):
                    raise ValueError(
                        f"Sample {b}: ch_mask has {signal.shape[0]} channels, "
                        f"its montage {len(montage)}."
                    )
                self._use(montage)
                if self.drop_non_eeg:
                    signal = signal.masked_fill(~self.electrode_mask[:, None], 0)
                signals.append(self.weight @ signal)
                observed.append(self.observed)
            return torch.stack(signals), torch.stack(observed)
        if ch_mask is not None:
            raise ValueError("ch_mask needs one montage per sample in chs_info.")
        if x.shape[1] != len(chs):
            raise ValueError(
                f"Input has {x.shape[1]} channels but the montage has {len(chs)}; they must match."
            )
        self._use(cast(list[dict], chs))
        if self.drop_non_eeg:
            x = x.masked_fill(~self.electrode_mask[:, None], 0)
        out = self.weight @ x
        if self.trainable:
            out = out + self.lead @ (self.parcel_mix @ (self.inverse @ x))
        if self.strategy == "latent" and self.lat_D.shape[1]:
            xf = x.to(torch.promote_types(x.dtype, torch.float32))  # fp16: 1e-12 = 0
            var = (xf - xf.mean(-1, keepdim=True)).square().sum(-1) / (x.shape[-1] - 1)
            stats = torch.stack([var, xf.diff(dim=-1).abs().mean(-1)], -1)
            stats = (stats + 1e-12).log()
            stats = stats - (self.lat_w.to(stats.dtype) @ stats)[:, None]  # (B, C, 2)
            keys = self.key_pos(self.src_feat) + self.key_stats(stats.to(x.dtype))
            q = self.query(self.tgt_feat)
            logits = q @ keys.transpose(1, 2) / keys.shape[-1] ** 0.5
            attn = logits.masked_fill(~self.lat_mask, float("-inf")).softmax(-1)
            out = out + self.lat_D @ (attn @ x)
        return out, self.observed


@lru_cache(maxsize=1)
def _reve_position_bank() -> dict[str, np.ndarray]:
    """REVE's electrode positions (m) by name, rebuilt from MNE's montages.

    The released bank (``brain-bzh/reve-positions``) holds MNE positions in the
    montage frame: the 10-05 family and BioSemi-128, the EGI HydroCel-129 net
    moved so that its vertex reference (``E129``) sits on 10-05 ``Cz``, and every
    BioSemi-128 channel again under a ``biosemi128_`` prefix. The montages below
    hold more names than the release; only those of ``_REVE_POSITION_NAMES`` are
    kept, each at the position of the MNE name it equals ignoring case (the
    first montage wins a clash).

    Returns
    -------
    dict[str, np.ndarray]
        Release names, in the release's order and spellings, to ``(3,)`` positions.
    """

    def positions(kind):
        return mne.channels.make_standard_montage(kind).get_positions()["ch_pos"]

    groups = [
        positions(kind)
        for kind in (
            "colin27_1005",
            "colin27_1020",
            "colin27_postfixed",
            "colin27_primed",
            "biosemi128",
        )
    ]
    egi = positions("GSN-HydroCel-129")
    shift = groups[0]["Cz"] - egi["Cz"]
    groups.append(
        {("E129" if ch == "Cz" else ch): xyz + shift for ch, xyz in egi.items()}
    )
    groups += [positions(kind) for kind in ("colin27_alphabetic", "colin27_prefixed")]
    groups.append(
        {f"biosemi128_{ch}": xyz for ch, xyz in positions("biosemi128").items()}
    )
    by_upper: dict[str, np.ndarray] = {}
    for group in groups:
        for ch, xyz in group.items():
            by_upper.setdefault(ch.upper(), xyz)
    return {name: by_upper[name.upper()] for name in _REVE_POSITION_NAMES}


# The 543 electrode names of REVE's released position bank (``positions.json`` of
# https://huggingface.co/brain-bzh/reve-positions), in the release's order and
# spellings: both ``Cz`` and ``CZ``, ``Cp1`` as well as ``CP1`` (18 pairs differ
# only in case). ``_reve_position_bank`` places each on MNE's montages.
_REVE_POSITION_NAMES = """
    A1 A2 C3 C4 CZ F3 F4 F7 F8 FP1 FP2 FZ O1 O2 P3 P4 PZ T3 T4 T5 T6 OZ Cz Fpz
    Fz P7 P8 Pz T7 T8 C1 C2 C5 C6 CP1 CP2 CP3 CP4 CPz FC1 FC2 FC3 FC4 FCz P1 P2
    POz AFz P5 P6 PO3 PO4 AF3 AF4 AF7 AF8 CP5 CP6 F1 F2 F5 F6 FC5 FC6 FT7 FT8
    Fp1 Fp2 Iz Oz P10 P9 PO7 PO8 TP7 TP8 F10 F9 FT10 FT9 FTT10h FTT9h PO10 PO9
    TP10 TP9 TPP10h TPP8h TPP9h TTP7h CCP1h CCP2h CCP3h CCP4h CCP5h CCP6h CPP1h
    CPP2h CPP3h CPP4h CPP5h CPP6h FCC1h FCC2h FCC3h FCC4h FCC5h FCC6h FFC1h
    FFC2h FFC3h FFC4h FFC5h FFC6h FTT7h FTT8h PPO1h PPO2h TTP8h T10 T9 AFF1h
    AFF2h AFF5h AFF6h AFp1 AFp2 POO1 POO2 PO5 PO6 AFF1 AFF2 AFp3h AFp4h FFT7h
    FFT8h I1 I2 M1 M2 OI1h OI2h POO10h POO3h POO4h POO9h PPO1 PPO10h PPO2 PPO5h
    PPO6h PPO9h TPP7h A3 A4 A5 A6 A7 A8 A9 A10 A11 A12 A13 A14 A15 A16 A17 A18
    A19 A20 A21 A22 A23 A24 A25 A26 A27 A28 A29 A30 A31 A32 B1 B2 B3 B4 B5 B6 B7
    B8 B9 B10 B11 B12 B13 B14 B15 B16 B17 B18 B19 B20 B21 B22 B23 B24 B25 B26
    B27 B28 B29 B30 B31 B32 C7 C8 C9 C10 C11 C12 C13 C14 C15 C16 C17 C18 C19 C20
    C21 C22 C23 C24 C25 C26 C27 C28 C29 C30 C31 C32 D1 D2 D3 D4 D5 D6 D7 D8 D9
    D10 D11 D12 D13 D14 D15 D16 D17 D18 D19 D20 D21 D22 D23 D24 D25 D26 D27 D28
    D29 D30 D31 D32 CPZ FCZ FPZ POZ E1 E10 E100 E101 E102 E103 E104 E105 E106
    E107 E108 E109 E11 E110 E111 E112 E113 E114 E115 E116 E117 E118 E119 E12
    E120 E121 E122 E123 E124 E125 E126 E127 E128 E13 E14 E15 E16 E17 E18 E19 E2
    E20 E21 E22 E23 E24 E25 E26 E27 E28 E29 E3 E30 E31 E32 E33 E34 E35 E36 E37
    E38 E39 E4 E40 E41 E42 E43 E44 E45 E46 E47 E48 E49 E5 E50 E51 E52 E53 E54
    E55 E56 E57 E58 E59 E6 E60 E61 E62 E63 E64 E65 E66 E67 E68 E69 E7 E70 E71
    E72 E73 E74 E75 E76 E77 E78 E79 E8 E80 E81 E82 E83 E84 E85 E86 E87 E88 E89
    E9 E90 E91 E92 E93 E94 E95 E96 E97 E98 E99 E129 Cp1 Cp2 Cp5 Cp6 Fc1 Fc2 Fc5
    Fc6 biosemi128_A1 biosemi128_A2 biosemi128_A3 biosemi128_A4 biosemi128_A5
    biosemi128_A6 biosemi128_A7 biosemi128_A8 biosemi128_A9 biosemi128_A10
    biosemi128_A11 biosemi128_A12 biosemi128_A13 biosemi128_A14 biosemi128_A15
    biosemi128_A16 biosemi128_A17 biosemi128_A18 biosemi128_A19 biosemi128_A20
    biosemi128_A21 biosemi128_A22 biosemi128_A23 biosemi128_A24 biosemi128_A25
    biosemi128_A26 biosemi128_A27 biosemi128_A28 biosemi128_A29 biosemi128_A30
    biosemi128_A31 biosemi128_A32 biosemi128_B1 biosemi128_B2 biosemi128_B3
    biosemi128_B4 biosemi128_B5 biosemi128_B6 biosemi128_B7 biosemi128_B8
    biosemi128_B9 biosemi128_B10 biosemi128_B11 biosemi128_B12 biosemi128_B13
    biosemi128_B14 biosemi128_B15 biosemi128_B16 biosemi128_B17 biosemi128_B18
    biosemi128_B19 biosemi128_B20 biosemi128_B21 biosemi128_B22 biosemi128_B23
    biosemi128_B24 biosemi128_B25 biosemi128_B26 biosemi128_B27 biosemi128_B28
    biosemi128_B29 biosemi128_B30 biosemi128_B31 biosemi128_B32 biosemi128_C1
    biosemi128_C2 biosemi128_C3 biosemi128_C4 biosemi128_C5 biosemi128_C6
    biosemi128_C7 biosemi128_C8 biosemi128_C9 biosemi128_C10 biosemi128_C11
    biosemi128_C12 biosemi128_C13 biosemi128_C14 biosemi128_C15 biosemi128_C16
    biosemi128_C17 biosemi128_C18 biosemi128_C19 biosemi128_C20 biosemi128_C21
    biosemi128_C22 biosemi128_C23 biosemi128_C24 biosemi128_C25 biosemi128_C26
    biosemi128_C27 biosemi128_C28 biosemi128_C29 biosemi128_C30 biosemi128_C31
    biosemi128_C32 biosemi128_D1 biosemi128_D2 biosemi128_D3 biosemi128_D4
    biosemi128_D5 biosemi128_D6 biosemi128_D7 biosemi128_D8 biosemi128_D9
    biosemi128_D10 biosemi128_D11 biosemi128_D12 biosemi128_D13 biosemi128_D14
    biosemi128_D15 biosemi128_D16 biosemi128_D17 biosemi128_D18 biosemi128_D19
    biosemi128_D20 biosemi128_D21 biosemi128_D22 biosemi128_D23 biosemi128_D24
    biosemi128_D25 biosemi128_D26 biosemi128_D27 biosemi128_D28 biosemi128_D29
    biosemi128_D30 biosemi128_D31 biosemi128_D32
""".split()
