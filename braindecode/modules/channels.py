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
from collections import OrderedDict
from functools import lru_cache
from unittest import mock

import mne
import numpy as np
import torch
from mne.io.constants import FIFF
from scipy.cluster.vq import kmeans2
from scipy.linalg import solve
from scipy.spatial.distance import cdist
from torch import Tensor, nn

from braindecode.util import resolve_montage_name

STRATEGIES = (
    *("exact", "zero", "nearest", "idw", "spline", "field", "source"),
    *("wiener", "region", "latent"),
)
_MIN_POSITIONS = dict(zip(STRATEGIES, (0, 0, 1, 1, 4, 4, 4, 1, 2, 1)))  # positioned
_ELECTRODES = ("eeg", "seeg", "ecog", "dbs")
_FIFF_KINDS = {FIFF.FIFFV_EEG_CH: "eeg", FIFF.FIFFV_SEEG_CH: "seeg"}
_FIFF_KINDS.update({FIFF.FIFFV_ECOG_CH: "ecog", FIFF.FIFFV_DBS_CH: "dbs"})
_MU_LAMBDA = (  # Berg-Scherg fit of make_sphere_model's 4 shells (COBYLA, 0.5 s)
    np.array([0.9433448511080679, 0.663623934869853, 0.079878238156799]),
    np.array([0.4260455268056578, 2.0834380895598508, -0.05381815373454841]),
)


@lru_cache(maxsize=1)
def _standard_positions() -> dict[str, np.ndarray]:
    """Lower-case ``standard_1005`` name (T3 = T7 included) -> position, own frame."""
    std = mne.channels.make_standard_montage(resolve_montage_name("standard_1005"))
    return {n.lower(): p for n, p in std.get_positions()["ch_pos"].items()}


def _resolve(chs_info: list[dict]):
    """Names, positions ``(C, 3)`` (given ``loc``, else ``standard_1005``, else NaN),
    standard positions of the names, channel types and which ``loc`` were given."""
    names = [str(ch["ch_name"]) for ch in chs_info]
    types = [  # missing kind = EEG; FIFF kind ints as in raw.info
        k.lower() if isinstance(k, str) else _FIFF_KINDS.get(int(k), str(k))
        for k in (ch.get("kind") or "eeg" for ch in chs_info)
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
    # ponytail: patches a private MNE helper; drop the patch if MNE renames it.
    fit = mock.patch.object(  # deterministic fit: reuse its result
        mne.bem,
        "_fwd_eeg_fit_berg_scherg",
        lambda m, *_: m.update(zip(("mu", "lambda"), _MU_LAMBDA), nfit=3) or 0.0,
    )
    with mne.utils.use_log_level("ERROR"), fit:
        sphere = mne.make_sphere_model(r0=(0.0, 0.0, 0.04), head_radius=0.09)
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
        self._cache, self._key, self._chs = OrderedDict(), None, None
        self.register_buffer("weight", torch.zeros(0, 0), persistent=False)
        tnames = [str(c["ch_name"]) for c in self.target]
        mono = list(dict.fromkeys(m for n in tnames for m in n.split("-")))
        named = {str(c["ch_name"]): c for c in self.target}
        _, tpos, tstd, *_ = _resolve([named.get(m, {"ch_name": m}) for m in mono])
        D = np.array(
            [
                [(m == n.split("-")[0]) - (m == n.split("-")[-1] != n) for m in mono]
                for n in tnames
            ],
            float,
        )
        self._mono, self._tpos, self._tstd, self._D = mono, tpos, tstd, D
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
        """Make the maps of ``chs`` the buffers (identity, then LRU, then build)."""
        # ponytail: identity fast path; a montage list mutated in place is not seen.
        if chs is self._chs:
            return
        key = tuple(
            (str(c["ch_name"]), c.get("kind"), np.float64(c.get("loc")).tobytes())
            for c in chs
        )
        if key != self._key:
            maps = self._cache.pop(key, None) or self._build(chs)
            self._cache[key] = maps  # most recent last
            if len(self._cache) > 16:
                self._cache.popitem(last=False)
            for name, t in maps.items():
                self.register_buffer(name, t, persistent=False)
            self._key = key
        self._chs = chs

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
        mono, tpos, tstd, D = self._mono, self._tpos, self._tstd, self._D
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
        if self.trainable:
            Lt, S = _source(pos[use], tpos[todo], **self.kwargs)
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
        maps["observed"] = torch.as_tensor((np.abs(D) @ ~observed) == 0, device=dev)
        return maps

    def _apply(self, fn, recurse=True):
        for maps in self._cache.values():  # cached maps follow .to() / .half()
            maps.update({k: fn(t) for k, t in maps.items()})
        super()._apply(fn, recurse)
        self._buffers.update(self._cache.get(self._key, {}))
        return self

    def fit(self, X, chs_info_dense: list[dict]) -> ChannelLayer:
        """``wiener``: fit the spatial covariance on ``X`` ``(n_samples, n_dense)``
        recorded with ``chs_info_dense`` (matched to the montages by position)."""
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
        self._key = self._chs = None
        return self

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        self._cache.clear()  # maps are rebuilt after a load
        self._key = self._chs = None
        for k in ("cov", "dense_pos"):  # fitted buffers change size
            if hasattr(self, k) and prefix + k in state_dict:
                setattr(self, k, torch.empty_like(state_dict[prefix + k]))
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

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
        self._use(chs)  # steady state: no host work, every map already on device
        out = self.weight @ x
        if self.trainable:
            out = out + self.lead @ (self.parcel_mix @ (self.inverse @ x))
        if self.strategy == "latent" and self.lat_D.shape[1]:
            var = (x - x.mean(-1, keepdim=True)).square().sum(-1) / (x.shape[-1] - 1)
            stats = torch.stack([var, x.diff(dim=-1).abs().mean(-1)], -1)
            stats = (stats + 1e-12).log()
            stats = stats - (self.lat_w @ stats)[:, None]  # (B, C, 2), no sync
            keys = self.key_pos(self.src_feat) + self.key_stats(stats)
            q = self.query(self.tgt_feat)
            logits = q @ keys.transpose(1, 2) / keys.shape[-1] ** 0.5
            attn = logits.masked_fill(~self.lat_mask, float("-inf")).softmax(-1)
            out = out + self.lat_D @ (attn @ x)
        return out, self.observed
