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
import re
import warnings
from collections import OrderedDict
from functools import lru_cache
from typing import NamedTuple
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

try:  # MNE's published aliases (Cb1 = POO7, T1 = T9, M1 = TP9, VREF = Cz, ...)
    from mne._fiff.constants import CHANNEL_LOC_ALIASES
except ImportError:  # pragma: no cover - private module moved
    CHANNEL_LOC_ALIASES = {"T3": "T7", "T4": "T8"}

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


# --------------------------------------------------------------------------
# Channel positions from a name -> position bank (REVE, and later the other
# name- or position-indexed checkpoints).
# --------------------------------------------------------------------------

#: Distance (m) under which a channel's ``loc`` agrees with a bank entry
#: (registration inliers; snapping beyond it warns).
POSITION_TOLERANCE = 0.015
#: Distance (m) beyond which a named channel's registered ``loc`` means another
#: site than the bank entry of that name. MNE's 10-10 montages (BioSemi-32/64,
#: EasyCap-M1, BrainProducts RNP-BA-128) put a site up to 16.2 mm from REVE's
#: entry of its name; BioSemi-128's reused names start at 37.7 mm (C2).
COLLISION_DISTANCE = 0.03
_ON_MISSING = ("error", "drop", "nearest")
# 10-10 names of the legacy 10-20 temporal sites (T5 = P7, not MNE's T9).
_ALIASES = {k.lower(): v for k, v in CHANNEL_LOC_ALIASES.items()}
_ALIASES.update({"t3": "T7", "t4": "T8", "t5": "P7", "t6": "P8"})
_PREFIX = re.compile(r"^eeg[\s_-]+", re.IGNORECASE)
_SUFFIX = re.compile(r"-(ref|le|ar)$", re.IGNORECASE)
_BIPOLAR = re.compile(r"\s*[-\u2013]\s*")


class ResolvedPositions(NamedTuple):
    """Result of :func:`resolve_channel_positions`.

    Attributes
    ----------
    positions : numpy.ndarray
        ``(K, 3)`` positions of the kept channels, in the bank's frame (m).
    index : numpy.ndarray
        ``(K,)`` indices of the kept channels in ``chs_info``; ``arange(C)``
        unless ``on_missing="drop"`` removed channels.
    provenance : list of dict
        One entry per input channel (dropped ones included): ``ch_name``,
        ``rule`` (``"montage"``, ``"exact"``, ``"normalised"``, ``"loc"``,
        ``"bipolar"`` or ``"dropped"``), ``name`` (the bank entry used, or
        the one a ``"nearest"`` channel snapped to; ``None`` otherwise),
        ``distance`` (m: snap distance, or the registered ``loc``'s distance
        to the named bank entry; ``None`` when not measured) and ``note``.
    """

    positions: np.ndarray
    index: np.ndarray
    provenance: list


def _normalise_name(name: str) -> str:
    """``"EEG Fp1-REF"`` -> ``"Fp1"`` (prefix ``EEG``, suffixes -REF/-LE/-AR)."""
    return _SUFFIX.sub("", _PREFIX.sub("", name.strip())).strip()


def _similarity(src: np.ndarray, dst: np.ndarray):
    """Least-squares ``dst ~ s * src @ R.T + t`` (Umeyama, 1991)."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    a, b = src - mu_s, dst - mu_d
    U, S, Vt = np.linalg.svd(b.T @ a / len(src))
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(U) * np.linalg.det(Vt)) or 1.0])
    R = U @ D @ Vt
    s = float(np.trace(np.diag(S) @ D) / ((a**2).sum() / len(src)))
    return s, R, mu_d - s * R @ mu_s


def _register(src: np.ndarray, dst: np.ndarray, tol: float):
    """Fit the montage's ``loc`` frame onto the bank's on the channels named in
    both, ignoring the ones more than ``tol`` off (name collisions).

    Returns the transform, or ``None`` when the channels do not determine a
    frame (fewer than 4, collinear, scale outside [0.5, 2]) or fewer than half
    of them agree with their bank entries.
    """
    if len(src) < 4:
        return None
    sv = np.linalg.svd(src - src.mean(0), compute_uv=False)
    if sv[1] <= 1e-3 * sv[0]:  # collinear: no rotation about that line
        return None
    # raw.set_montage gives ``loc`` in the head frame; banks such as REVE's are
    # in the montage files' own frames (standard_1005: 46-57 mm apart).
    inl = np.linalg.norm(src - dst, axis=1) <= tol
    if inl.sum() < max(4, len(src) / 2):
        inl = np.ones(len(src), bool)
    for _ in range(20):  # trimmed least squares
        s, R, t = _similarity(src[inl], dst[inl])
        res = np.linalg.norm(s * src @ R.T + t - dst, axis=1)
        new = res <= tol
        if new.sum() < 4 or (new == inl).all():
            break
        inl = new
    if not 0.5 <= s <= 2.0 or (res <= tol).sum() < max(4, len(src) / 2):
        return None
    return lambda x: s * np.asarray(x) @ R.T + t


def _template_registration(table: np.ndarray, keys: list[str]):
    """Head frame -> bank frame from the bank's own ``standard_1005`` names
    (MNE's positions in the head frame, as ``raw.set_montage`` gives them)."""
    std = _standard_positions()
    rows = [i for i, k in enumerate(keys) if k.lower() in std]
    src = np.array([std[keys[i].lower()] for i in rows]).reshape(-1, 3)
    return _register(src, table[rows], POSITION_TOLERANCE)


def resolve_channel_positions(
    chs_info: list[dict],
    bank: dict,
    montage: str | None = None,
    on_missing: str = "error",
) -> ResolvedPositions:
    """Position of each channel of ``chs_info`` in a name -> position bank.

    The shared rule of the models whose checkpoint indexes channels by name or
    by position (:class:`braindecode.models.REVE`). Each channel takes the first
    rule that applies:

    1. ``montage`` given: the montage-qualified bank name (``montage="biosemi128"``:
       ``A1`` -> ``biosemi128_A1``).
    2. The exact bank name (the checkpoint's own lookup). If the channel's
       ``loc``, registered onto the bank's frame, lies more than 30 mm
       (``COLLISION_DISTANCE``) from that entry, the name means another site
       in this montage (REVE's ``A1``/``A2`` are earlobes, BioSemi's are near
       the vertex): the registered ``loc`` is used, with a warning.
    3. The normalised name: case-insensitive, without an ``EEG`` prefix or a
       ``-REF``/``-LE``/``-AR`` suffix, MNE's location aliases (``M1`` = ``TP9``)
       and the legacy 10-20 names (``T3``/``T4``/``T5``/``T6`` =
       ``T7``/``T8``/``P7``/``P8``).
    4. The channel's ``loc`` (finite, non-zero), registered onto the bank's
       frame.
    5. A bipolar name ``A-B`` (or ``A–B``): the midpoint of ``A`` and ``B``,
       both resolved by rules 1-4.
    6. Otherwise ``on_missing`` decides.

    Registration: ``raw.set_montage`` gives ``loc`` in the head frame, while a
    bank such as REVE's holds the montage files' own frames (46-57 mm apart for
    ``standard_1005``). A similarity transform fitted on the channels whose
    ``loc`` and bank name are both known (at least 4, not collinear; channels
    more than 15 mm off are left out of the fit, and at least half must agree)
    maps ``loc`` onto the bank. Otherwise the transform fitted between MNE's
    ``standard_1005`` in the head frame and the bank's entries of the same
    names is used (template frames differ by up to ~30 mm); if the bank has
    none, ``loc`` is used as given, with a warning. Names are checked against
    ``loc`` (rule 2) only when at least 4 channels have both: with fewer, the
    bank name wins (pass ``montage`` for a small BioSemi subset); with 4 or
    more of which fewer than half agree, the montage reuses the bank's names
    for other sites (GSN-HydroCel-256 or BioSemi-256 against REVE's
    GSN-HydroCel-129 and BioSemi-128 names) and each disagreeing channel
    takes its ``loc``.

    A position is never invented: a channel without a matching name and
    without a ``loc`` gets none, and name similarity is only suggested.

    Parameters
    ----------
    chs_info : list of dict
        Channels (``ch_name``, optional ``loc`` in metres), as in ``raw.info["chs"]``.
    bank : dict
        Bank name -> ``(3,)`` position (metres).
    montage : str, optional
        Prefix of the bank's montage-qualified names (e.g. ``"biosemi128"``).
    on_missing : {"error", "drop", "nearest"}
        Channels without a position: ``"error"`` raises a :class:`ValueError`
        listing them with the closest bank names (suggestions only);
        ``"drop"`` removes them with a warning. ``"nearest"`` (for models
        that need a bank entry per channel) also snaps every channel placed by
        its ``loc`` or as a bipolar midpoint to the nearest bank entry, warning
        beyond 15 mm; it still raises for a channel without a position.

    Returns
    -------
    ResolvedPositions
        ``positions`` ``(K, 3)``, ``index`` of the kept channels and the
        per-channel ``provenance``.
    """
    if on_missing not in _ON_MISSING:
        raise ValueError(
            f"on_missing must be one of {_ON_MISSING}; got {on_missing!r}."
        )
    keys = list(bank)
    table = np.array([bank[k] for k in keys], float).reshape(-1, 3)
    if not len(keys) or not np.isfinite(table).all():
        raise ValueError("The position bank must hold finite (3,) positions.")
    row = {k: i for i, k in enumerate(keys)}
    lower: dict[str, list[int]] = {}
    for i, k in enumerate(keys):
        lower.setdefault(k.lower(), []).append(i)

    def caseless(name):  # one position under all spellings, else no match
        hits = lower.get(name.lower(), [])
        if hits and np.ptp(table[hits], axis=0).max() < 1e-6:
            return hits[0]
        return None

    if montage is not None:
        qualifiers = sorted({k.split("_")[0] for k in keys if "_" in k})
        if not any(k.lower().startswith(f"{montage}_".lower()) for k in keys):
            raise ValueError(
                f"The position bank has no names qualified by montage {montage!r}; "
                f"its montages: {qualifiers}."
            )

    def by_name(name):  # rules 1-3: (rule, bank row, note) or None
        norm = _normalise_name(name)
        if montage is not None:
            for n in dict.fromkeys((name, norm)):
                j = row.get(f"{montage}_{n}", caseless(f"{montage}_{n}"))
                if j is not None:
                    return "montage", j, ""
        if name in row:
            return "exact", row[name], ""
        j = row.get(norm, caseless(norm))
        if j is not None:
            return "normalised", j, f"{name!r} read as {keys[j]!r}"
        alias = _ALIASES.get(norm.lower())
        j = None if alias is None else row.get(alias, caseless(alias))
        if j is not None:
            return "normalised", j, f"{name!r} read as {keys[j]!r} (alias)"
        return None

    names, pos_or_std, _, _, given = _resolve(chs_info)
    loc = np.where(given[:, None], pos_or_std, np.nan)
    named = [by_name(n) for n in names]
    anchors = [i for i, m in enumerate(named) if m is not None and given[i]]
    target = table[[named[i][1] for i in anchors]].reshape(-1, 3)
    transform = None
    if given.any():
        transform = _register(loc[anchors], target, POSITION_TOLERANCE)
        if transform is None:  # too few or disagreeing names: the template
            transform = _template_registration(table, keys)
    residual = {}
    if transform is not None and len(anchors) >= 4:
        d = np.linalg.norm(transform(loc[anchors]) - target, axis=1)
        residual = dict(zip(anchors, d.tolist()))

    pos = np.full((len(names), 3), np.nan)
    prov = [
        dict(ch_name=n, rule=None, name=None, distance=None, note="") for n in names
    ]
    collided, as_given = [], []
    for i, m in enumerate(named):
        if m is not None and residual.get(i, 0.0) <= COLLISION_DISTANCE:
            pos[i] = table[m[1]]
            prov[i].update(rule=m[0], name=keys[m[1]], note=m[2])
            if i in residual:
                prov[i]["distance"] = float(residual[i])
        elif given[i]:
            pos[i] = loc[i] if transform is None else transform(loc[i])
            prov[i]["rule"] = "loc"
            if m is not None:  # rule 2: the name means another site here
                collided.append(f"{names[i]} ({residual[i] * 1e3:.0f} mm)")
                prov[i]["distance"] = float(residual[i])
                prov[i]["note"] = f"name collision with bank {keys[m[1]]!r}"
            elif transform is None:
                as_given.append(names[i])
                prov[i]["note"] = "loc used as given (not registered)"
    if collided:
        if len(collided) > 10:
            collided = [*collided[:10], f"... {len(collided) - 10} more"]
        warnings.warn(
            f"The 'loc' of channels {collided} lies more than "
            f"{COLLISION_DISTANCE * 1e3:.0f} mm from the bank entry of the same "
            "name (a montage that reuses another site's name, e.g. BioSemi A1/A2): "
            "their 'loc' is used. Pass montage=... if the bank qualifies its names.",
            UserWarning,
            stacklevel=2,
        )
    if as_given:
        warnings.warn(
            f"Channels {as_given} are placed at their 'loc' as given: neither "
            "their montage nor the bank's standard_1005 names register it onto "
            "the bank's frame, so 'loc' must already be in it.",
            UserWarning,
            stacklevel=2,
        )

    placed = {  # rule-4 channels by name, for bipolar halves
        n.lower(): pos[i] for i, n in enumerate(names) if prov[i]["rule"] == "loc"
    }
    for i, n in enumerate(names):  # rule 5: bipolar midpoint
        parts = _BIPOLAR.split(_normalise_name(n))
        if prov[i]["rule"] is not None or len(parts) != 2 or not all(parts):
            continue
        ends = []
        for p in parts:
            m = by_name(p)
            ends.append(table[m[1]] if m is not None else placed.get(p.lower()))
        if all(e is not None for e in ends):
            pos[i] = (ends[0] + ends[1]) / 2
            prov[i].update(
                rule="bipolar", note=f"midpoint of {parts[0]} and {parts[1]}"
            )

    missing = [i for i in range(len(names)) if prov[i]["rule"] is None]
    if missing and on_missing != "drop":
        hint = {
            names[i]: difflib.get_close_matches(_normalise_name(names[i]), keys, n=3)
            for i in missing
        }
        why = (
            "on_missing='nearest' snaps only channels that have a position (name "
            "similarity never places a channel)"
            if on_missing == "nearest"
            else "Rename them, supply their 'loc', pass montage=..."
        )
        raise ValueError(
            f"No position in the position bank for channels "
            f"{[names[i] for i in missing]}: no matching name and no finite "
            f"non-zero 'loc'. Closest bank names (suggestions only, never "
            f"applied): {hint}. {why}; or pass on_missing='drop' to remove them."
        )
    if missing:
        warnings.warn(
            f"Channels {[names[i] for i in missing]} have no position in the "
            "bank and no 'loc'; they are dropped.",
            UserWarning,
            stacklevel=2,
        )
        for i in missing:
            prov[i]["rule"] = "dropped"

    if on_missing == "nearest":  # positioned channels -> nearest bank entry
        far = []
        for i, p in enumerate(prov):
            if p["rule"] in ("loc", "bipolar"):
                d = np.linalg.norm(table - pos[i], axis=1)
                j = int(np.argmin(d))
                pos[i] = table[j]
                p.update(name=keys[j], distance=float(d[j]))
                p["note"] = (p["note"] + "; " if p["note"] else "") + "snapped"
                if d[j] > POSITION_TOLERANCE:
                    far.append(f"{p['ch_name']} -> {keys[j]} ({d[j] * 1e3:.0f} mm)")
        if far:
            warnings.warn(
                f"Channels snapped to a bank entry more than "
                f"{POSITION_TOLERANCE * 1e3:.0f} mm away: {far}.",
                UserWarning,
                stacklevel=2,
            )

    keep = np.setdiff1d(np.arange(len(names)), missing).astype(int)
    return ResolvedPositions(pos[keep], keep, prov)


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
    # ponytail: patches a private MNE helper; drop the patch if MNE renames it.
    fit = mock.patch.object(  # deterministic fit: reuse its result
        mne.bem,
        "_fwd_eeg_fit_berg_scherg",
        lambda m, *_: m.update(zip(("mu", "lambda"), _MU_LAMBDA), nfit=3) or 0.0,
    )
    with mne.utils.use_log_level("ERROR"), fit:
        sphere = mne.make_sphere_model(r0=_SPHERE_R0, head_radius=_SPHERE_RADIUS)
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
        # A-B is copied when measured, else derived from A and B (both columns).
        mono = list(dict.fromkeys(m for n in tnames for m in (n, *n.split("-"))))
        named = {n: c for n, c in zip(tnames, self.target) if "-" not in n}
        _, tpos, tstd, *_ = _resolve([named.get(m, {"ch_name": m}) for m in mono])
        self._whole = [mono.index(n) for n in tnames]
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
        W = np.zeros((len(mono), len(names)))
        for k, m in enumerate(mono):
            for hit in (low == m.lower(), alias[k], near[k] <= 0.015):
                if (hit & keep).any():
                    j = np.flatnonzero(hit & keep)
                    W[k, j[np.argmin(near[k][j])]] = 1.0
                    break
        observed = W.sum(1) > 0
        copied = observed[self._whole][:, None]  # a measured A-B target: copy it
        D = np.where(copied, np.eye(len(mono))[self._whole], self._D)
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
        maps["observed"] = torch.as_tensor(seen, device=dev)
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
