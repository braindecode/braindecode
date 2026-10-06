# Authors: Bruno Aristimunha <b.aristimunha@gmail.com>
#
# License: BSD (3-clause)
"""Stage 1 of the channel layer: user montage -> canonical names, positions, types.

Model independent. Every channel strategy starts from a
:class:`ResolvedMontage`, so name aliases, the EEG-type check and position
filling happen in exactly one place.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from functools import lru_cache
from typing import Sequence

import numpy as np

# Legacy / equivalent electrode names mapped to the 10-20 spelling used by the
# model vocabularies. Extend here, once, rather than per model.
CHANNEL_NAME_ALIASES: dict[str, str] = {
    "t3": "t7",
    "t4": "t8",
    "t5": "p7",
    "t6": "p8",
    "a1": "m1",
    "a2": "m2",
}

#: Electrode kinds the layer can place: scalp EEG and intracranial contacts.
ELECTRODE_KINDS: tuple[str, ...] = ("eeg", "seeg", "ecog", "dbs")

# mne.io.constants.FIFF.FIFFV_{EEG,SEEG,ECOG,DBS}_CH
_FIFF_KINDS = {2: "eeg", 802: "seeg", 902: "ecog", 803: "dbs"}


def canon_name(name: str) -> str:
    """Lower-case a channel name and resolve it through the alias table."""
    key = name.lower()
    return CHANNEL_NAME_ALIASES.get(key, key)


@lru_cache(maxsize=1)
def _standard_1005() -> dict[str, np.ndarray]:
    """Lower-cased ``standard_1005`` names (and their aliases) -> position (m)."""
    import mne

    try:
        montage = mne.channels.make_standard_montage("standard_1005")
    except ValueError:  # MNE >= 1.13 renamed the standard montages
        montage = mne.channels.make_standard_montage("colin27_1005")
    pos = {
        k.lower(): np.asarray(v, float)
        for k, v in montage.get_positions()["ch_pos"].items()
    }
    for old, new in CHANNEL_NAME_ALIASES.items():
        if new in pos:
            pos.setdefault(old, pos[new])
        if old in pos:
            pos.setdefault(new, pos[old])
    return pos


def standard_position(name: str) -> np.ndarray:
    """``standard_1005`` position of ``name`` (aliases accepted), NaN if unknown."""
    return _standard_1005().get(name.lower(), np.full(3, np.nan))


def _kind_name(kind) -> str:
    """Lower-case channel kind of a ``chs_info`` entry (string or FIFF code)."""
    # Missing kind = EEG: many users build chs_info by hand without it.
    if kind is None:
        return "eeg"
    if isinstance(kind, str):
        return kind.lower()
    return _FIFF_KINDS.get(int(kind), f"kind {int(kind)}")


def _check_kinds(kinds: Sequence[str]) -> tuple[str, ...]:
    out = tuple(k.lower() for k in kinds)
    bad = sorted(set(out) - set(ELECTRODE_KINDS))
    if not out or bad:
        raise ValueError(
            f"kinds must be a non-empty subset of {ELECTRODE_KINDS}; got {tuple(kinds)}."
        )
    return out


def _loc(ch: dict) -> np.ndarray:
    """First three ``loc`` entries; NaN when missing, non-finite or all zero."""
    loc = ch.get("loc")
    if loc is None or len(loc) < 3:
        return np.full(3, np.nan)
    xyz = np.asarray(loc[:3], dtype=float)
    if not np.isfinite(xyz).all() or np.abs(xyz).max() < 1e-8:
        return np.full(3, np.nan)
    return xyz


@dataclass(frozen=True, eq=False)
class ResolvedMontage:
    """A user montage after resolution.

    Attributes
    ----------
    names : tuple of str
        Channel names as given (non-EEG channels removed if requested).
    canon : tuple of str
        Lower-cased, alias-resolved names.
    positions : ndarray, shape (C, 3)
        Positions in metres: the given ``loc`` when usable, else the
        ``standard_1005`` position of the name, else NaN.
    has_position : ndarray of bool, shape (C,)
        Whether the user supplied the position (``False`` = filled or NaN).
    key : str
        Hash of the names and the positions rounded to 0.1 mm; identical
        montages share it (cache key of the channel layer).
    picks : ndarray of int, shape (C,)
        Index of each kept channel in the input ``chs_info``.
    n_input : int
        Number of channels in the input ``chs_info`` (before dropping).
    kinds : tuple of str
        Channel kind of each kept channel (``"eeg"``, ``"seeg"``, ...).
    """

    names: tuple[str, ...]
    canon: tuple[str, ...]
    positions: np.ndarray
    has_position: np.ndarray
    key: str
    picks: np.ndarray
    n_input: int
    kinds: tuple[str, ...] = ()

    @property
    def positioned(self) -> np.ndarray:
        """Boolean mask of channels with a known (given or filled) position."""
        return np.isfinite(self.positions).all(axis=1)


def resolve_montage(
    chs_info: list[dict],
    *,
    kinds: Sequence[str] = ("eeg",),
    drop_non_eeg: bool = False,
    fill_positions: bool = True,
) -> ResolvedMontage:
    """Resolve MNE-style ``chs_info`` into a :class:`ResolvedMontage`.

    Parameters
    ----------
    chs_info : list of dict
        ``info["chs"]``-like dicts with ``"ch_name"``, optional ``"loc"`` and
        optional ``"kind"`` (string or FIFF code; missing = EEG).
    kinds : sequence of str
        Accepted channel kinds, a subset of :data:`ELECTRODE_KINDS`. The
        default accepts scalp EEG only; intracranial models pass
        ``("eeg", "seeg", "ecog", "dbs")``.
    drop_non_eeg : bool
        Drop channels of another kind (EOG, ECG, stim, ...) instead of raising.
    fill_positions : bool
        Fill missing positions from ``standard_1005`` by name.

    Raises
    ------
    ValueError
        On a channel of a kind not in ``kinds`` (unless ``drop_non_eeg``), on
        duplicated names (case-insensitive) or when no channel is left.
    """
    accepted = _check_kinds(kinds)
    what = "EEG" if accepted == ("eeg",) else f"one of {accepted}"
    picks, names, kept_kinds = [], [], []
    for i, ch in enumerate(chs_info):
        kind = _kind_name(ch.get("kind"))
        if kind not in accepted:
            if drop_non_eeg:
                continue
            raise ValueError(
                f"Channel {ch.get('ch_name')!r} has kind={ch.get('kind')!r}, not "
                f"{what}. Remove it (e.g. raw.pick('eeg')) or pass drop_non_eeg=True."
            )
        picks.append(i)
        names.append(str(ch["ch_name"]))
        kept_kinds.append(kind)
    if not names:
        raise ValueError(f"No {what} channel in chs_info.")
    seen: dict[str, str] = {}
    for n in names:
        if n.lower() in seen:
            raise ValueError(
                f"Duplicate channel name {n!r} (also given as {seen[n.lower()]!r}); "
                f"channel names must be unique (case-insensitive)."
            )
        seen[n.lower()] = n

    given = np.stack([_loc(chs_info[i]) for i in picks])
    has_position = np.isfinite(given).all(axis=1)
    positions = given.copy()
    if fill_positions:
        for i in np.flatnonzero(~has_position):
            positions[i] = standard_position(names[i])

    rounded = np.round(positions * 1e4)  # 0.1 mm
    ident: tuple = (tuple(names), rounded.tolist())
    if any(k != "eeg" for k in kept_kinds):  # EEG-only keys stay as they were
        ident += (tuple(kept_kinds),)
    key = hashlib.sha1(repr(ident).encode()).hexdigest()
    return ResolvedMontage(
        names=tuple(names),
        canon=tuple(canon_name(n) for n in names),
        positions=positions,
        has_position=has_position,
        key=key,
        picks=np.asarray(picks, dtype=int),
        n_input=len(chs_info),
        kinds=tuple(kept_kinds),
    )


def match_names(names: Sequence[str], vocabulary: Sequence[str]) -> np.ndarray:
    """Index of each name in ``vocabulary`` (``-1`` = absent).

    Exact case-insensitive match first, then through the alias table, so a
    vocabulary holding both ``T3`` and ``T7`` keeps them distinct.
    """
    exact: dict[str, int] = {}
    alias: dict[str, int] = {}
    for j, v in enumerate(vocabulary):
        exact.setdefault(v.lower(), j)
        alias.setdefault(canon_name(v), j)
    return np.array(
        [exact.get(n.lower(), alias.get(canon_name(n), -1)) for n in names],
        dtype=int,
    )


def nearest_vocabulary(
    positions: np.ndarray, vocab_positions: np.ndarray, max_mm: float = 15.0
) -> np.ndarray:
    """Index of the nearest vocabulary position within ``max_mm`` (``-1`` = none).

    NaN positions (on either side) never match.
    """
    positions = np.asarray(positions, float).reshape(-1, 3)
    vocab_positions = np.asarray(vocab_positions, float).reshape(-1, 3)
    if len(vocab_positions) == 0:
        return np.full(len(positions), -1, dtype=int)
    d = np.linalg.norm(positions[:, None] - vocab_positions[None], axis=-1)
    d = np.where(np.isnan(d), np.inf, d)
    j = d.argmin(axis=1)
    ok = d[np.arange(len(d)), j] <= max_mm * 1e-3
    return np.where(ok, j, -1).astype(int)
