# Authors: Bruno Aristimunha <b.aristimunha@gmail.com>
#
# License: BSD (3-clause)
"""Stage 1 of the channel layer: montage -> names, positions, kinds; the target contract."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal, NamedTuple, Optional, Sequence, get_args

import numpy as np
from scipy.spatial.distance import cdist

from braindecode.util import resolve_montage_name

# Legacy electrode names -> the 10-20 spelling of the model vocabularies.
CHANNEL_NAME_ALIASES = {
    "t3": "t7",
    "t4": "t8",
    "t5": "p7",
    "t6": "p8",
    "a1": "m1",
    "a2": "m2",
}
#: Electrode kinds the layer can place: scalp EEG and intracranial contacts.
ELECTRODE_KINDS = ("eeg", "seeg", "ecog", "dbs")
_FIFF_KINDS = {2: "eeg", 802: "seeg", 902: "ecog", 803: "dbs"}  # FIFF.FIFFV_*_CH


def canon_name(name: str) -> str:
    """Lower-case a channel name and resolve it through the alias table."""
    return CHANNEL_NAME_ALIASES.get(name.lower(), name.lower())


@lru_cache(maxsize=1)
def _standard_1005() -> dict[str, np.ndarray]:
    """Lower-cased ``standard_1005`` names (and their aliases) -> position (m)."""
    import mne

    montage = mne.channels.make_standard_montage(resolve_montage_name("standard_1005"))
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


def _check_kinds(kinds: Sequence[str]) -> tuple[str, ...]:
    out = tuple(k.lower() for k in kinds)
    if not out or set(out) - set(ELECTRODE_KINDS):
        raise ValueError(
            f"kinds must be a non-empty subset of {ELECTRODE_KINDS}; got {tuple(kinds)}."
        )
    return out


def _kind(ch: dict) -> str:
    kind = ch.get("kind")  # missing = EEG: hand-built chs_info often omit it
    if kind is None or isinstance(kind, str):
        return (kind or "eeg").lower()
    return _FIFF_KINDS.get(int(kind), f"kind {int(kind)}")


def _loc(ch: dict) -> np.ndarray:
    """First three ``loc`` entries; NaN when missing, non-finite or all zero."""
    loc = ch.get("loc")
    xyz = (
        np.asarray(loc[:3], float) if loc is not None and len(loc) >= 3 else np.zeros(3)
    )
    return (
        xyz
        if np.isfinite(xyz).all() and np.abs(xyz).max() >= 1e-8
        else np.full(3, np.nan)
    )


@dataclass(frozen=True, eq=False)
class ResolvedMontage:
    """A montage after resolution.

    ``positions`` (C, 3) in metres: the given ``loc``, else the ``standard_1005``
    position of the name, else NaN; ``has_position`` marks given ones. ``key``
    hashes names and positions (0.1 mm) and keys the map cache. ``picks`` are
    the kept indices of the ``n_input`` input channels, ``kinds`` their kinds.
    """

    names: tuple[str, ...]
    positions: np.ndarray
    has_position: np.ndarray
    key: str
    picks: np.ndarray
    n_input: int
    kinds: tuple[str, ...] = ()

    @property
    def positioned(self) -> np.ndarray:
        return np.isfinite(self.positions).all(axis=1)


def resolve_montage(
    chs_info: list[dict], *, kinds: Sequence[str] = ("eeg",), drop_non_eeg: bool = False
) -> ResolvedMontage:
    """Resolve MNE-style ``chs_info`` into a :class:`ResolvedMontage`.

    Raises a ``ValueError`` on a channel whose kind is not in ``kinds`` (unless
    ``drop_non_eeg``, which drops it), on duplicated names (case-insensitive)
    and when no channel is left.
    """
    accepted = _check_kinds(kinds)
    what = "EEG" if accepted == ("eeg",) else f"one of {accepted}"
    picks = []
    for i, ch in enumerate(chs_info):
        if _kind(ch) in accepted:
            picks.append(i)
        elif not drop_non_eeg:
            raise ValueError(
                f"Channel {ch.get('ch_name')!r} has kind={ch.get('kind')!r}, not {what}. "
                f"Remove it (e.g. raw.pick('eeg')) or pass drop_non_eeg=True."
            )
    if not picks:
        raise ValueError(f"No {what} channel in chs_info.")
    names = [str(chs_info[i]["ch_name"]) for i in picks]
    seen: dict[str, str] = {}
    for n in names:
        if n.lower() in seen:
            raise ValueError(
                f"Duplicate channel name {n!r} (also given as {seen[n.lower()]!r}); "
                f"channel names must be unique (case-insensitive)."
            )
        seen[n.lower()] = n
    kept_kinds = tuple(_kind(chs_info[i]) for i in picks)
    positions = np.stack([_loc(chs_info[i]) for i in picks])
    has_position = np.isfinite(positions).all(axis=1)
    for i in np.flatnonzero(~has_position):
        positions[i] = standard_position(names[i])
    ident: tuple = (tuple(names), np.round(positions * 1e4).tolist())
    if any(k != "eeg" for k in kept_kinds):  # EEG-only keys stay as they were
        ident += (kept_kinds,)
    return ResolvedMontage(
        names=tuple(names),
        positions=positions,
        has_position=has_position,
        key=hashlib.sha1(repr(ident).encode()).hexdigest(),
        picks=np.asarray(picks, dtype=int),
        n_input=len(chs_info),
        kinds=kept_kinds,
    )


def match_names(names: Sequence[str], vocabulary: Sequence[str]) -> np.ndarray:
    """Index of each name in ``vocabulary`` (-1 = absent): exact first, then alias.

    So a vocabulary holding both ``T3`` and ``T7`` keeps them distinct.
    """
    exact: dict[str, int] = {}
    alias: dict[str, int] = {}
    for j, v in enumerate(vocabulary):
        exact.setdefault(v.lower(), j)
        alias.setdefault(canon_name(v), j)
    return np.array(
        [exact.get(n.lower(), alias.get(canon_name(n), -1)) for n in names], dtype=int
    )


def nearest_vocabulary(positions, vocab_positions, max_mm: float = 15.0) -> np.ndarray:
    """Index of the nearest vocabulary position within ``max_mm`` (-1 = none, NaN never matches)."""
    positions = np.asarray(positions, float).reshape(-1, 3)
    vocab_positions = np.asarray(vocab_positions, float).reshape(-1, 3)
    if len(vocab_positions) == 0:
        return np.full(len(positions), -1, dtype=int)
    d = np.nan_to_num(cdist(positions, vocab_positions), nan=np.inf)
    j = d.argmin(axis=1)
    return np.where(d[np.arange(len(d)), j] <= max_mm * 1e-3, j, -1)


Interface = Literal["montage", "ids", "positions", "slots", "free"]


class TargetSensors(NamedTuple):
    """The ``K`` electrodes a strategy must produce for a target."""

    names: tuple[str, ...]
    positions: np.ndarray  # (K, 3) metres, NaN = unknown
    channel_ids: Optional[np.ndarray]  # (K,) vocabulary ids for ``ids``
    non_electrode: np.ndarray  # (K,) bool, never interpolated


@dataclass(frozen=True, eq=False)
class ChannelTarget:
    """Channel contract of a backbone (what it consumes).

    Parameters
    ----------
    interface : {"montage", "ids", "positions", "slots", "free"}
        ``montage``: ``(B, K, T)`` in the order of ``chs_info``. ``ids``: ``x``
        plus ids into ``vocabulary``. ``positions``: ``x`` plus ``(K, 3)``
        coordinates (onto ``chs_info`` if given, else pass-through).
        ``slots``: the first ``n_slots`` channels of ``chs_info``. ``free``:
        any ``C``, no channel identity.
    chs_info : list of dict, optional
        Training montage (required for ``montage`` and ``slots``; vocabulary
        positions for ``ids``). Any electrode kind is accepted.
    vocabulary : tuple of str, optional
        Names of the embedding table (``ids``).
    n_slots : int, optional
        Number of slots (``slots``).
    non_electrode : tuple of str
        Targets that are not electrodes (BENDR's ``SCALE``): never
        interpolated, zero unless the input carries them by name.
    """

    interface: Interface
    chs_info: Optional[list[dict]] = None
    vocabulary: Optional[tuple[str, ...]] = None
    n_slots: Optional[int] = None
    non_electrode: tuple[str, ...] = ()

    def __post_init__(self):
        if self.interface not in get_args(Interface):
            raise ValueError(
                f"Unknown channel interface {self.interface!r}; expected one of {get_args(Interface)}."
            )
        need = {"montage": "chs_info", "slots": "chs_info", "ids": "vocabulary"}.get(
            self.interface
        )
        if need is not None and getattr(self, need) is None:
            raise ValueError(f"ChannelTarget({self.interface!r}) requires {need}.")
        if self.interface == "slots" and self.n_slots is None:
            raise ValueError("ChannelTarget('slots') requires n_slots.")

    def sensors(self) -> Optional[TargetSensors]:
        """Electrodes to produce, or ``None`` for a pass-through target."""
        if self.interface == "free" or (
            self.interface == "positions" and self.chs_info is None
        ):
            return None
        ids = None
        if self.interface == "ids":
            names = tuple(self.vocabulary or ())
            pos = np.stack([standard_position(n) for n in names])
            if self.chs_info is not None:
                given = resolve_montage(self.chs_info, kinds=ELECTRODE_KINDS)
                j = match_names(names, given.names)
                pos[j >= 0] = given.positions[j[j >= 0]]
            ids = np.arange(len(names))
        else:
            chs = self.chs_info or []
            given = resolve_montage(
                chs[: self.n_slots] if self.interface == "slots" else chs,
                kinds=ELECTRODE_KINDS,
            )
            names, pos = given.names, given.positions
        non_el = {n.lower() for n in self.non_electrode}
        return TargetSensors(
            names, pos, ids, np.array([n.lower() in non_el for n in names], dtype=bool)
        )
