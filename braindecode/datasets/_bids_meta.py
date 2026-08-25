# Authors: Bruno Aristimunha <b.aristimunha@gmail.com>
#
# License: BSD (3-clause)
"""Generalized BIDS metadata -> braindecode description fields.

The emg2pose conversion (and any future BIDS-backed dataset) carries a
long tail of source-specific metadata columns (``stage``, ``side``,
``moving_hand``, ``held_out_user``, ...) that we do not want to hardcode
into braindecode. This module provides one declarative mechanism used by
both the converter (writing ``*_emg.json`` sidecars) and the dataset
classes (building per-recording ``description``):

- ``collect_fields`` maps/merges arbitrary source dicts into a flat
  description dict;
- everything unknown is carried through verbatim ("auto" mode), so new
  upstream columns appear automatically instead of being dropped.

Pure functions only: no filesystem access, trivially unit-testable.
"""

from __future__ import annotations

import json
from typing import Any

# Keys braindecode treats as first-class BIDS entities. These are always
# kept (when present) regardless of ``extra_fields``.
CORE_FIELDS = frozenset(
    {
        "subject",
        "session",
        "task",
        "run",
        "acquisition",
        "recording",
        "space",
        "suffix",
        "extension",
    }
)

EXTRA_MODES = ("auto", "mapped_only")


def _coerce(value: Any) -> Any:
    """Make an arbitrary JSON value pandas/description friendly."""
    if isinstance(value, (dict, list)):
        return json.dumps(value, separators=(",", ":"))
    return value


def collect_fields(
    *sources: dict | None,
    field_map: dict[str, str] | None = None,
    extra_fields: str | list[str] | None = "auto",
    exclude: list[str] | tuple[str, ...] = (),
) -> dict[str, Any]:
    """Merge source dicts into a description dict with generic rules.

    Parameters
    ----------
    *sources : dict | None
        Ordered source mappings (later sources win on conflicts), e.g.
        a participants row followed by a ``*_emg.json`` record.
    field_map : dict[str, str] | None
        Renames applied before anything else: source key -> description
        key. Renamed keys are always included.
    extra_fields : "auto" | list[str] | None
        Policy for unmapped keys. ``"auto"`` carries every key not
        explicitly excluded (forward compatible); a list selects exactly
        those keys; ``None`` keeps only CORE_ + renamed fields.
    exclude : list[str] | tuple[str, ...]
        Source keys never copied (e.g. bulky internal fields).

    Returns
    -------
    dict[str, Any]
        Sorted-key flat dict; container values are JSON-stringified so
        they survive pandas Series round-trips cleanly.
    """
    merged: dict[str, Any] = {}
    for src in sources:
        if src:
            merged.update(src)

    field_map = field_map or {}
    out: dict[str, Any] = {}

    # 1) renamed keys (always included when present)
    for src_key, dst_key in field_map.items():
        if src_key in merged:
            out[dst_key] = _coerce(merged[src_key])

    # 2) core entity keys (always included when present)
    for key in CORE_FIELDS:
        if key in merged:
            out[key] = _coerce(merged[key])

    # 3) policy-driven remainder
    if extra_fields is None:
        pass
    elif extra_fields == "auto":
        for key, value in merged.items():
            if key in field_map or key in exclude or key in CORE_FIELDS:
                continue
            out[key] = _coerce(value)
    elif isinstance(extra_fields, (list, tuple, set)):
        wanted = set(extra_fields)
        for key in wanted:
            if key in exclude:
                continue
            if key in merged:
                out[field_map.get(key, key)] = _coerce(merged[key])
    else:
        raise ValueError(
            f"extra_fields must be 'auto', a list of keys, or None; got {extra_fields!r}"
        )

    return {k: out[k] for k in sorted(out)}
