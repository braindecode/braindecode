# Authors: Bruno Aristimunha <b.aristimunha@gmail.com>
#
# License: BSD-3
#
# Ported from https://github.com/OpenTSLab/BrainOmni (MIT License, 2025 OpenTSLab).
# SEANet/conv/LSTM submodules derive from Meta's EnCodec (MIT License).
from __future__ import annotations

import numpy as np

# MNE FIFF constants (avoid importing mne at module load for a tiny lookup)
_FIFF_MEG_CH = 1
_FIFF_EEG_CH = 2
# sensor_type integer codes consumed by _BrainSensorModule
_SENSOR_EEG, _SENSOR_MAG, _SENSOR_GRAD = 0, 1, 2


def _resolve_kind(kind) -> int:
    """Return FIFF int kind from an MNE ``info['chs']`` ``kind`` field.

    Accepts the harness/string form (``"eeg"``, ``"mag"``, ``"grad"``) and the
    real MNE integer constants.
    """
    if isinstance(kind, str):
        k = kind.lower()
        if k == "eeg":
            return _FIFF_EEG_CH
        if k in ("mag", "grad", "meg"):
            return _FIFF_MEG_CH
        raise ValueError(f"Unsupported channel kind string: {kind!r}")
    return int(kind)


def _sensor_type_of(ch: dict) -> int:
    kind = _resolve_kind(ch.get("kind", "eeg"))
    if kind == _FIFF_EEG_CH:
        return _SENSOR_EEG
    if kind not in (_FIFF_EEG_CH, _FIFF_MEG_CH):
        raise ValueError(
            f"Unsupported channel kind {kind!r}; pass only EEG/MEG channels "
            "(e.g. raw.pick(['eeg', 'meg']))."
        )
    # MEG: distinguish MAG vs GRAD. Prefer the string kind when present,
    # else use coil_type (planar/grad coils contain "GRAD"/"PLANAR").
    raw_kind = ch.get("kind")
    if isinstance(raw_kind, str) and raw_kind.lower() == "grad":
        return _SENSOR_GRAD
    coil = str(ch.get("coil_type", ""))
    if "PLANAR" in coil or "GRAD" in coil:
        return _SENSOR_GRAD
    return _SENSOR_MAG


def _orientation_of(ch: dict, sensor_type: int) -> np.ndarray:
    """3-D orientation vector extracted from ``ch['loc']``.

    EEG: zeros (3,).
    GRAD: ``loc[3:6]`` (first direction triplet).
    MAG: ``loc[9:12]`` (third direction triplet).
    """
    if sensor_type == _SENSOR_EEG:
        return np.zeros(3, dtype=np.float64)
    loc = np.asarray(ch["loc"], dtype=np.float64)
    dir_idx = 1 if sensor_type == _SENSOR_GRAD else 3
    return loc[3 * dir_idx : 3 * (dir_idx + 1)]


def _normalize_pos(pos: np.ndarray, sensor_type: np.ndarray) -> np.ndarray:
    """Per-modality position normalization (upstream ``normalize_pos``).

    EEG positions and MEG positions are each mean-centered then divided by
    ``sqrt(3 * mean(squared_norm))``.
    """
    pos = pos.copy()
    eeg = sensor_type == _SENSOR_EEG
    meg = (sensor_type == _SENSOR_MAG) | (sensor_type == _SENSOR_GRAD)
    for mask in (eeg, meg):
        if not mask.any():
            continue
        xyz = pos[mask, :3]
        xyz = xyz - xyz.mean(axis=0, keepdims=True)
        scale = np.sqrt(3 * np.mean(np.sum(xyz**2, axis=1)))
        scale = scale if scale > 0 else 1.0
        pos[mask, :3] = xyz / scale
    return pos


def _geometry_from_chs_info(chs_info):
    """Derive ``(pos (C, 6) float32, sensor_type (C,) int64)`` from ``chs_info``.

    Raises ``ValueError`` if any channel has a missing/non-finite ``loc``.
    """
    if not chs_info:
        raise ValueError("chs_info is empty; at least one channel is required.")
    pos, stype = [], []
    for ch in chs_info:
        if "loc" not in ch:
            raise ValueError(
                f"Channel {ch.get('ch_name', '?')!r} has no 'loc'. "
                "BrainOmni needs sensor positions; call raw.set_montage(...)."
            )
        loc = np.asarray(ch["loc"], dtype=np.float64)
        if loc.shape != (12,):
            raise ValueError(
                f"Channel {ch.get('ch_name', '?')!r} has loc of shape "
                f"{loc.shape}; expected (12,)."
            )
        s = _sensor_type_of(ch)
        ori = _orientation_of(ch, s)
        vec = np.concatenate([loc[:3], ori])
        if not np.all(np.isfinite(vec)):
            raise ValueError(
                f"Channel {ch.get('ch_name', '?')!r} has non-finite position. "
                "BrainOmni needs sensor positions; call raw.set_montage(...)."
            )
        pos.append(vec)
        stype.append(s)
    pos = np.stack(pos).astype(np.float32)
    sensor_type = np.asarray(stype, dtype=np.int64)
    pos = _normalize_pos(pos, sensor_type).astype(np.float32)
    return pos, sensor_type
