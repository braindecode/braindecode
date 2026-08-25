# Authors: Bruno Aristimunha <b.aristimunha@gmail.com>
#
# License: BSD (3-clause)
"""emg2pose benchmark as a BIDS dataset.

Wraps a BIDS tree produced from Meta's emg2pose release
(:footcite:`Salter2024`): 16-channel 2 kHz wrist sEMG plus 20 joint
angles per hand/stage, converted to BrainVision/EDF by
``scripts/export_emg2pose_bids.py``. Recordings pair with an optional
``*_desc-pose.json`` skeleton sidecar that the embedded eegdash-viewer
renders as a synchronized hand panel (:ref:`docs/pose-sidecar` upstream).
"""

from __future__ import annotations

import json
import re
from glob import glob
from pathlib import Path

import mne
import pandas as pd

from ._bids_meta import collect_fields
from ._viewer_server import build_iframe_html, embed_html, get_viewer_server
from .base import BaseConcatDataset, RawDataset
from .registry import register_dataset

_ENTITY_RE = re.compile(r"(?:^|_)(sub|ses|task|run|acq|rec|space)-([^_]+)")

# BIDS short entity -> braindecode/core field name
_ENTITY_TO_FIELD = {
    "sub": "subject",
    "ses": "session",
    "task": "task",
    "run": "run",
    "acq": "acquisition",
    "rec": "recording",
    "space": "space",
}


def _parse_entities(path: Path) -> dict[str, str]:
    return {
        _ENTITY_TO_FIELD[key]: value for key, value in _ENTITY_RE.findall(path.stem)
    }


_READERS = {
    ".vhdr": mne.io.read_raw_brainvision,
    ".edf": mne.io.read_raw_edf,
    ".bdf": mne.io.read_raw_bdf,
    ".set": mne.io.read_raw_eeglab,
    ".fif": mne.io.read_raw_fif,
}


def _load_participants(root: Path) -> dict[str, dict]:
    """Map normalized subject id -> participant-level fields."""
    tsv = root / "participants.tsv"
    if not tsv.is_file():
        return {}
    table = pd.read_csv(tsv, sep="\t", dtype=str).fillna("")
    rows = {}
    for _, row in table.iterrows():
        pid = str(row.get("participant_id", ""))
        subject = pid.removeprefix("sub-")
        rows[subject] = {k: v for k, v in row.items() if k != "participant_id"}
    return rows


@register_dataset
class EMG2Pose(BaseConcatDataset):
    """emg2pose sEMG hand-pose benchmark, BIDS layout.

    Each HDF5 stage file of the original release becomes one BIDS
    recording (16 ``emg`` channels + 20 ``misc`` joint-angle channels at
    2000 Hz). Metadata columns of the source release (``stage``,
    ``side``, ``moving_hand``, ``held_out_user``, ``split``, ...) live
    in the ``*_emg.json`` sidecars and flow into each dataset's
    ``description`` through the generic field mechanism in
    :mod:`braindecode.datasets._bids_meta`: known entities always map,
    everything else is carried verbatim unless restricted.

    Parameters
    ----------
    root : pathlib.Path | str
        Root of the converted BIDS dataset.
    subjects : list[str] | None
        Restrict to these subject ids (without ``sub-`` prefix).
    filters : dict | None
        Exact-match filters applied to merged metadata, e.g.
        ``{"split": "train", "side": "left"}``.
    field_map : dict[str, str] | None
        Optional renames forwarded to :func:`collect_fields`.
    extra_fields : "auto" | list[str] | None
        Forwarded to :func:`collect_fields`; ``"auto"`` keeps every
        sidecar/participants column (default).
    exclude_fields : list[str]
        Sidecar keys never copied into descriptions.

    Notes
    -----
    Conversion/download is intentionally out of scope here; run
    ``scripts/export_emg2pose_bids.py`` against the official release.

    References
    ----------
    .. footbibliography::
    """

    def __init__(
        self,
        root: str | Path,
        subjects: list[str] | None = None,
        filters: dict | None = None,
        field_map: dict[str, str] | None = None,
        extra_fields="auto",
        exclude_fields: list[str] | tuple[str, ...] = (),
        target_name=None,
        transform=None,
    ):
        self.root = Path(root).resolve()
        # Only header files a reader can open: skips BrainVision's .eeg
        # data / .vmrk marker siblings that share the *_emg.* pattern.
        files = [
            f
            for f in sorted(
                glob(str(self.root / "sub-*" / "**" / "*_emg.*"), recursive=True)
            )
            if Path(f).suffix.lower() in _READERS
        ]
        if not files:
            raise ValueError(
                f"No *_emg.<ext> recordings found under {self.root}. "
                "Convert the emg2pose release first."
            )

        participants = _load_participants(self.root)
        records = []
        datasets = []
        for fname in files:
            path = Path(fname)
            subject = _parse_entities(path).get("subject", "")
            if subjects is not None and subject not in subjects:
                continue
            record_json = {}
            sidecar = path.with_suffix(".json")
            if sidecar.is_file():
                record_json = json.loads(sidecar.read_text())
            description = collect_fields(
                _parse_entities(path),
                participants.get(subject),
                record_json,
                field_map=field_map,
                extra_fields=extra_fields,
                exclude=exclude_fields,
            )
            description["path"] = str(path.relative_to(self.root))
            records.append(description)

            reader = _READERS[path.suffix.lower()]
            raw = reader(path, preload=False, verbose="ERROR")
            datasets.append(
                RawDataset(
                    raw,
                    description=description,
                    target_name=target_name,
                    transform=transform,
                )
            )

        if filters:

            def keep(rec: dict) -> bool:
                return all(rec.get(k) == v for k, v in filters.items())

            pairs = [(d, r) for d, r in zip(datasets, records) if keep(r)]
            if not pairs:
                raise ValueError(f"Filters {filters} matched no recordings.")
            datasets, records = (list(x) for x in zip(*pairs))

        self.records = pd.DataFrame(records)
        super().__init__(datasets)

    # -- visualization -----------------------------------------------------

    def _pose_sidecar_rel(self, recording_rel: str) -> str | None:
        rec_path = Path(recording_rel)
        stem = rec_path.stem.rsplit("_", 1)[0]
        candidate = rec_path.with_name(f"{stem}_desc-pose.json")
        # recording_rel is root-relative; existence must be checked
        # against the dataset root, not the process CWD.
        if (self.root / candidate).is_file():
            return str(candidate)
        return None

    def plot(
        self,
        index: int = 0,
        height: int = 420,
        viewer_url: str | None = None,
        data_url: str | None = None,
    ):
        """Open the embedded eegdash-viewer for one recording.

        Returns an :class:`IPython.display.HTML` holding an iframe bound
        to a localhost server for this dataset (kernel and browser must
        share a host). When a ``*_desc-pose.json`` skeleton sidecar sits
        next to the recording it is passed as ``&pose=``, enabling the
        synchronized hand panel (toggle with ``p``).

        Parameters
        ----------
        index : int
            Recording to display (row of :attr:`records`).
        height : int
            Iframe height in pixels.
        viewer_url : str | None
            Base URL of a *hosted* eegdash-viewer deployment (e.g.
            ``https://viewer.eegdash.org``). Skips serving the vendored
            assets locally.
        data_url : str | None
            Web-reachable base for this dataset's BIDS root. When both
            ``viewer_url`` and ``data_url`` are given, no kernel-side
            server is started at all — useful on remote kernels/Colab
            where localhost iframes cannot reach the kernel host.
        """
        from IPython.display import HTML

        rec_rel = self.records.iloc[index]["path"]
        pose_rel = self._pose_sidecar_rel(rec_rel)

        if viewer_url is not None or data_url is not None:
            # Mixed/hosted mode. The data base defaults to the local
            # server's /data tree; the viewer base defaults to it too,
            # so any single override composes with the other half.
            server = get_viewer_server(self.root)
            data_base = data_url.rstrip("/") if data_url else f"{server.base}/data"
            viewer_base = (
                viewer_url.rstrip("/") if viewer_url else f"{server.base}/viewer"
            )
            pose_url = None if pose_rel is None else f"{data_base}/{pose_rel}"
            html = embed_html(
                viewer_base, f"{data_base}/{rec_rel}", pose_url, height=height
            )
            return HTML(html)

        server = get_viewer_server(self.root)
        html = build_iframe_html(server.base, rec_rel, pose_rel, height=height)
        return HTML(html)
