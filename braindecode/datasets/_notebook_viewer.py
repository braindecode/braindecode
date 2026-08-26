# Authors: Bruno Aristimunha <b.aristimunha@gmail.com>
#
# License: BSD (3-clause)
"""Serverless in-notebook viewer for file-backed recordings.

The recording bytes are inlined in the cell output as base64 (the
"papaya pattern") and pushed into the deployed eegdash-viewer through
its ``postMessage`` host bridge (``docs/embedding.md`` in
https://github.com/eegdash/eegdash-viewer). No local server, no ports,
no CORS. The output is an iframe driven by an inline script: it renders
when the cell ran in your session or the saved notebook is trusted
(``jupyter trust``); GitHub previews and untrusted notebooks show an
empty cell. What is shown are the bytes on disk behind a dataset element
(a preprocessed dataset must be saved first).
"""

from __future__ import annotations

import base64
import json
import os
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import mne_bids
from mne.utils import _soft_import

CDN = "https://eegdash.github.io/eegdash-viewer"
MAX_BYTES = 64 * 2**20  # base64 output per call; it is saved with the notebook
HEIGHT = 520
# Formats the viewer reads from in-memory files (single file or small siblings).
EXTENSIONS = frozenset({".set", ".edf", ".bdf", ".vhdr", ".fif", ".snirf", ".nwb"})
_BIDS_SUFFIXES = frozenset({"eeg", "ieeg", "emg", "meg", "nirs"})
_SIBLINGS = {".vhdr": (".eeg", ".vmrk"), ".set": (".fdt",)}  # travel with the header
_HEADER_OF = {
    ".eeg": ".vhdr",
    ".fdt": ".set",
}  # mne names the data file, the viewer wants the header

_SCRIPT = """
<script>
(function () {
  // The iframe is normally the element right before this script (JupyterLab
  // re-runs scripts in place, so a duplicated output still finds its own);
  // VS Code / nbclassic run the script elsewhere, hence the id fallback.
  var self = document.currentScript, id = %(id)s;
  var frame = (self && self.previousElementSibling && self.previousElementSibling.tagName === "IFRAME")
    ? self.previousElementSibling : document.getElementById(id);
  if (!frame) { console.error("eegdash viewer: output iframe " + id + " not found"); return; }
  var payload = %(payload)s;
  var origin = %(origin)s;
  var files = null, pose = null;
  function decode(b64) {
    if (Uint8Array.fromBase64) return Uint8Array.fromBase64(b64);
    var bin = atob(b64), out = new Uint8Array(bin.length);
    for (var i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
    return out;
  }
  function send(target) {
    try {
      if (!files) {
        files = payload.files.map(function (f) { return new File([decode(f.b64)], f.name); });
        pose = payload.pose ? "data:application/json;base64," + payload.pose : null;
        payload = null;   // the File objects carry the bytes from here on
      }
      frame.contentWindow.postMessage({ type: "eegdash-viewer:open", files: files, pose: pose }, target || origin);
    } catch (err) {
      var note = document.createElement("div");
      note.style.cssText = "font:12px system-ui,sans-serif;color:#b3261e;padding:4px 0";
      note.textContent = "eegdash viewer: could not hand over the recording (" + err.message + ")";
      frame.insertAdjacentElement("afterend", note);
    }
  }
  window.addEventListener("message", function onMessage(e) {
    if (e.source === frame.contentWindow && e.data && e.data.type === "eegdash-viewer:ready") {
      send(e.origin);   // every ready: the viewer re-announces after a reload
    } else if (!frame.isConnected) {
      window.removeEventListener("message", onMessage);   // output gone: release the bytes
    }
  });
  frame.src = %(src)s;   // after the listener: the viewer's "ready" can never precede it
})();
</script>"""


def viewer_name(recording: Path) -> str:
    """File name to hand the viewer: BIDS names as-is, others as ``<stem>_eeg<ext>``.

    The viewer picks the recording by its ``*_<datatype>.<ext>`` name, so a
    braindecode-saved ``0-raw.fif`` or a plain ``session1.edf`` is posted as
    ``0-raw_eeg.fif`` / ``session1_eeg.edf`` (the name only selects the
    reader; channel types still come from the file).
    """
    rec = Path(recording)
    token = rec.stem.rsplit("_", 1)[-1]
    return (
        rec.name if token in _BIDS_SUFFIXES else f"{rec.stem}_eeg{rec.suffix.lower()}"
    )


def _posted_name(path: Path, recording: Path) -> str:
    """Name a payload entry: the header via :func:`viewer_name`, an EEGLAB ``.fdt``
    next to the posted ``.set`` name (the viewer probes ``<prefix>_eeg.fdt``),
    everything else (BrainVision siblings referenced from the header, sidecars) as-is."""
    if path == recording:
        return viewer_name(recording)
    if path.suffix.lower() == ".fdt":
        return viewer_name(recording).rsplit("_", 1)[0] + "_eeg.fdt"
    return path.name


def check_viewable(recording: Path) -> None:
    """Raise ``ValueError`` when the viewer cannot open ``recording`` from memory."""
    rec = Path(recording)
    if not rec.exists():
        raise ValueError(
            f"{rec.name}: file not found (a git-annex/datalad symlink may need `datalad get`)"
        )
    if rec.is_dir():
        raise ValueError(f"{rec.name}: directory-based recordings cannot be inlined")
    if rec.suffix.lower() not in EXTENSIONS or rec.stem.endswith("_epo"):
        raise ValueError(
            f"{rec.name}: the viewer opens raw recordings in "
            f"{' '.join(sorted(EXTENSIONS))} from memory"
        )


def recording_files(recording: Path) -> tuple[Path, tuple[Path, ...], Path | None]:
    """``(recording, sidecars, pose)`` for a recording file.

    A BIDS-named file gets its channels/events sidecars by BIDS inheritance
    (``mne_bids``); other names, oddly named neighbours or unparsable trees
    get none. The hand-pose sidecar (``<prefix>_desc-pose.json``) sits next
    to the recording. The path is not resolved, so git-annex/datalad
    symlinks keep their BIDS name.
    """
    rec = Path(recording)
    sidecars: tuple[Path, ...] = ()
    try:
        bids_path = mne_bids.get_bids_path_from_fname(rec, check=False)
        if bids_path.subject is not None:  # hyphen-free names parse, but are not BIDS
            sidecars = tuple(
                p
                for suffix in ("channels", "events")
                if (
                    p := bids_path.find_matching_sidecar(
                        suffix=suffix, extension=".tsv", on_error="ignore"
                    )
                )
            )
    except (
        KeyError,
        ValueError,
    ):  # not a BIDS name (0-raw.fif) / unknown entity in the tree
        pass
    pose = rec.with_name(rec.stem.rsplit("_", 1)[0] + "_desc-pose.json")
    return rec, sidecars, pose if pose.is_file() else None


def collect_files(recording: Path, sidecars: Sequence[Path] = ()) -> list[Path]:
    """Recording first, then split-format siblings and the given sidecars that exist."""
    rec = Path(recording)
    candidates = [rec.with_suffix(ext) for ext in _SIBLINGS.get(rec.suffix.lower(), ())]
    candidates += [Path(p) for p in sidecars]
    out = [rec]
    for p in candidates:
        if p.is_symlink() and not p.exists():
            raise ValueError(
                f"{p.name}: dangling symlink (a datalad tree may need `datalad get`)"
            )
        if p.is_file() and p not in out:
            out.append(p)
    return out


def _check_cdn(cdn: str) -> tuple[str, str]:
    """Validate the viewer base URL; return (base, origin)."""
    parts = urlsplit(cdn)
    if parts.scheme not in ("http", "https") or not parts.netloc:
        raise ValueError(f"cdn must be an absolute http(s) URL, got {cdn!r}")
    if (
        parts.query
        or parts.fragment
        or parts.path.endswith(("index.html", "index.htm"))
    ):
        raise ValueError(
            f"cdn must be the viewer's base URL (no query, fragment or index.html), got {cdn!r}"
        )
    # geturl() drops empty `?`/`#` delimiters left in a pasted URL.
    return parts.geturl().rstrip("/"), f"{parts.scheme}://{parts.netloc}"


def build_viewer_html(
    recording: Path,
    pose_sidecar: Path | None = None,
    *,
    sidecars: Sequence[Path] = (),
    height: int = HEIGHT,
    cdn: str = CDN,
    max_bytes: int = MAX_BYTES,
) -> str:
    """HTML for one recording: viewer iframe + inlined bytes + bridge glue.

    ``max_bytes`` bounds the base64 payload (what the notebook file grows
    by); everything is checked before any bytes are read.
    """
    base, origin = _check_cdn(cdn)
    check_viewable(recording)
    rec = Path(recording)
    files = collect_files(rec, sidecars)
    inlined = files + ([Path(pose_sidecar)] if pose_sidecar is not None else [])
    encoded = sum(4 * -(-p.stat().st_size // 3) for p in inlined)
    if encoded > max_bytes:
        raise ValueError(
            f"{rec.name}: {encoded / 2**20:.1f} MiB of base64 would be inlined into "
            f"the notebook output (max_bytes={max_bytes / 2**20:.1f} MiB). Crop or downsample "
            "and export a smaller file, or pass a larger max_bytes."
        )
    payload = {
        "files": [{"name": _posted_name(p, rec), "b64": _b64(p)} for p in files],
        "pose": _b64(pose_sidecar) if pose_sidecar is not None else None,
    }
    uid = f"eegdash-viewer-{uuid.uuid4().hex[:8]}"
    return "".join(
        (
            f'<iframe id="{uid}" title="eegdash trace viewer" style="width:100%;height:{int(height)}px;'
            'border:1px solid var(--jp-border-color1,#d9dce1);border-radius:6px;background:transparent"></iframe>',
            _SCRIPT
            % {
                "id": _js(uid),
                "payload": _js(payload),
                "origin": _js(origin),
                "src": _js(f"{base}/index.html?embed=1"),
            },
        )
    )


def _js(value: Any) -> str:
    """A JSON literal safe inside an inline <script> (`<` can never open a tag)."""
    return json.dumps(value).replace("<", "\\u003c")


def _b64(path: Path) -> str:
    return base64.b64encode(Path(path).read_bytes()).decode()


class ViewerMixin:
    """``plot()`` for every :class:`~braindecode.datasets.BaseConcatDataset`.

    :meth:`_viewer_recording` names the file behind ``self.datasets[index]``:
    the file its mne ``raw`` reads from (accessing ``raw`` is where lazily
    downloading datasets, e.g. eegdash, fetch it; a data file maps back to
    its header) or, when the raw carries no file, the element's recorded
    ``description["path"]``. Datasets with another notion of "the recording
    file" override the hook and return :func:`recording_files`.
    """

    datasets: Sequence[Any]  # provided by the concat dataset

    def _viewer_recording(
        self, index: int
    ) -> tuple[Path, tuple[Path, ...], Path | None]:
        ds = self.datasets[index]
        raw = getattr(ds, "raw", None)
        names = [
            Path(f)
            for f in getattr(raw, "filenames", None) or ()
            if isinstance(f, (str, os.PathLike))
        ]
        if len(names) > 1:
            raise ValueError(
                f"{type(self).__name__}[{index}] is a split recording ({len(names)} files); "
                "the viewer reads single-file recordings"
            )
        path = names[0] if names else None
        if path is None:
            desc = getattr(ds, "description", None)  # dict or pandas Series
            recorded = desc.get("path") if desc is not None else None
            path = Path(recorded) if isinstance(recorded, (str, os.PathLike)) else None
        if path is None:
            raise ValueError(
                f"{type(self).__name__}[{index}] is not backed by a recording file; "
                "nothing to show in the viewer"
            )
        header = path.with_suffix(_HEADER_OF.get(path.suffix.lower(), path.suffix))
        return recording_files(header if header.is_file() else path)

    def plot(
        self,
        index: int = 0,
        *,
        height: int = HEIGHT,
        cdn_url: str = CDN,
        max_bytes: int = MAX_BYTES,
    ):
        """Show one recording in the eegdash-viewer inside a Jupyter cell.

        Serverless: the recording bytes (as on disk) are inlined in the output
        and pushed into the viewer (loaded from ``cdn_url``) over
        ``postMessage``; it renders when the cell ran in your session or the
        saved notebook is trusted. A ``*_desc-pose.json`` sidecar next to the
        recording adds the synchronized hand-pose panel. Needs IPython (soft
        dependency). See
        https://github.com/eegdash/eegdash-viewer/blob/main/docs/embedding.md.

        Parameters
        ----------
        index : int
            Recording to display.
        height : int
            Viewer height in pixels.
        cdn_url : str
            Base URL of a deployed eegdash-viewer.
        max_bytes : int
            Refuse to inline more than this much base64 (default 64 MiB); the
            payload is saved with the notebook and, like any cell output,
            stays referenced by IPython's ``Out`` history for the session.

        Returns
        -------
        IPython.display.HTML
        """
        ipython = _soft_import("IPython", purpose=f"{type(self).__name__}.plot()")
        recording, sidecars, pose = self._viewer_recording(index)
        out = (
            ipython.display.HTML()
        )  # data assigned after: HTML(data) stats the whole string
        out.data = build_viewer_html(
            recording,
            pose,
            sidecars=sidecars,
            height=height,
            cdn=cdn_url,
            max_bytes=max_bytes,
        )
        return out
