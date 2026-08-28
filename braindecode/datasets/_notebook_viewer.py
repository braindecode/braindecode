# Authors: Bruno Aristimunha <b.aristimunha@gmail.com>
#
# License: BSD (3-clause)
"""Serverless in-notebook viewer for file-backed recordings.

The bytes on disk behind a dataset element are inlined in the cell output
as base64 and handed to the deployed eegdash-viewer over its ``postMessage``
bridge (``docs/embedding.md`` in https://github.com/eegdash/eegdash-viewer):
no server, no CORS. The output is an iframe plus an inline script, so it
renders when the cell ran in your session or the saved notebook is trusted
(``jupyter trust``).
"""

from __future__ import annotations

import base64
import json
import os
import uuid
from pathlib import Path
from urllib.parse import urlsplit

import mne_bids
from mne.utils import _soft_import

CDN = "https://eegdash.github.io/eegdash-viewer"
MAX_BYTES = 64 * 2**20  # base64 output per call; it is saved with the notebook
EXTENSIONS = {
    ".set",
    ".edf",
    ".bdf",
    ".vhdr",
    ".fif",
    ".snirf",
    ".nwb",
}  # viewer readers
_BIDS = {"eeg", "ieeg", "emg", "meg", "nirs"}
_SIBLINGS = {".vhdr": (".eeg", ".vmrk"), ".set": (".fdt",)}  # travel with the header
_HEADER = {s: h for h, ss in _SIBLINGS.items() for s in ss}  # data file -> header

_SCRIPT = """<iframe id=%(id)s title="eegdash trace viewer" style="width:100%%;height:%(height)spx;
border:1px solid var(--jp-border-color1,#d9dce1);border-radius:6px;background:transparent"></iframe>
<script>
(function () {
  var self = document.currentScript, id = %(id)s;   // Lab re-runs scripts in place; VS Code/nbclassic elsewhere
  var frame = (self && self.previousElementSibling && self.previousElementSibling.tagName === "IFRAME")
    ? self.previousElementSibling : document.getElementById(id);
  if (!frame) { console.error("eegdash viewer: output iframe " + id + " not found"); return; }
  var payload = %(payload)s, origin = %(origin)s, files = null, pose = null;
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
        payload = null;
      }
      frame.contentWindow.postMessage({ type: "eegdash-viewer:open", files: files, pose: pose }, target || origin);
    } catch (err) {
      frame.insertAdjacentHTML("afterend", '<div style="font:12px system-ui;color:#b3261e">eegdash viewer: '
        + String(err.message).replace(/</g, "&lt;") + "</div>");
    }
  }
  window.addEventListener("message", function onMessage(e) {
    if (e.source === frame.contentWindow && e.data && e.data.type === "eegdash-viewer:ready") send(e.origin);
    else if (!frame.isConnected) window.removeEventListener("message", onMessage);
  });
  frame.src = %(src)s;   // after the listener, so "ready" can never precede it
})();
</script>"""


def recording_files(recording: Path) -> tuple[list[Path], Path | None]:
    """``(files, pose)`` to inline for a recording: the header first, then the
    split-format siblings, the BIDS-inherited ``_channels.tsv``/``_events.tsv``
    (``mne_bids`` parses the name; plain names get none) and, separately, the
    ``<prefix>_desc-pose.json`` hand-pose sidecar next to it. Symlinks keep
    their name (git-annex/datalad)."""
    rec = Path(recording)
    if not rec.exists():
        raise ValueError(
            f"{rec.name}: file not found (a datalad symlink may need `datalad get`)"
        )
    if (
        rec.is_dir()
        or rec.suffix.lower() not in EXTENSIONS
        or rec.stem.endswith("_epo")
    ):
        raise ValueError(
            f"{rec.name}: the viewer opens raw recordings in {' '.join(sorted(EXTENSIONS))}"
        )
    sidecars: list[Path | None] = []
    try:
        bids = mne_bids.get_bids_path_from_fname(rec, check=False)
        if bids.subject is not None:  # hyphen-free names parse, but are not BIDS
            sidecars = [
                bids.find_matching_sidecar(
                    suffix=s, extension=".tsv", on_error="ignore"
                )
                for s in ("channels", "events")
            ]
    except (
        KeyError,
        ValueError,
    ):  # not a BIDS name / unknown entity somewhere in the tree
        pass
    files = [rec]
    for p in [
        rec.with_suffix(e) for e in _SIBLINGS.get(rec.suffix.lower(), ())
    ] + sidecars:
        if p and p.is_file():
            if p not in files:
                files.append(p)
        elif p and p.is_symlink():  # dangling: git-annex/datalad content not fetched
            raise ValueError(f"{p.name}: dangling symlink (try `datalad get`)")
    stem, _, token = rec.stem.rpartition("_")
    pose = rec.with_name((stem if token in _BIDS else rec.stem) + "_desc-pose.json")
    return files, pose if pose.is_file() else None


def build_viewer_html(
    recording: Path,
    *,
    height: int = 520,
    cdn_url: str = CDN,
    max_bytes: int = MAX_BYTES,
) -> str:
    """Viewer iframe + inlined bytes + bridge script for one recording."""
    url = urlsplit(cdn_url)
    if (
        url.scheme not in ("http", "https")
        or not url.netloc
        or url.username
        or url.query
        or url.fragment
        or url.path.endswith(("index.html", "index.htm"))
    ):
        raise ValueError(
            f"cdn_url must be the viewer's base http(s) URL, got {cdn_url!r}"
        )
    files, pose = recording_files(recording)
    encoded = sum(
        4 * -(-p.stat().st_size // 3) for p in files + ([pose] if pose else [])
    )
    if encoded > max_bytes:
        raise ValueError(
            f"{files[0].name}: {encoded / 2**20:.1f} MiB of base64 would be inlined into the "
            f"notebook output (max_bytes={max_bytes / 2**20:.1f} MiB); crop/downsample or raise it"
        )
    rec = files[0]
    # The viewer picks the recording by its *_<datatype>.<ext> name: plain names
    # are posted as <stem>_eeg<ext>, an EEGLAB .fdt next to the posted .set name.
    head = (
        rec.name
        if rec.stem.rpartition("_")[2] in _BIDS
        else f"{rec.stem}_eeg{rec.suffix.lower()}"
    )
    names = [head] + [
        head.rsplit("_", 1)[0] + "_eeg.fdt" if p.suffix.lower() == ".fdt" else p.name
        for p in files[1:]
    ]
    b64 = [base64.b64encode(p.read_bytes()).decode() for p in files]
    literals = {
        "id": f"eegdash-viewer-{uuid.uuid4().hex[:8]}",
        "height": int(height),
        "payload": {
            "files": [{"name": n, "b64": b} for n, b in zip(names, b64)],
            "pose": base64.b64encode(pose.read_bytes()).decode() if pose else None,
        },
        "origin": f"{url.scheme}://{url.netloc}",
        "src": f"{url.geturl().rstrip('/')}/index.html?embed=1",
    }
    return _SCRIPT % {
        k: json.dumps(v).replace("<", "\\u003c") for k, v in literals.items()
    }


def _recording(dataset, index: int) -> Path:
    """File behind ``dataset.datasets[index]``: the one its mne ``raw`` reads (a
    data file maps back to its header; lazily downloading datasets fetch it when
    ``raw`` is accessed) or the recorded ``description["path"]``."""
    ds = dataset.datasets[index]
    names = [
        Path(f)
        for f in getattr(getattr(ds, "raw", None), "filenames", None) or ()
        if isinstance(f, (str, os.PathLike))
    ]
    if len(names) > 1:
        raise ValueError(
            f"{type(dataset).__name__}[{index}]: split recordings are not supported"
        )
    desc = getattr(ds, "description", None)  # dict or pandas Series
    recorded = desc.get("path") if desc is not None else None
    path = (
        names[0]
        if names
        else Path(recorded)
        if isinstance(recorded, (str, os.PathLike))
        else None
    )
    if path is None:
        raise ValueError(
            f"{type(dataset).__name__}[{index}] is not backed by a recording file"
        )
    header = path.with_suffix(_HEADER.get(path.suffix.lower(), path.suffix))
    return header if header.is_file() else path


def plot(
    dataset,
    index: int = 0,
    *,
    height: int = 520,
    cdn_url: str = CDN,
    max_bytes: int = MAX_BYTES,
):
    """Show one recording in the eegdash-viewer inside a Jupyter cell.

    Serverless: the recording bytes (as on disk) are inlined in the output and
    pushed to the viewer at ``cdn_url`` over ``postMessage``; the output renders
    when the cell ran in your session or the notebook is trusted. A
    ``*_desc-pose.json`` sidecar next to the recording adds the synchronized
    hand-pose panel. Needs IPython (soft dependency).

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
        payload is saved with the notebook and, like any cell output, stays
        referenced by IPython's ``Out`` history for the session.

    Returns
    -------
    IPython.display.HTML
    """
    ipython = _soft_import("IPython", purpose=f"{type(dataset).__name__}.plot()")
    return ipython.display.HTML(
        build_viewer_html(
            _recording(dataset, index),
            height=height,
            cdn_url=cdn_url,
            max_bytes=max_bytes,
        )
    )
