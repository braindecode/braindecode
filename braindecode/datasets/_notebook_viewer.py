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
empty cell.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from urllib.parse import urlsplit

import mne_bids
from mne.utils import _soft_import

CDN = "https://eegdash.github.io/eegdash-viewer"
MAX_BYTES = 64 * 2**20  # base64 output per call; it is saved with the notebook
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
(function () {{
  var frame = document.currentScript.previousElementSibling;
  var payload = {payload};
  var files = null;
  function decode(b64) {{
    if (Uint8Array.fromBase64) return Uint8Array.fromBase64(b64);
    var bin = atob(b64), out = new Uint8Array(bin.length);
    for (var i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
    return out;
  }}
  function send() {{
    try {{
      if (!files) {{
        files = payload.files.map(function (f) {{ return new File([decode(f.b64)], f.name); }});
        files.pose = payload.pose ? "data:application/json;base64," + payload.pose : null;
        payload = null;   // the File objects carry the bytes from here on
      }}
      frame.contentWindow.postMessage({{ type: "eegdash-viewer:open", files: files, pose: files.pose }}, {origin});
    }} catch (err) {{
      var note = document.createElement("div");
      note.style.cssText = "font:12px system-ui,sans-serif;color:#b3261e;padding:4px 0";
      note.textContent = "eegdash viewer: could not hand over the recording (" + err.message + ")";
      frame.insertAdjacentElement("afterend", note);
    }}
  }}
  window.addEventListener("message", function onMessage(e) {{
    if (!frame.isConnected) {{ window.removeEventListener("message", onMessage); return; }}
    if (e.source === frame.contentWindow && e.data && e.data.type === "eegdash-viewer:ready") send();
  }});
  frame.src = {src};   // after the listener: the viewer's "ready" can never precede it
}})();
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


def check_viewable(recording: Path) -> None:
    """Raise ``ValueError`` when the viewer cannot open ``recording`` from memory."""
    rec = Path(recording)
    if rec.is_dir():
        raise ValueError(f"{rec.name}: directory-based recordings cannot be inlined")
    if rec.suffix.lower() not in EXTENSIONS or rec.stem.endswith("_epo"):
        raise ValueError(
            f"{rec.name}: the viewer opens raw recordings in "
            f"{' '.join(sorted(EXTENSIONS))} from memory"
        )


def recording_files(recording: Path) -> tuple[Path, tuple[Path, ...], Path | None]:
    """``(recording, sidecars, pose)`` for a recording file.

    A BIDS-named file (any ``mne_bids.BIDSPath``-parsable name) gets its
    channels/events sidecars by BIDS inheritance; the hand-pose sidecar
    (``<prefix>_desc-pose.json``) sits next to the recording. The path is
    not resolved, so git-annex/datalad symlinks keep their BIDS name.
    """
    rec = Path(recording)
    try:
        bids_path = mne_bids.get_bids_path_from_fname(rec, check=False)
    except (KeyError, ValueError):  # not a BIDS name (0-raw.fif, SC4001E0-PSG.edf)
        bids_path = None
    sidecars = tuple(
        p
        for suffix in ("channels", "events")
        if bids_path
        and (
            p := bids_path.find_matching_sidecar(
                suffix=suffix, extension=".tsv", on_error="ignore"
            )
        )
    )
    pose = rec.with_name(rec.stem.rsplit("_", 1)[0] + "_desc-pose.json")
    return rec, sidecars, pose if pose.is_file() else None


def collect_files(recording: Path, sidecars: tuple[Path, ...] = ()) -> list[Path]:
    """Recording first, then split-format siblings and the given sidecars that exist."""
    rec = Path(recording)
    candidates = [rec.with_suffix(ext) for ext in _SIBLINGS.get(rec.suffix.lower(), ())]
    candidates += [Path(p) for p in sidecars]
    out = [rec]
    for p in candidates:
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
    sidecars: tuple[Path, ...] = (),
    height: int = 520,
    cdn: str = CDN,
    max_bytes: int = MAX_BYTES,
) -> str:
    """HTML for one recording: viewer iframe + inlined bytes + bridge glue.

    ``max_bytes`` bounds the base64 payload (what the notebook file grows
    by); everything is checked before any bytes are read.
    """
    base, origin = _check_cdn(cdn)
    check_viewable(recording)
    files = collect_files(recording, sidecars)
    inlined = files + ([Path(pose_sidecar)] if pose_sidecar is not None else [])
    encoded = sum(4 * -(-p.stat().st_size // 3) for p in inlined)
    if encoded > max_bytes:
        raise ValueError(
            f"{Path(recording).name}: {encoded / 2**20:.1f} MiB of base64 would be inlined into "
            f"the notebook output (max_bytes={max_bytes / 2**20:.1f} MiB). Crop or downsample "
            "and export a smaller file, or pass a larger max_bytes."
        )
    # Splice the base64 straight into the JSON text: only the names need
    # escaping and the large strings are never copied through json.dumps.
    entries = ",".join(
        f'{{"name":{_js(viewer_name(p) if p == files[0] else p.name)},"b64":"{_b64(p)}"}}'
        for p in files
    )
    pose = f'"{_b64(pose_sidecar)}"' if pose_sidecar is not None else "null"
    return (
        f'<iframe title="eegdash trace viewer" style="width:100%;height:{int(height)}px;'
        'border:1px solid var(--jp-border-color1,#d9dce1);border-radius:6px;background:transparent"></iframe>'
        + _SCRIPT.format(
            payload=f'{{"files":[{entries}],"pose":{pose}}}',
            origin=_js(origin),
            src=_js(f"{base}/index.html?embed=1"),
        )
    )


def _js(value: str) -> str:
    """A JSON string literal safe inside an inline <script>."""
    return json.dumps(value).replace("<", "\\u003c")


def _b64(path: Path) -> str:
    return base64.b64encode(Path(path).read_bytes()).decode()


class ViewerMixin:
    """``plot()`` for every :class:`~braindecode.datasets.BaseConcatDataset`.

    :meth:`_viewer_recording` names the file behind ``self.datasets[index]``:
    ``description["path"]`` when the element records it (``BIDSDataset``),
    else the file behind its mne ``raw`` (accessing ``raw`` is what lazily
    downloading datasets, e.g. eegdash, hook into); split-format data files
    map back to their header. Datasets with another notion of "the
    recording file" override the hook and return :func:`recording_files`.
    """

    def _viewer_recording(
        self, index: int
    ) -> tuple[Path, tuple[Path, ...], Path | None]:
        ds = self.datasets[index]
        desc = getattr(ds, "description", None)  # dict or pandas Series
        path = desc.get("path") if desc is not None else None
        if path is None:
            names = [
                f
                for f in getattr(getattr(ds, "raw", None), "filenames", None) or ()
                if f
            ]
            if not names:
                raise ValueError(
                    f"{type(self).__name__}[{index}] is not backed by a recording file; "
                    "nothing to show in the viewer"
                )
            path = Path(names[0])
            path = path.with_suffix(_HEADER_OF.get(path.suffix.lower(), path.suffix))
        return recording_files(Path(path))

    def plot(
        self,
        index: int = 0,
        *,
        height: int = 520,
        cdn_url: str = CDN,
        max_bytes: int = MAX_BYTES,
    ):
        """Show one recording in the eegdash-viewer inside a Jupyter cell.

        Serverless: the recording bytes are inlined in the output and pushed
        into the viewer (loaded from ``cdn_url``) over ``postMessage``; it
        renders when the cell ran in your session or the saved notebook is
        trusted. A ``*_desc-pose.json`` sidecar next to the recording adds
        the synchronized hand-pose panel. Needs IPython (soft dependency).
        See https://github.com/eegdash/eegdash-viewer/blob/main/docs/embedding.md.

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
        return ipython.display.HTML(
            build_viewer_html(
                recording,
                pose,
                sidecars=sidecars,
                height=height,
                cdn=cdn_url,
                max_bytes=max_bytes,
            )
        )
