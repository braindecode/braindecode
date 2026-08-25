# Authors: Bruno Aristimunha <b.aristimunha@gmail.com>
#
# License: BSD (3-clause)
"""Serverless in-notebook viewer for BIDS recordings.

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
import html
import json
import uuid
from pathlib import Path
from urllib.parse import urlsplit

CDN = "https://eegdash.github.io/eegdash-viewer"
MAX_BYTES = 64 * 2**20  # base64 output per call; it is saved with the notebook
# What the viewer opens from in-memory files: BIDS electrophysiology
# suffixes and the single-file (or small-sibling) formats it reads.
SUFFIXES = frozenset({"eeg", "ieeg", "emg", "meg", "nirs"})
EXTENSIONS = frozenset({".set", ".edf", ".bdf", ".vhdr", ".fif", ".snirf", ".nwb"})
# Split-file formats travel with their siblings.
_SIBLINGS = {".vhdr": (".eeg", ".vmrk"), ".set": (".fdt",)}

_TEMPLATE = """<iframe id="{uid}" src="{src}" title="eegdash trace viewer"
  style="width:100%;height:{height}px;border:1px solid var(--jp-border-color1,#d9dce1);border-radius:6px;background:transparent"></iframe>
<script>
(function () {{
  // The iframe is the element right before this script, so a cell output
  // rendered twice (linked view, display() twice) still finds its own.
  var self = document.currentScript;
  var frame = (self && self.previousElementSibling && self.previousElementSibling.tagName === "IFRAME")
    ? self.previousElementSibling : document.getElementById({uid_json});
  var payload = {payload};
  var origin = {origin_json};
  var files = null, readyGot = false;
  function decode(b64) {{
    if (Uint8Array.fromBase64) return Uint8Array.fromBase64(b64);
    var bin = atob(b64), out = new Uint8Array(bin.length);
    for (var i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
    return out;
  }}
  function materialize() {{
    if (files) return files;
    files = payload.files.map(function (f) {{ return new File([decode(f.b64)], f.name); }});
    files.pose = payload.pose ? "data:application/json;base64," + payload.pose : null;
    payload = null;   // the File objects carry the bytes from here on
    return files;
  }}
  function send(target) {{
    try {{
      var fs = materialize();
      frame.contentWindow.postMessage({{ type: "eegdash-viewer:open", files: fs, pose: fs.pose }}, target || origin);
    }} catch (err) {{
      var note = document.createElement("div");
      note.style.cssText = "font:12px system-ui,sans-serif;color:#b3261e;padding:4px 0";
      note.textContent = "eegdash viewer: could not hand over the recording (" + err.message + ")";
      frame.insertAdjacentElement("afterend", note);
    }}
  }}
  function onMessage(e) {{
    if (!frame.isConnected) {{ window.removeEventListener("message", onMessage); return; }}
    if (e.source !== frame.contentWindow || !e.data || e.data.type !== "eegdash-viewer:ready") return;
    readyGot = true;
    send(e.origin);   // every ready: the viewer re-announces after a reload
  }}
  window.addEventListener("message", onMessage);
  // Fallback when `ready` was posted before this script ran (cached
  // iframe finished during the parse of the payload): the viewer's open
  // handler is idempotent, so a second hand-over only costs a reload.
  setTimeout(function () {{ if (!readyGot) send(); }}, 2000);
}})();
</script>"""


def bids_prefix(recording: Path) -> str:
    """``sub-01_ses-02_task-x_run-17`` for ``.../sub-01_ses-02_task-x_run-17_emg.bdf``."""
    return Path(recording).stem.rsplit("_", 1)[0]


def pose_sidecar_for(recording: Path) -> Path | None:
    """The ``*_desc-pose.json`` hand-pose sidecar next to ``recording``, if any."""
    rec = Path(recording)
    pose = rec.with_name(bids_prefix(rec) + "_desc-pose.json")
    return pose if pose.is_file() else None


def check_viewable(recording: Path) -> None:
    """Raise ``ValueError`` when the viewer cannot open ``recording`` from memory."""
    rec = Path(recording)
    suffix = rec.stem.rsplit("_", 1)[-1]
    if rec.is_dir():
        raise ValueError(f"{rec.name}: directory-based recordings cannot be inlined")
    if suffix not in SUFFIXES or rec.suffix.lower() not in EXTENSIONS:
        raise ValueError(
            f"{rec.name}: the viewer opens *_{{{','.join(sorted(SUFFIXES))}}} recordings in "
            f"{' '.join(sorted(EXTENSIONS))} from memory"
        )


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
    payload = {
        "files": [{"name": p.name, "b64": _b64(p)} for p in files],
        "pose": _b64(pose_sidecar) if pose_sidecar is not None else None,
    }
    uid = f"eegdash-viewer-{uuid.uuid4().hex[:8]}"
    return _TEMPLATE.format(
        uid=uid,
        uid_json=json.dumps(uid),
        src=html.escape(f"{base}/index.html?embed=1", quote=True),
        origin_json=json.dumps(origin),
        height=int(height),
        payload=json.dumps(payload).replace("</", "<\\/"),
    )


def _b64(path: Path) -> str:
    return base64.b64encode(Path(path).read_bytes()).decode()
