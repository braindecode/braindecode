# Authors: Bruno Arististimunha <b.aristimunha@gmail.com>
#
# License: BSD (3-clause)
"""Localhost server that bridges braindecode datasets to eegdash-viewer.

The viewer is a static JS app whose format readers fetch windows over
HTTP *Range* requests. To embed it in a Jupyter cell (``EMG2Pose.plot``)
we serve two route trees from one ephemeral port on 127.0.0.1:

- ``/viewer/*`` -> packaged static assets of eegdash-viewer
  (populated once via ``scripts/sync_viewer_assets.py``);
- ``/data/*``   -> the dataset's BIDS root, read-only.

Same origin means zero CORS friction; the kernel and the browser must
share a host (the normal local-Jupyter case).

Python's ``http.server`` has no Range support, so this module ships a
minimal single-interval implementation — enough for the viewer's lazy
window reads without pulling third-party dependencies.
"""

from __future__ import annotations

import atexit
import os
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import quote, unquote, urlparse

__all__ = [
    "ViewerServer",
    "get_viewer_server",
    "build_iframe_html",
    "embed_html",
]

_ASSETS_ENV_VAR = "BRAINSDECODE_VIEWER_ASSETS"
_ASSETS_DIRNAME = Path(__file__).parent / "viewer_static"

_RANGE_RE = re.compile(r"^bytes=(\d*)-(\d*)$")

_CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".json": "application/json",
    ".tsv": "text/tab-separated-values",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".vhdr": "text/plain",
    ".vmrk": "text/plain",
    ".edf": "application/octet-stream",
    ".bdf": "application/octet-stream",
    ".eeg": "application/octet-stream",
}


class _RouteHandler(BaseHTTPRequestHandler):
    """Read-only GET/HEAD with single-range support for /data and /viewer."""

    server_version = "braindecode-viewer/1.0"
    # Deliberately HTTP/1.0 (connection-per-request). Keep-alive over
    # loopback proved flaky across process boundaries on macOS sandboxes
    # (responses acknowledged server-side but never drained to peers),
    # and connection setup costs ~50 µs locally — irrelevant next to the
    # viewer's windowed reads.
    protocol_version = "HTTP/1.0"

    # populated by ViewerServer via functools.partial-free pattern:
    assets_dir: Path = Path(".")
    data_dir: Path = Path(".")

    def log_message(self, fmt, *args):
        import os
        if os.environ.get("BRAINSDECODE_VIEWER_LOG"):
            super().log_message(fmt, *args)

    def _resolve(self) -> tuple[Path, str] | None:
        prefix, _, rest = unquote(urlparse(self.path).path).lstrip("/").partition("/")
        base = (
            self.assets_dir
            if prefix == "viewer"
            else (self.data_dir if prefix == "data" else None)
        )
        if base is None or not rest:
            return None
        candidate = (base / rest).resolve()
        # traversal guard: realpath must stay under the served root
        if not str(candidate).startswith(str(base.resolve()) + os.sep):
            return None
        return candidate, prefix

    def _file_bytes(self, path: Path, rng: str | None):
        """Return (status, headers, fileobj, length) honoring one range."""
        if not path.is_file():
            return 404, {}, None, 0
        size = path.stat().st_size
        ctype = _CONTENT_TYPES.get(path.suffix.lower(), "application/octet-stream")
        start, end = 0, size - 1
        status = 200
        extra: dict[str, str] = {"Accept-Ranges": "bytes"}
        if rng:
            m = _RANGE_RE.match(rng.strip())
            if m and (m.group(1) or m.group(2)):
                if m.group(1):
                    start = int(m.group(1))
                    end = int(m.group(2)) if m.group(2) else size - 1
                else:  # suffix form bytes=-N
                    start = max(0, size - int(m.group(2)))
                    end = size - 1
                if start >= size or start > end:
                    return (
                        416,
                        {"Content-Range": f"bytes */{size}"},
                        None,
                        0,
                    )
                status = 206
                extra["Content-Range"] = f"bytes {start}-{end}/{size}"
        length = end - start + 1
        headers = {
            "Content-Type": ctype,
            "Content-Length": str(length),
            **extra,
        }
        try:
            f = open(path, "rb")
        except OSError:
            return 404, {}, None, 0
        f.seek(start)
        return status, headers, f, length

    def _serve(self, body: bool):
        rng = self.headers.get("Range")
        resolved = self._resolve()
        if resolved is None:
            self._plain(404, b"not found\n", body)
            return
        status, headers, fobj, length = self._file_bytes(resolved[0], rng)
        self.send_response(status)
        for k, v in headers.items():
            self.send_header(k, v)
        if status == 416:
            self.end_headers()
            return
        self.end_headers()
        if not body or fobj is None:
            if fobj is not None:
                fobj.close()
            return
        try:
            # Send EXACTLY `length` bytes: shutil.copyfileobj has no
            # count parameter — using it here would stream to EOF,
            # overshooting the promised Content-Length and desyncing
            # HTTP/1.1 keep-alive connections (browsers then hang on
            # the next pipelined request).
            remaining = length
            while remaining > 0:
                chunk = fobj.read(min(65536, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)
        except (BrokenPipeError, ConnectionResetError):  # aborted fetches
            pass
        finally:
            fobj.close()

    def _plain(self, code: int, payload: bytes, body: bool):
        self.send_response(code)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        if body:
            self.wfile.write(payload)

    def do_GET(self):  # noqa: N802 (stdlib naming)
        self._serve(body=True)

    def do_HEAD(self):  # noqa: N802
        self._serve(body=False)


def _default_assets_dir() -> Path:
    env = os.environ.get(_ASSETS_ENV_VAR)
    if env and Path(env).is_dir():
        return Path(env).resolve()
    if (_ASSETS_DIRNAME / "index.html").is_file():
        return _ASSETS_DIRNAME.resolve()
    raise RuntimeError(
        "eegdash-viewer assets not found. Run "
        "`python scripts/sync_viewer_assets.py --src <path-to-eegdash-viewer>` "
        f"or set {_ASSETS_ENV_VAR}."
    )


class ViewerServer:
    """One threaded HTTP server per (BIDS root); shut down at exit."""

    def __init__(
        self, root: str | os.PathLike, assets_dir: str | os.PathLike | None = None
    ):
        self.data_dir = Path(root).resolve()
        self.assets_dir = (
            Path(assets_dir).resolve() if assets_dir else _default_assets_dir()
        )

        handler = type(
            "_BoundHandler",
            (_RouteHandler,),
            {"assets_dir": self.assets_dir, "data_dir": self.data_dir},
        )
        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self._httpd.daemon_threads = True
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()
        port = self._httpd.server_address[1]
        self.base = f"http://127.0.0.1:{port}"

    def shutdown(self):
        self._httpd.shutdown()
        self._httpd.server_close()


_SERVERS: dict[str, ViewerServer] = {}


def get_viewer_server(root: str | os.PathLike, assets_dir=None) -> ViewerServer:
    """Return (creating if needed) the singleton server for ``root``."""
    key = str(Path(root).resolve())
    srv = _SERVERS.get(key)
    if srv is None:
        srv = ViewerServer(root, assets_dir)
        _SERVERS[key] = srv

        def _cleanup(srv=srv):
            try:
                srv.shutdown()
            except Exception:
                pass

        atexit.register(_cleanup)
    return srv


def build_iframe_html(
    base_url: str,
    recording_rel: str,
    pose_rel: str | None = None,
    height: int = 420,
) -> str:
    """Compose an iframe against a served root using relative paths."""
    return embed_html(
        f"{base_url}/viewer",
        f"{base_url}/data/{recording_rel}",
        None if pose_rel is None else f"{base_url}/data/{pose_rel}",
        height=height,
    )


def embed_html(
    viewer_base: str,
    recording_url: str,
    pose_url: str | None = None,
    height: int = 420,
) -> str:
    """Compose the ``?embed=1`` iframe HTML from absolute URLs.

    Pure string builder (unit-tested without a live server); parameters
    match eegdash-viewer's url-resolver grammar — recordings are passed
    via the modality parameter (``?emg=`` for ``*_emg.*`` files,
    ``?eeg=`` otherwise) plus the F10 ``pose`` extension. Works
    identically for the localhost server (vendored assets + local data)
    and for hosted deployments.
    """
    stem = Path(unquote(urlparse(recording_url).path)).stem.lower()
    param = "eeg"
    for token in ("ieeg", "emg", "meg", "nirs"):
        if stem.endswith(f"_{token}"):
            param = token
            break
    params = [
        f"{param}={quote(recording_url, safe='')}",
        "embed=1",
    ]
    if pose_url:
        params.append(f"pose={quote(pose_url, safe='')}")
    src = f"{viewer_base.rstrip('/')}/index.html?{'&'.join(params)}"
    return (
        f'<iframe src="{src}" '
        f'style="width:100%;height:{int(height)}px;'
        "border:1px solid var(--jp-border-color0, #ccc);"
        'border-radius:4px" '
        'title="eegdash trace viewer" loading="lazy"></iframe>'
    )
