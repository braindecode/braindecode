import base64
import json
import re

import pytest

from braindecode.datasets import _notebook_viewer as nv

REC = "sub-1_task-a_emg"
CDN = "https://viewer.test/v"


@pytest.fixture
def files(tmp_path):
    """Write ``tmp_path/<name>`` for each name and return the paths by name."""

    def _files(*names, content=bytes(range(16))):
        out = {}
        for name in names:
            out[name] = tmp_path / name
            out[name].write_bytes(content)
        return out

    return _files


def _payload(html):
    return json.loads(re.search(r"var payload = (\{.*?\});\n", html).group(1))


@pytest.mark.parametrize(
    ("recording", "present", "sidecars", "expected"),
    [
        (
            f"{REC}.vhdr",
            (f"{REC}.eeg", f"{REC}.vmrk"),
            (),
            (f"{REC}.eeg", f"{REC}.vmrk"),
        ),
        (
            f"{REC}.vhdr",
            (f"{REC}.eeg",),
            ("sub-1_task-a_channels.tsv",),
            (f"{REC}.eeg",),
        ),
        (
            f"{REC}.set",
            (f"{REC}.fdt", "x_channels.tsv"),
            ("x_channels.tsv", "missing.tsv", "x_channels.tsv"),
            (f"{REC}.fdt", "x_channels.tsv"),
        ),
        (f"{REC}.bdf", (f"{REC}.eeg",), (), ()),
    ],
    ids=["brainvision-trio", "partial-trio", "eeglab+dedup-sidecars", "single-file"],
)
def test_collect_files(files, tmp_path, recording, present, sidecars, expected):
    paths = files(recording, *present)
    got = nv.collect_files(paths[recording], tuple(tmp_path / s for s in sidecars))
    assert got == [paths[recording]] + [tmp_path / e for e in expected]


@pytest.mark.parametrize("with_pose", [False, True], ids=["no-pose", "pose"])
def test_build_viewer_html_payload(files, tmp_path, with_pose):
    rec = files(f"{REC}.bdf")[f"{REC}.bdf"]
    pose = (tmp_path / "sub-1_task-a_desc-pose.json") if with_pose else None
    if pose:
        pose.write_text("{}")
    html = nv.build_viewer_html(rec, pose, height=300, cdn=CDN + "/")

    assert f'src="{CDN}/index.html?embed=1"' in html and "height:300px" in html
    assert 'var origin = "https://viewer.test";' in html
    assert "eegdash-viewer:open" in html and "eegdash-viewer:ready" in html
    assert "document.currentScript" in html and "localhost" not in html
    payload = _payload(html)
    assert [f["name"] for f in payload["files"]] == [rec.name]
    assert base64.b64decode(payload["files"][0]["b64"]) == bytes(range(16))
    assert payload["pose"] == (base64.b64encode(b"{}").decode() if with_pose else None)


@pytest.mark.parametrize(
    ("raw_bytes", "pose_bytes", "max_bytes", "ok"),
    [
        (48, None, 64, True),
        (48, None, 63, False),
        (48, 30, 64, False),
        (0, None, 0, True),
    ],
    ids=["exact-fit", "one-short", "pose-counted", "empty"],
)
def test_size_guard_counts_base64(
    files, tmp_path, raw_bytes, pose_bytes, max_bytes, ok
):
    rec = files(f"{REC}.edf", content=bytes(raw_bytes))[f"{REC}.edf"]
    pose = None
    if pose_bytes is not None:
        pose = tmp_path / "sub-1_task-a_desc-pose.json"
        pose.write_bytes(b"x" * pose_bytes)
    if ok:
        assert nv.build_viewer_html(rec, pose, max_bytes=max_bytes)
    else:
        with pytest.raises(ValueError, match="max_bytes"):
            nv.build_viewer_html(rec, pose, max_bytes=max_bytes)


@pytest.mark.parametrize(
    "name",
    ["sub-1_epo.fif", "sub-1_eeg.cdt", "sub-1_ieeg.mef", "sub-1_task-a_beh.tsv"],
)
def test_check_viewable_rejects_unsupported(files, name):
    with pytest.raises(ValueError, match="viewer opens"):
        nv.build_viewer_html(files(name)[name])


def test_check_viewable_rejects_directories(tmp_path):
    (tmp_path / "sub-1_task-a_meg.ds").mkdir()
    with pytest.raises(ValueError, match="directory"):
        nv.build_viewer_html(tmp_path / "sub-1_task-a_meg.ds")


@pytest.mark.parametrize(
    "cdn",
    [
        "viewer",
        "file:///tmp/viewer",
        "//cdn.example.org/v",
        f"{CDN}?x=1",
        f"{CDN}#frag",
        f"{CDN}/index.html",
    ],
)
def test_bad_cdn_fails_before_reading(tmp_path, cdn):
    with pytest.raises(ValueError, match="cdn must be"):
        nv.build_viewer_html(tmp_path / f"{REC}.edf", cdn=cdn)  # file never created


def test_recording_files_from_bids_path_and_plain_path(files, tmp_path):
    rec = files(f"{REC}.bdf")[f"{REC}.bdf"]
    pose = tmp_path / "sub-1_task-a_desc-pose.json"
    pose.write_text("{}")
    assert nv.recording_files(rec) == (rec, (), pose)

    class FakeBIDSPath:  # duck-typed like mne_bids.BIDSPath
        fpath = rec

        def find_matching_sidecar(self, suffix, extension, on_error):
            return (
                tmp_path / f"sub-1_{suffix}{extension}" if suffix == "events" else None
            )

    assert nv.recording_files(FakeBIDSPath()) == (
        rec,
        (tmp_path / "sub-1_events.tsv",),
        pose,
    )
