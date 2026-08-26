import base64
import json
import re

import pytest

from braindecode.datasets import _notebook_viewer as nv

REC = "sub-1_task-a_emg"
CDN = "https://viewer.test/v"


@pytest.fixture
def files(tmp_path):
    """Write ``tmp_path/<name>`` for each name and return the paths."""

    def _files(*names, content=bytes(range(16))):
        paths = [tmp_path / n for n in names]
        for p in paths:
            p.write_bytes(content)
        return paths

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
    rec, *_ = files(recording, *present)
    got = nv.collect_files(rec, tuple(tmp_path / s for s in sidecars))
    assert got == [rec] + [tmp_path / e for e in expected]


@pytest.mark.parametrize("with_pose", [False, True], ids=["no-pose", "pose"])
def test_build_viewer_html_payload(files, tmp_path, with_pose):
    (rec,) = files(f"{REC}.bdf")
    pose = (tmp_path / "sub-1_task-a_desc-pose.json") if with_pose else None
    if pose:
        pose.write_text("{}")
    html = nv.build_viewer_html(rec, pose, height=300, cdn=CDN + "/")

    assert f'frame.src = "{CDN}/index.html?embed=1"' in html and "height:300px" in html
    assert '"https://viewer.test");' in html  # postMessage target origin
    assert "eegdash-viewer:open" in html and "eegdash-viewer:ready" in html
    assert "document.currentScript" in html
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
    (rec,) = files(f"{REC}.edf", content=bytes(raw_bytes))
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
    ("name", "posted_as"),
    [
        ("sub-1_task-a_emg.bdf", "sub-1_task-a_emg.bdf"),  # BIDS name kept
        ("0-raw.fif", "0-raw_eeg.fif"),  # braindecode BaseConcatDataset.save()
        ("session1.EDF", "session1_eeg.edf"),  # plain file, any case
        ("sub-1_epo.fif", ValueError),  # epochs, not a raw recording
        ("sub-1_eeg.cdt", ValueError),  # no in-memory reader
        ("sub-1_task-a_beh.tsv", ValueError),
    ],
)
def test_recordings_are_posted_under_a_viewer_name_or_rejected(files, name, posted_as):
    (rec,) = files(name)
    if posted_as is ValueError:
        with pytest.raises(ValueError, match="viewer opens"):
            nv.build_viewer_html(rec)
    else:
        assert _payload(nv.build_viewer_html(rec))["files"][0]["name"] == posted_as


def test_directory_recordings_are_rejected(tmp_path):
    (tmp_path / "sub-1_task-a_meg.ds").mkdir()
    with pytest.raises(ValueError, match="directory"):
        nv.build_viewer_html(tmp_path / "sub-1_task-a_meg.ds")


@pytest.mark.parametrize(
    ("cdn", "expect"),
    [
        ("viewer", ValueError),
        ("javascript:alert(1)", ValueError),
        ("file:///tmp/viewer", ValueError),
        ("//cdn.example.org/v", ValueError),
        (f"{CDN}?x=1", ValueError),
        (f"{CDN}#frag", ValueError),
        (f"{CDN}/index.html", ValueError),
        (
            f"{CDN}/app?",
            f'frame.src = "{CDN}/app/index.html?embed=1"',
        ),  # empty delimiters dropped
        (f"{CDN}/app#", f'frame.src = "{CDN}/app/index.html?embed=1"'),
        (
            f'{CDN}/"><script>alert(1)</script>',
            "\\u003cscript>alert",
        ),  # JSON-escaped, never a tag
    ],
    ids=[
        "relative",
        "javascript",
        "file",
        "protocol-relative",
        "query",
        "fragment",
        "index.html",
        "empty-query",
        "empty-fragment",
        "escaped",
    ],
)
def test_cdn_handling(files, tmp_path, cdn, expect):
    if expect is ValueError:  # fails before any file is read
        with pytest.raises(ValueError, match="cdn must be"):
            nv.build_viewer_html(tmp_path / f"{REC}.edf", cdn=cdn)
    else:
        (rec,) = files(f"{REC}.edf")
        html = nv.build_viewer_html(rec, cdn=cdn)
        assert expect in html and "<script>alert" not in html


def test_recording_files_uses_bids_inheritance(tmp_path):
    root = tmp_path / "ds"
    eeg_dir = root / "sub-01" / "ses-1" / "eeg"
    eeg_dir.mkdir(parents=True)
    rec = eeg_dir / "sub-01_ses-1_task-x_eeg.bdf"
    rec.touch()
    events = root / "sub-01" / "sub-01_ses-1_events.tsv"  # session level
    events.touch()
    pose = eeg_dir / "sub-01_ses-1_task-x_desc-pose.json"
    pose.write_text("{}")
    assert nv.recording_files(rec) == (rec, (events,), pose)
    plain = tmp_path / "0-raw.fif"
    plain.touch()
    assert nv.recording_files(plain) == (
        plain,
        (),
        None,
    )  # not a BIDS name: no inheritance
