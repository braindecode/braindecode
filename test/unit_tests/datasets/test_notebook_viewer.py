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
    return json.loads(re.search(r"var payload = (\{.*?\}), origin", html).group(1))


@pytest.mark.parametrize(
    ("recording", "present", "expected"),
    [
        (f"{REC}.vhdr", (f"{REC}.eeg", f"{REC}.vmrk"), (f"{REC}.eeg", f"{REC}.vmrk")),
        (f"{REC}.vhdr", (f"{REC}.eeg",), (f"{REC}.eeg",)),
        (f"{REC}.set", (f"{REC}.fdt",), (f"{REC}.fdt",)),
        (f"{REC}.bdf", (f"{REC}.eeg",), ()),
    ],
    ids=["brainvision-trio", "partial-trio", "eeglab", "single-file"],
)
def test_recording_files_siblings(files, tmp_path, recording, present, expected):
    rec, *_ = files(recording, *present)
    assert nv.recording_files(rec) == ([rec] + [tmp_path / e for e in expected], None)


def test_recording_files_bids_inheritance_and_pose(tmp_path):
    root = tmp_path / "ds"
    eeg_dir = root / "sub-01" / "ses-1" / "eeg"
    eeg_dir.mkdir(parents=True)
    rec = eeg_dir / "sub-01_ses-1_task-x_eeg.bdf"
    rec.touch()
    events = root / "sub-01" / "sub-01_ses-1_events.tsv"  # session level
    events.touch()
    pose = eeg_dir / "sub-01_ses-1_task-x_desc-pose.json"
    pose.write_text("{}")
    assert nv.recording_files(rec) == ([rec, events], pose)
    (root / "sub-01" / "sub-01_backup-1_events.tsv").touch()  # unknown entity
    assert nv.recording_files(rec) == ([rec], pose)  # mne_bids gives up: no crash
    plain = (
        tmp_path / "my_recording_01.edf"
    )  # not BIDS: no inheritance, pose after the whole stem
    plain.touch()
    (tmp_path / "task-rest_events.tsv").touch()
    (tmp_path / "my_recording_01_desc-pose.json").write_text("{}")
    assert nv.recording_files(plain) == (
        [plain],
        tmp_path / "my_recording_01_desc-pose.json",
    )


@pytest.mark.parametrize(
    ("name", "expect"),
    [
        ("sub-1_task-a_emg.bdf", "sub-1_task-a_emg.bdf"),  # BIDS name kept
        ("0-raw.fif", "0-raw_eeg.fif"),  # braindecode BaseConcatDataset.save()
        ("session1.EDF", "session1_eeg.edf"),  # plain file, any case
        ("sub-1_epo.fif", ValueError),  # epochs, not a raw recording
        ("sub-1_eeg.cdt", ValueError),  # no in-memory reader
        ("sub-1_task-a_beh.tsv", ValueError),
    ],
)
def test_posted_name_or_rejection(files, name, expect):
    (rec,) = files(name)
    if expect is ValueError:
        with pytest.raises(ValueError, match="viewer opens"):
            nv.build_viewer_html(rec)
    else:
        assert _payload(nv.build_viewer_html(rec))["files"][0]["name"] == expect


def test_eeglab_fdt_follows_the_posted_set_name(files):
    rec, _ = files("session1.set", "session1.fdt")
    names = [f["name"] for f in _payload(nv.build_viewer_html(rec))["files"]]
    assert names == [
        "session1_eeg.set",
        "session1_eeg.fdt",
    ]  # the viewer probes <prefix>_eeg.fdt


def test_missing_dangling_or_directory_recordings_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="file not found"):
        nv.build_viewer_html(tmp_path / "sub-1_task-a_eeg.edf")
    (tmp_path / "sub-1_task-a_meg.ds").mkdir()
    with pytest.raises(ValueError, match="viewer opens"):
        nv.build_viewer_html(tmp_path / "sub-1_task-a_meg.ds")
    vhdr = tmp_path / "sub-1_task-a_eeg.vhdr"
    vhdr.touch()
    (tmp_path / "sub-1_task-a_eeg.eeg").symlink_to(tmp_path / "annex" / "gone")
    with pytest.raises(ValueError, match="dangling symlink"):
        nv.build_viewer_html(vhdr)


@pytest.mark.parametrize("with_pose", [False, True], ids=["no-pose", "pose"])
def test_build_viewer_html_payload(files, tmp_path, with_pose):
    (rec,) = files(f"{REC}.bdf")
    if with_pose:
        (tmp_path / "sub-1_task-a_desc-pose.json").write_text("{}")
    html = nv.build_viewer_html(rec, height=300, cdn_url=CDN + "/")
    assert f'frame.src = "{CDN}/index.html?embed=1"' in html and "height:300px" in html
    assert 'origin = "https://viewer.test"' in html
    payload = _payload(html)
    assert base64.b64decode(payload["files"][0]["b64"]) == bytes(range(16))
    assert payload["pose"] == (base64.b64encode(b"{}").decode() if with_pose else None)


@pytest.mark.parametrize(
    ("raw_bytes", "pose_bytes", "max_bytes", "ok"),
    [(48, None, 64, True), (48, None, 63, False), (48, 30, 64, False)],
    ids=["exact-fit", "one-short", "pose-counted"],
)
def test_size_guard_counts_base64(
    files, tmp_path, raw_bytes, pose_bytes, max_bytes, ok
):
    (rec,) = files(f"{REC}.edf", content=bytes(raw_bytes))
    if pose_bytes is not None:
        (tmp_path / "sub-1_task-a_desc-pose.json").write_bytes(b"x" * pose_bytes)
    if ok:
        assert nv.build_viewer_html(rec, max_bytes=max_bytes)
    else:
        with pytest.raises(ValueError, match="max_bytes"):
            nv.build_viewer_html(rec, max_bytes=max_bytes)


@pytest.mark.parametrize(
    ("cdn", "expect"),
    [
        ("viewer", ValueError),
        ("javascript:alert(1)", ValueError),
        ("file:///tmp/viewer", ValueError),
        (f"{CDN}?x=1", ValueError),
        (f"{CDN}#frag", ValueError),
        (f"{CDN}/index.html", ValueError),
        (
            "https://user:pw@viewer.test/v",
            ValueError,
        ),  # origins never carry credentials
        (
            f"{CDN}/app?",
            f'frame.src = "{CDN}/app/index.html?embed=1"',
        ),  # empty delimiter dropped
        (
            f'{CDN}/"><script>alert(1)</script>',
            "\\u003cscript>alert",
        ),  # JSON-escaped, never a tag
    ],
)
def test_cdn_handling(files, tmp_path, cdn, expect):
    if expect is ValueError:  # fails before any file is touched
        with pytest.raises(ValueError, match="cdn_url must be"):
            nv.build_viewer_html(tmp_path / f"{REC}.edf", cdn_url=cdn)
    else:
        (rec,) = files(f"{REC}.edf")
        html = nv.build_viewer_html(rec, cdn_url=cdn)
        assert expect in html and "<script>alert" not in html
