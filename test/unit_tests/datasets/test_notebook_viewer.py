import base64
import json
import re

import pytest

from braindecode.datasets import _notebook_viewer as nv


def _write(path, n_bytes=16):
    path.write_bytes(bytes(range(n_bytes)))
    return path


def _payload(html):
    return json.loads(re.search(r"var payload = (\{.*?\});\n", html).group(1))


def test_collect_files_brainvision_trio_and_given_sidecars(tmp_path):
    rec = _write(tmp_path / "sub-1_task-a_emg.vhdr")
    eeg = _write(tmp_path / "sub-1_task-a_emg.eeg")
    channels = _write(tmp_path / "sub-1_task-a_channels.tsv")
    missing = tmp_path / "sub-1_task-a_events.tsv"
    assert nv.collect_files(rec, (channels, missing, channels)) == [rec, eeg, channels]


def test_collect_files_single_file_recording(tmp_path):
    rec = _write(tmp_path / "sub-1_emg.bdf")
    assert nv.collect_files(rec) == [rec]


def test_build_viewer_html_iframe_and_payload(tmp_path):
    rec = _write(tmp_path / "sub-1_task-a_emg.bdf")
    pose = tmp_path / "sub-1_task-a_desc-pose.json"
    pose.write_text("{}")
    html = nv.build_viewer_html(rec, pose, height=300, cdn="https://viewer.test/v/")

    assert 'src="https://viewer.test/v/index.html?embed=1"' in html
    assert "height:300px" in html
    assert 'var origin = "https://viewer.test";' in html  # postMessage target origin
    assert "eegdash-viewer:open" in html and "eegdash-viewer:ready" in html
    assert "document.currentScript" in html  # a duplicated output finds its own iframe
    assert "localhost" not in html and "127.0.0.1" not in html

    payload = _payload(html)
    assert [f["name"] for f in payload["files"]] == ["sub-1_task-a_emg.bdf"]
    assert base64.b64decode(payload["files"][0]["b64"]) == bytes(range(16))
    assert payload["pose"] == base64.b64encode(b"{}").decode()


def test_build_viewer_html_without_pose(tmp_path):
    rec = _write(tmp_path / "sub-1_emg.edf")
    assert _payload(nv.build_viewer_html(rec))["pose"] is None


def test_size_guard_counts_base64_and_the_pose_sidecar(tmp_path):
    rec = _write(tmp_path / "sub-1_emg.edf", 48)  # 48 raw bytes -> 64 base64 chars
    assert nv.build_viewer_html(rec, max_bytes=64)
    with pytest.raises(ValueError, match="max_bytes"):
        nv.build_viewer_html(rec, max_bytes=63)
    pose = tmp_path / "sub-1_desc-pose.json"
    pose.write_text("x" * 30)
    with pytest.raises(ValueError, match="max_bytes"):
        nv.build_viewer_html(rec, pose, max_bytes=64)


def test_pose_sidecar_for_uses_the_bids_prefix(tmp_path):
    rec = _write(tmp_path / "sub-1_ses-2_task-a_run-3_emg.bdf")
    assert nv.pose_sidecar_for(rec) is None
    pose = tmp_path / "sub-1_ses-2_task-a_run-3_desc-pose.json"
    pose.write_text("{}")
    assert nv.pose_sidecar_for(rec) == pose


@pytest.mark.parametrize(
    "name",
    ["sub-1_epo.fif", "sub-1_eeg.cdt", "sub-1_ieeg.mef", "sub-1_task-a_beh.tsv"],
)
def test_check_viewable_rejects_what_the_viewer_cannot_open(tmp_path, name):
    rec = _write(tmp_path / name)
    with pytest.raises(ValueError, match="viewer opens"):
        nv.build_viewer_html(rec)


def test_check_viewable_rejects_directory_recordings(tmp_path):
    ds = tmp_path / "sub-1_task-a_meg.ds"
    ds.mkdir()
    with pytest.raises(ValueError, match="directory"):
        nv.build_viewer_html(ds)


@pytest.mark.parametrize(
    "cdn",
    [
        "viewer",
        "file:///tmp/viewer",
        "//cdn.example.org/v",
        "https://viewer.test/v?x=1",
        "https://viewer.test/v#frag",
        "https://viewer.test/v/index.html",
    ],
)
def test_build_viewer_html_rejects_bad_cdn_before_reading(tmp_path, cdn):
    rec = tmp_path / "sub-1_emg.edf"  # never created: a bad cdn must fail first
    with pytest.raises(ValueError, match="cdn must be"):
        nv.build_viewer_html(rec, cdn=cdn)
