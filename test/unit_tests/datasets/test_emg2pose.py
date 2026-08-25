# Authors: Bruno Aristimunha <b.aristimunha@gmail.com>
#
# License: BSD (3-clause)
import json
import urllib.request

import mne
import numpy as np
import pytest

from braindecode.datasets._bids_meta import collect_fields
from braindecode.datasets._viewer_server import (
    ViewerServer,
    build_iframe_html,
    get_viewer_server,
)
from braindecode.datasets.emg2pose import EMG2Pose


# ==================== collect_fields (generalized extras) ====================

PARTICIPANTS = {"n_sessions": "4", "handedness": "R"}
RECORD = {
    "subject": "01",
    "task": "emg2pose",
    "stage": "counting",
    "side": "left",
    "split": "train",
    "moving_hand": "left",
    "held_out_user": False,
    "nested": {"a": 1},
}


def test_collect_fields_auto_carries_everything():
    out = collect_fields(PARTICIPANTS, RECORD)
    assert out["stage"] == "counting"
    assert out["handedness"] == "R"
    assert out["held_out_user"] is False
    # containers are JSON-stringified for pandas-friendliness
    assert json.loads(out["nested"]) == {"a": 1}
    # deterministic order
    assert list(out) == sorted(out)


def test_collect_fields_mapped_only_and_renames():
    out = collect_fields(
        PARTICIPANTS,
        RECORD,
        field_map={"side": "hand"},
        extra_fields=None,
    )
    assert "stage" not in out and "split" not in out
    assert out["hand"] == "left"          # renamed
    assert out["subject"] == "01"         # core kept even in mapped_only


def test_collect_fields_explicit_list_and_exclude():
    out = collect_fields(
        PARTICIPANTS, RECORD, extra_fields=["stage", "split"], exclude=[]
    )
    assert set(out) >= {"stage", "split", "subject"}
    assert "moving_hand" not in out

    auto = collect_fields(PARTICIPANTS, RECORD, exclude=("split",))
    assert "split" not in auto and "stage" in auto


def test_collect_fields_later_sources_win():
    out = collect_fields({"side": "left"}, {"side": "right"})
    assert out["side"] == "right"


# ============================ viewer server ==================================

@pytest.fixture()
def served_tree(tmp_path):
    root = tmp_path / "bids"
    (root / "sub-01" / "emg").mkdir(parents=True)
    (root / "sub-01" / "emg" / "rec.bin").write_bytes(bytes(range(256)))
    assets = tmp_path / "assets"
    assets.mkdir()
    (assets / "index.html").write_text("<html>viewer</html>")
    return ViewerServer(root, assets_dir=assets), root


def test_server_serves_full_and_ranged(served_tree):
    srv, _ = served_tree
    url = f"{srv.base}/data/sub-01/emg/rec.bin"

    full = urllib.request.urlopen(url)
    assert full.status == 200
    assert full.read() == bytes(range(256))

    req = urllib.request.Request(url, headers={"Range": "bytes=10-19"})
    part = urllib.request.urlopen(req)
    assert part.status == 206
    assert part.headers["Content-Range"] == "bytes 10-19/256"
    assert part.read() == bytes(range(10, 20))

    suffix = urllib.request.urlopen(
        urllib.request.Request(url, headers={"Range": "bytes=-5"})
    )
    assert suffix.status == 206
    assert suffix.read() == bytes(range(251, 256))


def test_server_blocks_traversal_and_missing(served_tree):
    srv, _ = served_tree
    with pytest.raises(urllib.error.HTTPError):
        urllib.request.urlopen(f"{srv.base}/data/../secrets.txt")
    with pytest.raises(urllib.error.HTTPError):
        urllib.request.urlopen(f"{srv.base}/data/sub-01/nope.bin")
    with pytest.raises(urllib.error.HTTPError):
        urllib.request.urlopen(f"{srv.base}/other/x")


def test_get_viewer_server_is_singleton(tmp_path):
    a = get_viewer_server(tmp_path, assets_dir=str(tmp_path))
    b = get_viewer_server(str(tmp_path), assets_dir=str(tmp_path))
    assert a is b


def test_iframe_html_includes_pose_when_present():
    html = build_iframe_html("http://127.0.0.1:1", "s/t/rec_emg.vhdr")
    assert "emg=http%3A%2F%2F127.0.0.1%3A1%2Fdata%2Fs%2Ft%2Frec_emg.vhdr" in html
    assert "pose=" not in html

    html = build_iframe_html(
        "http://127.0.0.1:1", "s/t/rec_emg.vhdr", pose_rel="s/t/rec_desc-pose.json"
    )
    assert "pose=" in html and "embed=1" in html


# ============================ EMG2Pose dataset ===============================

@pytest.fixture()
def bids_emg2pose(tmp_path):
    """Minimal converted tree: one subject, one 0.5 s recording."""

    root = tmp_path / "emg2pose-bids"
    ch_dir = root / "sub-01" / "ses-s1" / "emg"
    ch_dir.mkdir(parents=True)

    sfreq = 100
    data = np.random.randn(3, sfreq // 2).astype(np.float64) * 1e-6
    info = mne.create_info(["EMG1", "EMG2", "ja_0"], sfreq, ["emg", "emg", "misc"])
    raw = mne.io.RawArray(data, info)
    mne.export.export_raw(
        ch_dir / "sub-01_ses-s1_task-counting_emg.vhdr", raw, fmt="brainvision", overwrite=True
    )

    prefix = "sub-01_ses-s1_task-counting"
    (ch_dir / f"{prefix}_emg.json").write_text(json.dumps({
        "SamplingFrequency": 2000.0,
        "side": "right",
        "stage": "counting",
        "split": "train",
        "moving_hand": "right",
    }))
    (root / "participants.tsv").write_text(
        "participant_id\thandedness\nsub-01\tR\n"
    )
    return root


def test_emg2pose_generalized_description(bids_emg2pose):
    ds = EMG2Pose(bids_emg2pose)
    desc = ds.records.iloc[0]
    # core entities parsed from the filename
    assert desc["subject"] == "01"
    assert desc["session"] == "s1"
    assert desc["task"] == "counting"
    # sidecar + participants carried verbatim by default ("auto")
    assert desc["side"] == "right"
    assert desc["split"] == "train"
    assert desc["handedness"] == "R"
    # underlying RawDataset description mirrors it
    assert ds.datasets[0].description["stage"] == "counting"


def test_emg2pose_filters(bids_emg2pose):
    ds = EMG2Pose(bids_emg2pose, filters={"split": "train"})
    assert len(ds.datasets) == 1
    with pytest.raises(ValueError, match="matched no recordings"):
        EMG2Pose(bids_emg2pose, filters={"split": "val"})


def test_emg2pose_field_map_restricts_extras(bids_emg2pose):
    ds = EMG2Pose(
        bids_emg2pose,
        field_map={"side": "hand"},
        extra_fields=["stage"],
    )
    desc = ds.records.iloc[0]
    assert desc["hand"] == "right"
    assert "stage" in desc
    assert "moving_hand" not in desc  # restricted away


def test_emg2pose_pose_sidecar_detection(bids_emg2pose):
    ds = EMG2Pose(bids_emg2pose)
    assert ds._pose_sidecar_rel(ds.records.iloc[0]["path"]) is None
    pose = bids_emg2pose / "sub-01" / "ses-s1" / "emg" / (
        "sub-01_ses-s1_task-counting_desc-pose.json"
    )
    pose.write_text("{}")
    rel = ds._pose_sidecar_rel(ds.records.iloc[0]["path"])
    assert rel is not None and rel.endswith("_desc-pose.json")


def test_embed_html_hosted_mode_no_server():
    from braindecode.datasets._viewer_server import embed_html

    html = embed_html(
        "https://viewer.eegdash.org",
        "https://data.example.org/ds/sub-01/emg/rec_emg.vhdr",
        "https://data.example.org/ds/sub-01/emg/rec_desc-pose.json",
    )
    assert html.startswith('<iframe src="https://viewer.eegdash.org/index.html?')
    assert "emg=https%3A%2F%2Fdata.example.org" in html
    assert "pose=" in html


def test_plot_remote_mode_skips_local_server(bids_emg2pose):
    ds = EMG2Pose(bids_emg2pose)
    html = ds.plot(
        0,
        viewer_url="https://viewer.eegdash.org",
        data_url="https://data.example.org/ds",
    ).data
    assert "127.0.0.1" not in html
    assert "https://viewer.eegdash.org/index.html?" in html

    # viewer_url only: data still proxied through the local server, and a
    # pose sidecar (written here) rides along via the same /data tree.
    (bids_emg2pose / "sub-01" / "ses-s1" / "emg" /
     "sub-01_ses-s1_task-counting_desc-pose.json").write_text("{}")
    ds = EMG2Pose(bids_emg2pose)
    html = ds.plot(0, viewer_url="https://viewer.eegdash.org").data
    from urllib.parse import unquote
    assert "127.0.0.1" in html and "/data/" in unquote(html) and "pose=" in unquote(html)


def test_server_range_respects_content_length_on_keepalive(served_tree):
    """Regression: a ranged response must not leak extra bytes into an
    HTTP/1.1 keep-alive connection (the viewer's worker pipelines many
    range fetches over one connection)."""
    import http.client

    srv, _ = served_tree
    host = srv.base.replace("http://", "")
    hostname, port = host.split(":")
    conn = http.client.HTTPConnection(hostname, int(port))

    conn.request("GET", "/data/sub-01/emg/rec.bin", headers={"Range": "bytes=10-19"})
    r1 = conn.getresponse()
    body = r1.read()
    assert r1.status == 206
    assert len(body) == 10 and body == bytes(range(10, 20))

    # Fresh connection (server is deliberately HTTP/1.0): the earlier bug
    # leaked oversized bodies, visible as truncated/garbled follow-ups.
    conn.close()
    conn = http.client.HTTPConnection(hostname, int(port))
    conn.request("GET", "/data/sub-01/emg/rec.bin")
    r2 = conn.getresponse()
    assert r2.status == 200 and len(r2.read()) == 256
    conn.close()
