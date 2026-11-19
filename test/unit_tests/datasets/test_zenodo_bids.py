"""CHBMIT and SIENA: download-or-root handling and event names, shared by both
datasets. SeizeIT2: download-or-root handling."""

import mne
import numpy as np
import pytest

from braindecode.datasets import BIDSDataset, RawDataset
from braindecode.datasets import seizeit2 as seizeit2_module
from braindecode.datasets.chb_mit import CHBMIT, CHB_MIT_dataset_params
from braindecode.datasets.seizeit2 import SeizeIT2
from braindecode.datasets.siena import SIENA, SIENA_dataset_params

CASES = [
    (CHBMIT, CHB_MIT_dataset_params, "chb_mit_bids.zip", "BIDS_CHB-MIT"),
    (SIENA, SIENA_dataset_params, "SIENA.zip", "BIDS_Siena"),
]


@pytest.fixture
def captured(monkeypatch):
    calls = {}

    def fake_init(self, *args, **kwargs):
        calls["bids"] = kwargs

    monkeypatch.setattr(BIDSDataset, "__init__", fake_init)
    return calls


@pytest.mark.parametrize("cls,params,archive,subdir", CASES)
def test_root_without_participants_raises(cls, params, archive, subdir, tmp_path):
    with pytest.raises(ValueError, match="missing participants.tsv"):
        cls(root=tmp_path)


@pytest.mark.parametrize("cls,params,archive,subdir", CASES)
def test_given_root_is_passed_to_bids(cls, params, archive, subdir, tmp_path, captured):
    (tmp_path / "participants.tsv").write_text("participant_id\n")
    cls(root=tmp_path, subjects=["01"])
    assert captured["bids"] == dict(
        root=tmp_path, subjects=["01"], extensions=".edf", check=False
    )


@pytest.mark.parametrize("cls,params,archive,subdir", CASES)
def test_no_root_fetches_and_corrects_the_path(
    cls, params, archive, subdir, monkeypatch, captured
):
    calls = []
    namespace = cls.__init__.__globals__  # where __init__ looks the helpers up
    monkeypatch.setitem(
        namespace,
        "fetch_dataset",
        lambda **kw: calls.append(("fetch", kw)) or "/data/raw",
    )
    monkeypatch.setitem(
        namespace,
        "_correct_dataset_path",
        lambda *a: calls.append(("correct", a)) or "/data/bids",
    )
    cls()
    assert calls == [
        (
            "fetch",
            dict(
                dataset_params=params, path=None, processor="unzip", force_update=False
            ),
        ),
        ("correct", ("/data/raw", archive, subdir)),
    ]
    assert captured["bids"] == dict(root="/data/bids", extensions=".edf", check=False)


@pytest.mark.parametrize("cls,params,archive,subdir", CASES)
def test_annotations_are_named_after_event_type(
    cls, params, archive, subdir, monkeypatch
):
    # mne-bids reads an events.tsv without trial_type as "n/a" annotations and
    # keeps the other columns in the extras; EDF+ gaps carry no extras.
    raw = mne.io.RawArray(np.zeros((1, 2560)), mne.create_info(1, 256.0, "eeg"))
    raw.set_annotations(
        mne.Annotations(
            [1, 3, 5],
            [1, 1, 1],
            ["n/a", "n/a", "BAD_ACQ_SKIP"],
            extras=[{"eventType": "bckg"}, {"eventType": "sz"}, {}],
        )
    )
    monkeypatch.setattr(
        BIDSDataset, "_get_dataset", lambda self, path: RawDataset(raw, {})
    )
    dataset = cls.__new__(cls)._get_dataset(None)
    assert list(dataset.raw.annotations.description) == ["bckg", "sz", "BAD_ACQ_SKIP"]
    _, event_id = mne.events_from_annotations(dataset.raw)
    assert set(event_id) == {"bckg", "sz"}


def test_seizeit2_given_root_reads_only_eeg(tmp_path, captured):
    (tmp_path / "participants.tsv").write_text("participant_id\n")
    SeizeIT2(root=tmp_path, subjects=["001"])
    assert captured["bids"] == dict(
        root=tmp_path,
        subjects=["001"],
        datatypes="eeg",
        extensions=".edf",
        check=False,
    )


def test_seizeit2_no_root_downloads_the_subjects(monkeypatch, captured, tmp_path):
    (tmp_path / "participants.tsv").write_text("participant_id\n")
    calls = []
    monkeypatch.setattr(
        seizeit2_module,
        "_fetch_seizeit2",
        lambda subjects: calls.append(subjects) or tmp_path,
    )
    SeizeIT2(subjects="001")
    assert calls == ["001"]
    assert captured["bids"] == dict(
        root=tmp_path,
        subjects="001",
        datatypes="eeg",
        extensions=".edf",
        check=False,
    )


def test_seizeit2_fetch_gets_top_level_files_and_eeg_of_the_subjects(
    monkeypatch, tmp_path
):
    files = {
        "ds005873/": ["ds005873/participants.tsv", "ds005873/events.json"],
        "ds005873/sub-001/": [
            "ds005873/sub-001/ses-01/eeg/sub-001_ses-01_run-01_eeg.edf",
            "ds005873/sub-001/ses-01/eeg/sub-001_ses-01_run-01_events.tsv",
            "ds005873/sub-001/ses-01/ecg/sub-001_ses-01_run-01_ecg.edf",
        ],
    }
    monkeypatch.setattr(
        seizeit2_module,
        "_list_files",
        lambda prefix, delimiter=None: [(key, "abc") for key in files[prefix]],
    )
    retrieved = []
    monkeypatch.setattr(
        seizeit2_module.pooch,
        "retrieve",
        lambda url, known_hash, fname, path: retrieved.append(
            (url, known_hash, f"{path}/{fname}")
        ),
    )
    root = seizeit2_module._fetch_seizeit2(["001"], path=tmp_path)
    assert root == f"{tmp_path}/ds005873"
    s3 = "https://s3.amazonaws.com/openneuro.org/ds005873"
    assert retrieved == [
        (f"{s3}/{name}", "md5:abc", f"{root}/{name}")
        for name in [
            "participants.tsv",
            "events.json",
            "sub-001/ses-01/eeg/sub-001_ses-01_run-01_eeg.edf",
            "sub-001/ses-01/eeg/sub-001_ses-01_run-01_events.tsv",
        ]
    ]
