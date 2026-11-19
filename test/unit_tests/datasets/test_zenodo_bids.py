"""CHBMIT and SIENA: download-or-root handling and event names, shared by both
datasets."""

import mne
import numpy as np
import pytest

from braindecode.datasets import BIDSDataset, RawDataset
from braindecode.datasets.chb_mit import CHBMIT, CHB_MIT_dataset_params
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


def test_szcore_bids_falls_back_to_trial_type(tmp_path):
    from mne_bids import BIDSPath, write_raw_bids

    pytest.importorskip("edfio")
    raw = mne.io.RawArray(np.zeros((1, 500)), mne.create_info(["C3"], 100, "eeg"))
    path = BIDSPath(subject="01", task="monitoring", datatype="eeg", root=tmp_path)
    write_raw_bids(raw, path, allow_preload=True, format="EDF", verbose="ERROR")
    events = path.copy().update(suffix="events", extension=".tsv")
    events.fpath.write_text(
        "onset\tduration\ttrial_type\teventType\n"
        "1\t0.2\tbaseline\tn/a\n"
        "3\t0.5\tseizure\tsz\n",
        encoding="utf-8",
    )

    loaded = CHBMIT(root=tmp_path).datasets[0].raw
    np.testing.assert_array_equal(loaded.annotations.description, ["baseline", "sz"])
    samples, ids = mne.events_from_annotations(loaded)
    np.testing.assert_array_equal(samples[:, 0], [100, 300])
    assert ids == {"baseline": 1, "sz": 2}
