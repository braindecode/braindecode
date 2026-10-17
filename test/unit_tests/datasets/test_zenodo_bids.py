"""CHBMIT and SIENA: download-or-root handling, shared by both datasets."""

import pytest

from braindecode.datasets import BIDSDataset
from braindecode.datasets.chb_mit import CHB_MIT_dataset_params
from braindecode.datasets.chb_mit import CHBMIT
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
