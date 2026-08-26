# Authors: Pierre Guetschel <pierre.guetschel@gmail.com>
#
# License: BSD (3-clause)

import sys
from types import SimpleNamespace

import pandas as pd
import pytest
from moabb.datasets import FakeDataset
from moabb.paradigms import LeftRightImagery

from braindecode.datasets import BIDSDataset, BIDSEpochsDataset


@pytest.fixture(scope="module")
def bids_dataset_root(tmpdir_factory):
    tmp_path = tmpdir_factory.mktemp("bids_root")
    dataset = FakeDataset(
        n_subjects=1,
        n_sessions=1,
        n_runs=1,
        stim=True,
        annotations=True,
        event_list=["left_hand", "right_hand"],
    )
    paradigm = LeftRightImagery()
    cache_config = dict(
        save_raw=True,
        save_epochs=True,
        save_array=False,
        use=False,
        overwrite_raw=False,
        overwrite_epochs=False,
        overwrite_array=False,
        path=tmp_path,
    )
    _ = paradigm.get_data(dataset, return_epochs=True, cache_config=cache_config)
    return (
        tmp_path
        / "MNE-BIDS-fake-dataset-imagery-1-1--60--120--lefthand-righthand--c3-cz-c4"
    )


def test_bids_dataset(bids_dataset_root):
    dataset = BIDSDataset(bids_dataset_root)
    assert len(dataset.datasets) == 1
    assert len(dataset.datasets[0].raw.ch_names) == 3
    assert len(dataset.datasets[0].raw.annotations) == 60
    assert set(dataset.datasets[0].raw.annotations.description) == {
        "left_hand",
        "right_hand",
    }


def test_bids_epochs_dataset(bids_dataset_root):
    dataset = BIDSEpochsDataset(bids_dataset_root)
    assert len(dataset) == 60
    x, y, _ = dataset[0]
    assert x.shape[0] == 3
    assert x.ndim == 2
    assert y in ["left_hand", "right_hand"]


@pytest.fixture(scope="module")
def emg_bids_root(tmp_path_factory):
    """One real emg2pose-style BrainVision recording + a session-level events sidecar."""
    import mne
    import numpy as np

    pytest.importorskip("pybv")  # mne's BrainVision writer

    root = tmp_path_factory.mktemp("emg2pose") / "bids"
    eeg_dir = root / "sub-893" / "ses-s1" / "emg"
    eeg_dir.mkdir(parents=True)
    info = mne.create_info(["EMG1", "EMG2", "ja0"], 100, ["emg", "emg", "misc"])
    raw = mne.io.RawArray(np.random.randn(3, 50) * 1e-6, info, verbose="ERROR")
    mne.export.export_raw(
        eeg_dir / "sub-893_ses-s1_task-fist_acq-right_emg.vhdr",
        raw,
        fmt="brainvision",
        overwrite=True,
        verbose="ERROR",
    )
    (root / "sub-893" / "sub-893_ses-s1_events.tsv").write_text(
        "onset\tduration\ttrial_type\n0.1\t0.0\tfist\n"
    )
    return root


def test_bids_dataset_plot_inlines_trio_and_inherited_sidecars(emg_bids_root):
    html = BIDSDataset(emg_bids_root, suffixes="emg", datatypes="emg").plot(0).data
    assert (
        'frame.src = "https://eegdash.github.io/eegdash-viewer/index.html?embed=1"'
        in html
    )
    for name in (
        "sub-893_ses-s1_task-fist_acq-right_emg.vhdr",  # header first
        "sub-893_ses-s1_task-fist_acq-right_emg.eeg",
        "sub-893_ses-s1_task-fist_acq-right_emg.vmrk",
        "sub-893_ses-s1_events.tsv",  # inherited from the session level
    ):
        assert name in html


def test_bids_dataset_plot_keeps_the_bids_name_of_a_symlink(tmp_path):
    """git-annex/datalad trees: the symlink's BIDS name is posted, not the target's."""
    root = tmp_path / "bids"
    eeg_dir = root / "sub-1" / "eeg"
    eeg_dir.mkdir(parents=True)
    target = tmp_path / "annex" / "MD5E-s0--abc.edf"
    target.parent.mkdir()
    target.touch()
    link = eeg_dir / "sub-1_task-a_eeg.edf"
    try:
        link.symlink_to(target)
    except OSError as error:
        pytest.skip(f"symlinks are unavailable: {error}")
    ds = object.__new__(BIDSDataset)
    ds.datasets = [SimpleNamespace(raw=SimpleNamespace(filenames=(link,)))]
    html = ds.plot(0).data
    assert link.name in html and "MD5E" not in html


def test_bids_dataset_plot_explains_missing_ipython(emg_bids_root, monkeypatch):
    ds = BIDSDataset(emg_bids_root, suffixes="emg", datatypes="emg")
    monkeypatch.setitem(sys.modules, "IPython", None)  # soft import: not declared
    with pytest.raises(
        RuntimeError, match=r"(?s)BIDSDataset.plot\(\).*IPython.*pip install IPython"
    ):
        ds.plot(0)


def test_plot_comes_with_base_concat_dataset(emg_bids_root):
    """Any BaseConcatDataset element: the raw's file (data file mapped to its header),
    the recorded provenance path, or nothing to show; split raws are refused."""
    from braindecode.datasets import BaseConcatDataset

    vhdr = (
        emg_bids_root
        / "sub-893"
        / "ses-s1"
        / "emg"
        / "sub-893_ses-s1_task-fist_acq-right_emg.vhdr"
    )

    def raw_of(*names):
        return SimpleNamespace(filenames=names)

    ds = object.__new__(BaseConcatDataset)
    ds.datasets = [
        SimpleNamespace(
            raw=raw_of(vhdr.with_suffix(".eeg"))
        ),  # mne names the .eeg data file
        SimpleNamespace(
            raw=raw_of(None), description={"path": str(vhdr)}
        ),  # recorded provenance
        SimpleNamespace(
            raw=raw_of(None), description=pd.Series({"path": float("nan")})
        ),  # MOABB
        SimpleNamespace(raw=raw_of(vhdr, vhdr)),  # split raw
    ]
    for index in (0, 1):
        assert vhdr.name in ds.plot(index).data
    with pytest.raises(ValueError, match="not backed by a recording file"):
        ds.plot(2)
    with pytest.raises(ValueError, match="split recordings"):
        ds.plot(3)


def test_plot_forwards_its_keyword_arguments(emg_bids_root):
    ds = BIDSDataset(emg_bids_root, suffixes="emg", datatypes="emg")
    html = ds.plot(0, height=333, cdn_url="https://viewer.example.org/app").data
    assert "height:333px" in html
    assert 'frame.src = "https://viewer.example.org/app/index.html?embed=1"' in html
    with pytest.raises(ValueError, match="max_bytes=0.0 MiB"):
        ds.plot(0, max_bytes=1)
