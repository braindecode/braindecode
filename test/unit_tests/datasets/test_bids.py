# Authors: Pierre Guetschel <pierre.guetschel@gmail.com>
#
# License: BSD (3-clause)

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


# ===================== BIDSDataset.plot (eegdash-viewer) =====================


def _make_emg_bids_tree(tmp_path):
    """One minimal emg2pose-style recording: 2 emg + 1 joint-angle ch."""
    import mne
    import numpy as np

    root = tmp_path / "emg2pose-bids"
    ch_dir = root / "sub-893" / "ses-s1" / "emg"
    ch_dir.mkdir(parents=True)
    info = mne.create_info(["EMG1", "EMG2", "ja0"], 100, ["emg", "emg", "misc"])
    raw = mne.io.RawArray(np.random.randn(3, 50) * 1e-6, info, verbose="ERROR")
    mne.export.export_raw(
        ch_dir / "sub-893_ses-s1_task-fist_acq-right_emg.vhdr",
        raw,
        fmt="brainvision",
        overwrite=True,
        verbose="ERROR",
    )
    return root


@pytest.mark.parametrize("with_pose", [False, True])
def test_bids_dataset_plot_iframe(tmp_path, with_pose):
    root = _make_emg_bids_tree(tmp_path)
    ds = BIDSDataset(root, suffixes="emg", datatypes="emg")
    assert len(ds.datasets) == 1
    assert ds.bids_paths[0].suffix == "emg"

    if with_pose:  # optional hand-pose skeleton sidecar
        (
            root
            / "sub-893"
            / "ses-s1"
            / "emg"
            / "sub-893_ses-s1_task-fist_acq-right_desc-pose.json"
        ).write_text("{}")

    html = ds.plot(
        0,
        viewer_url="https://viewer.example.org",
        base_url="https://data.example.org/ds",
    ).data
    assert html.startswith('<iframe src="https://viewer.example.org/index.html?emg=')
    assert "data.example.org%2Fds%2Fsub-893" in html and "embed=1" in html
    assert ("pose=" in html) is with_pose
