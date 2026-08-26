# Authors: Pierre Guetschel <pierre.guetschel@gmail.com>
#
# License: BSD (3-clause)

import sys
from pathlib import Path
from types import SimpleNamespace

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
    assert len(dataset.bids_paths) == 1
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


def _make_plot_dataset(tmp_path, *, symlink_recording=False):
    """Minimal dataset stub with one (empty) BrainVision recording."""
    root = tmp_path / "emg2pose-bids"
    ch_dir = root / "sub-893" / "ses-s1" / "emg"
    ch_dir.mkdir(parents=True)
    recording = ch_dir / "sub-893_ses-s1_task-fist_acq-right_emg.vhdr"
    if symlink_recording:
        target = tmp_path / "external" / recording.name
        target.parent.mkdir()
        target.touch()
        try:
            recording.symlink_to(target)
        except OSError as error:
            pytest.skip(f"symlinks are unavailable: {error}")
    else:
        recording.touch()

    dataset = object.__new__(BIDSDataset)
    dataset.root = root
    dataset.bids_paths = [SimpleNamespace(fpath=recording)]
    return dataset, recording


@pytest.fixture
def plot_dataset(tmp_path, request):
    """``(dataset, recording)``; ``request.param`` is {"symlink", "pose"} ⊆ options."""
    opts = set(getattr(request, "param", ()))
    dataset, recording = _make_plot_dataset(
        tmp_path, symlink_recording="symlink" in opts
    )
    if "pose" in opts:
        recording.with_name(
            "sub-893_ses-s1_task-fist_acq-right_desc-pose.json"
        ).write_text("{}")
    return dataset, recording


@pytest.mark.parametrize(
    "plot_dataset",
    [(), ("pose",), ("symlink", "pose")],
    indirect=True,
    ids=["plain", "pose", "symlink+pose"],
)
def test_bids_dataset_plot_inlines_the_bids_name(plot_dataset, request):
    dataset, recording = plot_dataset
    html = dataset.plot(0).data
    assert 'src="https://eegdash.github.io/eegdash-viewer/index.html?embed=1"' in html
    assert (
        recording.name in html and "external" not in html
    )  # BIDS name, never the symlink target
    assert "localhost" not in html and "127.0.0.1" not in html
    assert ('"pose": "e30="' in html) is (
        "pose" in request.node.callspec.id
    )  # base64 of "{}"


@pytest.mark.parametrize(
    ("cdn_url", "expect"),
    [
        ("javascript:alert(1)", ValueError),
        ("file:///tmp/viewer", ValueError),
        ("https://viewer.example.org/app?theme=dark", ValueError),
        ("https://viewer.example.org#recording", ValueError),
        (
            "https://viewer.example.org/app?",
            'src="https://viewer.example.org/app/index.html?embed=1"',
        ),
        (
            "https://viewer.example.org/app#",
            'src="https://viewer.example.org/app/index.html?embed=1"',
        ),
        (
            'https://viewer.example.org/"><script>alert(1)</script>',
            "&quot;&gt;&lt;script&gt;",
        ),
    ],
    ids=[
        "javascript",
        "file",
        "query",
        "fragment",
        "empty-query",
        "empty-fragment",
        "escaped",
    ],
)
def test_bids_dataset_plot_cdn_url_handling(plot_dataset, cdn_url, expect):
    dataset, _ = plot_dataset
    if expect is ValueError:
        with pytest.raises(ValueError, match="cdn must be"):
            dataset.plot(0, cdn_url=cdn_url)
    else:
        html = dataset.plot(0, cdn_url=cdn_url).data
        assert expect in html and "<script>alert" not in html


def test_make_plot_dataset_skips_when_symlinks_are_unavailable(tmp_path, monkeypatch):
    monkeypatch.setattr(
        Path,
        "symlink_to",
        lambda *a, **k: (_ for _ in ()).throw(OSError("no symlinks")),
    )
    with pytest.raises(pytest.skip.Exception, match="symlinks are unavailable"):
        _make_plot_dataset(tmp_path, symlink_recording=True)


def test_bids_dataset_plot_explains_missing_ipython(plot_dataset, monkeypatch):
    dataset, _ = plot_dataset
    monkeypatch.setitem(sys.modules, "IPython", None)  # soft import: not declared
    # mne's _soft_import raises RuntimeError with the pip hint
    with pytest.raises(
        RuntimeError, match=r"(?s)BIDSDataset.plot\(\).*IPython.*pip install IPython"
    ):
        dataset.plot(0)


def test_plot_comes_with_base_concat_dataset(plot_dataset):
    """Any BaseConcatDataset plots: bidspath elements (eegdash) or file-backed raws."""
    from braindecode.datasets import BaseConcatDataset
    from braindecode.datasets._notebook_viewer import ViewerMixin

    assert issubclass(BaseConcatDataset, ViewerMixin)
    _, recording = plot_dataset
    touched = []

    class LazyRecord:  # eegdash-style element: BIDSPath + download-on-access raw
        bidspath = SimpleNamespace(
            fpath=recording.with_name("sub-893_task-later_emg.vhdr")
        )

        @property
        def raw(self):
            touched.append("download")
            self.bidspath.fpath.write_bytes(b"")
            return None

    class FileBacked:  # any RawDataset whose mne Raw came from a file
        raw = SimpleNamespace(filenames=(str(recording),))

    class MemoryOnly:
        raw = SimpleNamespace(filenames=())

    ds = object.__new__(BaseConcatDataset)
    ds.datasets = [LazyRecord(), FileBacked(), MemoryOnly()]
    assert LazyRecord.bidspath.fpath.name in ds.plot(0).data and touched == ["download"]
    assert recording.name in ds.plot(1).data
    with pytest.raises(ValueError, match="not backed by a recording file"):
        ds.plot(2)


def test_bids_dataset_plot_real_tree_inlines_trio_and_inherited_sidecars(tmp_path):
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
    (root / "sub-893" / "sub-893_ses-s1_events.tsv").write_text(
        "onset\tduration\ttrial_type\n0.1\t0.0\tfist\n"
    )
    html = BIDSDataset(root, suffixes="emg", datatypes="emg").plot(0).data
    for name in (
        "sub-893_ses-s1_task-fist_acq-right_emg.vhdr",
        "sub-893_ses-s1_task-fist_acq-right_emg.eeg",
        "sub-893_ses-s1_task-fist_acq-right_emg.vmrk",
        "sub-893_ses-s1_events.tsv",  # inherited from the session level
    ):
        assert name in html
