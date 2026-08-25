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
    """Make a minimal dataset stub for testing URL generation only."""
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


def _plot(dataset):
    return dataset.plot(
        0,
        viewer_url="https://viewer.example.org",
        base_url="https://data.example.org/ds",
    )


@pytest.mark.parametrize("with_pose", [False, True])
def test_bids_dataset_plot_iframe(tmp_path, with_pose):
    dataset, recording = _make_plot_dataset(tmp_path)

    if with_pose:  # optional hand-pose skeleton sidecar
        recording.with_name(
            "sub-893_ses-s1_task-fist_acq-right_desc-pose.json"
        ).write_text("{}")

    html = _plot(dataset).data
    assert html.startswith('<iframe src="https://viewer.example.org/index.html?emg=')
    assert "data.example.org%2Fds%2Fsub-893" in html and "embed=1" in html
    assert ("pose=" in html) is with_pose


def test_bids_dataset_plot_preserves_symlink_path(tmp_path):
    dataset, recording = _make_plot_dataset(tmp_path, symlink_recording=True)
    recording.with_name("sub-893_ses-s1_task-fist_acq-right_desc-pose.json").write_text(
        "{}"
    )

    html = _plot(dataset).data

    assert "data.example.org%2Fds%2Fsub-893%2Fses-s1%2Femg" in html
    assert "pose=" in html
    assert "external" not in html


def test_make_plot_dataset_skips_when_symlinks_are_unavailable(
    tmp_path, monkeypatch
):
    def _raise_permission_error(*args, **kwargs):
        raise OSError("symlink privilege is unavailable")

    monkeypatch.setattr(Path, "symlink_to", _raise_permission_error)

    with pytest.raises(pytest.skip.Exception, match="symlinks are unavailable"):
        _make_plot_dataset(tmp_path, symlink_recording=True)


def test_bids_dataset_plot_is_defined_on_class():
    assert BIDSDataset.plot.__qualname__ == "BIDSDataset.plot"


def test_bids_dataset_plot_explains_missing_ipython(tmp_path, monkeypatch):
    dataset, _ = _make_plot_dataset(tmp_path)
    monkeypatch.setitem(sys.modules, "IPython.display", None)

    with pytest.raises(ImportError, match="BIDSDataset.plot requires IPython"):
        _plot(dataset)


@pytest.mark.parametrize(
    ("viewer_url", "data_url"),
    [
        ("javascript:alert(1)", "https://data.example.org/ds"),
        ("https://viewer.example.org", "file:///tmp/data"),
        (
            "https://viewer.example.org/app?theme=dark",
            "https://data.example.org/ds",
        ),
        ("https://viewer.example.org", "https://data.example.org/ds#recording"),
    ],
)
def test_bids_dataset_plot_rejects_non_web_urls(tmp_path, viewer_url, data_url):
    dataset, _ = _make_plot_dataset(tmp_path)

    with pytest.raises(ValueError, match="http or https URL"):
        dataset.plot(0, viewer_url=viewer_url, base_url=data_url)


def test_bids_dataset_plot_escapes_viewer_url(tmp_path):
    dataset, _ = _make_plot_dataset(tmp_path)

    html = dataset.plot(
        0,
        viewer_url='https://viewer.example.org/"><script>alert(1)</script>',
        base_url="https://data.example.org/ds",
    ).data

    assert "<script>" not in html
    assert "&quot;&gt;&lt;script&gt;" in html


@pytest.mark.parametrize(
    ("viewer_url", "data_url"),
    [
        ("https://viewer.example.org/app?", "https://data.example.org/ds"),
        ("https://viewer.example.org/app", "https://data.example.org/ds#"),
    ],
)
def test_bids_dataset_plot_normalizes_empty_url_delimiters(
    tmp_path, viewer_url, data_url
):
    dataset, _ = _make_plot_dataset(tmp_path)

    html = dataset.plot(0, viewer_url=viewer_url, base_url=data_url).data

    assert html.startswith(
        '<iframe src="https://viewer.example.org/app/index.html?emg='
    )
    assert "data.example.org%2Fds%2Fsub-893" in html
