"""Unsupported private conversion options must not silently disappear."""

from types import SimpleNamespace

import pytest

from braindecode.datasets import (
    BaseConcatDataset,
    EEGWindowsDataset,
    RawDataset,
    WindowsDataset,
)
from braindecode.datasets.bids import hub
from braindecode.datautil.hub_formats import load_from_zarr


@pytest.mark.parametrize("preload", [False, True])
@pytest.mark.parametrize("bounded", [False, True])
def test_public_hub_entry_rejects_bounded_option_before_download(
    tmp_path, monkeypatch, preload, bounded
):
    """Reviewer's offline probe: reject rather than silently use default slabs."""
    (tmp_path / "dataset.zarr").mkdir()
    downloads, received = [], []

    def download(**kwargs):
        downloads.append(kwargs)
        return str(tmp_path)

    def load(path, preload, **kwargs):
        received.append(kwargs)
        return SimpleNamespace(datasets=[])

    # Only replace IO/dependencies. The real public entry validates the option.
    monkeypatch.setattr(hub, "huggingface_hub", SimpleNamespace(snapshot_download=download))
    monkeypatch.setattr(hub, "zarr", object())
    monkeypatch.setattr(BaseConcatDataset, "_load_from_zarr_inline", staticmethod(load))
    monkeypatch.setattr(
        BaseConcatDataset, "_load_bids_metadata", staticmethod(lambda *args: None)
    )
    with pytest.raises(TypeError, match="bounded_conversion.*private.*converter"):
        BaseConcatDataset.pull_from_hub(
            "review/offline", preload=preload, bounded_conversion=bounded
        )
    assert downloads == []
    assert received == []


@pytest.mark.parametrize(
    "loader,args",
    [
        (load_from_zarr, ("absent.zarr",)),
        (BaseConcatDataset._load_from_zarr_inline, ("absent.zarr", False)),
        (RawDataset._from_zarr, ("absent.zarr", "recording_0", {}, {})),
        (EEGWindowsDataset._from_zarr, ("absent.zarr", "recording_0", None, {}, {})),
        (WindowsDataset._from_zarr, ("absent.zarr", "recording_0", None, {}, {})),
    ],
)
def test_other_zarr_loader_entries_reject_private_option(loader, args):
    """Closed signatures reject before filesystem access/metadata reconstruction."""
    with pytest.raises(TypeError, match="bounded_conversion"):
        loader(*args, bounded_conversion=True)
