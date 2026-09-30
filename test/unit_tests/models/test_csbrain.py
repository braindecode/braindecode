"""Targeted tests for the CSBrain port beyond the generic model tests."""

import pytest
import torch

from braindecode.models.csbrain import (
    REGION_CENTRAL,
    REGION_FRONTAL,
    CSBrain,
    build_region_attention_mask,
    derive_brain_regions,
    make_area_config,
    region_of_electrode,
)


def test_region_of_electrode_prefixes():
    assert region_of_electrode("Fpz") == REGION_FRONTAL
    assert region_of_electrode("AF7") == REGION_FRONTAL
    assert region_of_electrode("Fz") == REGION_FRONTAL
    assert region_of_electrode("FC3") == REGION_FRONTAL
    assert region_of_electrode("C3") == REGION_CENTRAL
    assert region_of_electrode("CPz") == REGION_CENTRAL
    assert region_of_electrode("P8") == 1  # parietal
    assert region_of_electrode("PO7") == 3  # occipital
    assert region_of_electrode("Oz") == 3
    assert region_of_electrode("T7") == 2  # temporal
    # Generic labels fall back to the central region instead of failing.
    assert region_of_electrode("EEG 021") == REGION_CENTRAL
    assert region_of_electrode("") == REGION_CENTRAL


def test_derive_brain_regions_sorts_regions_contiguous():
    names_2a = [
        "Fz",
        "FC3", "FC1", "FCz", "FC2", "FC4",
        "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
        "CP3", "CP1", "CPz", "CP2", "CP4",
        "P1", "Pz", "P2", "POz",
    ]
    chs_info = [{"ch_name": n, "kind": "eeg"} for n in names_2a]
    ordered, sorted_indices = derive_brain_regions(chs_info)

    # Regions are contiguous and ordered by identifier: frontal(6), parietal(3),
    # occipital(1, POz), central(12).
    region_sizes = [make_area_config(ordered)[k]["channels"] for k in sorted(make_area_config(ordered))]
    assert region_sizes == [6, 3, 1, 12]
    # Permutation brings the same channels back.
    assert sorted(sorted_indices) == list(range(len(names_2a)))
    # Inside the central region the original order is preserved.
    central_origin = [names_2a[i] for i in sorted_indices if ordered[sorted_indices.index(i)] == REGION_CENTRAL]
    assert central_origin == [
        "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
        "CP3", "CP1", "CPz", "CP2", "CP4",
    ]


def test_region_attention_mask_groups_electrodes():
    # Two regions of 2 electrodes each -> 2 groups of 2.
    area_config = {
        "region_0": {"channels": 2, "slice": slice(0, 2)},
        "region_4": {"channels": 2, "slice": slice(2, 4)},
    }
    mask = build_region_attention_mask(area_config, n_channels=4)
    assert mask.shape == (4, 4)
    # Each electrode attends to itself and exactly one electrode per region.
    n_allowed = (mask == 0).sum(dim=1)
    assert torch.all(n_allowed == 2)
    # Mask rows are symmetric groups (either both allowed or both blocked).
    assert torch.equal((mask == 0), (mask == 0).T)


def test_csbrain_without_channel_names_degenerates_to_full_attention():
    model = CSBrain(n_outputs=2, n_chans=4, n_times=400, sfreq=200.0, n_layer=1)
    assert model.sorted_indices is None
    assert model.area_config == {}
    out = model(torch.randn(2, 4, 400))
    assert out.shape == (2, 2)


def test_csbrain_forward_with_region_structure():
    names_2a = [
        "Fz", "FC3", "FC1", "FCz", "FC2", "FC4",
        "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
        "CP3", "CP1", "CPz", "CP2", "CP4",
        "P1", "Pz", "P2", "POz",
    ]
    chs_info = [{"ch_name": n, "kind": "eeg"} for n in names_2a]
    model = CSBrain(
        n_outputs=4, chs_info=chs_info, n_times=800, sfreq=200.0, n_layer=2
    )
    out = model(torch.randn(2, 22, 800))
    assert out.shape == (2, 4)
    feats = model(torch.randn(1, 22, 800), return_features=True)
    assert feats["features"].shape == (1, 22, 4, 200)


def test_csbrain_masked_forward_replaces_patches():
    model = CSBrain(n_outputs=2, n_chans=3, n_times=400, sfreq=200.0, n_layer=1)
    x = torch.randn(1, 3, 400)
    mask = torch.zeros(1, 3, 2, dtype=torch.bool)
    mask[:, :, 0] = True
    out = model(x, mask=mask)
    assert out.shape == (1, 2)


@pytest.mark.parametrize("batch_size", [1, 3])
def test_csbrain_train_mode_small_batches(batch_size):
    model = CSBrain(n_outputs=2, n_chans=3, n_times=400, sfreq=200.0, n_layer=1)
    model.train()
    out = model(torch.randn(batch_size, 3, 400))
    assert out.shape == (batch_size, 2)
