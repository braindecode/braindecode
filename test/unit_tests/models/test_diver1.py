# Authors: The braindecode contributors.
#
# License: BSD-3

import mne
import pytest
import torch

from braindecode.models import DIVER1

N_TIMES = 1000
SFREQ = 500.0


def make_chs_info(n_chans, kind="seeg", located=True):
    """Channel info of a montage of ``n_chans`` electrodes of one kind."""
    info = mne.create_info([f"{kind}{i}" for i in range(n_chans)], SFREQ, kind)
    if located:
        for i, ch in enumerate(info["chs"]):
            ch["loc"][:3] = [0.01 * i, 0.02, -0.03]
    return info["chs"]


SUBJECT_A = make_chs_info(6)
SUBJECT_B = make_chs_info(9, kind="ecog")


@pytest.fixture
def model():
    return DIVER1(
        n_outputs=4,
        chs_info=SUBJECT_A,
        n_times=N_TIMES,
        sfreq=SFREQ,
        pooling="mean",
        d_model=64,
        n_layers=2,
    ).eval()


def test_channel_metadata_reads_kind_and_position():
    metadata = DIVER1.channel_metadata(make_chs_info(2, kind="ecog"))
    assert metadata.shape == (2, 5)
    # Millimetres, from the metres MNE stores.
    torch.testing.assert_close(metadata[1, :3], torch.tensor([10.0, 20.0, -30.0]))
    # ECoG is intracranial (modality slot 1) and a grid (sub-modality slot 0).
    assert metadata[:, 3].tolist() == [1.0, 1.0]
    assert metadata[:, 4].tolist() == [0.0, 0.0]


def test_channel_metadata_flags_unknown_sub_modality_and_position():
    metadata = DIVER1.channel_metadata(make_chs_info(2, kind="eeg", located=False))
    # Scalp EEG has no electrode sub-modality, and no montage was set.
    assert metadata[:, 3].tolist() == [0.0, 0.0]
    assert (metadata[:, 4] < 0).all()
    assert torch.isnan(metadata[:, :3]).all()


def test_channel_metadata_rejects_an_undeterminable_modality():
    with pytest.raises(ValueError, match="cannot determine the recording modality"):
        DIVER1.channel_metadata(make_chs_info(2, kind="misc"))


def test_one_model_reads_another_subject(model):
    """A montage the model was not built for goes through."""
    x = torch.randn(2, len(SUBJECT_B), N_TIMES)
    with torch.no_grad():
        y = model(x, DIVER1.channel_metadata(SUBJECT_B))
    assert y.shape == (2, 4)


def test_passing_the_default_montage_changes_nothing(model):
    x = torch.randn(2, len(SUBJECT_A), N_TIMES)
    with torch.no_grad():
        implicit = model(x)
        explicit = model(x, DIVER1.channel_metadata(SUBJECT_A))
    torch.testing.assert_close(implicit, explicit)


def test_channel_order_does_not_change_the_output(model):
    perm = torch.tensor([4, 0, 3, 1, 5, 2])
    x = torch.randn(2, len(SUBJECT_A), N_TIMES)
    with torch.no_grad():
        expected = model(x)
        permuted = model(x[:, perm], model.default_chan_metadata[perm])
    torch.testing.assert_close(expected, permuted, atol=1e-5, rtol=1e-5)


def test_switching_subjects_does_not_leak_between_calls(model):
    xa = torch.randn(2, len(SUBJECT_A), N_TIMES)
    xb = torch.randn(2, len(SUBJECT_B), N_TIMES)
    metadata_b = DIVER1.channel_metadata(SUBJECT_B)
    with torch.no_grad():
        first_a, first_b = model(xa), model(xb, metadata_b)
        again_b, again_a = model(xb, metadata_b), model(xa)
    torch.testing.assert_close(first_a, again_a)
    torch.testing.assert_close(first_b, again_b)


def test_flatten_pooling_stays_tied_to_its_montage():
    model = DIVER1(
        n_outputs=4,
        chs_info=SUBJECT_A,
        n_times=N_TIMES,
        sfreq=SFREQ,
        pooling="flatten",
        d_model=64,
        n_layers=2,
    ).eval()
    metadata_b = DIVER1.channel_metadata(SUBJECT_B)
    with pytest.raises(ValueError, match="pooling='flatten'"):
        model(torch.randn(1, len(SUBJECT_B), N_TIMES), metadata_b)


@pytest.mark.parametrize(
    "chan_metadata,match",
    [
        (None, "built for 6 channels but got input with 9"),
        (torch.zeros(3, 5), r"shape \(9, 5\)"),
        (torch.zeros(9, 4), r"shape \(9, 5\)"),
        (torch.full((9, 5), 7.0), "modality column"),
        (torch.zeros(9, 5).index_fill_(1, torch.tensor([4]), 3.0), "sub-modality"),
    ],
)
def test_bad_channel_metadata_is_rejected(model, chan_metadata, match):
    with pytest.raises(ValueError, match=match):
        model(torch.randn(1, len(SUBJECT_B), N_TIMES), chan_metadata)
