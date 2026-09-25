# Authors: The braindecode contributors.
#
# License: BSD-3

import pytest
import torch

from braindecode.models import MAPA

SUBJECT_A = ["LA1", "LA2", "LA5", "LB3", "LB7"]
SUBJECT_B = ["RH2", "RH3", "RH8", "LT1", "LT2", "LT3", "LT4", "RX5"]


@pytest.fixture
def model():
    return MAPA(
        n_outputs=4,
        n_chans=len(SUBJECT_A),
        n_times=2048,
        sfreq=2048,
        contact_labels=SUBJECT_A,
        d_model=64,
    ).eval()


def test_sensor_indices_reads_array_and_contact_number():
    indices = MAPA.sensor_indices(SUBJECT_A, ["ctx-lh-insula"] + [None] * 4)
    assert indices.tolist() == [
        [0, 1, 7],
        [0, 2, 74],
        [0, 5, 74],
        [1, 3, 74],
        [1, 7, 74],
    ]


@pytest.mark.parametrize("n_times", [448, 2048, 4096])
def test_one_model_reads_another_subject(model, n_times):
    """A montage and a window the model was not built for both go through."""
    x = torch.randn(2, len(SUBJECT_B), n_times)
    with torch.no_grad():
        y = model(x, MAPA.sensor_indices(SUBJECT_B))
    assert y.shape == (2, 4)


def test_channel_order_does_not_change_the_output(model):
    perm = torch.tensor([4, 0, 3, 1, 2])
    x = torch.randn(2, len(SUBJECT_A), 2048)
    with torch.no_grad():
        expected = model(x)
        permuted = model(x[:, perm], model.default_sensor_indices[perm])
    torch.testing.assert_close(expected, permuted, atol=1e-5, rtol=1e-5)


def test_switching_subjects_does_not_leak_between_calls(model):
    """The cached token layout must not survive a change of montage."""
    xa = torch.randn(2, len(SUBJECT_A), 2048)
    xb = torch.randn(2, len(SUBJECT_B), 2048)
    indices_b = MAPA.sensor_indices(SUBJECT_B)
    with torch.no_grad():
        first_a, first_b = model(xa), model(xb, indices_b)
        again_b, again_a = model(xb, indices_b), model(xa)
    torch.testing.assert_close(first_a, again_a)
    torch.testing.assert_close(first_b, again_b)


def test_flatten_pooling_stays_tied_to_its_montage():
    model = MAPA(
        n_outputs=4,
        n_chans=len(SUBJECT_A),
        n_times=2048,
        sfreq=2048,
        contact_labels=SUBJECT_A,
        d_model=64,
        pooling="flatten",
    ).eval()
    with pytest.raises(ValueError, match="pooling='flatten'"):
        model(torch.randn(1, len(SUBJECT_B), 2048), MAPA.sensor_indices(SUBJECT_B))


@pytest.mark.parametrize(
    "sensor_indices,match",
    [
        (None, "got input with 8 channels"),
        (torch.zeros(3, 3, dtype=torch.long), r"shape \(8, 3\)"),
        (torch.zeros(8, 3), "must hold integers"),
        (torch.full((8, 3), 75), "region slots below 75"),
        (torch.full((8, 3), -1), "must be non-negative"),
    ],
)
def test_bad_sensor_indices_are_rejected(model, sensor_indices, match):
    with pytest.raises(ValueError, match=match):
        model(torch.randn(1, len(SUBJECT_B), 2048), sensor_indices)


def test_window_shorter_than_one_slow_token_is_rejected(model):
    with pytest.raises(ValueError, match="at least 448 samples"):
        model(torch.randn(1, len(SUBJECT_A), 256))
