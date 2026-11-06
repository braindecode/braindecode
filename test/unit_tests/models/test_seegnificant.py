# Authors: Julien Gadonneix <juliengado.2001@gmail.com>
#
# License: BSD-3

import pytest
import torch

from braindecode.models import SEEGnificant


def test_seegnificant_matches_the_reference():
    """Values computed with the authors' code (gmentz/seegnificant at 4e1e98c)
    for the same weights and inputs; electrodes 20-27 of the first window are
    padding."""
    torch.manual_seed(0)
    model = SEEGnificant(n_chans=28, n_times=600, n_outputs=1, n_subjects=2).eval()
    g = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(0.0, 0.3, generator=g)
    x = torch.randn(2, 28, 600, generator=g)
    pos = 0.04 * torch.randn(2, 28, 3, generator=g)
    x[0, 20:], pos[0, 20:] = 0.0, 0.0
    subject_ids = torch.tensor([1, 0])
    with torch.no_grad():
        features = model(x, pos=pos, subject_ids=subject_ids, return_features=True)
        out = model(x, pos=pos, subject_ids=subject_ids)
    torch.testing.assert_close(
        features["features"].norm(dim=1), torch.tensor([126.41360, 125.95853]),
        rtol=1e-5, atol=0,
    )
    torch.testing.assert_close(
        out.flatten(), torch.tensor([19.34845, 35.83667]), rtol=1e-5, atol=0
    )


def test_seegnificant_reads_each_window_with_its_subject_head():
    model = SEEGnificant(n_chans=3, n_times=600, n_outputs=2, n_subjects=3).eval()
    x, pos = torch.randn(4, 3, 600), 0.05 * torch.randn(3, 3)
    subject_ids = torch.tensor([2, 0, 1, 2])
    with torch.no_grad():
        features = model(x, pos=pos, return_features=True)["features"]
        out = model(x, pos=pos, subject_ids=subject_ids)
        heads = model.final_layer.heads
        expected = torch.stack(
            [heads[int(s)](f) for f, s in zip(features, subject_ids)]
        )
    torch.testing.assert_close(out, expected)
    with pytest.raises(ValueError, match="subject_ids is needed"):
        model(x, pos=pos)
