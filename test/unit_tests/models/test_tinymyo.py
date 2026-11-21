import pytest
import torch

from braindecode.models import TinyMyo


@pytest.mark.parametrize("kwargs", [{"n_chans": 0}, {"num_heads": 0}, {"embed_dim": 0}])
def test_tinymyo_requires_positive_geometry(kwargs):
    config = dict(
        n_chans=2, n_outputs=3, n_times=40, embed_dim=12, num_heads=2, depth=1
    )
    with pytest.raises(ValueError, match="n_chans|num_heads|embed_dim"):
        TinyMyo(**(config | kwargs))


@pytest.mark.network
@pytest.mark.huggingface
@pytest.mark.parametrize(
    "repo, revision, n_chans, n_outputs, expected",
    [
        ("tinymyo-pretrained", "d1d45075fe195e00b3432205dc9144ed37a449c4", 8, 4,
         [0.154599, 0.452840, 0.624193, -0.952561, 0.019963]),
        ("tinymyo-db5", "9f347a12a83927c927f9b9cad653d2f6c6f24a3e", 16, 53,
         [5.257892, -0.917425, -1.100922, -0.951650, -0.962309]),
        ("tinymyo-epn612", "ec76952479ef6d6f6bd551be5196e692820b9242", 8, 6,
         [0.077005, -1.300985, -0.276001, -1.294532, 0.148218]),
        ("tinymyo-uci-emg", "486089fe3db62d207f343a04283f19b420c6b447", 8, 6,
         [2.667200, -1.019451, -0.740182, -0.926584, -1.576798]),
    ],
)  # fmt: skip
def test_tinymyo_pretrained_matches_the_release(
    repo, revision, n_chans, n_outputs, expected
):
    """The re-hosted PulpBio/TinyMyo files (bc14906) give the outputs of the
    originals: the features of the encoder, whose head is not released, and the
    logits of the three gesture classifiers."""
    pytest.importorskip("huggingface_hub")
    model = TinyMyo.from_pretrained(
        f"braindecode/{repo}",
        revision=revision,
        n_chans=n_chans,
        n_outputs=n_outputs,
        n_times=1000,
    ).eval()
    x = torch.arange(n_chans * 1000.0).reshape(1, n_chans, 1000).mul(0.01).sin()
    encoder_only = repo == "tinymyo-pretrained"
    with torch.no_grad():
        out = model(x, return_features=encoder_only)
    out = out["features"] if encoder_only else out
    torch.testing.assert_close(
        out.flatten()[:5], torch.tensor(expected), rtol=0, atol=1e-5
    )
