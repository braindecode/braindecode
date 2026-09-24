import mne
import pytest
import torch

from braindecode.models import BaRISTA

SMALL = dict(patch_size=32, d_model=16, n_layers=1, num_heads=2, cnn_depth=1)


def _model(spatial_scale, n_chans=4, pooling="mean", **kwargs):
    return BaRISTA(
        n_outputs=2,
        n_chans=n_chans,
        n_times=128,
        spatial_scale=spatial_scale,
        pooling=pooling,
        **SMALL,
        **kwargs,
    ).eval()


@pytest.mark.parametrize(
    "spatial_scale, indices",
    [("parcels", [1, 5, 120, 0]), ("lobes", [0, 3, 20, 7]), ("none", None)],
)
def test_region_scales(spatial_scale, indices):
    model = _model(spatial_scale, spatial_indices=indices)
    assert model(torch.randn(3, 4, 128)).shape == (3, 2)


def test_forward_indices_other_montage():
    """Mean pooling lets one model read recordings with other montages."""
    model = _model("parcels", spatial_indices=[1, 2, 3, 4])
    out = model(torch.randn(2, 6, 128), spatial_indices=torch.tensor([1, 2, 3, 4, 5, 6]))
    assert out.shape == (2, 2)


@pytest.mark.parametrize("bad", [[1, 2, 3, 121], [1, 2, 3, -1]])
def test_forward_rejects_out_of_range_indices(bad):
    model = _model("parcels", spatial_indices=[1, 2, 3, 4])
    with pytest.raises(ValueError, match=r"\[0, 121\)"):
        model(torch.randn(1, 4, 128), spatial_indices=torch.tensor(bad))


def test_learned_pooling_needs_construction_grid():
    model = _model("parcels", pooling="learned", spatial_indices=[1, 2, 3, 4])
    with pytest.raises(ValueError, match="pooling='mean'"):
        model(torch.randn(1, 6, 128), spatial_indices=torch.tensor([1, 2, 3, 4, 5, 6]))


def test_coords_fallback_uses_left_inferior_posterior_order():
    info = mne.create_info(["a"], 256.0, "seeg")
    info["chs"][0]["loc"][:3] = [0.010, 0.020, 0.030]  # RAS metres
    model = BaRISTA(
        n_outputs=2, chs_info=info["chs"], n_times=128, pooling="mean", **SMALL
    )
    left, inferior, posterior = model.spatial_emb.default_indices[0].tolist()
    centre = model.spatial_emb.tables[0].num_embeddings // 2
    assert (left, inferior, posterior) == (centre - 10, centre - 30, centre - 20)
