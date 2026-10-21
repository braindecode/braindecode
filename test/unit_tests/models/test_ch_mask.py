"""Recordings with different numbers of channels in one batch (``ch_mask``).

Each model gets recordings padded to a common channel count and their
``ch_mask``, and must give every recording the output it gives alone,
whatever the padding holds.
"""

import mne
import numpy as np
import pytest
import torch

from braindecode.models import (
    DIVER1,
    BaRISTA,
    BrainBERT,
    Brant,
    EpiNT,
    MVPFormer,
    PopulationTransformer,
)
from braindecode.models.diver1 import channel_metadata_from_chs_info

N_REAL = [1, 3, 2]  # real channels of the three recordings, padded to 3


def _seeg_chs(n_chans, kind="seeg", seed=0):
    rng = np.random.default_rng(seed)
    info = mne.create_info([f"E{i}" for i in range(n_chans)], 500.0, kind)
    for ch in info["chs"]:
        ch["loc"][:3] = rng.uniform(-0.05, 0.05, 3)
    return info["chs"]


def _coords(i, n_chans):
    return torch.randint(0, 200, (n_chans, 3), generator=torch.Generator().manual_seed(i))


def _parcels(i, n_chans):
    return torch.randint(0, 121, (n_chans,), generator=torch.Generator().manual_seed(i))


def _diver1_metadata(i, n_chans):
    return channel_metadata_from_chs_info(_seeg_chs(n_chans, ("seeg", "ecog")[i % 2], i))


# name: (model, n_times, per-recording input (name, builder, batched alone) or None)
_CASES = {
    "BrainBERT": lambda: (BrainBERT(n_chans=2, n_outputs=3, n_times=2048), 2048, None),
    "BrainBERT-all-frames": lambda: (
        BrainBERT(n_chans=2, n_outputs=3, n_times=2048, pool_n_frames=None),
        2048,
        None,
    ),
    "EpiNT": lambda: (
        EpiNT(n_chans=2, n_outputs=3, n_times=512, patch_size=64, embed_dim=32,
              n_layers=1, n_heads=4, ffn_dim=64),
        512,
        None,
    ),
    "Brant": lambda: (
        Brant(n_chans=2, n_outputs=3, n_times=128, patch_size=32, embed_dim=32,
              ffn_dim=64, temporal_n_layers=1, spatial_n_layers=1, n_heads=4),
        128,
        None,
    ),
    "PopulationTransformer": lambda: (
        PopulationTransformer(n_chans=2, n_outputs=3, n_times=48, hidden_dim=32,
                              ffn_dim=64, n_layers=1, n_heads=4),
        48,
        ("coords", _coords, True),
    ),
    "BaRISTA": lambda: (
        BaRISTA(n_chans=2, n_outputs=3, n_times=128, spatial_scale="parcels",
                patch_size=32, d_model=16, n_layers=1, num_heads=2, cnn_depth=1,
                pooling="mean"),
        128,
        ("spatial_indices", _parcels, False),
    ),
    "DIVER1": lambda: (
        DIVER1(chs_info=_seeg_chs(2), n_outputs=3, n_times=200, patch_size=50,
               d_model=64, n_layers=1, pooling="mean"),
        200,
        ("chan_metadata", _diver1_metadata, False),
    ),
    "MVPFormer": lambda: (
        MVPFormer(n_chans=2, n_outputs=3, n_times=256, segment_len=64, d_model=32,
                  n_layers=1, n_heads=4, n_head_kv=2, d_inner=64, local_window=2,
                  max_channels=16),
        256,
        None,
    ),
}


@pytest.mark.parametrize("name", list(_CASES))
def test_ch_mask_matches_each_recording_alone(name):
    torch.manual_seed(0)
    model, n_times, per_recording = _CASES[name]()
    model.eval()
    xs = [torch.randn(n_real, n_times) for n_real in N_REAL]
    mask = torch.arange(max(N_REAL)) < torch.tensor(N_REAL)[:, None]
    x = torch.full((len(N_REAL), max(N_REAL), n_times), float("nan"))
    for i, xi in enumerate(xs):
        x[i, : len(xi)] = xi
    batch_kwargs, alone_kwargs = {}, [{} for _ in N_REAL]
    if per_recording is not None:
        key, build, batched = per_recording
        alone = [build(i, n_real) for i, n_real in enumerate(N_REAL)]
        padded = torch.zeros((len(N_REAL), max(N_REAL)) + alone[0].shape[1:])
        for i, value in enumerate(alone):
            padded[i, : len(value)] = value
        batch_kwargs = {key: padded.to(alone[0].dtype)}
        alone_kwargs = [{key: value[None] if batched else value} for value in alone]

    x.requires_grad_(True)
    out = model(x, ch_mask=mask, **batch_kwargs)
    out.sum().backward()
    assert torch.equal(x.grad[~mask], torch.zeros_like(x.grad[~mask]))
    assert all(p.grad.isfinite().all() for p in model.parameters() if p.grad is not None)
    with torch.no_grad():  # inference fast paths (e.g. nested tensors)
        expected = torch.cat(
            [model(xi[None], **kwargs) for xi, kwargs in zip(xs, alone_kwargs)]
        )
        torch.testing.assert_close(model(x, ch_mask=mask, **batch_kwargs), expected)
    torch.testing.assert_close(out.detach(), expected)
    with torch.no_grad():
        scripted = torch.jit.script(model)(x.detach(), ch_mask=mask, **batch_kwargs)
    torch.testing.assert_close(scripted, expected)


# Heads tied to the construction-time channels do not take a ch_mask.
_FIXED_HEADS = {
    "BaRISTA-learned": lambda: (
        BaRISTA(n_chans=3, n_outputs=3, n_times=128, spatial_scale="none",
                patch_size=32, d_model=16, n_layers=1, num_heads=2, cnn_depth=1,
                pooling="learned"),
        128,
    ),
    "DIVER1-flatten": lambda: (
        DIVER1(chs_info=_seeg_chs(3), n_outputs=3, n_times=200, patch_size=50,
               d_model=64, n_layers=1, pooling="flatten"),
        200,
    ),
    "MVPFormer-concat": lambda: (
        MVPFormer(n_chans=3, n_outputs=3, n_times=256, segment_len=64, d_model=32,
                  n_layers=1, n_heads=4, n_head_kv=2, d_inner=64, local_window=2,
                  max_channels=16, pooling="concat"),
        256,
    ),
}


@pytest.mark.parametrize("name", list(_FIXED_HEADS))
def test_ch_mask_is_refused_by_fixed_size_heads(name):
    model, n_times = _FIXED_HEADS[name]()
    x = torch.randn(len(N_REAL), 3, n_times)
    mask = torch.arange(3) < torch.tensor(N_REAL)[:, None]
    with pytest.raises(ValueError, match="ch_mask needs pooling='mean'"):
        model(x, ch_mask=mask)
    with pytest.raises(torch.jit.Error, match="ch_mask needs pooling='mean'"):
        torch.jit.script(model)(x, ch_mask=mask)


def test_diver1_ch_mask_needs_per_sample_metadata():
    model, n_times, (_, build, _) = _CASES["DIVER1"]()
    x = torch.randn(2, 2, n_times)
    mask = torch.tensor([[True, False], [True, True]])
    for metadata in (None, build(0, 2)):  # default montage, shared montage
        with pytest.raises(ValueError, match="per-sample chan_metadata"):
            model(x, chan_metadata=metadata, ch_mask=mask)
