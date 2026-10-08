# Authors: Bruno Aristimunha <b.aristimunha@gmail.com>
#
# License: BSD-3
"""BrainOmni / BrainTokenizer behaviour the shared model suites do not cover."""

from pathlib import Path

import mne
import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from mne.io.constants import FIFF

from braindecode.models import BrainOmni, BrainTokenizer
from braindecode.models.brainomni import (
    _geometry_from_chs_info,
    _RotaryPositionalEmbedding,
    _SpatialTemporalBlock,
)
from braindecode.modules.quantization import EMACodebook as _Codebook
from braindecode.modules.quantization import ResidualVectorQuantizer

_SMALL = dict(
    emb_dim=16,
    n_neuro=3,
    n_filters=8,
    codebook_dim=16,
    codebook_size=32,
    num_quantizers=2,
    tokenizer_num_heads=4,
)


def _loc(*values):
    loc = np.zeros(12)
    loc[: len(values)] = values
    return loc


def _eeg_chs_info(n):
    rng = np.random.default_rng(0)
    return [
        {"ch_name": f"C{i}", "kind": "eeg", "loc": _loc(*rng.random(3))}
        for i in range(n)
    ]


def _small_tokenizer(n_times=512, chs_info=None, **kwargs):
    chs_info = _eeg_chs_info(4) if chs_info is None else chs_info
    kw = _SMALL | kwargs
    return BrainTokenizer(chs_info=chs_info, n_times=n_times, sfreq=256.0, **kw)


def _small_brainomni(chs_info=None, sfreq=256.0, **kwargs):
    return BrainOmni(
        chs_info=_eeg_chs_info(4) if chs_info is None else chs_info,
        n_outputs=3,
        n_times=512,
        sfreq=sfreq,
        **(_SMALL | dict(lm_dim=16, num_heads=4, depth=2) | kwargs),
    )


def _meg_info_chs(n_triplets=4, n_eeg=3, axial_coil=None, plain_int=False):
    """MNE ``info["chs"]``: VectorView triplets (or ``axial_coil``) plus EEG."""
    names, types, coils = [], [], []
    for i in range(n_triplets):
        names += [f"MEG{i:03d}{j}" for j in (1, 2, 3)]
        if axial_coil is None:
            types += ["mag", "grad", "grad"]
            coils += [
                FIFF.FIFFV_COIL_VV_MAG_T3,
                FIFF.FIFFV_COIL_VV_PLANAR_T1,
                FIFF.FIFFV_COIL_VV_PLANAR_T1,
            ]
        else:
            types += ["mag"] * 3  # MNE stores axial gradiometers with unit T
            coils += [axial_coil] * 3
    names += [f"EEG{i:03d}" for i in range(n_eeg)]
    types += ["eeg"] * n_eeg
    coils += [FIFF.FIFFV_COIL_EEG] * n_eeg
    info = mne.create_info(names, 256.0, types)
    rng = np.random.default_rng(7)
    for ch, coil in zip(info["chs"], coils):
        ch["coil_type"] = int(coil) if plain_int else coil
        if plain_int:
            ch["kind"] = int(ch["kind"])
        direction = rng.normal(size=3)
        direction /= np.linalg.norm(direction)
        ex = np.cross(direction, [0.0, 0.0, 1.0])
        ex /= np.linalg.norm(ex)
        ch["loc"] = np.concatenate([0.1 * direction, ex, np.cross(direction, ex), direction])
    return info["chs"]


# ---- geometry ------------------------------------------------------------------


@pytest.mark.parametrize(
    "ch, expected",
    [
        ({"kind": "eeg"}, 0),
        ({"ch_type": "eeg"}, 0),
        ({"kind": "grad", "coil_type": 3012}, 2),
        ({"kind": "mag", "coil_type": 3022}, 1),
    ],
)
def test_geometry_sensor_type(ch, expected):
    ch = {"ch_name": "A", "loc": _loc(1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0, 0, 0, 0, 0, 1)} | ch
    assert _geometry_from_chs_info([ch])[1].tolist() == [expected]


@pytest.mark.parametrize(
    "kind, coil_type, axes",
    [("grad", 3012, slice(3, 6)), ("grad", 5001, slice(9, 12)), ("mag", 3022, slice(9, 12))],
    ids=["planar_grad", "ctf_axial_grad", "mag"],
)
def test_geometry_meg_orientation_axis(kind, coil_type, axes):
    loc = _loc(0.1, 0.2, 0.3)
    loc[axes] = (0.4, 0.5, 0.6)
    ch = {"ch_name": "M1", "kind": kind, "coil_type": coil_type, "loc": loc}
    np.testing.assert_allclose(_geometry_from_chs_info([ch])[0][0, 3:], (0.4, 0.5, 0.6))
    loc[axes] = np.nan
    with pytest.raises(ValueError, match="finite MEG coil orientation"):
        _geometry_from_chs_info([ch])


@pytest.mark.parametrize("plain_int", [False, True], ids=["named_int", "plain_int"])
def test_geometry_vectorview_meg_info(plain_int):
    chs = _meg_info_chs(plain_int=plain_int)
    pos, sensor_type = _geometry_from_chs_info(chs)
    assert sensor_type.tolist() == [1, 2, 2] * 4 + [0] * 3
    np.testing.assert_allclose(pos[-3:, 3:], 0.0)
    for mask in (sensor_type == 0, sensor_type > 0):  # EEG and MEG each normalized
        xyz = pos[mask, :3]
        np.testing.assert_allclose(xyz.mean(axis=0), 0.0, atol=1e-6)
        assert np.isclose(np.sqrt(3 * np.mean(np.sum(xyz**2, axis=1))), 1.0)


@pytest.mark.parametrize(
    "coil, expected",
    [(FIFF.FIFFV_COIL_CTF_GRAD, 2), (FIFF.FIFFV_COIL_KIT_GRAD, 2), (FIFF.FIFFV_COIL_MAGNES_GRAD, 1)],
    ids=["ctf_5001", "kit_6001", "magnes_4002"],
)
@pytest.mark.parametrize("plain_int", [False, True], ids=["named_int", "plain_int"])
def test_geometry_axial_gradiometer_types_follow_release(plain_int, coil, expected):
    # MNE types axial gradiometers "mag"; the release says GRAD unless the coil
    # name contains "MAG" (FIFFV_COIL_MAGNES_GRAD).
    chs = _meg_info_chs(n_eeg=0, axial_coil=coil, plain_int=plain_int)
    assert _geometry_from_chs_info(chs)[1].tolist() == [expected] * len(chs)


@pytest.mark.parametrize("loc", [np.full(12, np.nan), None], ids=["nan", "absent"])
def test_geometry_rejects_bad_channels(loc):
    ch = {"ch_name": "A", "kind": "eeg"} | ({} if loc is None else {"loc": loc})
    with pytest.raises(ValueError, match="set_montage"):
        _geometry_from_chs_info([ch])
    chs = _meg_info_chs(n_triplets=1, n_eeg=0)
    chs[0]["kind"] = FIFF.FIFFV_REF_MEG_CH
    with pytest.raises(ValueError, match="Unsupported channel type"):
        _geometry_from_chs_info(chs)


@pytest.mark.parametrize("axial_coil", [None, FIFF.FIFFV_COIL_CTF_GRAD], ids=["vv", "ctf"])
def test_forward_on_meg_and_eeg_info(axial_coil):
    chs = _meg_info_chs(axial_coil=axial_coil)
    x = torch.randn(2, len(chs), 512)
    assert _small_brainomni(chs_info=chs).eval()(x).shape == (2, 3)
    assert _small_tokenizer(chs_info=chs).eval()(x).shape == x.shape


# ---- quantizer -------------------------------------------------------------------


def _distributed_codebook_update(rank, world_size, init_file, output_dir):
    """One EMA update with different data on each CPU rank."""
    dist.init_process_group(
        backend="gloo", init_method=f"file://{init_file}", rank=rank, world_size=world_size
    )
    try:
        codebook = _Codebook(dim=2, codebook_size=2, decay=0.5, threshold_ema_dead_code=0)
        codebook.inited.fill_(1)
        codebook.cluster_size.copy_(torch.tensor([4.0, 4.0]))
        codebook.embed.copy_(torch.tensor([[-1.0, 0.0], [1.0, 0.0]]))
        codebook.embed_avg.copy_(torch.tensor([[-4.0, 0.0], [4.0, 0.0]]))
        samples = torch.tensor([[[-2.0, 0.0], [-1.0, 0.0]]] if rank == 0 else [[[1.0, 0.0], [2.0, 0.0]]])
        codebook.train()(samples)
        torch.manual_seed(rank)
        fresh = _Codebook(
            dim=2, codebook_size=2, decay=0.5, threshold_ema_dead_code=0, kmeans_iters=2
        )
        fresh.train()(samples)
        state = dict(codebook.state_dict())
        state |= {f"fresh_{k}": v for k, v in fresh.state_dict().items()}
        torch.save(state, Path(output_dir) / f"rank-{rank}.pt")
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed is unavailable")
def test_quantizer_distributed_ema_uses_global_statistics(tmp_path):
    mp.spawn(
        _distributed_codebook_update,
        args=(2, str(tmp_path / "init"), str(tmp_path)),
        nprocs=2,
        join=True,
    )
    states = [torch.load(tmp_path / f"rank-{r}.pt", weights_only=True) for r in (0, 1)]
    for key in states[0]:
        torch.testing.assert_close(states[0][key], states[1][key])
    torch.testing.assert_close(states[0]["cluster_size"], torch.tensor([3.0, 3.0]))
    torch.testing.assert_close(
        states[0]["embed"], torch.tensor([[-7.0 / 6.0, 0.0], [7.0 / 6.0, 0.0]])
    )
    assert states[0]["fresh_inited"].item() == 1


def _broadcast_kmeans_reference(codebook, samples):
    """``EMACodebook._kmeans`` before #1239: explicit squared distances."""
    samples = samples.reshape(-1, samples.shape[-1])
    dim, dtype = samples.shape[1], samples.dtype
    if samples.shape[0] < codebook.codebook_size:
        noise = torch.randn(codebook.codebook_size - samples.shape[0], dim, dtype=dtype)
        samples = torch.cat([samples, noise], dim=0)
    centers = codebook._sample_vectors(samples, codebook.codebook_size)
    bins = torch.ones(codebook.codebook_size, dtype=torch.long)
    for _ in range(codebook.kmeans_iters):
        distances = torch.cat(
            [((c.unsqueeze(1) - centers) ** 2).sum(dim=-1) for c in samples.split(256)]
        )
        buckets = distances.argmin(dim=-1)
        bins = torch.bincount(buckets, minlength=codebook.codebook_size)
        empty = bins == 0
        bins[empty] = 1
        new_centers = torch.zeros(codebook.codebook_size, dim, dtype=dtype)
        new_centers.scatter_add_(0, buckets.unsqueeze(-1).expand(-1, dim), samples)
        new_centers = new_centers / bins.unsqueeze(-1)
        centers = torch.where(empty.unsqueeze(-1), centers, new_centers)
    return centers, bins


@pytest.mark.parametrize(
    "make_samples",
    [
        lambda: torch.randn(600, 16),
        lambda: torch.randint(-2, 3, (600, 4)).float(),  # many exact distance ties
        lambda: torch.randint(0, 2, (20, 3)).float(),  # fewer samples than codes
    ],
    ids=["gaussian", "tie_heavy_grid", "fewer_samples_than_codes"],
)
def test_codebook_kmeans_matches_broadcast_reference(make_samples):
    codebook = _Codebook(dim=16, codebook_size=32, kmeans_iters=10)
    torch.manual_seed(0)
    samples = make_samples()
    torch.manual_seed(1)
    centers, bins = codebook._kmeans(samples)
    torch.manual_seed(1)
    ref_centers, ref_bins = _broadcast_kmeans_reference(codebook, samples)
    torch.testing.assert_close(bins, ref_bins, rtol=0, atol=0)
    torch.testing.assert_close(centers, ref_centers, rtol=0, atol=0)


@pytest.mark.parametrize("train", [True, False])
def test_quantizer_codebook_ema_only_in_train(train):
    torch.manual_seed(0)
    quantizer = ResidualVectorQuantizer(16, 16, 32, 2).train(train)
    codebook = quantizer.layers[0]._codebook
    quantizer(torch.randn(8, 10, 16))  # K-means initialization of a fresh codebook
    before = codebook.embed.clone()
    quantizer(torch.randn(8, 10, 16))
    assert (not torch.equal(before, codebook.embed)) is train


# ---- BrainTokenizer ---------------------------------------------------------------


@pytest.mark.parametrize("n_times", [300, 512, 600])
def test_braintokenizer_reconstruction_zero_fills_dropped_tail(n_times):
    model = _small_tokenizer(n_times=n_times).eval()
    reconstruction = model(torch.randn(1, 4, n_times))
    assert reconstruction.shape == (1, 4, n_times)
    assert torch.count_nonzero(reconstruction[..., 512:]) == 0


@pytest.mark.parametrize(
    "n_times, overlap_ratio, n_windows",
    [(300, 0.0, 1), (600, 0.0, 1), (600, 0.25, 2), (1024, 0.25, 3)],
)
def test_braintokenizer_windows_follow_release(n_times, overlap_ratio, n_windows):
    """Short input padded, non-overlap tail dropped, overlap tail padded."""
    feat, indices = _small_tokenizer().tokenize(
        torch.randn(1, 4, n_times), overlap_ratio=overlap_ratio
    )
    assert feat.shape == (1, 3, n_windows * 8, 16)  # 8 tokens per 512 samples
    assert indices.shape == (1, 3, n_windows * 8, 2)


def test_braintokenizer_first_training_forward_keeps_codebooks_bounded():
    torch.manual_seed(0)
    model = _small_tokenizer().train()
    codebook = model.quantizer.layers[0]._codebook
    assert torch.isfinite(model(torch.randn(2, 4, 512))).all()
    assert codebook.inited.item() == 1
    assert codebook.embed.norm(dim=-1).max() < 1.1


@pytest.mark.parametrize("window_length, ratios", [(5, (5,)), (1, (1,))])
def test_braintokenizer_odd_ratio_and_one_sample_window(window_length, ratios):
    model = _small_tokenizer(
        n_times=window_length, window_length=window_length, ratios=ratios
    ).eval()
    x = torch.randn(1, 4, window_length)
    assert model(x).shape == x.shape


def test_braintokenizer_crops_each_window_when_hop_does_not_divide_it():
    """SEANet decodes 12 samples for a 10-sample window at hop 4; each window is
    cropped before joining, so window 2 lands at samples 10-19."""
    torch.manual_seed(0)
    model = _small_tokenizer(n_times=30, window_length=10, ratios=(4,)).eval()
    x = torch.randn(1, 4, 30)
    with torch.no_grad():
        full = model(x)
        reconstruction, _, _ = model.encode_decode(x)
        window = model(x[..., 10:20])
    torch.testing.assert_close(full[..., 10:20], window)
    torch.testing.assert_close(reconstruction, full, rtol=0, atol=0)


def test_braintokenizer_tokenize_restores_train_mode():
    model = _small_tokenizer().train()
    feat, indices = model.tokenize(torch.randn(2, 4, 512))
    assert model.training and not feat.requires_grad
    assert feat.shape == (2, 3, 8, 16) and indices.shape == (2, 3, 8, 2)


@pytest.mark.parametrize(
    "build, match",
    [
        (lambda: _small_tokenizer(n_filters=1), "n_filters"),
        (lambda: _small_tokenizer(emb_dim=15), "tokenizer_num_heads"),
        (lambda: _small_brainomni(overlap_ratio=1.0), "overlap_ratio"),
        (lambda: _small_brainomni(overlap_ratio=-0.1), "overlap_ratio"),
        (lambda: _small_brainomni(num_heads=3), "num_heads"),
        (lambda: _small_brainomni(lm_dim=12, num_heads=4), "num_heads"),
    ],
    ids=["n_filters", "emb_dim", "overlap_1", "overlap_neg", "odd_heads", "odd_head_dim"],
)
def test_rejects_invalid_arguments(build, match):
    with pytest.raises(ValueError, match=match):
        build()


@pytest.mark.parametrize(
    "build",
    [
        lambda: _small_tokenizer(tokenizer_num_heads=0),
        lambda: _small_tokenizer(emb_dim=0),
        lambda: _small_brainomni(num_heads=0),
    ],
    ids=["tokenizer_heads", "emb_dim", "lm_heads"],
)
def test_rejects_non_positive_head_dims(build):
    with pytest.raises(ValueError, match="must be positive"):
        build()


@pytest.mark.parametrize("overlap_ratio", [-0.1, 1.0, 0.999])
def test_braintokenizer_tokenize_rejects_bad_overlap(overlap_ratio):
    with pytest.raises(ValueError, match="overlap_ratio"):
        _small_tokenizer().tokenize(torch.randn(1, 4, 512), overlap_ratio)


def test_sfreq_warning():
    with pytest.warns(UserWarning, match="256 Hz"):
        _small_brainomni(sfreq=128.0)


# ---- RoPE and attention ---------------------------------------------------------


def _complex_rope_reference(x, n_dim, rotate=None):
    """Released RoPE: one complex frequency ladder split across heads."""
    if rotate is None:
        freqs = 1.0 / (10000 ** (torch.arange(0, n_dim, 2).float() / n_dim))
        angles = torch.outer(torch.arange(x.shape[1]).float(), freqs)
        rotate = torch.polar(torch.ones_like(angles), angles)
    rotate = rotate[: x.shape[1]].reshape(x.shape[1], x.shape[2], -1).unsqueeze(0)
    x_ = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    return torch.view_as_real(x_ * rotate).flatten(3).type_as(x)


@pytest.mark.parametrize("seq", [1, 7, 300])
def test_rope_matches_released_complex_rotation(seq):
    rope = _RotaryPositionalEmbedding(n_dim=32)
    q = torch.randn(2, seq, 4, 8)
    torch.testing.assert_close(rope(q, q)[0], _complex_rope_reference(q, 32))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_rope_cast_keeps_float32_cache_and_rebuilds_in_dtype(dtype):
    """As released: the cache stays float32, ``freqs`` follows the cast."""
    rope = _RotaryPositionalEmbedding(n_dim=32).to(dtype)
    assert rope.freqs.dtype == dtype and rope.rotate.dtype == torch.float32
    q = torch.randn(2, 9, 4, 8).to(dtype)
    torch.testing.assert_close(rope(q, q)[0], _complex_rope_reference(q, 32))
    q = torch.randn(2, 300, 4, 8).to(dtype)
    angles = torch.outer(torch.arange(300).type_as(rope.freqs), rope.freqs).float()
    released = torch.polar(torch.ones_like(angles), angles)
    torch.testing.assert_close(
        rope(q, q)[0], _complex_rope_reference(q, 32, rotate=released)
    )


def test_rope_released_cosine_only_cache_then_rebuilt():
    """Released cache (cosines, zero sine) up to 240 positions; past it the
    cache is rebuilt from ``freqs`` and kept; a reload restores the loaded one."""
    exact = 1.0 / (10000 ** (torch.arange(0, 32, 2).float() / 32))
    cosines = torch.cos(torch.outer(torch.arange(240).float(), exact))
    state = {
        "freqs": exact.bfloat16().float(),
        "rotate": torch.stack((cosines, torch.zeros_like(cosines)), dim=-1),
    }
    rope = _RotaryPositionalEmbedding(n_dim=32)
    rope.load_state_dict(state, strict=True)
    cosine_only = torch.complex(cosines, torch.zeros_like(cosines))
    angles = torch.outer(torch.arange(250).float(), state["freqs"])
    rebuilt = torch.polar(torch.ones_like(angles), angles)
    for seq, released in ((240, cosine_only), (250, rebuilt), (5, rebuilt)):
        q = torch.randn(2, seq, 4, 8)
        torch.testing.assert_close(
            rope(q, q)[0], _complex_rope_reference(q, 32, rotate=released)
        )
    assert {k: v.shape for k, v in rope.state_dict().items()} == {
        "freqs": (16,),
        "rotate": (240, 16, 2),
    }
    rope.load_state_dict(state, strict=True)
    torch.testing.assert_close(
        rope(q, q)[0], _complex_rope_reference(q, 32, rotate=cosine_only)
    )


def test_spatial_temporal_block_matches_frozen_released_output():
    """``SpatialTemporalAttentionBlock(16, 4, 0.0, False)`` of OpenTSLab/BrainOmni
    340d6b5 (complex RoPE, CPU float32) on the same state dict (cosine-only
    ``rotate``, bfloat16-rounded ``freqs``) and input."""
    block = _SpatialTemporalBlock(16, 4, 0.0).eval()
    gen = torch.Generator().manual_seed(0)
    state = {}
    for key, value in block.state_dict().items():
        if "rope_embedding_layer" not in key:
            offset = 1.0 if key.endswith("norm.weight") else 0.0
            state[key] = 0.3 * torch.randn(value.shape, generator=gen) + offset
    exact = 1.0 / (10000 ** (torch.arange(0, 8, 2).float() / 8))
    cosines = torch.cos(torch.outer(torch.arange(240).float(), exact))
    state["time_attn.rope_embedding_layer.freqs"] = exact.bfloat16().float()
    state["time_attn.rope_embedding_layer.rotate"] = torch.stack(
        (cosines, torch.zeros_like(cosines)), dim=-1
    )
    block.load_state_dict(state, strict=True)
    with torch.no_grad():
        out = block(torch.randn(2, 3, 5, 16, generator=gen))
    expected = torch.tensor(
        [2.487884044647217, -2.8454644680023193, 2.999533176422119, 7.002948760986328,
         -0.9435640573501587, 6.7119903564453125, -1.1264961957931519, 3.6791319847106934]
    )  # fmt: skip
    torch.testing.assert_close(out.flatten()[:8], expected, rtol=1e-5, atol=1e-5)
    assert out.sum().item() == pytest.approx(-245.95851135253906, abs=1e-3)
    assert out.abs().sum().item() == pytest.approx(1331.4229736328125, abs=1e-3)


# ---- BrainOmni ------------------------------------------------------------------


def test_brainomni_encode_is_l2_normalized():
    feat = _small_brainomni().eval().encode(torch.randn(2, 4, 512))
    assert feat.shape == (2, 3, 8, 16)
    torch.testing.assert_close(feat.norm(dim=-1), torch.ones(2, 3, 8))


def test_brainomni_reset_head_follows_mode_and_dtype():
    model = _small_brainomni().double().eval()
    model.reset_head(5)
    assert not any(module.training for module in model.final_layer.modules())
    assert next(model.final_layer.parameters()).dtype == torch.float64
    x = torch.randn(2, 4, 512, dtype=torch.float64)
    with torch.no_grad():
        torch.testing.assert_close(model(x), model(x), rtol=0, atol=0)
    model.train().reset_head(2)
    assert all(module.training for module in model.final_layer.modules())


def test_brainomni_dropouts_and_default_head_init():
    """Tokenizer, transformer and head dropouts are separate, as released; the
    head keeps PyTorch's default init (non-zero biases), unlike the trunk."""
    model = _small_brainomni(drop_prob=0.0)
    assert model.tokenizer.encoder.backwardsolution.dropout == 0.0
    assert model.blocks[0].time_attn.dropout == 0.0
    assert model.final_layer[0].p == 0.1
    biases = [model.final_layer[i].bias for i in (1, 3)]
    model.reset_head(5)
    biases += [model.final_layer[i].bias for i in (1, 3)]
    assert all(bias.abs().max() > 0 for bias in biases)
    assert model.blocks[0].ff[0].bias.abs().max() == 0


def test_brainomni_tokenizer_is_frozen_during_train_step():
    torch.manual_seed(0)
    model = _small_brainomni().train()
    assert not any(p.requires_grad for p in model.tokenizer.parameters())
    x = torch.randn(2, 4, 512)
    model.tokenizer.tokenize(x)  # one-time initialization of a fresh codebook
    codebook = model.tokenizer.quantizer.layers[0]._codebook
    before = codebook.embed.clone()
    model(x).sum().backward()
    assert torch.equal(before, codebook.embed)


# ---- released weights (braindecode Hub repositories) ------------------------------

_POS = torch.tensor([[0.1, 0.2, 0.3, 0, 0, 0], [-0.2, 0.1, 0.4, 0, 0, 0]])


def _from_hub(cls, repo, **kwargs):
    pytest.importorskip("huggingface_hub")
    model = cls.from_pretrained(
        f"braindecode/{repo}",
        chs_info=_eeg_chs_info(2),
        n_times=512,
        sfreq=256.0,
        strict=True,
        **kwargs,
    ).eval()
    tokenizer = model if cls is BrainTokenizer else model.tokenizer
    tokenizer.pos.copy_(_POS)
    torch.manual_seed(123)
    return model, torch.randn(1, 2, 512)


# Reference values: the released code (OpenTSLab/BrainOmni 340d6b5, attention
# dropout 0, CPU float32) on the released weights and the same input.
@pytest.mark.network
@pytest.mark.huggingface
def test_braintokenizer_pretrained_matches_release():
    model, x = _from_hub(BrainTokenizer, "braintokenizer-pretrained")
    feat, indices = model.tokenize(x)
    expected = torch.tensor(
        [-0.048095703125, 0.039093017578125, 0.0056915283203125, -0.0263824462890625,
         0.063201904296875, 0.07301521301269531, -0.024904251098632812,
         -0.02942657470703125]
    )  # fmt: skip
    torch.testing.assert_close(feat.flatten()[:8], expected, rtol=1e-5, atol=1e-5)
    assert feat.sum().item() == pytest.approx(-52.497127532958984, abs=1e-4)
    assert indices.sum().item() == 138744


@pytest.mark.network
@pytest.mark.huggingface
@pytest.mark.parametrize(
    "size, first, total",
    [
        (
            "tiny",
            [0.006593634374439716, -0.009288209490478039, -0.017851131036877632,
             -0.047316037118434906],
            51.254783630371094,
        ),
        (
            "base",  # braindecode 517aa14f on the released base checkpoint
            [0.007863267324864864, -0.08868950605392456, -0.07359513640403748,
             -0.03522023186087608],
            -11.090436935424805,
        ),
    ],
)  # fmt: skip
def test_brainomni_pretrained_matches_release(size, first, total):
    model, x = _from_hub(BrainOmni, f"brainomni-{size}-pretrained", n_outputs=3)
    feat = model.encode(x)
    torch.testing.assert_close(
        feat.flatten()[:4], torch.tensor(first), rtol=1e-5, atol=1e-5
    )
    assert feat.sum().item() == pytest.approx(total, abs=1e-4)
    assert model(x).shape == (1, 3)
