# Authors: Bruno Aristimunha <b.aristimunha@gmail.com>
#
# License: BSD-3
"""Focused tests for the BrainOmni port (BrainTokenizer and BrainOmni)."""

import hashlib
import json
from pathlib import Path

import mne
import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from mne.io.constants import FIFF

from braindecode.models import BrainOmni, BrainTokenizer
from braindecode.models.base import EEGModuleMixin
from braindecode.models.brainomni import (
    _BackwardSolution,
    _ForwardSolution,
    _geometry_from_chs_info,
    _MultiHeadAttentionRoPE,
    _rename_official_key,
    _RotaryPositionalEmbedding,
    _SEANetDecoder,
    _SEANetEncoder,
    _SensorEmbedding,
    _SpatialTemporalBlock,
    _TokenizerEncoder,
)
from braindecode.modules.quantization import EMACodebook as _Codebook
from braindecode.modules.quantization import ResidualVectorQuantizer as _ResidualVQ

# Shared small-model config (keeps every BrainOmni/BrainTokenizer build fast).
_BRAINOMNI_KW = dict(
    emb_dim=16,
    n_neuro=3,
    n_filters=8,
    codebook_dim=16,
    codebook_size=32,
    num_quantizers=2,
    tokenizer_num_heads=4,
)
_HF_REVISION = "9a4d3c70495370397ccfbfd6d2496f25647545a5"


def _loc(x=0.0, y=0.0, z=0.0, *rest):
    arr = np.zeros(12, dtype=np.float64)
    arr[:3] = (x, y, z)
    for i, v in enumerate(rest):
        arr[3 + i] = v
    return arr


def _eeg_chs_info(n):
    rng = np.random.default_rng(0)
    return [
        {"ch_name": f"C{i}", "kind": "eeg", "loc": _loc(*rng.random(3))}
        for i in range(n)
    ]


def _mixed_chs_info():
    rng = np.random.default_rng(1)
    return [
        {"ch_name": "E1", "kind": "eeg", "loc": _loc(*rng.random(3))},
        {"ch_name": "E2", "kind": "eeg", "loc": _loc(*rng.random(3))},
        {"ch_name": "M1", "kind": "mag", "coil_type": 3022, "loc": _loc(*rng.random(6))},
        {"ch_name": "G1", "kind": "grad", "coil_type": 3012, "loc": _loc(*rng.random(6))},
    ]


def _real_chs(ch_names, ch_types):
    """Build MNE channel dicts with a minimal finite loc (identity rotation)."""
    finite_loc = np.array([0.1, 0.0, 0.1, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
    info = mne.create_info(ch_names, 256.0, ch_types)
    for ch in info["chs"]:
        ch["loc"] = finite_loc.copy()
    return info["chs"]


def _small_tokenizer(n_chans=4, n_times=512, chs_info=None, sfreq=256.0):
    return BrainTokenizer(
        chs_info=chs_info if chs_info is not None else _eeg_chs_info(n_chans),
        n_times=n_times,
        sfreq=sfreq,
        **_BRAINOMNI_KW,
    )


def _small_brainomni(n_chans=4, n_outputs=3, n_times=512, sfreq=256.0, chs_info=None):
    return BrainOmni(
        chs_info=chs_info if chs_info is not None else _eeg_chs_info(n_chans),
        n_outputs=n_outputs,
        n_times=n_times,
        sfreq=sfreq,
        lm_dim=16,
        num_heads=4,
        depth=2,
        **_BRAINOMNI_KW,
    )


def _quantizer(num_quantizers=2):
    return _ResidualVQ(
        dim=16,
        codebook_dim=16,
        codebook_size=32,
        num_quantizers=num_quantizers,
        rotation_trick=True,
        quantize_optimize_method="ema",
    )


def _distributed_codebook_update(rank, world_size, init_file, output_dir):
    """Exercise one EMA update with different data on each CPU rank."""
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        codebook = _Codebook(
            dim=2,
            codebook_size=2,
            decay=0.5,
            threshold_ema_dead_code=0,
        )
        codebook.inited.fill_(1)
        codebook.cluster_size.copy_(torch.tensor([4.0, 4.0]))
        codebook.embed.copy_(torch.tensor([[-1.0, 0.0], [1.0, 0.0]]))
        codebook.embed_avg.copy_(torch.tensor([[-4.0, 0.0], [4.0, 0.0]]))
        local_samples = (
            torch.tensor([[[-2.0, 0.0], [-1.0, 0.0]]])
            if rank == 0
            else torch.tensor([[[1.0, 0.0], [2.0, 0.0]]])
        )
        codebook.train()(local_samples)
        torch.manual_seed(rank)
        fresh_codebook = _Codebook(
            dim=2,
            codebook_size=2,
            decay=0.5,
            threshold_ema_dead_code=0,
            kmeans_iters=2,
        )
        fresh_codebook.train()(local_samples)
        torch.save(
            {
                "cluster_size": codebook.cluster_size,
                "embed_avg": codebook.embed_avg,
                "embed": codebook.embed,
                "fresh_inited": fresh_codebook.inited,
                "fresh_cluster_size": fresh_codebook.cluster_size,
                "fresh_embed_avg": fresh_codebook.embed_avg,
                "fresh_embed": fresh_codebook.embed,
            },
            Path(output_dir) / f"rank-{rank}.pt",
        )
    finally:
        dist.destroy_process_group()


def _hub_file(filename, tmp_path):
    """Download one file of the pinned official release (network tests only)."""
    hf_hub = pytest.importorskip("huggingface_hub")
    return Path(
        hf_hub.hf_hub_download(
            "OpenTSLab/BrainOmni", filename, revision=_HF_REVISION, cache_dir=tmp_path
        )
    )


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---- geometry derivation -----------------------------------------------------


@pytest.mark.parametrize(
    "chs_info, expected",
    [
        ([{"ch_name": "C1", "kind": "eeg", "loc": _loc(1.0)}], [0]),
        ([{"ch_name": "C1", "ch_type": "eeg", "loc": _loc(1.0)}], [0]),
        (_real_chs(["MEG0111"], ["grad"]), [2]),
        (_real_chs(["MEG0112"], ["mag"]), [1]),
        (_real_chs(["E1"], ["eeg"]), [0]),
    ],
    ids=["eeg_simplified", "eeg_ch_type", "grad", "mag", "eeg_real"],
)
def test_geometry_sensor_type(chs_info, expected):
    _, sensor_type = _geometry_from_chs_info(chs_info)
    assert sensor_type.tolist() == expected


def test_geometry_eeg_orientation_and_centering():
    pos, _ = _geometry_from_chs_info(
        [
            {"ch_name": "A", "kind": "eeg", "loc": _loc(1.0, 0.0, 0.0)},
            {"ch_name": "B", "kind": "eeg", "loc": _loc(-1.0, 0.0, 0.0)},
        ]
    )
    assert pos.shape == (2, 6)
    assert np.allclose(pos[:, 3:], 0.0)  # EEG orientation columns are zero
    assert np.allclose(pos[:, :3].mean(axis=0), 0.0, atol=1e-6)  # mean-centered


@pytest.mark.parametrize(
    "kind, coil_type, orientation_slice",
    [
        ("grad", 3012, slice(3, 6)),  # planar gradiometer
        ("grad", 5001, slice(9, 12)),  # CTF axial gradiometer
        ("grad", 6001, slice(9, 12)),  # KIT axial gradiometer
        ("mag", 3022, slice(9, 12)),
    ],
)
def test_geometry_meg_orientation_uses_mne_loc_axes(kind, coil_type, orientation_slice):
    loc = _loc(0.1, 0.2, 0.3)
    loc[orientation_slice] = (0.4, 0.5, 0.6)
    pos, _ = _geometry_from_chs_info(
        [{"ch_name": "M1", "kind": kind, "coil_type": coil_type, "loc": loc}]
    )
    assert np.allclose(pos[0, 3:], (0.4, 0.5, 0.6))


@pytest.mark.parametrize(
    "kind, coil_type, orientation_slice",
    [
        ("grad", 3012, slice(3, 6)),
        ("grad", 5001, slice(9, 12)),
        ("grad", 6001, slice(9, 12)),
        ("mag", 3022, slice(9, 12)),
    ],
)
def test_geometry_nonfinite_meg_orientation_raises(kind, coil_type, orientation_slice):
    loc = _loc(0.1, 0.2, 0.3)
    loc[orientation_slice] = np.nan
    with pytest.raises(ValueError, match="finite coil orientation"):
        _geometry_from_chs_info(
            [{"ch_name": "M1", "kind": kind, "coil_type": coil_type, "loc": loc}]
        )


def _meg_info_chs(
    n_triplets=4,
    n_eeg=3,
    axial=False,
    plain_int=False,
    axial_coil=FIFF.FIFFV_COIL_CTF_GRAD,
):
    """MNE ``info["chs"]`` like the sample data: integer FIFF kind/coil codes.

    VectorView triplets (MAG 3022 + two planar GRAD 3012), or axial
    gradiometers (``axial_coil``, CTF 5001 by default, unit T) with
    ``axial=True``, plus EEG. Locations are a sensor helmet (head frame,
    metres) with an orthonormal coil frame per sensor.
    """
    names, types, coils = [], [], []
    for i in range(n_triplets):
        if axial:
            names += [f"MLC{i}1", f"MLC{i}2", f"MLC{i}3"]
            types += ["mag"] * 3  # MNE stores CTF axial gradiometers with unit T
            coils += [axial_coil] * 3
        else:
            names += [f"MEG{i:03d}1", f"MEG{i:03d}2", f"MEG{i:03d}3"]
            types += ["mag", "grad", "grad"]
            coils += [
                FIFF.FIFFV_COIL_VV_MAG_T3,
                FIFF.FIFFV_COIL_VV_PLANAR_T1,
                FIFF.FIFFV_COIL_VV_PLANAR_T1,
            ]
    names += [f"EEG{i:03d}" for i in range(n_eeg)]
    types += ["eeg"] * n_eeg
    coils += [FIFF.FIFFV_COIL_EEG] * n_eeg
    info = mne.create_info(names, 256.0, types)
    rng = np.random.default_rng(7)
    for ch, coil in zip(info["chs"], coils):
        ch["coil_type"] = int(coil) if plain_int else coil
        direction = rng.normal(size=3)
        direction /= np.linalg.norm(direction)
        ex = np.cross(direction, [0.0, 0.0, 1.0])
        ex /= np.linalg.norm(ex)
        ey = np.cross(direction, ex)
        ch["loc"] = np.concatenate([0.1 * direction, ex, ey, direction])
        if plain_int:
            ch["kind"] = int(ch["kind"])
    return info["chs"]


@pytest.mark.parametrize("plain_int", [False, True], ids=["named_int", "plain_int"])
def test_geometry_vectorview_meg_info(plain_int):
    chs = _meg_info_chs(plain_int=plain_int)
    pos, sensor_type = _geometry_from_chs_info(chs)
    assert sensor_type.tolist() == [1, 2, 2] * 4 + [0] * 3
    for i, ch in enumerate(chs):
        if sensor_type[i] == 2:  # planar GRAD: in-plane ex axis
            assert np.allclose(pos[i, 3:], ch["loc"][3:6])
        elif sensor_type[i] == 1:  # MAG: coil normal
            assert np.allclose(pos[i, 3:], ch["loc"][9:12])
        else:
            assert np.allclose(pos[i, 3:], 0.0)
    # EEG and MEG (MAG + GRAD together) are each centred and scaled.
    for mask in (sensor_type == 0, sensor_type > 0):
        xyz = pos[mask, :3]
        assert np.allclose(xyz.mean(axis=0), 0.0, atol=1e-6)
        assert np.isclose(np.sqrt(3 * np.mean(np.sum(xyz**2, axis=1))), 1.0)


@pytest.mark.parametrize(
    "coil, expected",
    [
        (FIFF.FIFFV_COIL_CTF_GRAD, 2),
        (FIFF.FIFFV_COIL_KIT_GRAD, 2),
        (FIFF.FIFFV_COIL_MAGNES_GRAD, 1),
    ],
    ids=["ctf_5001", "kit_6001", "magnes_4002"],
)
@pytest.mark.parametrize("plain_int", [False, True], ids=["named_int", "plain_int"])
def test_geometry_axial_gradiometer_types_follow_release(plain_int, coil, expected):
    # mne.channel_type says "mag" for axial gradiometers (unit T); the released
    # extract_pos_sensor_type says GRAD unless the coil name contains "MAG"
    # (Magnes 4002 is FIFFV_COIL_MAGNES_GRAD, so it stays MAG).
    chs = _meg_info_chs(n_eeg=0, axial=True, plain_int=plain_int, axial_coil=coil)
    assert {mne.channel_type({"chs": chs}, i) for i in range(len(chs))} == {"mag"}
    pos, sensor_type = _geometry_from_chs_info(chs)
    assert sensor_type.tolist() == [expected] * len(chs)
    assert np.allclose(pos[:, 3:], np.stack([ch["loc"][9:12] for ch in chs]))


def test_geometry_rejects_meg_reference_channels():
    chs = _meg_info_chs(n_triplets=1, n_eeg=0)
    chs[0]["kind"] = FIFF.FIFFV_REF_MEG_CH
    with pytest.raises(ValueError, match="Unsupported channel type"):
        _geometry_from_chs_info(chs)


@pytest.mark.parametrize("axial", [False, True], ids=["vectorview", "ctf"])
def test_brainomni_forward_on_meg_and_eeg_info(axial):
    chs = _meg_info_chs(axial=axial)
    model = _small_brainomni(chs_info=chs, n_outputs=2).eval()
    assert model.tokenizer.sensor_type.tolist() == (
        [2] * 12 if axial else [1, 2, 2] * 4
    ) + [0] * 3
    out = model(torch.randn(2, len(chs), 512))
    assert out.shape == (2, 2) and torch.isfinite(out).all()
    tokenizer = _small_tokenizer(chs_info=chs).eval()
    x = torch.randn(2, len(chs), 512)
    assert tokenizer(x).shape == x.shape


@pytest.mark.parametrize("loc", [np.full(12, np.nan), None], ids=["nan", "absent"])
def test_geometry_bad_loc_raises(loc):
    ch = {"ch_name": "A", "kind": "eeg"}
    if loc is not None:
        ch["loc"] = loc
    with pytest.raises(ValueError, match="set_montage"):
        _geometry_from_chs_info([ch])


# ---- tokenizer submodules ----------------------------------------------------


@pytest.mark.parametrize(
    "build, make_inputs, exp_shape",
    [
        (
            lambda: _SensorEmbedding(n_dim=16),
            lambda: (torch.randn(2, 5, 6), torch.zeros(2, 5, dtype=torch.long)),
            (2, 5, 16),
        ),
        (
            lambda: _TokenizerEncoder(
                n_filters=8,
                ratios=[8, 4, 2],
                kernel_size=5,
                last_kernel_size=5,
                n_dim=16,
                n_head=4,
                dropout=0.0,
                n_neuro=3,
            ),
            lambda: (torch.randn(2, 5, 1, 512), torch.randn(2, 5, 16)),
            (2, 3, 1, 8, 16),  # channels (5) collapse to n_neuro (3); T = 512/64
        ),
    ],
    ids=["sensor_module", "tokenizer_encoder"],
)
def test_tokenizer_submodule_shapes(build, make_inputs, exp_shape):
    out = build()(*make_inputs())
    assert out.shape == exp_shape
    assert torch.isfinite(out).all()


def test_seanet_roundtrip_downsampling():
    kw = dict(
        channels=1,
        dimension=32,
        n_filters=8,
        ratios=[8, 4, 2],
        kernel_size=5,
        last_kernel_size=5,
    )
    enc, dec = _SEANetEncoder(**kw), _SEANetDecoder(**kw)
    z = enc(torch.randn(4, 1, 512))  # 512 / (8*4*2) = 8
    assert z.shape == (4, 32, 8)
    x_rec = dec(z)
    assert x_rec.shape == (4, 1, 512)
    assert torch.isfinite(x_rec).all()


def test_seanet_roundtrip_supports_odd_ratio():
    kw = dict(
        channels=1, dimension=8, n_filters=4, ratios=[5], kernel_size=5, last_kernel_size=5
    )
    encoder, decoder = _SEANetEncoder(**kw), _SEANetDecoder(**kw)
    x = torch.randn(1, 1, 25)
    reconstruction = decoder(encoder(x))
    assert reconstruction.shape == x.shape
    assert torch.isfinite(reconstruction).all()


def test_seanet_reflect_padding_supports_one_sample_input():
    encoder = _SEANetEncoder(
        channels=1, dimension=8, n_filters=4, ratios=[2], kernel_size=5, last_kernel_size=5
    )
    encoded = encoder(torch.randn(1, 1, 1))
    assert encoded.shape == (1, 8, 1)
    assert torch.isfinite(encoded).all()


# ---- residual vector quantization --------------------------------------------


def test_quantizer_shapes_and_loss():
    q = _quantizer(num_quantizers=4).eval()
    x_q, indices, loss = q(torch.randn(2, 5, 16))
    assert x_q.shape == (2, 5, 16)
    assert indices.shape == (2, 5, 4)  # num_quantizers
    assert torch.isfinite(loss)


def test_quantizer_initializes_codebook_from_first_batch():
    torch.manual_seed(0)
    quantizer = _quantizer(num_quantizers=1).train()
    codebook = quantizer.layers[0]._codebook
    assert codebook.inited.item() == 0
    assert torch.count_nonzero(codebook.cluster_size) == 0
    assert torch.count_nonzero(codebook.embed) == 0

    quantizer(torch.randn(8, 10, 16))

    assert codebook.inited.item() == 1
    assert codebook.cluster_size.sum() > 0
    assert torch.isfinite(codebook.embed).all()
    assert codebook.embed.norm(dim=-1).max() < 1.1


def _broadcast_kmeans_reference(codebook, samples):
    """``EMACodebook._kmeans`` before #1239: explicit squared distances."""
    samples = samples.reshape(-1, samples.shape[-1])
    dim, dtype = samples.shape[1], samples.dtype
    if samples.shape[0] < codebook.codebook_size:
        noise = torch.randn(
            codebook.codebook_size - samples.shape[0], dim, dtype=dtype
        )
        samples = torch.cat([samples, noise], dim=0)
    centers = codebook._sample_vectors(samples, codebook.codebook_size)
    bins = torch.ones(codebook.codebook_size, dtype=torch.long)
    for _ in range(codebook.kmeans_iters):
        distances = torch.cat(
            [
                ((chunk.unsqueeze(1) - centers) ** 2).sum(dim=-1)
                for chunk in samples.split(256)
            ]
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
    "make_samples, codebook_size",
    [
        (lambda: torch.randn(600, 16), 32),
        # Integer grid with repeated rows: many exact distance ties.
        (lambda: torch.randint(-2, 3, (600, 4)).float(), 32),
        # Fewer samples than codes: noise padding, duplicated centres, empty bins.
        (lambda: torch.randint(0, 2, (20, 3)).float(), 32),
    ],
    ids=["gaussian", "tie_heavy_grid", "fewer_samples_than_codes"],
)
def test_codebook_kmeans_matches_broadcast_reference(make_samples, codebook_size):
    codebook = _Codebook(dim=16, codebook_size=codebook_size, kmeans_iters=10)
    torch.manual_seed(0)
    samples = make_samples()
    torch.manual_seed(1)
    centers, bins = codebook._kmeans(samples)
    torch.manual_seed(1)
    ref_centers, ref_bins = _broadcast_kmeans_reference(codebook, samples)
    torch.testing.assert_close(bins, ref_bins, rtol=0, atol=0)
    torch.testing.assert_close(centers, ref_centers, rtol=0, atol=0)


@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed is unavailable")
def test_quantizer_distributed_ema_uses_global_statistics(tmp_path):
    world_size = 2
    mp.spawn(
        _distributed_codebook_update,
        args=(world_size, str(tmp_path / "init"), str(tmp_path)),
        nprocs=world_size,
        join=True,
    )
    states = [
        torch.load(tmp_path / f"rank-{rank}.pt", weights_only=True)
        for rank in range(world_size)
    ]
    for key in states[0]:
        torch.testing.assert_close(states[0][key], states[1][key])
    torch.testing.assert_close(states[0]["cluster_size"], torch.tensor([3.0, 3.0]))
    torch.testing.assert_close(
        states[0]["embed_avg"], torch.tensor([[-3.5, 0.0], [3.5, 0.0]])
    )
    torch.testing.assert_close(
        states[0]["embed"], torch.tensor([[-7.0 / 6.0, 0.0], [7.0 / 6.0, 0.0]])
    )
    assert states[0]["fresh_inited"].item() == 1


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"dim": 0}, "dim"),
        ({"codebook_dim": 0}, "codebook_dim"),
        ({"codebook_size": 0}, "codebook_size"),
        ({"num_quantizers": 0}, "num_quantizers"),
        ({"quantize_optimize_method": "sgd"}, "ema"),
    ],
)
def test_quantizer_rejects_invalid_arguments(kwargs, match):
    defaults = {"dim": 16, "codebook_dim": 16, "codebook_size": 32, "num_quantizers": 2}
    defaults.update(kwargs)
    with pytest.raises(ValueError, match=match):
        _ResidualVQ(**defaults)


@pytest.mark.parametrize(
    "train, expect_change",
    [(True, True), (False, False)],
    ids=["train_updates", "eval_frozen"],
)
def test_quantizer_codebook_ema(train, expect_change):
    torch.manual_seed(0)
    q = _quantizer(num_quantizers=2)
    q.train(train)
    codebook = q.layers[0]._codebook
    q(torch.randn(8, 10, 16))  # initialize fresh K-means codebooks
    before = codebook.embed.clone()
    for _ in range(3):
        q(torch.randn(8, 10, 16))
    assert (not torch.allclose(before, codebook.embed)) is expect_change


# ---- public BrainTokenizer ---------------------------------------------------


def test_braintokenizer_is_eeg_module():
    assert issubclass(BrainTokenizer, EEGModuleMixin)
    last_two = [name for name, _ in _small_tokenizer().named_children()][-2:]
    assert "final_layer" in last_two


@pytest.mark.parametrize("n_times", [300, 512, 600])
def test_braintokenizer_forward_reconstruction_shape(n_times):
    model = _small_tokenizer(n_times=n_times).eval()
    x = torch.randn(2, 4, n_times)
    assert model(x).shape == x.shape


def test_braintokenizer_mixed_eeg_meg_forward():
    chs_info = _mixed_chs_info()
    model = _small_tokenizer(chs_info=chs_info).eval()
    assert model.sensor_type.tolist() == [0, 0, 1, 2]
    out = model(torch.randn(2, len(chs_info), 512))
    assert out.shape == (2, 4, 512)
    assert torch.isfinite(out).all()


def test_braintokenizer_first_training_forward_keeps_codebooks_bounded():
    torch.manual_seed(0)
    model = _small_tokenizer().train()
    codebook = model.quantizer.layers[0]._codebook
    reconstruction = model(torch.randn(2, 4, 512))
    assert torch.isfinite(reconstruction).all()
    assert codebook.inited.item() == 1
    assert codebook.cluster_size.sum() > 0
    assert codebook.embed.norm(dim=-1).max() < 1.1


@pytest.mark.parametrize(
    "window_length, ratios",
    [(5, (5,)), (1, (1,))],
    ids=["odd_ratio", "one_sample_window"],
)
def test_braintokenizer_supports_documented_positive_ratios_and_windows(
    window_length, ratios
):
    model = BrainTokenizer(
        chs_info=_eeg_chs_info(2),
        n_times=window_length,
        sfreq=256.0,
        window_length=window_length,
        ratios=ratios,
        emb_dim=8,
        n_neuro=2,
        n_filters=4,
        tokenizer_num_heads=2,
        codebook_dim=8,
        codebook_size=8,
        num_quantizers=1,
    ).eval()
    x = torch.randn(1, 2, window_length)
    assert model(x).shape == x.shape


def test_braintokenizer_constructs_from_official_config():
    official_config = {
        "window_length": 8,
        "n_filters": 4,
        "ratios": [2],
        "kernel_size": 3,
        "last_kernel_size": 3,
        "n_dim": 8,
        "n_head": 2,
        "n_neuro": 2,
        "dropout": 0.0,
        "codebook_dim": 8,
        "codebook_size": 8,
        "num_quantizers": 1,
        "rotation_trick": True,
        "quantize_optimize_method": "ema",
    }
    original_config = dict(official_config)
    model = BrainTokenizer.from_opentslab_config(
        official_config, chs_info=_eeg_chs_info(2), n_times=8, sfreq=256.0
    )
    assert model.emb_dim == 8
    assert model.drop_prob == 0.0
    assert official_config == original_config


def test_braintokenizer_crops_each_window_when_hop_does_not_divide_it():
    """SEANet decodes 12 samples for a 10-sample window at hop 4; each window
    is cropped before joining, so window 2 lands at samples 10-19."""
    kw = dict(
        sfreq=256.0,
        window_length=10,
        ratios=(4,),
        emb_dim=8,
        n_neuro=2,
        n_filters=4,
        tokenizer_num_heads=2,
        codebook_dim=8,
        codebook_size=8,
        num_quantizers=1,
    )
    torch.manual_seed(0)
    model = BrainTokenizer(chs_info=_eeg_chs_info(2), n_times=30, **kw).eval()
    x = torch.randn(1, 2, 30)
    with torch.no_grad():
        full = model(x)
        reconstruction, _, _ = model.encode_decode(x)
        window = model(x[..., 10:20])
    assert full.shape == x.shape
    torch.testing.assert_close(full[..., 10:20], window)
    torch.testing.assert_close(reconstruction, full, rtol=0, atol=0)


def test_braintokenizer_reconstruction_zero_fills_dropped_tail():
    model = _small_tokenizer(n_times=600).eval()
    reconstruction = model(torch.randn(1, 4, 600))
    assert torch.count_nonzero(reconstruction[..., 512:]) == 0


def test_braintokenizer_encode_decode_and_tokenize():
    model = _small_tokenizer()
    x = torch.randn(2, 4, 512)
    recon, commit_loss, indices = model.encode_decode(x)
    assert recon.shape == x.shape
    assert torch.isfinite(commit_loss)
    assert indices.shape[-1] == 2  # num_quantizers
    model.train()
    feat, idx = model.tokenize(x)
    assert model.training  # tokenize restores the caller's mode
    assert feat.shape[:2] == (2, 3) and feat.shape[-1] == 16
    assert idx.shape[-1] == 2


@pytest.mark.parametrize(
    "n_times, overlap_ratio, expected_starts",
    [(300, 0.0, [0]), (600, 0.0, [0]), (600, 0.25, [0, 384])],
)
def test_braintokenizer_unfold_matches_released_source(
    n_times, overlap_ratio, expected_starts
):
    model = _small_tokenizer(n_times=n_times)
    x = torch.arange(n_times, dtype=torch.float32).reshape(1, 1, -1)
    windows = model._unfold(x, overlap_ratio=overlap_ratio)
    assert windows[0, 0, :, 0].tolist() == expected_starts


@pytest.mark.parametrize("overlap_ratio", [-0.1, 1.0 - 0.5 / 512, 1.0, 1.1])
def test_braintokenizer_rejects_invalid_overlap(overlap_ratio):
    model = _small_tokenizer()
    with pytest.raises(ValueError, match="overlap_ratio"):
        model.tokenize(torch.randn(1, 4, 512), overlap_ratio=overlap_ratio)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"window_length": 0}, "window_length"),
        ({"n_filters": 0}, "n_filters"),
        ({"n_filters": 1}, "n_filters must be at least 2"),
        ({"ratios": ()}, "ratios"),
        ({"ratios": (8, 0, 2)}, "ratios"),
        ({"kernel_size": 0}, "kernel_size"),
        ({"last_kernel_size": 0}, "last_kernel_size"),
        ({"emb_dim": 15}, "emb_dim.*tokenizer_num_heads"),
        ({"tokenizer_num_heads": 0}, "tokenizer_num_heads"),
        ({"n_neuro": 0}, "n_neuro"),
        ({"drop_prob": -0.1}, "drop_prob"),
        ({"drop_prob": 1.1}, "drop_prob"),
    ],
)
def test_braintokenizer_rejects_invalid_constructor_arguments(kwargs, match):
    with pytest.raises(ValueError, match=match):
        BrainTokenizer(
            chs_info=_eeg_chs_info(4),
            n_times=512,
            sfreq=256.0,
            **(_BRAINOMNI_KW | kwargs),
        )


def test_braintokenizer_sfreq_warning():
    with pytest.warns(UserWarning, match="BrainTokenizer pretrained weights.*256"):
        _small_tokenizer(sfreq=128.0)


def test_braintokenizer_native_state_dict_roundtrip():
    source = _small_tokenizer()
    target = _small_tokenizer()
    target.load_state_dict(source.state_dict(), strict=True)
    for key, value in source.state_dict().items():
        assert torch.equal(target.state_dict()[key], value)


def test_braintokenizer_official_key_names_load():
    """Official ``rvq``/``decoder``/``conv.conv`` names map onto the port."""
    source = _small_tokenizer()
    official = {}
    for key, value in source.state_dict().items():
        if key in {"pos", "sensor_type"}:
            continue  # the official artifact carries no geometry
        key = key.replace("quantizer.", "quantizer.rvq.")
        key = key.replace("final_layer.", "decoder.")
        key = key.replace(".conv.weight", ".conv.conv.weight")
        key = key.replace(".conv.bias", ".conv.conv.bias")
        key = key.replace(".convtr.", ".convtr.convtr.")
        key = key.replace("aggregate_mlp.0.", "aggregate_mlp.layer.0.")
        key = key.replace("aggregate_mlp.3.", "aggregate_mlp.layer.2.")
        official[key] = value
    target = _small_tokenizer()
    target.load_state_dict(official, strict=True)
    for key, value in source.state_dict().items():
        assert torch.equal(target.state_dict()[key], value), key


def test_braintokenizer_checkpoint_key_remap_rejects_collisions():
    model = _small_tokenizer()
    state_dict = model.state_dict()
    native_key = next(key for key in state_dict if ".conv.weight_g" in key)
    official_key = native_key.replace(".conv.weight_g", ".conv.conv.weight_g")
    state_dict[official_key] = state_dict[native_key].clone()
    with pytest.raises(ValueError, match="collide"):
        model.load_state_dict(state_dict, strict=True)


@pytest.mark.network
@pytest.mark.huggingface
def test_braintokenizer_released_checkpoint_strict_load_and_parity(tmp_path):
    """Gate the exact official tokenizer artifact and public loading path."""
    path = _hub_file("braintokenizer/BrainTokenizer.pt", tmp_path)
    config_path = _hub_file("braintokenizer/model_cfg.json", tmp_path)
    assert _sha256(path) == (
        "d41c44c14c3f3b11fd0fb660752e356dff4cb4bc5f32a05f470f503ffddc7b1a"
    )
    assert _sha256(config_path) == (
        "67d99edcfdc54285bed7f37757b6fc9732199c56ac48f86027af1c8a22e95183"
    )
    state_dict = torch.load(path, map_location="cpu", weights_only=True)
    model = BrainTokenizer.from_opentslab_config(
        json.loads(config_path.read_text()),
        chs_info=_eeg_chs_info(2),
        n_times=512,
        sfreq=256.0,
    ).eval()
    model.load_state_dict(state_dict, strict=True)

    # Values captured from pinned source 340d6b5 with the same released
    # artifact and deterministic input/geometry.
    model.pos.copy_(torch.tensor([[0.1, 0.2, 0.3, 0, 0, 0], [-0.2, 0.1, 0.4, 0, 0, 0]]))
    torch.manual_seed(123)
    feat, indices = model.tokenize(torch.randn(1, 2, 512))
    assert feat.shape == (1, 16, 8, 256)
    assert indices.shape == (1, 16, 8, 4)
    expected = torch.tensor(
        [
            -0.048095703125,
            0.039093017578125,
            0.0056915283203125,
            -0.0263824462890625,
            0.063201904296875,
            0.07301521301269531,
            -0.024904251098632812,
            -0.02942657470703125,
            -0.081573486328125,
            -0.005706787109375,
            -0.040435791015625,
            0.038818359375,
            -0.12237548828125,
            -0.0460357666015625,
            0.004184722900390625,
            0.00177764892578125,
        ]
    )
    torch.testing.assert_close(feat.flatten()[:16], expected, rtol=1e-5, atol=1e-5)
    assert feat.sum().item() == pytest.approx(-52.497127532958984, abs=1e-4)
    assert indices.sum().item() == 138744


# ---- BrainOmni private attention ---------------------------------------------


def _complex_rope_reference(x, n_dim, base=10000):
    """Released BrainOmni RoPE: one complex frequency ladder split across heads."""
    freqs = 1.0 / (base ** (torch.arange(0, n_dim, 2)[: (n_dim // 2)].float() / n_dim))
    angles = torch.outer(torch.arange(x.shape[1]).float(), freqs)
    rotate = torch.polar(torch.ones_like(angles), angles)
    rotate = rotate.reshape(x.shape[1], x.shape[2], -1).unsqueeze(0)
    x_ = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    return torch.view_as_real(x_ * rotate).flatten(3).type_as(x)


@pytest.mark.parametrize("seq", [1, 7, 300])
def test_rope_matches_released_complex_rotation(seq):
    """Real-valued RoPE equals the released complex rotation, per-head bands included."""
    n_heads, head_dim = 4, 8
    rope = _RotaryPositionalEmbedding(n_dim=n_heads * head_dim)
    q = torch.randn(2, seq, n_heads, head_dim)
    k = torch.randn(2, seq, n_heads, head_dim)
    q_out, k_out = rope(q, k)
    torch.testing.assert_close(q_out, _complex_rope_reference(q, n_heads * head_dim))
    torch.testing.assert_close(k_out, _complex_rope_reference(k, n_heads * head_dim))
    torch.testing.assert_close(q_out.norm(dim=-1), q.norm(dim=-1))


def test_rope_is_stateless_and_keeps_dtype():
    """No buffers to cast or load; half inputs are rotated in float32."""
    rope = _RotaryPositionalEmbedding(n_dim=16).half()
    q = torch.randn(2, 9, 4, 4)
    q_out, _ = rope(q.half(), q.half())
    assert rope.state_dict() == {}
    assert q_out.dtype == torch.float16
    torch.testing.assert_close(
        q_out.float(), _complex_rope_reference(q, 16), atol=1e-2, rtol=1e-2
    )


def test_rope_rejects_mismatched_heads():
    rope = _RotaryPositionalEmbedding(n_dim=16)
    with pytest.raises(ValueError, match="n_heads \\* head_dim"):
        rope(torch.randn(1, 3, 2, 4), torch.randn(1, 3, 2, 4))


@pytest.mark.parametrize("rope", [True, False])
def test_rope_attention_shape_and_eval_determinism(rope):
    module = _MultiHeadAttentionRoPE(16, 4, dropout=0.5, rope=rope).eval()
    x = torch.randn(2, 7, 16)
    out = module(x)
    assert out.shape == (2, 7, 16)
    torch.testing.assert_close(module(x), out)  # SDPA dropout is off in eval


def test_rope_attention_rejects_odd_head_dim():
    with pytest.raises(ValueError, match="head dimension.*even"):
        _MultiHeadAttentionRoPE(12, 4, dropout=0.0, rope=True)


@pytest.mark.parametrize(
    "build, match",
    [
        (lambda: _SpatialTemporalBlock(15, 4, 0.0, causal=False), "must be even"),
        (lambda: _SpatialTemporalBlock(16, 3, 0.0, causal=False), "must be even"),
        (lambda: _ForwardSolution(10, 4, 0.0), "divisible"),
        (lambda: _BackwardSolution(10, 4, 0.0), "divisible"),
    ],
    ids=["block_odd_dim", "block_odd_heads", "forward_solution", "backward_solution"],
)
def test_attention_blocks_reject_invalid_dimensions(build, match):
    with pytest.raises(ValueError, match=match):
        build()


def test_spatial_temporal_block_shape():
    out = _SpatialTemporalBlock(16, 4, 0.0, causal=False)(torch.randn(2, 3, 7, 16))
    assert out.shape == (2, 3, 7, 16)
    assert torch.isfinite(out).all()


# ---- public BrainOmni classifier ---------------------------------------------


@pytest.mark.parametrize(
    "chs_info, n_times, n_outputs",
    [
        (_eeg_chs_info(4), 512, 3),  # standard EEG
        (_eeg_chs_info(4), 300, 3),  # input shorter than window_length -> padded
        (_mixed_chs_info(), 512, 2),  # mixed EEG + MAG + GRAD
    ],
    ids=["standard", "short_input", "mixed_eeg_meg"],
)
def test_brainomni_forward_shape(chs_info, n_times, n_outputs):
    model = _small_brainomni(chs_info=chs_info, n_outputs=n_outputs, n_times=n_times)
    out = model.eval()(torch.randn(2, len(chs_info), n_times))
    assert out.shape == (2, n_outputs)


def test_brainomni_constructs_from_official_stage2_config():
    official_config = {
        "window_length": 8,
        "n_filters": 4,
        "ratios": [2],
        "kernel_size": 3,
        "last_kernel_size": 3,
        "n_dim": 8,
        "n_head": 2,
        "n_neuro": 2,
        "dropout": 0.0,
        "codebook_dim": 8,
        "codebook_size": 8,
        "num_quantizers": 1,
        "rotation_trick": True,
        "quantize_optimize_method": "ema",
        "overlap_ratio": 0.0,
        "lm_dim": 8,
        "lm_head": 2,
        "lm_depth": 2,
        "lm_dropout": 0.0,
        "mask_ratio": 0.5,
        "num_quantizers_used": 1,
    }
    original_config = dict(official_config)
    model = BrainOmni.from_opentslab_config(
        official_config, chs_info=_eeg_chs_info(2), n_outputs=3, n_times=8, sfreq=256.0
    )
    assert model.lm_dim == 8
    assert len(model.blocks) == 2
    assert model.tokenizer.emb_dim == 8
    assert official_config == original_config


def test_brainomni_encode_shape_and_normalized():
    feat = _small_brainomni().eval().encode(torch.randn(2, 4, 512))
    assert feat.ndim == 4 and feat.shape[1] == 3 and feat.shape[-1] == 16
    norms = feat.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)


def test_brainomni_reset_head_changes_only_head():
    model = _small_brainomni(n_outputs=3)
    tokenizer = model.tokenizer
    model.reset_head(5)
    assert model(torch.randn(2, 4, 512)).shape == (2, 5)
    assert model.tokenizer is tokenizer  # backbone untouched
    assert model.get_config()["n_outputs"] == 5


def test_brainomni_reset_head_follows_eval_mode():
    model = _small_brainomni(n_outputs=3).eval()
    model.reset_head(5)
    assert not any(module.training for module in model.final_layer.modules())
    x = torch.randn(2, 4, 512)
    with torch.no_grad():
        torch.testing.assert_close(model(x), model(x), rtol=0, atol=0)
    model.train()
    model.reset_head(2)
    assert all(module.training for module in model.final_layer.modules())


def test_brainomni_reset_head_preserves_dtype():
    model = _small_brainomni().double()
    model.reset_head(5)
    assert next(model.final_layer.parameters()).dtype == torch.float64


def test_brainomni_return_features_head_contract():
    model = _small_brainomni().eval()
    x = torch.randn(2, 4, 512)
    bundle = model(x, return_features=True)
    assert set(bundle) == {"features", "cls_token"}
    assert bundle["cls_token"] is None
    assert torch.allclose(model.final_layer(bundle["features"]), model(x))


def test_brainomni_released_dropout_configuration():
    """Tokenizer and Stage-2 dropout follow their separate released configs."""
    model = _small_brainomni()
    assert model.tokenizer.drop_prob == 0.0
    assert model.tokenizer.encoder.backwardsolution.dropout == 0.0
    assert model.tokenizer.final_layer.forwardsolution.dropout == 0.0
    assert model.drop_prob == 0.1
    assert model.blocks[0].time_attn.dropout == 0.1
    assert model.final_layer[0].p == 0.1


def test_brainomni_downstream_head_dropout_is_independent():
    """The released downstream head keeps 0.1 dropout when LM dropout changes."""
    model = BrainOmni(
        chs_info=_eeg_chs_info(4),
        n_outputs=3,
        n_times=512,
        sfreq=256.0,
        lm_dim=16,
        num_heads=4,
        depth=2,
        drop_prob=0.0,
        **_BRAINOMNI_KW,
    )
    assert model.blocks[0].time_attn.dropout == 0.0
    assert model.final_layer[0].p == 0.1


def test_brainomni_native_state_dict_roundtrip():
    source = _small_brainomni()
    target = _small_brainomni()
    target.load_state_dict(source.state_dict(), strict=True)
    for key, value in source.state_dict().items():
        assert torch.equal(target.state_dict()[key], value)


def test_brainomni_official_stage2_keys_load():
    """A Stage-2 state dict (pretraining head, RoPE caches) loads strictly."""
    source = _small_brainomni()
    official = {}
    for key, value in source.state_dict().items():
        if key.startswith("final_layer.") or key in {
            "tokenizer.pos",
            "tokenizer.sensor_type",
        }:
            continue  # not in the official artifact
        key = key.replace("tokenizer.quantizer.", "tokenizer.quantizer.rvq.")
        key = key.replace("tokenizer.final_layer.", "tokenizer.decoder.")
        key = key.replace(".conv.weight", ".conv.conv.weight")
        key = key.replace(".conv.bias", ".conv.conv.bias")
        key = key.replace(".convtr.", ".convtr.convtr.")
        key = key.replace("ff.0.", "ff.layer.0.").replace("ff.3.", "ff.layer.2.")
        key = key.replace("aggregate_mlp.0.", "aggregate_mlp.layer.0.")
        key = key.replace("aggregate_mlp.3.", "aggregate_mlp.layer.2.")
        official[key] = value
    official["mask_token"] = torch.zeros(16)
    official["predict_head.weight"] = torch.zeros(4, 16)
    official["blocks.0.time_attn.rope_embedding_layer.freqs"] = torch.zeros(4)
    official["blocks.0.time_attn.rope_embedding_layer.rotate"] = torch.zeros(7, 4)
    target = _small_brainomni()
    head = {k: v.clone() for k, v in target.state_dict().items() if "final_layer." in k}
    target.load_state_dict(official, strict=True)
    for key, value in source.state_dict().items():
        expected = head[key] if key.startswith("final_layer.") else value
        assert torch.equal(target.state_dict()[key], expected), key


def test_brainomni_official_decoder_rename_is_anchored():
    """Only a leading ``decoder.``/``tokenizer.decoder.`` is renamed."""
    assert _rename_official_key("decoder.weight") == "final_layer.weight"
    assert (
        _rename_official_key("tokenizer.decoder.model.0.weight")
        == "tokenizer.final_layer.model.0.weight"
    )
    assert _rename_official_key("head.tokenizer.decoder.w") == "head.tokenizer.decoder.w"


def test_brainomni_checkpoint_key_remap_rejects_collisions():
    model = _small_brainomni()
    state_dict = model.state_dict()
    native_key = next(
        key for key in state_dict if "tokenizer." in key and ".conv.weight_g" in key
    )
    official_key = native_key.replace(".conv.weight_g", ".conv.conv.weight_g")
    state_dict[official_key] = state_dict[native_key].clone()
    with pytest.raises(ValueError, match="collide"):
        model.load_state_dict(state_dict, strict=True)


def test_brainomni_tokenizer_is_frozen_during_train_step():
    torch.manual_seed(0)
    model = _small_brainomni().train()
    assert not any(p.requires_grad for p in model.tokenizer.parameters())
    assert all(p.requires_grad for p in model.blocks.parameters())
    x = torch.randn(2, 4, 512)
    model.tokenizer.tokenize(x)  # one-time initialization for a fresh tokenizer
    codebook = model.tokenizer.quantizer.layers[0]._codebook
    before = codebook.embed.clone()
    model(x).sum().backward()
    assert torch.equal(before, codebook.embed)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"overlap_ratio": -0.1}, "overlap_ratio"),
        ({"overlap_ratio": 1.0 - 0.5 / 512}, "overlap_ratio"),
        ({"overlap_ratio": 1.0}, "overlap_ratio"),
        ({"lm_dim": 15}, "lm_dim.*num_heads"),
        ({"num_heads": 3}, "num_heads.*even"),
        ({"depth": 0}, "depth"),
        ({"tokenizer_drop_prob": -0.1}, "tokenizer_drop_prob"),
        ({"tokenizer_drop_prob": 1.1}, "tokenizer_drop_prob"),
        ({"drop_prob": -0.1}, "drop_prob"),
        ({"drop_prob": 1.1}, "drop_prob"),
    ],
)
def test_brainomni_rejects_invalid_constructor_arguments(kwargs, match):
    model_kwargs = dict(lm_dim=16, num_heads=4, depth=2) | kwargs
    with pytest.raises(ValueError, match=match):
        BrainOmni(
            chs_info=_eeg_chs_info(4),
            n_outputs=3,
            n_times=512,
            sfreq=256.0,
            **(_BRAINOMNI_KW | model_kwargs),
        )


def test_brainomni_sfreq_warning():
    with pytest.warns(UserWarning, match="256"):
        _small_brainomni(sfreq=128.0)
    with pytest.warns(UserWarning, match="256"):
        _small_brainomni(sfreq=255.6)


@pytest.mark.network
@pytest.mark.huggingface
def test_brainomni_released_checkpoint_strict_load_and_parity(tmp_path):
    """Gate the exact official tiny artifact and deterministic encoder path."""
    path = _hub_file("tiny/BrainOmni.pt", tmp_path)
    config_path = _hub_file("tiny/model_cfg.json", tmp_path)
    assert _sha256(path) == (
        "62c67ba6a84ea0625e67a3b5e7463fe3930bfee88a612a225e9062a052542ffc"
    )
    assert _sha256(config_path) == (
        "4b994b38f1a8dccc8136f2b837a8ad2d3ebb028a1af0c527f9e0902a1a5c252e"
    )
    state_dict = torch.load(path, map_location="cpu", weights_only=True)
    model = BrainOmni.from_opentslab_config(
        json.loads(config_path.read_text()),
        chs_info=_eeg_chs_info(2),
        n_times=512,
        sfreq=256.0,
        n_outputs=3,
    ).eval()
    original_head = {
        key: value.clone()
        for key, value in model.state_dict().items()
        if key.startswith("final_layer.")
    }
    model.load_state_dict(state_dict, strict=True)
    assert all(
        torch.equal(model.state_dict()[key], value)
        for key, value in original_head.items()
    )
    # The release stores RoPE's frequencies rounded to bfloat16 and its complex
    # cache without the sine part; the loader drops both and the port recomputes
    # them in float32, as a freshly built upstream model does. The pinned
    # upstream signature below uses that path.
    model.tokenizer.pos.copy_(
        torch.tensor([[0.1, 0.2, 0.3, 0, 0, 0], [-0.2, 0.1, 0.4, 0, 0, 0]])
    )
    torch.manual_seed(123)
    feat = model.encode(torch.randn(1, 2, 512))
    assert feat.shape == (1, 16, 8, 256)
    expected = torch.tensor(
        [
            0.004591966513544321,
            -0.005511901341378689,
            -0.02144569717347622,
            -0.047614686191082,
            -0.057811133563518524,
            0.04927004501223564,
            -0.009117362089455128,
            -0.033146947622299194,
            -0.006567645352333784,
            -0.2256787121295929,
            -0.12158702313899994,
            0.009830539114773273,
            -0.045027319341897964,
            -0.0062219384126365185,
            0.016128726303577423,
            0.06308680027723312,
        ]
    )
    torch.testing.assert_close(feat.flatten()[:16], expected, rtol=1e-5, atol=1e-5)
    assert feat.sum().item() == pytest.approx(62.13804626464844, abs=1e-4)


@pytest.mark.network
@pytest.mark.huggingface
def test_brainomni_base_released_checkpoint_strict_load(tmp_path):
    """Gate the exact official base architecture and public loading path."""
    path = _hub_file("base/BrainOmni.pt", tmp_path)
    config_path = _hub_file("base/model_cfg.json", tmp_path)
    assert _sha256(path) == (
        "435db24e57a55df05aa7e16355def7b7ecbedb22aa1ec16063e7d14efd2386d0"
    )
    assert _sha256(config_path) == (
        "492e2229b1fb87d49330b23f482a1641ec7cdc0b41d38f76384f18fdef3696d5"
    )
    state_dict = torch.load(path, map_location="cpu", weights_only=True)
    model = BrainOmni.from_opentslab_config(
        json.loads(config_path.read_text()),
        chs_info=_eeg_chs_info(2),
        n_times=512,
        sfreq=256.0,
        n_outputs=3,
    )
    original_head = {
        key: value.clone()
        for key, value in model.state_dict().items()
        if key.startswith("final_layer.")
    }
    model.load_state_dict(state_dict, strict=True)
    assert torch.equal(model.projection.weight, state_dict["projection.weight"])
    assert all(
        torch.equal(model.state_dict()[key], value)
        for key, value in original_head.items()
    )
