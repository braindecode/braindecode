# Authors: Bruno Aristimunha <b.aristimunha@gmail.com>
#
# License: BSD-3
"""Focused tests for the BrainOmni port (BrainTokenizer)."""

import hashlib
import json
from pathlib import Path

import mne
import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from braindecode.models import BrainTokenizer
from braindecode.models.base import EEGModuleMixin
from braindecode.models.brainomni import (
    _geometry_from_chs_info,
    _SEANetDecoder,
    _SEANetEncoder,
    _SensorEmbedding,
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
        ("grad", 5001, slice(9, 12)),  # axial gradiometer
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
    with pytest.warns(UserWarning, match="256"):
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
