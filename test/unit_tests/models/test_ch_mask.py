"""Recordings with different numbers of channels in one batch (``ch_mask``).

Each model gets recordings padded to a common channel count and their
``ch_mask``, and must give every recording the output it gives alone,
whatever the padding holds.
"""

import inspect

import mne
import numpy as np
import pytest
import torch

from braindecode.models import (
    BIOT,
    CORTEG,
    DIVER1,
    LUNA,
    MAPA,
    MEGGPT,
    REVE,
    BaRISTA,
    BrainBERT,
    Brant,
    CBraMod,
    EpiNT,
    GeoNIRS,
    Labram,
    MtReNIRS,
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


def _positions(i, n_chans):
    return 0.1 * torch.rand((n_chans, 3), generator=torch.Generator().manual_seed(i))


def _mapa_indices(i, n_chans):
    labels = [f"{'LR'[i % 2]}{'AB'[k % 2]}{k // 2 + 1}" for k in range(n_chans)]
    regions = [None, "ctx-lh-insula", "Left-Hippocampus"][:n_chans]
    return MAPA.sensor_indices(labels, regions)


def _diver1_metadata(i, n_chans):
    return channel_metadata_from_chs_info(_seeg_chs(n_chans, ("seeg", "ecog")[i % 2], i))


def _corteg(**kwargs):
    model = CORTEG(n_chans=2, n_outputs=3, n_times=64, embed_dim=32, depth=2,
                   num_heads=2, n_merge_blocks=1, **kwargs)
    with torch.no_grad():  # off the zero init of the codebook, residual and gate
        for parameter in model.parameters():
            parameter.normal_(0.0, 0.1)
    return model


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
    "LUNA": lambda: (
        LUNA(n_chans=3, n_outputs=3, n_times=80, patch_size=20, embed_dim=32,
             num_queries=2, depth=1),
        80,
        ("channel_locations", _positions, True),
    ),
    "BIOT": lambda: (
        BIOT(n_chans=3, n_outputs=3, n_times=400, sfreq=200, embed_dim=32,
             num_heads=2, num_layers=1),
        400,
        None,
    ),
    "REVE": lambda: (
        REVE(n_chans=2, n_outputs=3, n_times=200, embed_dim=32, depth=1, heads=2,
             head_dim=8, mlp_dim_ratio=2, patch_size=50, patch_overlap=0,
             attention_pooling=True),
        200,
        ("pos", _positions, True),
    ),
    "MAPA": lambda: (
        MAPA(n_chans=2, n_outputs=3, n_times=512, sfreq=2048, d_model=64),
        512,
        ("sensor_indices", _mapa_indices, False),
    ),
    "CORTEG": lambda: (_corteg(), 64, ("pos", _positions, True)),
    "MtReNIRS": lambda: (
        MtReNIRS(n_chans=3, n_outputs=3, n_times=100, window_samples=50, d_model=32,
                 depth=1, n_heads=2, mlp_hidden=64, pool="channel_temporal_mean"),
        100,
        ("coordinates", lambda i, n: torch.rand((n, 4), generator=torch.Generator().manual_seed(i)), True),
    ),
}


def _assert_matches_alone(model, x, mask, xs, batch_kwargs, alone_kwargs):
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


def _padded_batch(n_times):
    xs = [torch.randn(n_real, n_times) for n_real in N_REAL]
    mask = torch.arange(max(N_REAL)) < torch.tensor(N_REAL)[:, None]
    x = torch.full((len(N_REAL), max(N_REAL), n_times), float("nan"))
    for i, xi in enumerate(xs):
        x[i, : len(xi)] = xi
    return x, mask, xs


def _padded_case(name):
    """Model, NaN-padded batch, its mask, the recordings alone and their kwargs."""
    torch.manual_seed(0)
    model, n_times, per_recording = _CASES[name]()
    model.eval()
    x, mask, xs = _padded_batch(n_times)
    batch_kwargs, alone_kwargs = {}, [{} for _ in N_REAL]
    if per_recording is not None:
        key, build, batched = per_recording
        alone = [build(i, n_real) for i, n_real in enumerate(N_REAL)]
        padded = torch.zeros((len(N_REAL), max(N_REAL)) + alone[0].shape[1:])
        for i, value in enumerate(alone):
            padded[i, : len(value)] = value
        batch_kwargs = {key: padded.to(alone[0].dtype)}
        alone_kwargs = [{key: value[None] if batched else value} for value in alone]
    return model, x, mask, xs, batch_kwargs, alone_kwargs


@pytest.mark.parametrize("merge_strategy", ["average", "layerwise_gate"])
def test_corteg_high_gamma_ch_mask_matches_each_recording_alone(merge_strategy):
    torch.manual_seed(0)
    model = _corteg(merge_strategy=merge_strategy).eval()
    x, mask, xs = _padded_batch(64)
    x_hi, _, his = _padded_batch(100)
    pos = [_positions(i, n_real) for i, n_real in enumerate(N_REAL)]
    padded_pos = torch.zeros(len(N_REAL), max(N_REAL), 3)
    for i, value in enumerate(pos):
        padded_pos[i, : len(value)] = value
    alone = [{"x_hi": hi[None], "pos": p[None]} for hi, p in zip(his, pos)]
    _assert_matches_alone(
        model, x, mask, xs, {"x_hi": x_hi, "pos": padded_pos}, alone
    )


def _labram_case():
    """As :func:`_padded_case`, for Labram, which reads channel names."""
    torch.manual_seed(0)
    model = Labram(n_chans=3, n_outputs=3, n_times=400, sfreq=200, patch_size=200,
                   embed_dim=32, num_layers=1, num_heads=2).eval()
    x, mask, xs = _padded_batch(400)
    ch_names = ["FP1", "FP2", "C3"]
    alone_kwargs = [{"ch_names": ch_names[:n_real]} for n_real in N_REAL]
    return model, x, mask, xs, {"ch_names": ch_names}, alone_kwargs


@pytest.mark.parametrize("name", list(_CASES))
def test_ch_mask_matches_each_recording_alone(name):
    _assert_matches_alone(*_padded_case(name))


def test_labram_ch_mask_matches_each_recording_alone():
    _assert_matches_alone(*_labram_case())


def test_geonirs_ch_mask_pairs_match_each_recording_alone():
    # GeoNIRS reads HbO/HbR pairs: recordings of 1, 3 and 2 interleaved pairs
    torch.manual_seed(0)
    model = GeoNIRS(n_chans=6, n_outputs=3, n_times=40, channel_order="interleaved",
                    embedding_dim=32, num_heads=4, temporal_layers=1,
                    num_latent_sources=4, output_feature_dim=8).eval()
    xs = [torch.randn(2 * n_real, 40) for n_real in N_REAL]
    geoms = [_positions(i, n_real).repeat(1, 2) for i, n_real in enumerate(N_REAL)]
    mask = torch.arange(6) < 2 * torch.tensor(N_REAL)[:, None]
    x = torch.full((len(N_REAL), 6, 40), float("nan"))
    padded = torch.zeros(len(N_REAL), 3, 6)
    for i, (xi, geom) in enumerate(zip(xs, geoms)):
        x[i, : len(xi)] = xi
        padded[i, : len(geom)] = geom
    _assert_matches_alone(
        model, x, mask, xs, {"geometry": padded}, [{"geometry": g[None]} for g in geoms]
    )


@pytest.mark.parametrize("name", ["REVE", "Labram"])
def test_reve_labram_tokens_match_each_recording_alone(name):
    case = _labram_case() if name == "Labram" else _padded_case(name)
    model, x, mask, xs, batch_kwargs, alone_kwargs = case
    with torch.no_grad():
        tokens = model(x, ch_mask=mask, return_features=True, **batch_kwargs)["tokens"]
        for i, (xi, kwargs) in enumerate(zip(xs, alone_kwargs)):
            alone = model(xi[None], return_features=True, **kwargs)["tokens"][0]
            torch.testing.assert_close(tokens[i, : len(xi)], alone)
        scripted = torch.jit.script(model)(
            x, ch_mask=mask, return_features=True, **batch_kwargs
        )
    assert torch.equal(tokens[~mask], torch.zeros_like(tokens[~mask]))
    torch.testing.assert_close(scripted["tokens"], tokens)


@pytest.mark.parametrize("name", list(_CASES))
def test_ch_mask_tokens_match_each_recording_alone(name):
    model, x, mask, xs, batch_kwargs, alone_kwargs = _padded_case(name)
    if "return_features" not in inspect.signature(type(model).forward).parameters:
        pytest.skip(f"{name} has no return_features")
    with torch.no_grad():
        features = model(x, ch_mask=mask, return_features=True, **batch_kwargs)
        if "tokens" not in features:
            pytest.skip(f"{name} pools its tokens inside the encoder")
        tokens = features["tokens"]
        scripted = torch.jit.script(model)(
            x, ch_mask=mask, return_features=True, **batch_kwargs
        )
        for i, (xi, kwargs) in enumerate(zip(xs, alone_kwargs)):
            alone = model(xi[None], return_features=True, **kwargs)["tokens"][0]
            torch.testing.assert_close(tokens[i, : len(xi)], alone)
    assert torch.equal(tokens[~mask], torch.zeros_like(tokens[~mask]))
    if isinstance(scripted, dict):  # most models ignore return_features when scripted
        torch.testing.assert_close(scripted["tokens"], tokens)


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
    "CBraMod-flatten": lambda: (
        CBraMod(n_outputs=3, n_chans=3, n_times=400, n_layer=1, nhead=2,
                dim_feedforward=64),
        400,
    ),
    "REVE-flatten": lambda: (
        REVE(n_chans=3, n_outputs=3, n_times=200, embed_dim=32, depth=1, heads=2,
             head_dim=8, mlp_dim_ratio=2, patch_size=50, patch_overlap=0),
        200,
    ),
    "Labram-decoder": lambda: (
        Labram(n_chans=3, n_outputs=3, n_times=400, sfreq=200, patch_size=200,
               embed_dim=32, num_layers=1, num_heads=2, neural_tokenizer=False),
        400,
    ),
    "MEGGPT": lambda: (_small_meggpt(), 16),
    "MtReNIRS-temporal_mean": lambda: (
        MtReNIRS(n_chans=3, n_outputs=3, n_times=100, window_samples=50, d_model=32,
                 depth=1, n_heads=2, mlp_hidden=64),
        100,
    ),
}


@pytest.mark.parametrize("name", list(_FIXED_HEADS))
def test_ch_mask_is_refused_by_fixed_size_heads(name, tmp_path, monkeypatch):
    # REVE reads its position bank at construction: a local one keeps
    # "REVE-flatten" offline (conftest marks only the "REVE" key as network).
    (tmp_path / "reve_positions.json").write_text('{"Cz": [0.0, 0.0, 0.09]}')
    monkeypatch.setenv("REVE_POSITIONS_PATH", str(tmp_path))
    model, n_times = _FIXED_HEADS[name]()
    x = torch.randn(len(N_REAL), 3, n_times)
    mask = torch.arange(3) < torch.tensor(N_REAL)[:, None]
    with pytest.raises(ValueError, match="ch_mask needs"):
        model(x, ch_mask=mask)
    with pytest.raises(torch.jit.Error, match="ch_mask needs"):
        torch.jit.script(model)(x, ch_mask=mask)


def test_cbramod_ch_mask_tokens_match_each_recording_alone():
    """CBraMod's heads flatten the channels; its encoder output is masked."""
    torch.manual_seed(0)
    model = CBraMod(n_outputs=3, n_chans=3, n_times=400, n_layer=1, nhead=2,
                    dim_feedforward=64, return_encoder_output=True).eval()
    xs = [torch.randn(n_real, 400) for n_real in N_REAL]
    mask = torch.arange(3) < torch.tensor(N_REAL)[:, None]
    x = torch.full((len(N_REAL), 3, 400), float("nan"))
    for i, xi in enumerate(xs):
        x[i, : len(xi)] = xi
    x.requires_grad_(True)
    out = model(x, ch_mask=mask)  # (batch, n_chans, n_patches, emb_dim)
    out.sum().backward()
    assert torch.equal(x.grad[~mask], torch.zeros_like(x.grad[~mask]))
    assert torch.equal(out[~mask], torch.zeros_like(out[~mask]))
    with torch.no_grad():
        for i, xi in enumerate(xs):
            torch.testing.assert_close(out[i, : len(xi)], model(xi[None])[0])
        features = model(x, ch_mask=mask, return_features=True)["features"]
        scripted = torch.jit.script(model)(x.detach(), ch_mask=mask)
    torch.testing.assert_close(features, out.detach())
    torch.testing.assert_close(scripted, out.detach())


def _small_meggpt():
    # Two layers: 8 steps per segment, then 2 patches of 4 steps -> 2 steps.
    return MEGGPT(n_chans=3, n_outputs=3, n_times=16, n_tokens=8, n_codes=16,
                  rnn_hidden_size=8, embed_dim=16, num_heads=2, feedforward_dim=32,
                  n_patches=(8, 2), patch_lens=(1, 4), n_last_steps=2,
                  patch_band=(2, None))


def test_meggpt_ch_mask_features_match_each_recording_alone():
    """MEGGPT's head flattens the channels; its features and tokens are masked."""
    torch.manual_seed(0)
    model = _small_meggpt().eval()
    x, mask, xs = _padded_batch(16)
    x.requires_grad_(True)
    out = model(x, ch_mask=mask, return_features=True)
    assert out["features"].shape == (3, 3, 16)
    assert out["tokens"].shape == (3, 3, 4, 16)
    torch.testing.assert_close(out["features"], out["tokens"].mean(dim=2))
    out["tokens"].sum().backward()
    assert x.grad is None  # the tokenizer's argmax passes no gradient
    assert all(p.grad.isfinite().all() for p in model.parameters() if p.grad is not None)
    with torch.no_grad():
        scripted = torch.jit.script(model)(x.detach(), ch_mask=mask, return_features=True)
        for i, xi in enumerate(xs):
            alone = model(xi[None], return_features=True)
            for key in ("features", "tokens"):
                torch.testing.assert_close(out[key][i, : len(xi)].detach(), alone[key][0])
    for key in ("features", "tokens"):
        assert out[key].shape[:2] == mask.shape
        assert torch.equal(out[key][~mask], torch.zeros_like(out[key][~mask]))
        torch.testing.assert_close(scripted[key], out[key].detach())


def test_luna_reconstruction_refuses_ch_mask():
    # The reconstruction decoder attends across every channel query.
    model = LUNA(n_chans=3, n_outputs=0, n_times=80, patch_size=20, embed_dim=32,
                 num_queries=2, depth=1)
    mask = torch.arange(3) < torch.tensor(N_REAL)[:, None]
    with pytest.raises(ValueError, match="ch_mask needs n_outputs > 0"):
        model(torch.randn(len(N_REAL), 3, 80), ch_mask=mask)


def test_ch_mask_is_refused_under_a_channel_strategy():
    # The channel layer would mix padded channels into the real ones.
    # FP1, F7, T7 form two of BIOT's bipolar pretraining channels.
    chs_info = [{"ch_name": name} for name in ("FP1", "F7", "T7")]
    model = BIOT(chs_info=chs_info, n_outputs=3, n_times=400, sfreq=200,
                 embed_dim=32, num_heads=2, num_layers=1, channel_strategy="zero")
    mask = torch.arange(3) < torch.tensor(N_REAL)[:, None]
    x = torch.randn(len(N_REAL), 3, 400)
    with pytest.raises(ValueError, match="ch_mask needs channel_strategy='native'"):
        model(x, ch_mask=mask)
    with pytest.raises(ValueError, match="ch_mask needs channel_strategy='native'"):
        model(x, False, mask)  # positional ch_mask


def test_diver1_ch_mask_needs_per_sample_metadata():
    model, n_times, (_, build, _) = _CASES["DIVER1"]()
    x = torch.randn(2, 2, n_times)
    mask = torch.tensor([[True, False], [True, True]])
    for metadata in (None, build(0, 2)):  # default montage, shared montage
        with pytest.raises(ValueError, match="per-sample chan_metadata"):
            model(x, chan_metadata=metadata, ch_mask=mask)


@pytest.mark.parametrize("spatial_scale", ["coords", "parcels", "lobes"])
@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("scripted", [False, True])
def test_barista_spatial_indices_batch_size(spatial_scale, masked, scripted):
    model = BaRISTA(
        n_chans=2, n_outputs=3, n_times=128, spatial_scale=spatial_scale,
        patch_size=32, d_model=16, n_layers=1, num_heads=2, cnn_depth=1,
        pooling="mean",
    ).eval()
    if scripted:
        model = torch.jit.script(model)
    x = torch.randn(1, 2, 128)
    mask = torch.tensor([[True, False]]) if masked else None
    indices = torch.tensor([1, 2])
    if spatial_scale == "coords":
        indices = indices[:, None].expand(-1, 3)
    with torch.no_grad():
        shared = model(x, spatial_indices=indices, ch_mask=mask)
        batched = model(x, spatial_indices=indices[None], ch_mask=mask)
        assert shared.shape == batched.shape == (1, 3)
        torch.testing.assert_close(shared, batched)
        wrong_batch = torch.stack([indices, indices])
        error = torch.jit.Error if scripted else ValueError
        with pytest.raises(error, match="spatial indices for batch size"):
            model(x, spatial_indices=wrong_batch, ch_mask=mask)


def test_epint_masked_nondivisible_window_normalizes_before_crop():
    torch.manual_seed(42)
    model = EpiNT(
        n_chans=2, n_outputs=2, n_times=100, patch_size=32,
        embed_dim=16, n_layers=1, n_heads=2, ffn_dim=32,
    ).eval()
    x = torch.randn(2, 2, 100)
    mask = torch.tensor([[True, False], [True, True]])
    x[~mask] = float("nan")
    shifted = x.clone()
    shifted[..., 96:] += 10.0
    with torch.no_grad():
        out = model(x, ch_mask=mask)
        expected = torch.cat([model(x[i:i + 1, mask[i]]) for i in range(2)])
        torch.testing.assert_close(out, expected)
        assert not torch.allclose(out, model(shifted, ch_mask=mask))
        torch.testing.assert_close(torch.jit.script(model)(x, ch_mask=mask), out)
    x.requires_grad_(True)
    model(x, ch_mask=mask).sum().backward()
    assert torch.isfinite(x.grad).all()
    assert torch.equal(x.grad[~mask], torch.zeros_like(x.grad[~mask]))


@pytest.mark.parametrize("policy,expected_indices", [
    ("pad", [6, 7, 14, 15, 22, 23]), ("crop", [6, 7, 14, 15]),
])
@pytest.mark.parametrize("scripted", [False, True])
def test_meggpt_next_token_metadata_and_training(policy, expected_indices, scripted):
    """A real vocab-head step selects hidden, valid shifted targets, not padding."""
    torch.manual_seed(0)
    model = _small_meggpt().eval()
    model.on_non_divisible = policy
    if scripted:
        model = torch.jit.script(model)
    ids = torch.randint(0, 8, (2, 3, 18))
    lengths = torch.tensor([9, 18])
    ch_mask = torch.tensor([[True, False, False], [True, True, True]])
    time = torch.arange(18)[None, None]
    valid = (time < lengths[:, None, None]) & ch_mask[:, :, None]
    ids[~valid.expand_as(ids)] = -999  # must be sanitized before embedding/CE
    hide = torch.zeros_like(ids, dtype=torch.bool)
    hide[..., 7] = True
    hide[..., 15] = True
    hide[~valid.expand_as(ids)] = True  # never contributes
    encoded = model.encode_tokens(ids, hide, ch_mask, lengths)
    indices = encoded["output_indices"]
    assert indices.tolist() == expected_indices
    assert encoded["input_indices"].tolist() == list(range(24 if policy == "pad" else 16))
    expected_valid = (indices[None, None] + 1 < lengths[:, None, None]) & ch_mask[:, :, None]
    assert torch.equal(encoded["target_valid"], expected_valid.expand_as(encoded["target_valid"]))
    for j, index in enumerate(expected_indices):
        if index + 1 < 18:
            real = encoded["target_valid"][..., j]
            assert torch.equal(encoded["next_token_targets"][..., j][real], ids[..., index + 1][real])
    assert not encoded["next_token_targets"][~encoded["target_valid"]].any()
    selected = encoded["prediction_mask"]
    assert selected.sum().item() == 7  # first recording1, second3 *2 targets
    assert not selected[~encoded["target_valid"]].any()
    head = torch.nn.Linear(16, 8)
    optimizer = torch.optim.SGD([*model.parameters(), *head.parameters()], lr=0.01)
    before = model.layers[0].attn.v_proj.weight.detach().clone()
    logits = head(encoded["tokens"])
    targets = encoded["next_token_targets"]
    loss = torch.nn.functional.cross_entropy(logits[selected], targets[selected])
    assert loss.isfinite()
    # CE is evaluated only at selected/valid entries: invalid sentinel labels
    # and arbitrary unselected logits cannot enter the training reduction.
    changed_targets = targets.clone()
    changed_targets[~selected] = -999
    changed_logits = logits.detach().clone()
    changed_logits[~selected] = float("nan")
    torch.testing.assert_close(loss.detach(), torch.nn.functional.cross_entropy(
        changed_logits[selected], changed_targets[selected]))
    optimizer.zero_grad()
    loss.backward()
    for parameter in (model.token_embedding.weight, model.layers[0].attn.v_proj.weight,
                      model.layers[1].attn.v_proj.weight, head.weight):
        assert parameter.grad is not None
        assert parameter.grad.isfinite().all() and parameter.grad.abs().sum() > 0
    assert all(not p.requires_grad and p.grad is None for p in model.tokenizer.parameters())
    optimizer.step()
    assert not torch.equal(before, model.layers[0].attn.v_proj.weight)
    # A defined empty selection: downstream training rejects instead of CE NaN.
    empty = model.encode_tokens(ids, torch.zeros_like(hide), ch_mask, lengths)
    assert not empty["prediction_mask"].any()


def test_meggpt_corruption_is_before_context_and_clean_targets_stay_clean():
    torch.manual_seed(0)
    model = _small_meggpt().eval()
    ids = torch.randint(0, 8, (2, 3, 16))
    hide = torch.zeros_like(ids, dtype=torch.bool)
    hide[..., 7] = True
    seen = []
    hook = model.layers[0].register_forward_pre_hook(lambda m, args: seen.append(args[0].detach()))
    clean = model.encode_tokens(ids)
    masked = model.encode_tokens(ids, hide)
    hook.remove()
    expected = model.token_embedding(model.to_segments(ids))
    expected = expected.masked_fill(model.to_segments(hide).unsqueeze(-1), 0.0)
    expected = expected + model.time_embedding[:, None]
    expected = expected + model.channel_embedding
    torch.testing.assert_close(seen[1], expected, rtol=0, atol=0)
    assert not torch.equal(seen[0], seen[1])
    assert torch.equal(clean["next_token_targets"], masked["next_token_targets"])
    assert not torch.equal(clean["tokens"], masked["tokens"])
    # Future corruption cannot affect the first segment's outputs.
    future = hide.clone().zero_()
    future[..., 15] = True
    out = model.encode_tokens(ids, future)
    torch.testing.assert_close(clean["tokens"][:, :, :2], out["tokens"][:, :, :2], rtol=0, atol=0)
    # Explicit error for a mask that would otherwise broadcast silently.
    with pytest.raises(ValueError, match="token_mask"):
        model.encode_tokens(ids, hide[:, :1])
    with pytest.raises(ValueError, match="token_mask"):
        model.encode_tokens(ids, hide.float())


@pytest.mark.parametrize("scripted", [False, True])
def test_meggpt_no_corruption_and_false_mask_match_including_rng_and_grad(scripted):
    torch.manual_seed(0)
    model = _small_meggpt().train()
    if scripted:
        model = torch.jit.script(model)
    x = torch.randn(2, 3, 16, requires_grad=True)
    mask = torch.zeros_like(x, dtype=torch.bool)
    outputs, states, grads = [], [], []
    for hide in (None, mask):
        torch.manual_seed(12)
        model.zero_grad()
        out = model(x, return_features=True, token_mask=hide)
        out["tokens"].sum().backward()
        outputs.append(out)
        states.append(torch.get_rng_state())
        grads.append({k: p.grad.clone() for k, p in model.named_parameters() if p.grad is not None})
    assert torch.equal(states[0], states[1])
    for key in ("features", "tokens"):
        torch.testing.assert_close(outputs[0][key], outputs[1][key], rtol=0, atol=0)
    assert grads[0].keys() == grads[1].keys()
    for key in grads[0]:
        torch.testing.assert_close(grads[0][key], grads[1][key], rtol=0, atol=0)
    assert x.grad is None
