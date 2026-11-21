import pytest
import torch

from braindecode.models import MEGGPT


def _small_meggpt(on_non_divisible="pad"):
    # segment_len = 8; the last 2 steps of each segment (positions 6, 7) are kept.
    # Two layers: 8 steps per segment, then 2 patches of 4 steps -> 2 steps.
    return MEGGPT(
        n_chans=3, n_outputs=3, n_times=16, n_tokens=8, n_codes=16,
        rnn_hidden_size=8, embed_dim=16, num_heads=2, feedforward_dim=32,
        n_patches=(8, 2), patch_lens=(1, 4), n_last_steps=2, patch_band=(2, None),
        on_non_divisible=on_non_divisible,
    )  # fmt: skip


@pytest.mark.parametrize(
    "n_times, n_real_steps",
    [(20, 4), (22, 4), (23, 5), (24, 6), (30, 6)],
)
def test_pad_features_skip_kept_steps_after_the_input(n_times, n_real_steps):
    # Window of 20 samples: the 3rd segment holds 4 real samples, so both of
    # its kept steps (positions 6 and 7) are padding and must not be averaged.
    torch.manual_seed(0)
    model = _small_meggpt().eval()
    x = torch.randn(2, 3, n_times)
    out = model(x, return_features=True)
    tokens = out["tokens"]  # (batch, n_chans, n_segments * n_last_steps, embed_dim)
    torch.testing.assert_close(out["features"], tokens[:, :, :n_real_steps].mean(dim=2))


def test_pad_features_of_a_window_shorter_than_the_kept_steps_are_finite():
    model = _small_meggpt().eval()
    out = model(torch.randn(2, 3, 4), return_features=True)
    assert torch.isfinite(out["features"]).all()


def test_raw_and_cached_features_use_the_available_channel_embeddings():
    model = _small_meggpt().eval()
    for n_chans in (2, 4):
        x = torch.randn(1, n_chans, 16)
        ids = model.tokenizer(x)
        if n_chans <= model.n_chans:
            torch.testing.assert_close(
                model(x, return_features=True)["tokens"],
                model.encode_tokens(ids)["tokens"],
            )
        else:
            with pytest.raises(ValueError, match="channels"):
                model(x, return_features=True)
            with pytest.raises(ValueError, match="channels"):
                model.encode_tokens(ids)


@pytest.mark.parametrize(
    "policy, expected_indices",
    [("pad", [6, 7, 14, 15, 22, 23]), ("crop", [6, 7, 14, 15])],
)
@pytest.mark.parametrize("scripted", [False, True])
def test_next_token_targets_select_hidden_valid_positions(
    policy, expected_indices, scripted
):
    """The targets are the clean ids one step after each output; only hidden
    ones of real channels and steps are selected, never the padding."""
    torch.manual_seed(0)
    model = _small_meggpt(policy).eval()
    if scripted:
        model = torch.jit.script(model)
    ids = torch.randint(0, 8, (2, 3, 18))
    lengths = torch.tensor([9, 18])
    ch_mask = torch.tensor([[True, False, False], [True, True, True]])
    valid = (torch.arange(18) < lengths[:, None, None]) & ch_mask[:, :, None]
    ids[~valid.expand_as(ids)] = -999  # must be sanitized before the embedding
    hide = torch.zeros_like(ids, dtype=torch.bool)
    hide[..., [7, 15]] = True
    hide[~valid.expand_as(ids)] = True  # never contributes
    encoded = model.encode_tokens(ids, hide, ch_mask, lengths)
    indices = encoded["output_indices"]
    assert indices.tolist() == expected_indices
    target_valid = encoded["target_valid"]
    available = (indices + 1 < lengths[:, None, None]) & ch_mask[:, :, None]
    assert torch.equal(target_valid, available)
    for j, index in enumerate(expected_indices):
        if index + 1 < 18:
            real = target_valid[..., j]
            assert torch.equal(encoded["next_token_targets"][..., j][real], ids[..., index + 1][real])
    assert not encoded["next_token_targets"][~target_valid].any()
    selected = encoded["prediction_mask"]
    assert selected.sum().item() == 7  # 1 target of the first recording, 3 * 2 of the second
    assert not selected[~target_valid].any()
    # Nothing hidden: nothing selected, so a downstream loss can reject it.
    assert not model.encode_tokens(ids, torch.zeros_like(hide), ch_mask, lengths)["prediction_mask"].any()


def test_token_mask_zeroes_embeddings_and_leaves_the_targets_clean():
    torch.manual_seed(0)
    model = _small_meggpt().eval()
    x = torch.randn(2, 3, 16)
    ids = torch.randint(0, 8, (2, 3, 16))
    hide = torch.zeros_like(ids, dtype=torch.bool)
    hide[..., 7] = True
    clean, masked = model.encode_tokens(ids), model.encode_tokens(ids, hide)
    assert not torch.equal(clean["tokens"], masked["tokens"])
    assert torch.equal(clean["next_token_targets"], masked["next_token_targets"])
    # Hiding zeroes the embedding: it is not the embedding of token id 0.
    assert not torch.equal(masked["tokens"], model.encode_tokens(ids.masked_fill(hide, 0))["tokens"])
    # An all-False mask changes nothing, in encode_tokens and in forward.
    none = torch.zeros_like(hide)
    assert torch.equal(model.encode_tokens(ids, none)["tokens"], clean["tokens"])
    features = model(x, return_features=True)
    assert torch.equal(model(x, return_features=True, token_mask=none)["tokens"], features["tokens"])
    # Causal: hiding a later segment leaves the first segment's outputs unchanged.
    future = torch.zeros_like(hide)
    future[..., 15] = True
    assert torch.equal(model.encode_tokens(ids, future)["tokens"][:, :, :2], clean["tokens"][:, :, :2])
    # A mask that would broadcast silently, or that is not boolean, is an error.
    for bad in (hide[:, :1], hide.float()):
        with pytest.raises(ValueError, match="token_mask"):
            model.encode_tokens(ids, bad)


def test_ch_mask_features_match_each_recording_alone():
    """Padded channels give zero features and tokens, and every recording gets
    what it gets alone, also scripted (the head is tied to ``n_chans``, so
    ``ch_mask`` needs ``return_features``)."""
    torch.manual_seed(0)
    model = _small_meggpt().eval()
    n_real = [1, 3, 2]
    mask = torch.arange(3) < torch.tensor(n_real)[:, None]  # (batch, n_chans)
    x = torch.randn(3, 3, 16).masked_fill(~mask[..., None], float("nan"))
    x.requires_grad_(True)
    out = model(x, ch_mask=mask, return_features=True)
    assert out["features"].shape == (3, 3, 16)
    assert out["tokens"].shape == (3, 3, 4, 16)
    out["tokens"].sum().backward()
    assert x.grad is None  # the argmax of the tokenizer passes no gradient
    assert all(p.grad.isfinite().all() for p in model.parameters() if p.grad is not None)
    with torch.no_grad():
        scripted = torch.jit.script(model)(x.detach(), ch_mask=mask, return_features=True)
        for i, k in enumerate(n_real):
            alone = model(x[i : i + 1, :k].detach(), return_features=True)
            for key in ("features", "tokens"):
                torch.testing.assert_close(out[key][i, :k].detach(), alone[key][0])
    for key in ("features", "tokens"):
        assert torch.equal(out[key][~mask], torch.zeros_like(out[key][~mask]))
        torch.testing.assert_close(scripted[key], out[key].detach())
