import pytest
import torch

from braindecode.models import MEGGPT


def _model(on_non_divisible="pad"):
    # segment_len = 8; the last 2 steps of each segment (positions 6, 7) are kept.
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
    model = _model().eval()
    x = torch.randn(2, 3, n_times)
    out = model(x, return_features=True)
    tokens = out["tokens"]  # (batch, n_chans, n_segments * n_last_steps, embed_dim)
    torch.testing.assert_close(out["features"], tokens[:, :, :n_real_steps].mean(dim=2))


def test_pad_features_of_a_window_shorter_than_the_kept_steps_are_finite():
    model = _model().eval()
    out = model(torch.randn(2, 3, 4), return_features=True)
    assert torch.isfinite(out["features"]).all()


def test_raw_and_cached_features_use_the_available_channel_embeddings():
    model = _model().eval()
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
