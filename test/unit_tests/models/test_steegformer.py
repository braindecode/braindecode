import pytest
import torch
from torch import nn

from braindecode.models import STEEGFormer
from braindecode.models.steegformer import _TemporalPositionalEncoding


def _small_steegformer(**kwargs):
    params = {
        "n_chans": 2,
        "n_times": 32,
        "n_outputs": 3,
        "patch_size": 16,
        "embed_dim": 8,
        "depth": 1,
        "num_heads": 2,
        "chan_pos_idx": [0, 1],
    }
    params.update(kwargs)
    return STEEGFormer(**params)


def test_temporal_patch_positions_start_at_zero():
    pos = _TemporalPositionalEncoding(embed_dim=8, max_len=4)

    patch_pos = pos(seq=3)

    torch.testing.assert_close(patch_pos[0, :, 0, :], pos.pe[:3])
    torch.testing.assert_close(pos.cls_token_encoding(), pos.pe[0])


def test_forward_rejects_windows_shorter_than_one_patch():
    model = _small_steegformer()

    with pytest.raises(ValueError, match="at least one full temporal patch"):
        model(torch.randn(1, 2, 15))


def test_mlp_ratio_must_be_whole_number():
    with pytest.raises(ValueError, match="whole number"):
        _small_steegformer(mlp_ratio=3.5)


class _AddOneNorm(nn.Module):
    def forward(self, x):
        return x + 1.0


def _tokens_before_final_norm(model, x):
    seq = x.shape[-1] // model.patch_size
    x = x[..., : seq * model.patch_size]
    tokens = model.patch_embed(x)
    tokens = tokens + model.temporal_pos(seq) + model.channel_pos(model.channel_indices)
    tokens = tokens.reshape(tokens.shape[0], -1, tokens.shape[-1])

    cls = model.cls_token + model.temporal_pos.cls_token_encoding()
    cls = cls.expand(tokens.shape[0], -1, -1)
    return torch.cat([cls, tokens], dim=1)


def test_return_features_and_average_pool_use_final_norm():
    torch.manual_seed(0)
    model = _small_steegformer()
    model.encoder = nn.Identity()
    model.norm = _AddOneNorm()
    model.final_layer = nn.Identity()
    model.eval()
    x = torch.randn(2, 2, 32)

    expected = _tokens_before_final_norm(model, x) + 1.0

    with torch.no_grad():
        out = model(x, return_features=True)
        logits = model(x)

    torch.testing.assert_close(out["features"], expected[:, 1:, :])
    torch.testing.assert_close(out["cls_token"], expected[:, 0, :])
    torch.testing.assert_close(logits, expected[:, 1:, :].mean(dim=1))
