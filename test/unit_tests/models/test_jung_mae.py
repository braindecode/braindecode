"""JungMAE against a transcription of the encoder path of the GeoNIRS authors'
masked autoencoder module (``model/`` of GeoNIRS@81f8995), which packs the
valid patches of each pair before the Transformer. Runs without the released
checkpoint."""

import math

import pytest
import torch
from torch import nn

from braindecode.models import JungMAE
from braindecode.models.util import _fnirs_chs_info


class _JungMAEReference(nn.Module):
    """``time_positions``, ``transformer``, ``patchify`` and ``_encode_visible``
    of the reference; ``valid`` is its ``valid_time_mask & valid_channel_mask``."""

    def __init__(self, patch_samples, fraction, dim, layers, heads, mlp_ratio):
        super().__init__()
        self.q = patch_samples
        self.min_valid = math.ceil(patch_samples * fraction)
        self.dim = dim
        self.patch_embed = nn.Linear(2 * patch_samples, dim)
        layer = nn.TransformerEncoderLayer(
            dim,
            heads,
            int(dim * mlp_ratio),
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, layers, norm=nn.LayerNorm(dim), enable_nested_tensor=False
        )

    def time_positions(self, count):
        position = torch.arange(count, dtype=torch.float32)[:, None]
        exponent = -math.log(10000) / self.dim
        frequency = torch.exp(torch.arange(0, self.dim, 2).float() * exponent)
        result = torch.empty(count, self.dim)
        result[:, 0::2] = (position * frequency).sin()
        result[:, 1::2] = (position * frequency).cos()
        return result

    def forward(self, hbo, hbr, valid):
        b, c, t = hbo.shape
        q = self.q
        signal = torch.where(valid[..., None], torch.stack((hbo, hbr), dim=-1), 0)
        patches = signal.reshape(b * c, t // q, q * 2)  # [HbO_t, HbR_t, ...]
        patch_valid = valid.reshape(b, c, t // q, q).sum(-1) >= self.min_valid
        visible = patch_valid.reshape(b * c, -1)
        n, p, _ = patches.shape
        output = patches.new_zeros(n, p, self.dim)
        active = visible.any(-1)
        selected, visibility = patches[active], visible[active]
        counts = visibility.sum(-1)
        length = int(counts.max())
        positions = torch.arange(p).expand(selected.shape[0], -1)
        indices = positions.masked_fill(~visibility, p).argsort(dim=-1)[:, :length]
        packed_valid = torch.arange(length)[None, :] < counts[:, None]
        index = indices[..., None].expand(-1, -1, selected.shape[-1])
        packed = selected.gather(1, index)
        tokens = self.patch_embed(packed) + self.time_positions(p)[indices]
        tokens = self.encoder(tokens, src_key_padding_mask=~packed_valid)
        tokens = tokens.masked_fill(~packed_valid[..., None], 0)
        full = tokens.new_zeros(selected.shape[0], p, self.dim)
        full.scatter_(1, indices[..., None].expand(-1, -1, self.dim), tokens)
        output[active] = full
        return output.reshape(b, c, p, -1), patch_valid


# The reference packs the valid tokens of each pair and the port masks the
# invalid ones: the same arithmetic in another order, so float32 rounding
# (up to 1.3e-6 on CI runners) depends on the machine
_ATOL = 1e-5


def _backbone(model):
    state = model.state_dict()
    return {k: v for k, v in state.items() if not k.startswith("final_layer.")}


def _missing_samples(x, n_pairs):
    """Stacked ``x``: patches above and below the validity threshold, a pair
    without data and a window without data."""
    x = x.clone()
    x[0, 0, 3:6] = float("nan")  # pair 1, patch 1: 7 of 10 HbO samples
    x[0, n_pairs, 12:14] = float("nan")  # pair 1, patch 2: 8 of 10 HbR samples
    x[0, n_pairs + 1, 20:25] = float("nan")  # pair 2, patch 3: 5 of 10 HbR
    x[1, 1] = float("nan")  # pair 2 without HbO
    x[2] = float("nan")  # a window without data
    return x


def _to_interleaved(x, n_pairs):
    """[HbO_1, ..., HbR_1, ...] -> MNE's [HbO_1, HbR_1, ...] order."""
    return x.reshape(x.shape[0], 2, n_pairs, -1).transpose(1, 2).flatten(1, 2)


def _reference_encode(reference, x, n_pairs):
    hbo, hbr = x[:, :n_pairs], x[:, n_pairs:]
    valid = hbo.isfinite() & hbr.isfinite()
    return reference(torch.where(valid, hbo, 0.0), torch.where(valid, hbr, 0.0), valid)


@pytest.mark.parametrize(
    "n_pairs, n_times, kw, interleaved",
    [
        (
            3,
            60,
            dict(
                embed_dim=32,
                depth=2,
                num_heads=4,
                mlp_ratio=2.0,
                # 7.5 valid samples: ceil gives 8, so a patch with 7 is invalid
                minimum_patch_valid_fraction=0.75,
            ),
            False,
        ),
        (
            4,
            50,
            dict(
                patch_size=5,
                minimum_patch_valid_fraction=0.6,
                embed_dim=16,
                depth=1,
                num_heads=2,
                activation=nn.GELU,
            ),
            True,
        ),
    ],
)
def test_jung_mae_matches_the_reference_encoder(n_pairs, n_times, kw, interleaved):
    """Strict load into the transcription; equal foundation, validity, logits
    from the flattened tokens, parameter and input gradients (eval and train
    mode without dropout), and two AdamW steps."""
    torch.manual_seed(0)
    chs = _fnirs_chs_info(n_pairs, interleaved=interleaved)
    model = JungMAE(chs_info=chs, n_outputs=3, n_times=n_times, drop_prob=0.0, **kw)
    reference = _JungMAEReference(
        model.patch_size,
        kw.get("minimum_patch_valid_fraction", 0.8),
        kw["embed_dim"],
        kw["depth"],
        kw["num_heads"],
        kw.get("mlp_ratio", 4.0),
    )
    reference.load_state_dict(_backbone(model), strict=True)
    stacked = _missing_samples(torch.randn(3, 2 * n_pairs, n_times), n_pairs)
    # the same data in the model's channel order
    x = _to_interleaved(stacked, n_pairs) if interleaved else stacked
    optimizers = [torch.optim.AdamW(m.parameters(), lr=1e-3) for m in (model, reference)]
    for step, mode in enumerate(("eval", "train", "train")):
        model.train(mode == "train")
        reference.train(mode == "train")
        ours_x = x.clone().requires_grad_(True)
        their_x = stacked.clone().requires_grad_(True)
        out = model.encode(ours_x)
        foundation, patch_valid = _reference_encode(reference, their_x, n_pairs)
        assert torch.equal(out["valid"], patch_valid)
        torch.testing.assert_close(out["foundation"], foundation, rtol=0, atol=_ATOL)
        if step == 0:
            with torch.no_grad():
                logits = model.final_layer(foundation.flatten(1))  # benchmark head
                torch.testing.assert_close(model(x), logits, rtol=0, atol=_ATOL)
        weight = torch.randn(foundation.shape)
        for opt in optimizers:
            opt.zero_grad()
        (out["foundation"] * weight).sum().backward()
        (foundation * weight).sum().backward()
        ours = dict(model.named_parameters())
        for name, param in reference.named_parameters():
            torch.testing.assert_close(ours[name].grad, param.grad, rtol=0, atol=1e-5)
        their_grad = their_x.grad
        if interleaved:
            their_grad = _to_interleaved(their_grad, n_pairs)
        torch.testing.assert_close(ours_x.grad, their_grad, rtol=0, atol=1e-5)
        if mode == "train":
            for m in (model, reference):
                # the softmax ignores the key bias: its gradient is rounding
                # noise, which Adam would scale up to the learning rate
                for attention in m.modules():
                    if isinstance(attention, nn.MultiheadAttention):
                        dim = attention.embed_dim
                        attention.in_proj_bias.grad[dim : 2 * dim] = 0
            for opt in optimizers:
                opt.step()
    ours = dict(model.named_parameters())
    for name, param in reference.named_parameters():
        torch.testing.assert_close(ours[name], param, rtol=0, atol=1e-5)


def test_jung_mae_defaults_are_the_released_configuration():
    """The defaults are jung_mae.pt's model_config (patch 10, threshold 0.8,
    192 wide, 4 layers, 8 heads, mlp_ratio 4, GELU, dropout 0.1) and the
    benchmark's head dropout 0.1."""
    torch.manual_seed(0)
    model = JungMAE(n_chans=4, n_outputs=2, n_times=60).eval()
    reference = _JungMAEReference(10, 0.8, 192, 4, 8, 4.0).eval()
    reference.load_state_dict(_backbone(model), strict=True)
    x = _missing_samples(torch.randn(3, 4, 60), 2)
    with torch.no_grad():
        foundation, patch_valid = _reference_encode(reference, x, 2)
        out = model.encode(x)
    assert torch.equal(out["valid"], patch_valid)
    torch.testing.assert_close(out["foundation"], foundation, rtol=0, atol=_ATOL)
    dropouts = [m.p for m in model.encoder.modules() if isinstance(m, nn.Dropout)]
    assert len(dropouts) == 4 * 3 and set(dropouts) == {0.1}
    assert model.head_dropout.p == 0.1


def test_jung_mae_head_dropout_comes_before_final_layer():
    torch.manual_seed(0)
    model = JungMAE(
        n_chans=4, n_outputs=2, n_times=60, depth=1, drop_prob=0.0, head_drop_prob=0.5
    )
    x = torch.randn(2, 4, 60)
    with torch.no_grad():
        plain = model.eval()(x, return_features=True)["features"]
        dropped = model.train()(x, return_features=True)["features"]
    kept = dropped != 0
    assert 0.3 < kept.float().mean() < 0.7
    torch.testing.assert_close(dropped[kept], 2 * plain[kept])


def _assert_xavier_init(model):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            fan_out, fan_in = module.weight.shape
            bound = math.sqrt(6 / (fan_in + fan_out))
            assert module.weight.abs().max() <= bound, name
            assert module.weight.abs().max() > 0.9 * bound, name
            assert not module.bias.any(), name
    weights = [layer.self_attn.in_proj_weight for layer in model.encoder.layers]
    assert not torch.equal(weights[0], weights[1])  # each layer drawn, not cloned
    for layer in model.encoder.layers:
        assert not layer.self_attn.in_proj_bias.any()


def test_jung_mae_initialization_follows_the_reference():
    """Xavier-uniform Linear weights, zero biases, after the constructor,
    reset_parameters and reset_head."""
    model = JungMAE(
        n_chans=4, n_outputs=2, n_times=60, embed_dim=32, depth=2, num_heads=4
    )
    _assert_xavier_init(model)
    model.reset_parameters()
    _assert_xavier_init(model)
    model.reset_head(5)
    _assert_xavier_init(model)


def test_jung_mae_loads_the_released_payload_layout():
    """The released payload as saved by the reference (weights under "model",
    with the pretraining decoder) loads strictly and keeps the model's head."""
    model = JungMAE(n_chans=4, n_outputs=2, n_times=60, depth=1)
    head = {k: v.clone() for k, v in model.final_layer.state_dict().items()}
    backbone = {k: torch.randn_like(v) for k, v in _backbone(model).items()}
    decoder = {
        "decoder_embed.weight": torch.zeros(96, 192),
        "mask_token": torch.zeros(1, 1, 96),
        "decoder.layers.0.linear1.weight": torch.zeros(384, 96),
        "prediction.weight": torch.zeros(20, 96),
    }
    payload = {
        "format_version": 1,
        "model_config": {"patch_samples": 10},
        "model": {**backbone, **decoder},
    }
    model.load_state_dict(payload, strict=True)
    state = model.state_dict()
    assert all(torch.equal(state[k], v) for k, v in backbone.items())
    assert all(torch.equal(state["final_layer." + k], v) for k, v in head.items())
    # the override keeps the strict load: unknown or missing keys still raise
    with pytest.raises(RuntimeError, match="Unexpected key"):
        model.load_state_dict({"model": {**backbone, "encoder.extra": torch.zeros(1)}})
    missing = {k: v for k, v in backbone.items() if k != "patch_embed.weight"}
    with pytest.raises(RuntimeError, match="Missing key"):
        model.load_state_dict({"model": missing})


def test_jung_mae_head_kept_from_the_model_is_not_reported_as_loaded():
    """Through ``from_pretrained``, the head the override supplies is new."""
    from braindecode.models.base import _RECORD_LOADED_KEYS

    model = JungMAE(n_chans=4, n_outputs=2, n_times=60, depth=1)
    backbone = {k: v.clone() for k, v in _backbone(model).items()}
    token = _RECORD_LOADED_KEYS.set(True)
    try:
        model.load_state_dict({"model": backbone}, strict=True)
    finally:
        _RECORD_LOADED_KEYS.reset(token)
    loaded = model.__dict__["_pretrained_provenance"]["loaded"]
    assert "patch_embed.weight" in loaded
    assert not any(key.startswith("final_layer.") for key in loaded)


def test_jung_mae_activation_reaches_the_feed_forward_layers():
    model = JungMAE(n_chans=4, n_outputs=2, n_times=60, depth=2, activation=nn.ELU)
    assert all(isinstance(layer.activation, nn.ELU) for layer in model.encoder.layers)


def test_jung_mae_padded_samples_are_missing():
    """``on_non_divisible="pad"`` equals the NaN-padded window: the last patch
    holds 5 of 10 samples, under the 0.8 threshold."""
    torch.manual_seed(0)
    padded = JungMAE(
        n_chans=4, n_outputs=2, n_times=55, depth=1, on_non_divisible="pad"
    ).eval()
    plain = JungMAE(n_chans=4, n_outputs=2, n_times=60, depth=1).eval()
    plain.load_state_dict(padded.state_dict(), strict=True)
    x = torch.randn(2, 4, 55)
    x_nan = torch.cat([x, torch.full((2, 4, 5), float("nan"))], dim=-1)
    with torch.no_grad():
        out = padded.encode(x)
        assert not out["valid"][..., -1].any()
        torch.testing.assert_close(padded(x), plain(x_nan), rtol=0, atol=0)


@pytest.mark.parametrize("fraction", [0.0, -0.1, 1.5])
def test_jung_mae_rejects_a_fraction_outside_zero_one(fraction):
    with pytest.raises(ValueError, match="minimum_patch_valid_fraction"):
        JungMAE(
            n_chans=4, n_outputs=2, n_times=60, minimum_patch_valid_fraction=fraction
        )


def test_jung_mae_rejects_a_non_positive_patch_size():
    with pytest.raises(ValueError, match="patch_size must be positive, got 0"):
        JungMAE(n_chans=4, n_outputs=2, n_times=60, patch_size=0)


class _PassThrough(nn.Module):
    def forward(self, tokens, src_key_padding_mask=None):
        return tokens


def test_jung_mae_bfloat16_positions_are_cast_from_float32():
    """The position table is computed in float32 and cast, as the reference."""
    torch.manual_seed(0)
    model = JungMAE(
        n_chans=4, n_outputs=2, n_times=60, embed_dim=32, depth=1, num_heads=4
    ).eval()
    zero = {"patch_embed.weight", "patch_embed.bias"}
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in zero:
                param.zero_()
        model.encoder = _PassThrough()  # tokens = patch_embed + positions
        reference = _JungMAEReference(10, 0.8, 32, 1, 4, 4.0)
        expected = reference.time_positions(6).to(torch.bfloat16)
        out = model.to(torch.bfloat16).encode(torch.randn(1, 4, 60).bfloat16())
    assert torch.equal(out["foundation"][0, 0], expected)
