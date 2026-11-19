"""TinyMyo against a transcription of the authors' classifier (``models/TinyMyo.py``
of pulp-bio/BioFoundation@d4484fe), and against logits of the authors' code on
the released EPN-612 checkpoint."""

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from braindecode.models import TinyMyo


class _Block(nn.Module):
    """``RotaryTransformerBlock`` with torchtune's RoPE, as in the reference."""

    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = nn.Module()
        self.attn.qkv, self.attn.proj = nn.Linear(dim, 3 * dim), nn.Linear(dim, dim)
        self.mlp = nn.Module()
        self.mlp.fc1, self.mlp.fc2 = nn.Linear(dim, 4 * dim), nn.Linear(4 * dim, dim)

    @staticmethod
    def rope(x, pos):  # x: (B, N, H, D)
        d = x.shape[-1]
        theta = 1.0 / 10000 ** (torch.arange(0, d, 2).float() / d)
        angle = pos.float()[:, None] * theta  # (N, D / 2)
        cos, sin = angle.cos()[:, None], angle.sin()[:, None]
        x = x.float().reshape(*x.shape[:-1], -1, 2)
        even, odd = x[..., 0], x[..., 1]
        return torch.stack(
            [even * cos - odd * sin, odd * cos + even * sin], -1
        ).flatten(3)

    def forward(self, x, pos):
        b, n, c = x.shape
        qkv = self.attn.qkv(self.norm1(x)).reshape(b, n, 3, self.heads, c // self.heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        q = self.rope(q.transpose(1, 2), pos).transpose(1, 2)
        k = self.rope(k.transpose(1, 2), pos).transpose(1, 2)
        a = F.scaled_dot_product_attention(q, k, v).transpose(2, 1).reshape(b, n, c)
        x = x + self.attn.proj(a)
        return x + self.mlp.fc2(F.gelu(self.mlp.fc1(self.norm2(x))))


class _TinyMyoReference(nn.Module):
    """``TinyMyo(task="classification")`` of the reference, in eval mode."""

    def __init__(self, n_chans, n_outputs, patch_size, dim, depth, heads):
        super().__init__()
        self.patch_size = patch_size
        self.channel_embed = nn.Parameter(torch.randn(1, 16, 1, dim))
        self.patch_embedding = nn.Module()
        self.patch_embedding.patch_embed = nn.Module()
        self.patch_embedding.patch_embed.proj = nn.Conv2d(
            1, dim, (1, patch_size), stride=(1, patch_size)
        )
        self.blocks = nn.ModuleList(_Block(dim, heads) for _ in range(depth))
        self.norm = nn.LayerNorm(dim)
        self.model_head = nn.Module()
        self.model_head.classifier = nn.Linear(n_chans * dim, n_outputs)

    def forward(self, x):
        b, c, t = x.shape
        p = t // self.patch_size
        x = self.patch_embedding.patch_embed.proj(x.unsqueeze(1))  # (B, D, C, P)
        x = x.permute(0, 2, 3, 1) + self.channel_embed[:, :c]  # (B, C, P, D)
        x = x.flatten(1, 2)  # "B C P D -> B (C P) D"
        pos = torch.arange(p).repeat(c)  # restarts at each channel
        for block in self.blocks:
            x = block(x, pos)
        x = self.norm(x)
        x = x.reshape(b, c, p, -1).permute(0, 2, 1, 3).flatten(2)  # b p (c d)
        return self.model_head.classifier(x.mean(1))


def _reference_name(key):
    key = key.replace("patch_embed.proj.", "patch_embedding.patch_embed.proj.")
    return key.replace("final_layer.", "model_head.classifier.")


def test_tinymyo_matches_the_reference():
    """``load_state_dict`` takes the reference keys; equal logits, input and
    parameter gradients."""
    torch.manual_seed(0)
    model = TinyMyo(
        n_chans=3,
        n_outputs=5,
        n_times=200,
        patch_size=10,
        embed_dim=32,
        depth=2,
        num_heads=2,
    ).eval()
    reference = _TinyMyoReference(3, 5, patch_size=10, dim=32, depth=2, heads=2)
    reference.eval()
    model.load_state_dict(
        {f"model.{k}": v for k, v in reference.state_dict().items()}, strict=True
    )

    x = torch.randn(4, 3, 200)
    ours_x, their_x = x.clone().requires_grad_(), x.clone().requires_grad_()
    ours, theirs = model(ours_x), reference(their_x)
    torch.testing.assert_close(ours, theirs, rtol=0, atol=1e-5)

    weight = torch.randn(ours.shape)
    (ours * weight).sum().backward()
    (theirs * weight).sum().backward()
    torch.testing.assert_close(ours_x.grad, their_x.grad, rtol=0, atol=1e-5)
    theirs = dict(reference.named_parameters())
    for key, param in model.named_parameters():
        their_grad = theirs[_reference_name(key)].grad.reshape(param.shape)
        torch.testing.assert_close(param.grad, their_grad, rtol=0, atol=1e-5)


@pytest.mark.parametrize("task_head", ["decoder_pred", "regressor.0"])
def test_tinymyo_loads_released_encoder_but_checks_native_head(task_head):
    torch.manual_seed(123)
    reference_bias = nn.Conv2d(1, 12, (1, 20), stride=(1, 20)).bias.detach()
    torch.manual_seed(123)
    model = TinyMyo(
        n_chans=2, n_outputs=3, n_times=40, embed_dim=12, num_heads=2, depth=1
    )
    torch.testing.assert_close(
        model.patch_embed.proj.bias, reference_bias, rtol=0, atol=0
    )
    reference = _TinyMyoReference(2, 3, patch_size=20, dim=12, depth=1, heads=2)
    weights = {
        f"model.{k}": v
        for k, v in reference.state_dict().items()
        if not k.startswith("model_head.")
    }
    weights[f"model.model_head.{task_head}.weight"] = torch.ones(12, 12)
    head = {k: v.clone() for k, v in model.final_layer.state_dict().items()}
    model.load_state_dict(weights, strict=True)
    for k, v in model.final_layer.state_dict().items():
        torch.testing.assert_close(v, head[k], rtol=0, atol=0)
    torch.testing.assert_close(
        model.channel_embed, reference.channel_embed, rtol=0, atol=0
    )

    native_encoder = {
        k: v for k, v in model.state_dict().items() if not k.startswith("final_layer.")
    }
    with pytest.raises(RuntimeError, match="final_layer.weight"):
        model.load_state_dict(native_encoder, strict=True)
    result = model.load_state_dict(native_encoder, strict=False)
    assert set(result.missing_keys) == {"final_layer.weight", "final_layer.bias"}


@pytest.mark.parametrize("kwargs", [{"n_chans": 0}, {"num_heads": 0}, {"embed_dim": 0}])
def test_tinymyo_requires_positive_geometry(kwargs):
    config = dict(
        n_chans=2, n_outputs=3, n_times=40, embed_dim=12, num_heads=2, depth=1
    )
    with pytest.raises(ValueError, match="n_chans|num_heads|embed_dim"):
        TinyMyo(**(config | kwargs))


# Logits of the authors' TinyMyo (BioFoundation@d4484fe) with
# EPN612/EPN_finetune_5sec.safetensors on the input below.
_EPN612_LOGITS = [
    [3.051263, -1.176474, -1.150631, -1.122424, -1.135589, -1.118205],
    [3.069420, -1.165260, -1.154662, -1.129658, -1.143181, -1.125813],
]


@pytest.mark.network
@pytest.mark.huggingface
def test_tinymyo_released_classifier_matches_the_authors_logits():
    pytest.importorskip("huggingface_hub")
    model = TinyMyo.from_pretrained(
        "PulpBio/TinyMyo",
        filename="EPN612/EPN_finetune_5sec.safetensors",
        revision="bc14906a46f74e77d44a14062e696bbb4bf72ae5",
        n_chans=8,
        n_outputs=6,
        n_times=1000,
    ).eval()
    x = torch.randn(2, 8, 1000, generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        logits = model(x)
    torch.testing.assert_close(logits, torch.tensor(_EPN612_LOGITS), rtol=0, atol=1e-5)
