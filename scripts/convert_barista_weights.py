# Authors: Bruno Aristimunha <b.aristimunha@gmail.com>
# License: USC academic, non-commercial; see NOTICE.txt.
"""Convert and check all three released BaRISTA encoders on CPU.

Run from an editable Braindecode checkout with the ``hub`` extra installed::

    python scripts/convert_barista_weights.py --output-dir /tmp/barista-converted

The source revision and checkpoint hashes are pinned below. The numerical check
uses the released forward equations with explicit PyTorch attention instead of
CUDA-only xformers. It checks float32 encoder tokens, not downstream accuracy or
mixed-precision equivalence. Pooling and classification weights are initialized
locally because the releases contain neither. Fine-tune these before prediction.
"""

import argparse
import json
import shutil
from pathlib import Path

import pooch
import torch
import torch.nn.functional as F

from braindecode.models import BaRISTA

REVISION = "83b27375eba60e9eba9da4e7dd8fb283baace376"
SOURCE = f"https://raw.githubusercontent.com/ShanechiLab/BaRISTA/{REVISION}"
CHECKPOINTS = {
    "coords": (
        "chans_chans.ckpt",
        "400eeacc0697004cb81c9ecf754859da184ffeea40afc8ee7b5930c3b997e1d0",
    ),
    "parcels": (
        "parcels_chans.ckpt",
        "6c234517d286a8e710b09716dc88c713618670df523cfffb89e4c9073f2657c1",
    ),
    "lobes": (
        "lobes_chans.ckpt",
        "d810338a4929df0fb2421f342b3ee859f9fef269e35fb4f2fd9c55347a63324a",
    ),
}


def convert_weights(source, model):
    """Rename tensors and fuse value/gate projections; check rotary buffers."""
    source = dict(source)
    for i in range(model.n_layers):
        prefix = f"backbone.layers.{i}."
        frequency = source.pop(prefix + "attention.rotary_emb.inv_freq")
        torch.testing.assert_close(frequency, model.backbone.inv_freq, rtol=0, atol=0)
        for suffix in ("weight", "bias"):
            source[prefix + "mlp.0." + suffix] = torch.cat(
                [
                    source.pop(prefix + "mlp.up_proj." + suffix),
                    source.pop(prefix + "mlp.gate_proj." + suffix),
                ]
            )
    converted = {}
    for key, tensor in source.items():
        key = key.replace(
            "tokenizer.temporal_encoder.feature_extractor.net.", "temporal_encoder."
        )
        key = key.replace("tokenizer.temporal_pooler.final_layer.", "temporal_pooler.")
        key = key.replace(
            "tokenizer.spatial_encoder.subcomponent_embeddings.", "spatial_emb.tables."
        )
        key = key.replace(".attention.", ".self_attn.").replace(
            ".mlp.down_proj.", ".mlp.3."
        )
        converted[key] = tensor
    missing, unexpected = model.load_state_dict(converted, strict=False)
    expected_missing = {
        "token_pooling.weight",
        "final_layer.weight",
        "final_layer.bias",
    }
    if model.token_pooling is None:
        expected_missing.remove("token_pooling.weight")
    if set(missing) != expected_missing or unexpected:
        raise RuntimeError(
            f"Incomplete conversion: missing={missing}, unexpected={unexpected}"
        )
    return len(converted), sorted(missing)


def reference_tokens(state, x, indices):
    """Evaluate released tokenizer/transformer equations without converted keys.

    Reference: barista/models/{TSEncoder2D,tokenizer,transformer,spatial_encoder}.py
    at REVISION. Defaults match config/model.yaml: 512 samples, 64 features,
    five CNN blocks, 12 transformer blocks, four heads and RMSNorm eps=1e-8.
    """
    batch, channels, _ = x.shape
    patches = x.unfold(-1, 512, 512).permute(0, 2, 1, 3)
    n_patches = patches.shape[1]
    h = patches.reshape(1, 1, -1, 512)
    for i in range(5):
        prefix = f"tokenizer.temporal_encoder.feature_extractor.net.{i}."
        residual = h
        if prefix + "projector.weight" in state:
            residual = F.conv2d(
                h, state[prefix + "projector.weight"], state[prefix + "projector.bias"]
            )
        for conv in ("conv1", "conv2"):
            h = F.conv2d(
                h,
                state[prefix + conv + ".conv.weight"],
                state[prefix + conv + ".conv.bias"],
                padding=(0, 2**i),
                dilation=(1, 2**i),
            )
            h = F.gelu(F.layer_norm(h, (512,)))
        h = h + residual
    h = F.linear(
        h.reshape(-1, 512), state["tokenizer.temporal_pooler.final_layer.weight"]
    )
    h = h.reshape(batch, n_patches * channels, 64)
    grid = indices.T if indices.ndim == 2 else indices[None]
    spatial = torch.stack(
        [
            F.embedding(
                axis,
                state[f"tokenizer.spatial_encoder.subcomponent_embeddings.{i}.weight"],
            )
            for i, axis in enumerate(grid)
        ]
    ).sum(0)
    h = h + spatial.repeat(n_patches, 1)[None]
    for i in range(12):
        prefix = f"backbone.layers.{i}."
        normalized = h * torch.rsqrt(h.square().mean(-1, keepdim=True) + 1e-8)
        normalized = normalized * state[prefix + "norm1.weight"]
        qkv = F.linear(
            normalized,
            state[prefix + "attention.qkv_proj.weight"],
            state[prefix + "attention.qkv_proj.bias"],
        )
        q, k, v = [
            part.reshape(batch, -1, 4, 16).transpose(1, 2) for part in qkv.chunk(3, -1)
        ]
        positions = torch.arange(n_patches).repeat_interleave(channels)
        angles = torch.outer(positions, state[prefix + "attention.rotary_emb.inv_freq"])
        angles = torch.cat((angles, angles), -1)[None, None]
        q = q * angles.cos() + torch.cat((-q[..., 8:], q[..., :8]), -1) * angles.sin()
        k = k * angles.cos() + torch.cat((-k[..., 8:], k[..., :8]), -1) * angles.sin()
        attention = ((q @ k.transpose(-1, -2)) / 4).softmax(-1) @ v
        attention = attention.transpose(1, 2).reshape(batch, -1, 64)
        h = h + F.linear(
            attention,
            state[prefix + "attention.o_proj.weight"],
            state[prefix + "attention.o_proj.bias"],
        )
        normalized = h * torch.rsqrt(h.square().mean(-1, keepdim=True) + 1e-8)
        normalized = normalized * state[prefix + "norm2.weight"]
        gate = F.gelu(
            F.linear(
                normalized,
                state[prefix + "mlp.gate_proj.weight"],
                state[prefix + "mlp.gate_proj.bias"],
            )
        )
        value = F.linear(
            normalized,
            state[prefix + "mlp.up_proj.weight"],
            state[prefix + "mlp.up_proj.bias"],
        )
        h = h + F.linear(
            gate * value,
            state[prefix + "mlp.down_proj.weight"],
            state[prefix + "mlp.down_proj.bias"],
        )
    return (
        h
        * torch.rsqrt(h.square().mean(-1, keepdim=True) + 1e-8)
        * state["backbone.norm.weight"]
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--cache-dir", type=Path, default=pooch.os_cache("braindecode") / "barista"
    )
    parser.add_argument("--n-chans", type=int, default=3)
    parser.add_argument("--n-times", type=int, default=6144)
    parser.add_argument("--n-outputs", type=int, default=2)
    parser.add_argument("--pooling", choices=("learned", "mean"), default="learned")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(0)
    report = {
        "source_revision": REVISION,
        "torch_version": torch.__version__,
        "checkpoints": {},
    }
    for scale, (filename, digest) in CHECKPOINTS.items():
        path = pooch.retrieve(
            f"{SOURCE}/pretrained_models/{filename}",
            known_hash=f"sha256:{digest}",
            fname=filename,
            path=args.cache_dir,
        )
        state = torch.load(path, map_location="cpu", weights_only=True)
        model = BaRISTA(
            n_chans=args.n_chans,
            n_times=args.n_times,
            n_outputs=args.n_outputs,
            spatial_scale=scale,
            pooling=args.pooling,
        ).eval()
        loaded, missing = convert_weights(state, model)
        # Synthetic indices exercise the whole table, including unknown region 0.
        slots = {"coords": 200, "parcels": 121, "lobes": 21}[scale]
        shape = (args.n_chans, 3) if scale == "coords" else (args.n_chans,)
        indices = (
            torch.linspace(0, slots - 1, args.n_chans * (3 if scale == "coords" else 1))
            .long()
            .reshape(shape)
        )
        x = torch.randn(2, args.n_chans, args.n_times)
        captured = []
        with (
            torch.no_grad(),
            model.backbone.register_forward_hook(lambda _m, _x, y: captured.append(y)),
        ):
            logits = model(x, spatial_indices=indices)
            expected = reference_tokens(state, x, indices)
        torch.testing.assert_close(captured[0], expected, rtol=1e-4, atol=1e-5)
        if not torch.isfinite(logits).all():
            raise RuntimeError(f"Non-finite output for {scale}")
        error = (captured[0] - expected).abs().max().item()
        destination = args.output_dir / scale
        model.save_pretrained(destination)
        shutil.copyfile(
            Path(__file__).resolve().parents[1] / "NOTICE.txt",
            destination / "NOTICE.txt",
        )
        restored = BaRISTA.from_pretrained(destination, strict=True).eval()
        with torch.no_grad():
            torch.testing.assert_close(
                restored(x, spatial_indices=indices), logits, rtol=0, atol=0
            )
        report["checkpoints"][scale] = {
            "filename": filename,
            "sha256": digest,
            "encoder_tensors_loaded": loaded,
            "new_head_tensors": missing,
            "max_abs_token_error": error,
            "rtol": 1e-4,
            "atol": 1e-5,
        }
        print(
            f"{scale}: {loaded} encoder tensors loaded; max token error {error:.3g}; saved to {destination}"
        )
    (args.output_dir / "conversion_report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(
        "All encoders and local Hub round trips passed. Downstream heads need fine-tuning."
    )


if __name__ == "__main__":
    main()
