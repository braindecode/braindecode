"""Convert released EEG-DINO checkpoints to braindecode HF format.

Released `.pt` files contain a DINO training graph under
`state_dict["module.teacher.*"]` / `["module.student.*"]`. This script keeps the
teacher encoder, strips the prefix, loads it into an `EEGDINO` instance (head is
freshly initialized), and saves a braindecode-style folder (config.json +
safetensors) ready for `huggingface-cli upload`.

Usage:
    python scripts/convert_eegdino_checkpoints.py \
        --src EEG-DINO/pre-trained-models/model_EEG_DINO_S.pt \
        --size small --out hf_export/eegdino-small
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from braindecode.models import EEGDINO
from braindecode.models.eegdino import EEGDINO_CONFIGS

_ENCODER_PREFIXES = ("patch_embedding.", "encoder_layers.", "global_tokens")
_DROP_SUFFIXES = ("relative_position_index",)


def extract_encoder_state_dict(ckpt: dict, source: str = "teacher") -> dict:
    """Return the encoder state_dict with the ``module.<source>.`` prefix stripped."""
    state = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    prefix = f"module.{source}."
    out = {}
    for key, value in state.items():
        if not key.startswith(prefix):
            continue
        stripped = key[len(prefix):]
        if stripped.endswith(_DROP_SUFFIXES):
            continue
        if stripped.startswith(_ENCODER_PREFIXES):
            out[stripped] = value
    return out


def convert(src: Path, size: str, out: Path, n_outputs: int = 2, source: str = "teacher"):
    ckpt = torch.load(src, map_location="cpu", weights_only=False)
    enc_sd = extract_encoder_state_dict(ckpt, source=source)
    cfg: dict[str, Any] = {} if size == "small" else EEGDINO_CONFIGS[size]
    model = EEGDINO(n_chans=19, n_outputs=n_outputs, n_times=1000, num_channels=19, **cfg)
    missing, unexpected = model.load_state_dict(enc_sd, strict=False)
    enc_missing = [m for m in missing if m.startswith(_ENCODER_PREFIXES)]
    if enc_missing or unexpected:
        raise RuntimeError(
            f"Encoder mismatch. missing={enc_missing} unexpected={unexpected}"
        )
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)  # writes config.json + safetensors via EEGModuleMixin
    print(f"Saved {size} to {out} (loaded {len(enc_sd)} encoder tensors)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", type=Path, required=True)
    p.add_argument("--size", choices=["small", "medium", "large"], required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--source", choices=["teacher", "student"], default="teacher")
    p.add_argument("--n-outputs", type=int, default=2)
    args = p.parse_args()
    convert(args.src, args.size, args.out, n_outputs=args.n_outputs, source=args.source)


if __name__ == "__main__":
    main()
