"""Convert and re-host EEG-DINO checkpoints on the Hugging Face Hub.

The released EEG-DINO ``.pt`` files contain a full DINO training graph; the
encoder lives under ``state_dict["module.teacher.*"]``. This script keeps the
teacher encoder, strips the prefix, loads it into an
:class:`~braindecode.models.EEGDINO` (the classification head is
re-initialized), and writes a braindecode-style folder (``config.json`` +
``model.safetensors``). With ``--push`` it uploads that folder to a per-size
braindecode Hub repo, matching the one-repo-per-checkpoint convention used by
the other foundation models (see ``generate_cards.py``):

    braindecode/eegdino-small-pretrained
    braindecode/eegdino-medium-pretrained

Run from the repo root (the released checkpoints must be available locally)::

    python hf_assets/model_cards/convert_eegdino_checkpoints.py \
        --src EEG-DINO/pre-trained-models/model_EEG_DINO_S.pt --size small
    python hf_assets/model_cards/convert_eegdino_checkpoints.py \
        --src EEG-DINO/pre-trained-models/model_EEG_DINO_M.pt --size medium

Add ``--push`` (needs write access to the ``braindecode`` HF org) to upload.
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
    state = ckpt.get("state_dict", ckpt)
    prefix = f"module.{source}."
    return {
        key[len(prefix) :]: value
        for key, value in state.items()
        if key.startswith(prefix)
        and key[len(prefix) :].startswith(_ENCODER_PREFIXES)
        and not key.endswith(_DROP_SUFFIXES)
    }


def convert(src: Path, size: str, out: Path, n_outputs: int = 2, source: str = "teacher"):
    """Convert one released ``.pt`` into a braindecode pretrained folder."""
    ckpt = torch.load(src, map_location="cpu", weights_only=False)
    encoder_state = extract_encoder_state_dict(ckpt, source=source)
    config: dict[str, Any] = {} if size == "small" else EEGDINO_CONFIGS[size]
    model = EEGDINO(n_chans=19, n_outputs=n_outputs, n_times=1000, **config)
    missing, unexpected = model.load_state_dict(encoder_state, strict=False)
    encoder_missing = [k for k in missing if k.startswith(_ENCODER_PREFIXES)]
    if encoder_missing or unexpected:
        raise RuntimeError(
            f"Encoder mismatch: missing={encoder_missing} unexpected={unexpected}"
        )
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)  # config.json + model.safetensors via EEGModuleMixin
    print(f"Saved {size} to {out} ({len(encoder_state)} encoder tensors loaded)")


def push(out: Path, size: str):
    """Upload a converted folder to ``braindecode/eegdino-<size>-pretrained``."""
    from huggingface_hub import HfApi

    repo_id = f"braindecode/eegdino-{size}-pretrained"
    api = HfApi()
    api.create_repo(repo_id, repo_type="model", exist_ok=True)
    api.upload_folder(repo_id=repo_id, folder_path=str(out))
    print(f"Pushed {out} -> {repo_id}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--size", choices=["small", "medium", "large"], required=True)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output folder (default: hf_export/eegdino-<size>-pretrained)",
    )
    parser.add_argument("--source", choices=["teacher", "student"], default="teacher")
    parser.add_argument("--n-outputs", type=int, default=2)
    parser.add_argument(
        "--push",
        action="store_true",
        help="upload to braindecode/eegdino-<size>-pretrained (needs org write access)",
    )
    args = parser.parse_args()
    out = args.out or Path("hf_export") / f"eegdino-{args.size}-pretrained"
    convert(args.src, args.size, out, n_outputs=args.n_outputs, source=args.source)
    if args.push:
        push(out, args.size)


if __name__ == "__main__":
    main()
