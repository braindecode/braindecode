# Authors: Bruno Aristimunha <b.aristimunha@gmail.com>
#
# License: MIT
"""Dev-only numerical-parity check: braindecode DANCE vs upstream `dance`.

Run in a throwaway venv (py>=3.12) with the upstream installed::

    pip install dance perceiver_pytorch x_transformers neuraltrain neuralset
    python scripts/dance_parity_check.py

NOT part of the CI suite. There is no brainmagick fallback: the upstream
modules live in `neuraltrain` (and `perceiver_pytorch`), both PyPI-installable.
"""
from __future__ import annotations

import importlib.util
import os
import sys

# Prefer the in-repo braindecode over any (possibly stale) site-packages install
# when this dev script is run as ``python scripts/dance_parity_check.py`` from
# the repo root -- otherwise sys.path[0] is ``scripts/`` and DANCE may be missing.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.isdir(os.path.join(_REPO_ROOT, "braindecode")):
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
import torch

from braindecode.models import DANCE
from braindecode.models.dance import _positions_from_chs_info

N_CHANS, N_CLASSES, N_TIMES = 19, 4, 2000
DURATION = N_TIMES / 200.0  # seconds; matcher/consistency use this


def _chs_info(n_chans):
    rng = np.random.default_rng(0)
    return [
        {"ch_name": f"E{i + 1}", "kind": "eeg", "loc": rng.random(12)}
        for i in range(n_chans)
    ]


def _build_braindecode(chs_info):
    return DANCE(
        n_outputs=N_CLASSES, n_chans=N_CHANS, chs_info=chs_info,
        n_times=N_TIMES, sfreq=200.0, input_window_seconds=DURATION,
    ).eval()


# UPSTREAM(neuraltrain) -> LOCAL(braindecode) parity weight map. This is a
# LOCAL dict (NOT DANCE.mapping). Decoder keys (x_transformers) and the dropped
# center_head/duration_head are intentionally OMITTED (decoder re-implemented).
def _build_rename_map(n_classes, depth=10):
    m = {
        # merger (nested under SimpleConv): encoder.encoder.merger.* -> conv.merger.*
        "encoder.encoder.merger.heads": "conv.merger.heads",
        "encoder.encoder.merger.embedding.pos": "conv.merger.embedding.pos",
        # initial 1x1 linear: encoder.encoder.initial_linear.0.* -> conv.initial_linear.0.*
        "encoder.encoder.initial_linear.0.weight": "conv.initial_linear.0.weight",
        "encoder.encoder.initial_linear.0.bias": "conv.initial_linear.0.bias",
        # dense head -> final_layer
        "dense_head.weight": "final_layer.weight",
        "dense_head.bias": "final_layer.bias",
    }
    # conv blocks: encoder.encoder.encoder.sequence.{k}.0.* -> conv.sequence.{k}.conv.*
    for k in range(depth):
        up = f"encoder.encoder.encoder.sequence.{k}.0"
        loc = f"conv.sequence.{k}.conv"
        m[f"{up}.weight"] = f"{loc}.weight"
        m[f"{up}.bias"] = f"{loc}.bias"
    # perceiver: our submodule names mirror perceiver_pytorch's layers.* keys
    # 1:1 (verified by test_perceiver_full_block_parity in Task 3), so the only
    # transform is the encoder.perceiver. -> perceiver. prefix swap; the
    # per-layer suffixes are IDENTICAL. Latents map directly.
    # (Built dynamically below from the actual upstream keys.)
    return m


def _copy_via_map(bd_model, up_state, n_classes):
    bd_state = bd_model.state_dict()
    rename = _build_rename_map(n_classes)
    mapped, skipped = 0, []
    for up_key, val in up_state.items():
        if up_key in rename:
            bd_key = rename[up_key]
        elif up_key.startswith("encoder.perceiver."):
            # encoder.perceiver.X -> perceiver.X (suffix identical to ours)
            bd_key = "perceiver." + up_key[len("encoder.perceiver."):]
        else:
            continue  # decoder / dropped heads: not parity-mapped
        if bd_key in bd_state and bd_state[bd_key].shape == val.shape:
            bd_state[bd_key] = val.clone()
            mapped += 1
        else:
            skipped.append((up_key, bd_key))
    bd_model.load_state_dict(bd_state, strict=False)
    if skipped:
        print(f"WARNING: {len(skipped)} keys could not be copied:")
        for u, b in skipped[:20]:
            print(f"  {u} -> {b}")
    return mapped


def main():
    if importlib.util.find_spec("neuraltrain") is None:
        print("neuraltrain not installed; cannot run parity. Skipping.")
        return 0
    from dance.dance import Dance  # type: ignore

    chs_info = _chs_info(N_CHANS)
    bd = _build_braindecode(chs_info)
    # IDENTICAL positions for BOTH sides (parity-load-bearing). braindecode
    # derives them from chs_info; feed the SAME tensor to upstream.
    positions = torch.as_tensor(
        _positions_from_chs_info(chs_info), dtype=torch.float32
    )
    pos_batched = positions.unsqueeze(0).expand(2, -1, -1)  # (B, n_chans, 2)
    subject_ids = None  # braindecode has no subjects (per_subject=False)

    up = Dance(
        n_channels=N_CHANS, n_classes=N_CLASSES, n_queries=100,
        duration=DURATION, use_channel_merger=True,
    ).eval()
    mapped = _copy_via_map(bd, up.state_dict(), N_CLASSES)
    print(f"mapped {mapped} tensors")

    x = torch.randn(2, N_CHANS, N_TIMES)
    with torch.no_grad():
        # 1) encoder c_out (strict). Upstream _Encoder.forward signature is
        #    forward(x, subject_ids=None, channel_positions=None); c_out is the
        #    key returned. Pass the SAME positions braindecode used.
        bd_enc = bd._encode(x)  # (B, num_latents, embed_dim)
        up_enc = up.encoder(
            x, subject_ids=subject_ids, channel_positions=pos_batched
        )["c_out"]
        torch.testing.assert_close(bd_enc, up_enc, rtol=1e-3, atol=1e-4)
        print("encoder parity OK")

        # 2) forward dense logits (strict). dense_head lives on the top-level
        #    Dance (NOT on the encoder): up.dense_head(c_out).
        bd_dense = bd(x)  # (B, num_latents, n_classes)
        up_dense = up.dense_head(up_enc)
        torch.testing.assert_close(bd_dense, up_dense, rtol=1e-3, atol=1e-4)
        print("forward dense parity OK")

        # 3) detect decoder dict (loose / best-effort; decoder re-implemented).
        bd_det = bd.detect(x)
        print("detect keys:", sorted(bd_det))
        print(
            "decoder parity is LOOSE by design (MultiHeadAttention vs "
            "x_transformers); not asserted."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
