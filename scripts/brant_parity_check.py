"""Dev-only numerical parity check for :class:`braindecode.models.Brant`.

Not part of CI. Verifies that braindecode's Brant encoders reproduce,
weight-for-weight, the upstream reference
(``Brant_src/pretrain/pre_model.py``, Apache-2.0,
https://huggingface.co/Daoze/Brant).

Usage
-----
Download the upstream code (gated) once::

    hf download Daoze/Brant --local-dir /path/to/brant-upstream

then run::

    python scripts/brant_parity_check.py --brant-src /path/to/brant-upstream/Brant_src

It builds each upstream encoder and its braindecode port with identical
dimensions, copies the upstream weights into the port, feeds identical inputs,
and reports ``max|diff|`` for the temporal and spatial encoders.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from braindecode.models.brant import (
    _BrantSpatialEncoder,
    _BrantTemporalEncoder,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--brant-src",
        type=Path,
        required=True,
        help="Path to the upstream 'Brant_src' directory (contains pretrain/).",
    )
    parser.add_argument("--atol", type=float, default=1e-5)
    return parser.parse_args()


def _import_upstream(brant_src: Path):
    """Import upstream encoders without leaking their path or modules."""
    old_path = sys.path.copy()
    old_modules = {
        name: module
        for name, module in sys.modules.items()
        if name == "pretrain" or name.startswith("pretrain.")
    }
    for name in old_modules:
        sys.modules.pop(name)
    sys.path.insert(0, str(brant_src))
    try:
        from pretrain.pre_model import ChannelEncoder, TimeEncoder
    finally:
        sys.path[:] = old_path
        for name in list(sys.modules):
            if name == "pretrain" or name.startswith("pretrain."):
                sys.modules.pop(name)
        sys.modules.update(old_modules)
    return TimeEncoder, ChannelEncoder


def main() -> int:
    args = _parse_args()
    try:
        TimeEncoder, ChannelEncoder = _import_upstream(args.brant_src)
    except ImportError as exc:  # pragma: no cover - dev tooling
        print(f"cannot import upstream Brant from {args.brant_src}: {exc}")
        return 1

    torch.manual_seed(0)
    # small dims keep the check fast; parity is dimension-independent
    batch, n_chans, seq_len = 2, 3, 4
    patch_size, d_model, dim_ff = 64, 32, 48
    n_bands, n_heads = 8, 4
    t_layers, s_layers = 2, 2

    data = torch.randn(batch, n_chans, seq_len, patch_size)
    power = torch.randn(batch, n_chans, seq_len, n_bands)

    # ----------------------------------------------------------------- temporal
    up_t = TimeEncoder(
        in_dim=patch_size, d_model=d_model, dim_feedforward=dim_ff,
        seq_len=seq_len, n_layer=t_layers, nhead=n_heads, band_num=n_bands,
        project_mode="linear", learnable_mask=False,
    ).eval()
    ours_t = _BrantTemporalEncoder(
        patch_size=patch_size, d_model=d_model, seq_len=seq_len, n_bands=n_bands,
        dim_feedforward=dim_ff, n_layers=t_layers, n_heads=n_heads, drop_prob=0.1,
    ).eval()
    ours_t.trans_enc.load_state_dict(up_t.trans_enc.state_dict())
    ours_t.input_embedding.proj.load_state_dict(up_t.input_embedding.proj.state_dict())
    ours_t.input_embedding.band_encoding.data.copy_(up_t.input_embedding.band_encoding.data)
    ours_t.input_embedding.positional_encoding.data.copy_(
        up_t.input_embedding.positional_encoding.data
    )

    with torch.no_grad():
        up_time = up_t(mask=None, data=data, power=power, need_mask=False, use_power=True)
        ours_time = ours_t(data, power)
    t_diff = (up_time - ours_time).abs().max().item()
    print(f"temporal encoder : max|diff| = {t_diff:.2e}")

    # ------------------------------------------------------------------ spatial
    time_z = up_time.reshape(batch, n_chans, seq_len, d_model)
    time_z = time_z.transpose(1, 2).reshape(batch * seq_len, n_chans, d_model)

    up_c = ChannelEncoder(
        out_dim=patch_size, d_model=d_model, dim_feedforward=dim_ff,
        n_layer=s_layers, nhead=n_heads,
    ).eval()
    ours_c = _BrantSpatialEncoder(
        d_model=d_model, out_dim=patch_size, dim_feedforward=dim_ff,
        n_layers=s_layers, n_heads=n_heads, drop_prob=0.1,
    ).eval()
    ours_c.trans_enc.load_state_dict(up_c.trans_enc.state_dict())
    ours_c.proj_out.load_state_dict(up_c.proj_out.state_dict())

    with torch.no_grad():
        up_ch, _ = up_c(time_z)
        ours_ch, _ = ours_c(time_z)
    c_diff = (up_ch - ours_ch).abs().max().item()
    print(f"spatial encoder  : max|diff| = {c_diff:.2e}")

    ok = t_diff < args.atol and c_diff < args.atol
    print("PARITY OK" if ok else "PARITY FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
