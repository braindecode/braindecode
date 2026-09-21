"""Validate the sparse-autoencoder utilities on LaBraM and the NMT corpus.

This follows Lehn-Schioler et al. (arXiv:2605.13930) on public data: fine-tune a
pretrained EEG transformer on a clinical target, freeze it, fit a Top-K SAE to
one layer's residual stream, then substitute the reconstruction back in and
re-measure the task with the model's own head.

LaBraM is used rather than a smaller transformer because its tokens are exact
non-overlapping one-second patches. That property is what a spectral decoder
needs later -- it learns ``token embedding -> FFT of the matching raw patch``,
a target that only exists when tokens are time-aligned. NMT also carries
``pathological``, ``age`` and ``gender``, three of the paper's five TCAV
concepts.

Everything imports from ``braindecode.visualization``, so a successful run
validates the library code rather than a parallel implementation.

The full NMT archive is ~13.4 GB and is downloaded on first use. Pass
``--data-root`` to point at an existing copy.

Usage
-----
    python validate_sae_labram_nmt.py --n-recordings 400
    python validate_sae_labram_nmt.py --data-root /data/nmt --layer 11
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Subset

from braindecode.datasets import NMT
from braindecode.models import Labram
from braindecode.models.labram import LABRAM_CHANNEL_ORDER
from braindecode.preprocessing import (
    Preprocessor,
    create_fixed_length_windows,
    preprocess,
)
from braindecode.visualization import (
    SparseAutoencoder,
    capture_activations,
    fit_sparse_autoencoder,
    run_with_activation_substitution,
    sae_diagnostics,
)

HF_REPO = "braindecode/labram-pretrained"
SFREQ = 200
EMBED_DIM = 200

# NMT stores these names with the older 10-20 convention; LaBraM's 128-name
# vocabulary only knows the modern ones, so T3/T4/T5/T6 must be renamed or they
# resolve to no position embedding at all.
CH_NAMES = [
    "FP1",
    "FP2",
    "F7",
    "F3",
    "FZ",
    "F4",
    "F8",
    "T3",
    "C3",
    "CZ",
    "C4",
    "T4",
    "T5",
    "P3",
    "PZ",
    "P4",
    "T6",
    "O1",
    "O2",
]
RENAME = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="directory holding Labels.csv; downloads if omitted",
    )
    p.add_argument("--cache", type=Path, default=Path("nmt_windows"))
    p.add_argument("--n-recordings", type=int, default=400)
    p.add_argument(
        "--delete-raw-after-cache",
        action="store_true",
        help="delete --data-root once the window cache is written; "
        "off by default, since it removes your data",
    )
    p.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=[11],
        help="blocks to sweep; the fine-tune is shared across them",
    )
    p.add_argument(
        "--ft-ckpt",
        type=Path,
        default=Path("labram_nmt_ft.pt"),
        help="fine-tuned weights; reused when present",
    )
    p.add_argument(
        "--expansions",
        type=int,
        nargs="+",
        default=[8],
        help="expansion ratios to sweep; k defaults to 8*E per cell",
    )
    p.add_argument("--k", type=int, default=None)  # defaults to 8 * expansion
    p.add_argument("--finetune-epochs", type=int, default=10)
    p.add_argument("--freeze-epochs", type=int, default=2)
    p.add_argument("--sae-epochs", type=int, default=20)
    p.add_argument(
        "--sae-log-every",
        type=int,
        default=5,
        help="print SAE loss/dead every N epochs; 0 to silence",
    )
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--head-lr", type=float, default=1e-3)
    p.add_argument("--full-lr", type=float, default=1e-4)
    p.add_argument("--max-sae-tokens", type=int, default=2_000_000)
    p.add_argument("--min-valid-auroc", type=float, default=0.75)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--resample-steps",
        type=int,
        nargs="+",
        default=[500],
        help="resampling intervals to sweep, in optimizer steps",
    )
    p.add_argument("--out", type=Path, default=Path("sae_labram_nmt_results.json"))
    return p.parse_args()


def _corpus_root(dataset):
    """Directory holding the extracted recordings, from a recording path.

    ``NMT`` picks its own download location, so the corpus root is only
    discoverable after loading. Returns ``None`` if the layout is unfamiliar,
    in which case nothing is deleted.
    """
    paths = dataset.description["path"]
    if paths.empty:
        return None
    for parent in Path(paths.iloc[0]).parents:
        if parent.name == "nmt_scalp_eeg_dataset":
            return parent
    return None


def build_windows(args):
    """Load NMT, preprocess to LaBraM's format, cut 15 s windows."""
    from braindecode.datautil import load_concat_dataset

    if args.cache.exists():
        windows = load_concat_dataset(str(args.cache), preload=False)
        print(f"loaded {len(windows)} windows from {args.cache}")
        return windows

    path = str(args.data_root) if args.data_root else None
    if path is None:
        print(
            "downloading NMT (~13.4 GB archive, plus the same again "
            "unzipped); pass --data-root to reuse an existing copy"
        )
    dataset = NMT(
        path=path,
        target_name="pathological",
        recording_ids=list(range(args.n_recordings)),
        preload=False,
        n_jobs=1,
    )
    print(f"{len(dataset.datasets)} recordings loaded")

    # NMT resolves the download location itself, so recover the corpus root
    # from a recording path rather than assuming where it landed. Without this
    # --delete-raw-after-cache could only clean up an explicit --data-root.
    raw_root = _corpus_root(dataset)

    # LaBraM's temporal embedding holds 16 positions, so 15 one-second patches
    # is the longest window that needs no interpolation of the checkpoint.
    window_samples = 15 * SFREQ
    want = set(CH_NAMES)
    keep = [
        i
        for i, d in enumerate(dataset.datasets)
        if want.issubset(set(d.raw.ch_names))
        and (d.raw.n_times - 1) / d.raw.info["sfreq"] >= 75
    ]
    if not keep:
        raise SystemExit("no recording has the full 19-channel montage")
    dataset = dataset.split(keep)["0"]
    print(f"{len(dataset.datasets)} recordings have the full montage")

    def crop_from(raw, tmin=60.0):
        end = (raw.n_times - 1) / raw.info["sfreq"]
        raw.crop(tmin=tmin, tmax=end, include_tmax=False)

    preprocessors = [
        Preprocessor(crop_from, tmin=60.0, apply_on_array=False),
        # Pick before referencing so the average is not polluted by channels
        # that are about to be dropped.
        Preprocessor("pick_channels", ch_names=CH_NAMES, ordered=True),
        Preprocessor("rename_channels", mapping=RENAME),
        Preprocessor("set_eeg_reference", ref_channels="average", ch_type="eeg"),
        Preprocessor(lambda d: d * 1e6, apply_on_array=True),  # V -> uV
        Preprocessor(lambda x: np.clip(x, -800, 800), apply_on_array=True),
        Preprocessor("resample", sfreq=SFREQ),
    ]
    # save_dir serialises each recording and reloads it lazily. Without it
    # preprocess() holds every recording in memory at once -- MNE works in
    # float64, so 400 recordings is tens of gigabytes of RAM.
    pp_dir = args.cache.parent / f"{args.cache.name}_pp"
    processed = preprocess(
        dataset, preprocessors, n_jobs=1, save_dir=str(pp_dir), overwrite=True
    )
    windows = create_fixed_length_windows(
        processed,
        window_size_samples=window_samples,
        window_stride_samples=window_samples,
        drop_last_window=True,
        n_jobs=1,
    )
    windows.save(str(args.cache), overwrite=True)
    print(f"cached {len(windows)} windows to {args.cache}")

    # Reload from the cache before touching anything on disk: the in-memory
    # object still reads the preprocessed files lazily.
    import shutil

    del processed, dataset, windows
    windows = load_concat_dataset(str(args.cache), preload=False)

    # The preprocessed copy is an intermediate; the window cache is what later
    # runs read, so it always goes.
    shutil.rmtree(pp_dir, ignore_errors=True)
    print(f"removed intermediate copy: {pp_dir}")

    if args.delete_raw_after_cache and raw_root is not None:
        shutil.rmtree(raw_root, ignore_errors=True)
        print(f"removed raw recordings: {raw_root}")
        # A downloaded run also leaves the archive behind, which is the larger
        # of the two.
        for archive in raw_root.parent.glob("NMT.zip*"):
            size_gb = archive.stat().st_size / 1024**3
            archive.unlink()
            print(f"removed archive: {archive} ({size_gb:.1f} GB)")
    return windows


def channel_names(windows):
    d0 = windows.datasets[0]
    return d0.windows.ch_names if hasattr(d0, "windows") else d0.raw.ch_names


def resolve_input_chans(ch_names):
    """Indices into ``position_embedding``: 0 is [CLS], 1..128 the channels."""
    canonical = {name.upper(): i for i, name in enumerate(LABRAM_CHANNEL_ORDER)}
    missing = [c for c in ch_names if c.upper() not in canonical]
    if missing:
        raise SystemExit(f"not in LaBraM's vocabulary: {missing}")
    idx = [0] + [canonical[c.upper()] + 1 for c in ch_names]
    return torch.tensor(idx, dtype=torch.long)


class LabramBinary(nn.Module):
    """LaBraM with a scalar head on mean-pooled patch tokens."""

    def __init__(self, encoder, input_chans):
        super().__init__()
        self.encoder = encoder
        self.register_buffer("input_chans", input_chans)
        self.head = nn.Linear(EMBED_DIM, 1)

    def features(self, x):
        tokens = self.encoder.forward_features(
            x, input_chans=self.input_chans, return_all_tokens=True
        )
        return tokens[:, 1:, :].mean(dim=1)  # strip CLS, mean over channel x time

    def forward(self, x):
        return self.head(self.features(x)).squeeze(-1)


def window_labels(windows):
    """Labels without touching signal data; indexing a lazy dataset hits disk."""
    return np.concatenate([np.asarray(d.y) for d in windows.datasets])


def recording_of_window(windows):
    sizes = [len(d) for d in windows.datasets]
    return np.repeat(np.arange(len(sizes)), sizes)


def stratified_group_split(groups, labels, seed, test_frac=0.2, valid_frac=0.2):
    """Split by recording, stratified by label.

    Grouping alone is not enough: an unstratified permutation lets the class
    balance drift, and a validation set that holds one positive recording
    reports how well the model recognises that recording rather than how well
    it generalises.
    """
    rng = np.random.default_rng(seed)
    rec_ids = np.unique(groups)
    rec_label = np.array([labels[groups == r][0] for r in rec_ids])

    train, valid, test = [], [], []
    for cls in np.unique(rec_label):
        cls_recs = rng.permutation(rec_ids[rec_label == cls])
        n_test = max(1, int(round(test_frac * len(cls_recs))))
        rest = cls_recs[n_test:]
        n_valid = max(1, int(round(valid_frac * len(rest))))
        test.extend(cls_recs[:n_test])
        valid.extend(rest[:n_valid])
        train.extend(rest[n_valid:])

    parts = {
        k: np.array(v) for k, v in (("train", train), ("valid", valid), ("test", test))
    }
    for a, b in (("train", "valid"), ("train", "test"), ("valid", "test")):
        assert not (set(parts[a]) & set(parts[b])), f"{a}/{b} share a recording"
    idx = {k: np.flatnonzero(np.isin(groups, v)) for k, v in parts.items()}
    return idx, parts


def measure_scale(windows, ids, n=64):
    """Signal scale, estimated on the training split only."""
    sample = torch.stack(
        [torch.as_tensor(windows[i][0]) for i in ids[: min(n, len(ids))]]
    ).float()
    return float(np.percentile(np.abs(sample.numpy()), 99))


@torch.no_grad()
def evaluate(model, loader, device, groups=None):
    """Window-level AUROC, plus recording level when groups are given."""
    model.eval()
    logits, ys = [], []
    for batch in loader:
        logits.append(model(batch[0].float().to(device)).cpu())
        ys.append(batch[1])
    model.train()
    logits = torch.cat(logits).numpy()
    ys = torch.cat(ys).numpy().astype(int)

    window = roc_auc_score(ys, logits)
    recording = float("nan")
    if groups is not None:
        recs = np.unique(groups)
        scores = np.array([logits[groups == r].mean() for r in recs])
        labels = np.array([ys[groups == r][0] for r in recs])
        if len(np.unique(labels)) > 1:
            recording = roc_auc_score(labels, scores)
    return window, recording


def finetune(model, loaders, groups, idx, args, device, pos_weight):
    """Two-phase fine-tune, with selection restarting at the unfreeze boundary.

    Head-phase validation scores are not comparable to full-model ones:
    unfreezing raises the loss briefly while the encoder adapts. Letting a
    lucky head-phase epoch win the comparison leaves the encoder at its
    pretrained weights while the run reports itself as fine-tuned.
    """
    torch.manual_seed(args.seed)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    for p in model.encoder.parameters():
        p.requires_grad = False
    opt = torch.optim.AdamW(model.head.parameters(), lr=args.head_lr, weight_decay=0.01)

    best, best_state, history = -1.0, None, []
    for epoch in range(1, args.finetune_epochs + 1):
        if epoch == args.freeze_epochs + 1:
            for p in model.encoder.parameters():
                p.requires_grad = True
            opt = torch.optim.AdamW(
                model.parameters(), lr=args.full_lr, weight_decay=0.05
            )
            best, best_state = -1.0, None
            print(f"--- epoch {epoch}: encoder unfrozen, selection restarted ---")

        model.train()
        total = 0.0
        for bi, batch in enumerate(loaders["train"]):
            x, y = batch[0], batch[1]
            loss = loss_fn(model(x.float().to(device)), y.float().to(device))
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.detach())
            if bi % 100 == 0:
                print(
                    f"  epoch {epoch:2d} batch {bi}/{len(loaders['train'])}", flush=True
                )

        win, rec = evaluate(model, loaders["valid"], device, groups[idx["valid"]])
        phase = "head" if epoch <= args.freeze_epochs else "full"
        history.append(
            {
                "epoch": epoch,
                "phase": phase,
                "loss": total / len(loaders["train"]),
                "valid_window_auroc": win,
                "valid_recording_auroc": rec,
            }
        )
        print(
            f"epoch {epoch:2d} [{phase}]  loss={total / len(loaders['train']):.4f}"
            f"  valid_window={win:.4f}  valid_recording={rec:.4f}"
        )

        # Select on the recording-level score: the label is a property of the
        # recording, not of any single 15 s window.
        score = rec if not np.isnan(rec) else win
        if score > best:
            best = score
            best_state = copy.deepcopy(
                {k: v.cpu() for k, v in model.state_dict().items()}
            )

    model.load_state_dict(best_state)
    return model.to(device), best, history


@torch.no_grad()
def collect_activations(model, loader, layer, device, max_tokens):
    model.eval()
    chunks, total = [], 0
    for batch in loader:
        captured = capture_activations(
            model,
            batch[0].float().to(device),
            {"layer": layer},
            forward_fn=model.features,
        )
        # Every token, CLS included. The dictionary has to cover whatever the
        # substitution hook will later feed it, and that hook replaces the
        # whole block output; training on a subset would leave the CLS token
        # reconstructed by features that never saw one.
        acts = captured["layer"]
        flat = acts.reshape(-1, acts.shape[-1]).cpu()
        chunks.append(flat)
        total += flat.shape[0]
        if total >= max_tokens:
            break
    return torch.cat(chunks)[:max_tokens]


def substitution_auroc(model, loader, layer, sae, device, groups):
    """AUROC with the layer replaced by its SAE reconstruction.

    The fine-tuned head is reused unchanged. Refitting a probe under each
    condition would let it re-adapt to whatever the SAE destroyed.
    """
    model.eval()
    logits, ys = [], []
    with torch.no_grad():
        for batch in loader:
            out = run_with_activation_substitution(
                model,
                batch[0].float().to(device),
                layer,
                lambda o: sae.reconstruct_activations(o),
            )
            logits.append(out.cpu())
            ys.append(batch[1])
    logits = torch.cat(logits).numpy()
    ys = torch.cat(ys).numpy().astype(int)
    window = roc_auc_score(ys, logits)
    recs = np.unique(groups)
    scores = np.array([logits[groups == r].mean() for r in recs])
    labels = np.array([ys[groups == r][0] for r in recs])
    recording = float("nan")
    if len(np.unique(labels)) > 1:
        recording = roc_auc_score(labels, scores)
    return window, recording


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    print(f"device: {device}")

    windows = build_windows(args)
    ch_names = channel_names(windows)
    groups = recording_of_window(windows)
    labels = window_labels(windows)
    idx, parts = stratified_group_split(groups, labels, args.seed)

    loaders = {
        name: DataLoader(
            Subset(windows, ids), batch_size=args.batch_size, shuffle=(name == "train")
        )
        for name, ids in idx.items()
    }
    for name in ("train", "valid", "test"):
        print(
            f"{name:5s}: {len(parts[name]):3d} recordings, {len(idx[name]):6d} "
            f"windows, {labels[idx[name]].mean():.1%} pathological"
        )

    p99 = measure_scale(windows, idx["train"])
    print(f"\ntrain-split |x| p99 = {p99:.3f} uV")

    input_chans = resolve_input_chans(ch_names).to(device)
    encoder = Labram.from_pretrained(HF_REPO)
    model = LabramBinary(encoder, input_chans).to(device)
    n_patches = int(encoder.patch_embed[0].n_patchs)
    print(
        f"encoder: {n_patches} patches x {len(ch_names)} channels = "
        f"{n_patches * len(ch_names)} tokens/window"
    )

    y_train = labels[idx["train"]]
    pos_weight = torch.tensor(
        (len(y_train) - y_train.sum()) / max(y_train.sum(), 1), dtype=torch.float32
    ).to(device)
    print(f"pos_weight: {pos_weight.item():.2f}\n")

    # Fine-tuning is the expensive stage and is identical for every layer, so
    # a sweep reuses one checkpoint rather than repeating it per layer.
    if args.ft_ckpt.exists():
        state = torch.load(args.ft_ckpt, map_location="cpu")
        model.load_state_dict(state["state_dict"])
        model = model.to(device)
        best_valid, history = state["best_valid"], state["history"]
        print(
            f"loaded fine-tuned weights from {args.ft_ckpt} "
            f"(valid recording AUROC {best_valid:.4f})"
        )
    else:
        model, best_valid, history = finetune(
            model, loaders, groups, idx, args, device, pos_weight
        )
        torch.save(
            {
                "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                "best_valid": best_valid,
                "history": history,
                "channels": ch_names,
            },
            args.ft_ckpt,
        )
        print(f"saved fine-tuned weights to {args.ft_ckpt}")
    print(f"\nbest validation recording AUROC: {best_valid:.4f}")

    # The gate reads validation and actually stops. Nothing about a model that
    # cannot do the task is worth interpreting.
    if best_valid < args.min_valid_auroc:
        print(
            f"BELOW GATE ({best_valid:.4f} < {args.min_valid_auroc:.2f}); "
            "stopping before the SAE stage."
        )
        raise SystemExit(1)

    results = {
        "best_valid_recording_auroc": best_valid,
        "finetune": history,
        "splits": {k_: sorted(v.tolist()) for k_, v in parts.items()},
        "cells": {},
    }

    def write_results():
        """Persist after every cell so a crash costs one cell, not the sweep."""
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=2, default=str))

    n_cells = len(args.layers) * len(args.expansions) * len(args.resample_steps)
    done = 0
    bar = "=" * 62

    for layer_idx in args.layers:
        layer = model.encoder.blocks[layer_idx]

        # Activations depend only on the layer, so they are collected once and
        # reused for every (expansion, resample interval) cell at that layer.
        # Re-extracting them per cell would dominate the runtime of the sweep.
        train_acts = collect_activations(
            model, loaders["train"], layer, device, args.max_sae_tokens
        )
        baselines = {
            name: evaluate(model, loaders[name], device, groups[idx[name]])
            for name in ("valid", "test")
        }

        for expansion in args.expansions:
            for resample_steps in args.resample_steps:
                done += 1
                k = args.k if args.k is not None else 8 * expansion
                n_feat = EMBED_DIM * expansion
                cell = f"layer{layer_idx}_e{expansion}_r{resample_steps}"
                print(f"\n{bar}\n[{done}/{n_cells}] {cell}\n{bar}", flush=True)
                print(
                    f"activations {tuple(train_acts.shape)} -> SAE "
                    f"{EMBED_DIM} -> {n_feat}, k={k} "
                    f"({len(train_acts) / n_feat:.0f} tokens/feature)"
                )

                # fit_sparse_autoencoder trains wherever the activations live,
                # and two million rows of 200 floats is only ~1.6 GB, so move
                # them to the GPU rather than training on CPU.
                sae, sae_history = fit_sparse_autoencoder(
                    train_acts.to(device),
                    expansion=expansion,
                    k=k,
                    epochs=args.sae_epochs,
                    batch_size=2048,
                    lr=1e-3,
                    seed=args.seed,
                    verbose=args.sae_log_every,
                    resample_steps=resample_steps,
                )
                sae = sae.to(device)
                diagnostics = sae_diagnostics(sae, train_acts)
                print(
                    f"SAE loss {sae_history['loss'][-1]:.4f}  diagnostics {diagnostics}"
                )

                # A randomly initialised SAE with the same shape and
                # normalisation is the control. If substituting it does not
                # hurt, the measurement is insensitive and a small drop for
                # the trained SAE means nothing.
                random_sae = SparseAutoencoder.from_config(sae.get_config()).to(device)
                random_sae.set_activation_normalization(
                    sae.activation_mean, sae.activation_std
                )

                entry = {
                    "layer": layer_idx,
                    "expansion": expansion,
                    "k": k,
                    "n_features": n_feat,
                    "resample_steps": resample_steps,
                    "diagnostics": diagnostics,
                    "tokens_per_feature": len(train_acts) / n_feat,
                    "sae_loss": sae_history["loss"],
                    "sae_dead": sae_history["dead"],
                    "sae_resampled": sae_history["resampled"],
                }
                for name in ("valid", "test"):
                    g = groups[idx[name]]
                    b_win, b_rec = baselines[name]
                    s_win, s_rec = substitution_auroc(
                        model, loaders[name], layer, sae, device, g
                    )
                    r_win, r_rec = substitution_auroc(
                        model, loaders[name], layer, random_sae, device, g
                    )
                    entry[name] = {
                        "baseline": {"window": b_win, "recording": b_rec},
                        "sae": {"window": s_win, "recording": s_rec},
                        "random_sae": {"window": r_win, "recording": r_rec},
                        "delta": {"window": b_win - s_win, "recording": b_rec - s_rec},
                    }
                    print(
                        f"  {name:5s} baseline {b_rec:.4f}  SAE {s_rec:.4f}  "
                        f"delta {b_rec - s_rec:+.4f}  random {r_rec:.4f}"
                    )

                results["cells"][cell] = entry
                write_results()
                del sae, random_sae
                if device == "cuda":
                    torch.cuda.empty_cache()

        del train_acts
        if device == "cuda":
            torch.cuda.empty_cache()

    header = (
        f"\n{'layer':>5} {'E':>3} {'resample':>9} {'dead':>7} "
        f"{'alive':>6} {'R2':>8} {'dWin':>8} {'rand':>7}"
    )
    print(header)
    for entry in results["cells"].values():
        g = entry["diagnostics"]
        print(
            f"{entry['layer']:>5} {entry['expansion']:>3} "
            f"{entry['resample_steps']:>9} {g['dead']:>6.2%} "
            f"{entry['n_features'] * (1 - g['dead']):>6.0f} {g['r2']:>8.4f} "
            f"{entry['test']['delta']['window']:>+8.4f} "
            f"{entry['test']['random_sae']['window']:>7.4f}"
        )

    write_results()
    print(f"\nwrote {args.out}")
    print(
        "Test-set window AUROC. A small delta only counts as evidence "
        "where the random column is clearly worse."
    )


if __name__ == "__main__":
    main()
