#!/usr/bin/env python
# Authors: Bruno Aristimunha <b.aristimunha@gmail.com>
#
# License: BSD (3-clause)
"""Convert the official emg2pose release into a BIDS tree.

Reference implementation pairing :class:`braindecode.datasets.EMG2Pose`
and the eegdash-viewer pose panel:

    python scripts/export_emg2pose_bids.py \
        --src ~/data/emg2pose/emg2pose_dataset_mini \
        --out ~/data/emg2pose-bids \
        [--subjects 893 542] [--with-pose]

Layout produced (one recording per hand x stage)::

    sub-<user>/
      ses-<session>/
        emg/
          sub-<user>_ses-<session>_task-<stage>-<side>_emg.vhdr|.eeg|.vmrk
          ..._channels.tsv
          ..._emg.json          # verbatim source metadata (generalized extras)
          ..._desc-pose.json    # skeleton sidecar (with --with-pose)
      participants.tsv

Sidecar keys are deliberately kept verbatim from ``metadata.csv``
(``stage``, ``side``, ``split``, ``moving_hand``, ...) so the generic
field mechanism in ``braindecode/datasets/_bids_meta.py`` flows them
into descriptions without any dataset-specific hardcoding.

Requires ``mne`` + ``h5py``. ``--with-pose`` additionally requires the
official ``emg2pose`` package (torch) for UmeTrack forward kinematics;
without it the viewer still shows EMG + raw joint-angle traces.
"""

import argparse
import base64
import json
from pathlib import Path

import h5py
import mne
import numpy as np
import pandas as pd

GROUP = "emg2pose/timeseries"
POSE_FS = 30  # skeleton sidecar frame rate


def _bids_task(stage: str, side: str) -> str:
    # `side` is folded into task because BIDS has no hand entity; this
    # keeps left/right recordings distinct under one subject/session.
    return f"{stage}-{side}"


def _convert_file(h5_path: Path, out_base: Path, row) -> Path:
    session = str(row["session"]).replace(":", "-")
    task = _bids_task(str(row["stage"]), str(row["side"]))
    stem = f"sub-{row['user']}_ses-{session}_task-{task}"
    ch_dir = out_base / f"sub-{row['user']}" / f"ses-{session}" / "emg"
    ch_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(h5_path, "r") as f:
        ts = f[GROUP]
        emg = ts["emg"][()].astype(np.float64)
        angles = ts["joint_angles"][()].astype(np.float64)
        fs = float(f.attrs["sample_rate"])

    data = np.concatenate([emg, angles], axis=1).T  # (n_ch, n_times)
    types = ["emg"] * emg.shape[1] + ["misc"] * angles.shape[1]
    names = [f"emg{i + 1}" for i in range(emg.shape[1])] + [
        f"ja{i}" for i in range(angles.shape[1])
    ]
    info = mne.create_info(names, fs, types)
    raw = mne.io.RawArray(data, info, verbose="ERROR")
    vhdr = ch_dir / f"{stem}_emg.vhdr"
    mne.export.export_raw(vhdr, raw, fmt="brainvision", overwrite=True, verbose="ERROR")

    channels = pd.DataFrame(
        {"name": names, "type": types,
         "unit": ["uV"] * emg.shape[1] + ["rad"] * angles.shape[1]}
    )
    channels.to_csv(ch_dir / f"{stem}_channels.tsv", sep="\t", index=False)

    record = {k: row[k] for k in (
        "stage", "side", "moving_hand",
        "held_out_user", "held_out_stage", "split", "generalization",
    ) if k in row.index}
    record["SamplingFrequency"] = fs
    record["TaskDescription"] = "emg2pose sEMG hand-pose stage"
    (ch_dir / f"{stem}_emg.json").write_text(json.dumps(record, indent=2))
    return vhdr


def _pose_sidecar(h5_path: Path, vhdr: Path, with_mesh: bool = False) -> bool:
    """Write the F10 skeleton sidecar via UmeTrack FK (needs emg2pose+torch)."""
    try:
        from emg2pose.hand_model import HandModel
        from emg2pose.kinematics import kinematic_positions
    except ImportError:
        print("  (pose skipped: install emg2pose[torch] for --with-pose)")
        return False

    with h5py.File(h5_path, "r") as f:
        angles = f[GROUP]["joint_angles"][()].astype(np.float32)
    valid = (~np.isnan(angles)).all(axis=1)

    # stride the 2 kHz timeline down to POSE_FS
    step = max(1, int(round(2000 / POSE_FS)))
    idx = np.arange(0, len(angles), step)
    angles_ds = angles[idx]
    valid_ds = valid[idx].astype(np.uint8)

    # kinematic_positions(hand_model, joint_angles, lengths=None) -> (T, J, 3)
    hand_model = HandModel()
    joints = np.asarray(kinematic_positions(hand_model, angles_ds), dtype=np.float32)

    parents = getattr(hand_model, "joint_parents", None)
    if parents is None:
        raise RuntimeError(
            "Cannot infer bone hierarchy: no `joint_parents` on HandModel; "
            "adapt `_pose_sidecar` to your emg2pose version."
        )
    bones = []
    for child, parent in enumerate(parents):
        if parent >= 0:
            bones.extend([int(parent), child])

    sidecar = {
        "format": "eegdash-pose",
        "version": 1,
        "fs": 2000 / step,
        "n_frames": int(len(idx)),
        "n_joints": int(joints.shape[1]),
        "bones": bones,
        "duration_s": float(len(idx)) * step / 2000.0,
        "positions": {
            "encoding": "base64-f32",
            "data": base64.b64encode(joints.reshape(-1).tobytes()).decode(),
        },
        "valid": base64.b64encode(valid_ds.tobytes()).decode(),
    }
    if with_mesh:
        sidecar["mesh"] = _extract_mesh_block(hand_model)
    stem = vhdr.with_suffix("").name.rsplit("_", 1)[0]
    (vhdr.parent / f"{stem}_desc-pose.json").write_text(json.dumps(sidecar))
    return True


def _f32(x):
    return base64.b64encode(np.asarray(x, dtype="<f4").tobytes()).decode()


def _u32(x):
    return base64.b64encode(np.asarray(x, dtype="<u4").ravel().tobytes()).decode()


def _extract_mesh_block(hand_model) -> dict:
    """Pull the LBS geometry off a UmeTrack HandModel.

    ``dense_bone_weights`` layout varies across emg2pose versions
    ((V,B) tensor, (B,V), or sparse); normalize to (V, B) dense before
    emitting the sparse triplet encoding the viewer expects.
    """
    import torch  # noqa: F401 — guaranteed by the emg2pose import upstream

    rest = np.asarray(hand_model.joint_rest_positions, dtype=np.float32)
    axes = np.asarray(hand_model.joint_rotation_axes, dtype=np.float32)
    verts = np.asarray(hand_model.mesh_vertices, dtype=np.float32)
    tris = np.asarray(hand_model.mesh_triangles, dtype=np.int64)

    w = hand_model.dense_bone_weights
    if hasattr(w, "to_dense"):
        w = w.to_dense()
    w = np.asarray(w, dtype=np.float32)
    if w.shape[0] == verts.shape[0] and w.shape[1] == rest.shape[0]:
        pass
    elif w.shape[1] == verts.shape[0]:
        w = w.T
    else:
        raise RuntimeError(
            f"Unrecognized dense_bone_weights shape {w.shape} for "
            f"{verts.shape[0]} vertices x {rest.shape[0]} joints."
        )
    rows, cols = np.nonzero(w > 1e-6)

    return {
        "mode": "umetrack-lbs",
        "rest_vertices": {"encoding": "base64-f32", "data": _f32(verts)},
        "triangles": {"encoding": "base64-u32", "data": _u32(tris)},
        "weight_vertex": {"encoding": "base64-u32", "data": _u32(rows)},
        "weight_bone": {"encoding": "base64-u32", "data": _u32(cols)},
        "weight_value": {"encoding": "base64-f32", "data": _f32(w[rows, cols])},
        "joint_axes": {"encoding": "base64-f32", "data": _f32(axes)},
        "joint_rest": {"encoding": "base64-f32", "data": _f32(rest)},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", required=True,
                    help="extracted emg2pose dataset dir (has metadata.csv)")
    ap.add_argument("--out", required=True, help="output BIDS root")
    ap.add_argument("--subjects", nargs="*", default=None, help="restrict to user ids")
    ap.add_argument("--with-pose", action="store_true",
                    help="export skeleton FK sidecars")
    ap.add_argument("--with-mesh", action="store_true",
                    help="add skinned-mesh blocks to pose sidecars (implies --with-pose)")
    args = ap.parse_args()
    if args.with_mesh:
        args.with_pose = True

    src, out = Path(args.src).resolve(), Path(args.out).resolve()
    meta = pd.read_csv(src / "metadata.csv")
    if args.subjects:
        meta = meta[meta["user"].astype(str).isin(map(str, args.subjects))]

    done = []
    for _, row in meta.iterrows():
        h5_path = src / row["filename"]
        if not h5_path.is_file():
            print(f"missing {row['filename']}, skipping")
            continue
        vhdr = _convert_file(h5_path, out, row)
        has_pose = args.with_pose and _pose_sidecar(h5_path, vhdr, with_mesh=args.with_mesh)
        done.append(row)
        print(f"converted {vhdr.name}{' (+pose)' if has_pose else ''}")

    participants = (
        pd.DataFrame(done)
        .drop_duplicates("user")[["user", "held_out_user"]]
        .rename(columns={"user": "participant_id"})
    )
    participants["participant_id"] = "sub-" + participants["participant_id"].astype(str)
    participants.to_csv(out / "participants.tsv", sep="\t", index=False)
    print(f"\nwrote {len(done)} recordings to {out}")


if __name__ == "__main__":
    main()
