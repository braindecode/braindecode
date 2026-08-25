""".. _emg2pose-viewer-example:

emg2pose BIDS Dataset and Interactive Viewer
============================================

In this example, we build a small emg2pose-style BIDS tree, load it with
:class:`~braindecode.datasets.EMG2Pose` and preview a recording with the
embedded `eegdash-viewer <https://github.com/eegdash/eegdash-viewer>`_
trace viewer.

The dataset carries 16 wrist sEMG channels plus 20 joint-angle channels
per recording. Source metadata columns (``stage``, ``side``, ``split``,
``moving_hand``, ...) flow into each recording's ``description``
verbatim — no dataset-specific hardcoding required.
"""

# Authors: Bruno Aristimunha <b.aristimunha@gmail.com>
#
# License: BSD (3-clause)

import json
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np

from braindecode.datasets import EMG2Pose

###############################################################################
# For documentation purposes we synthesize one tiny recording instead of
# downloading the official release (431 GB). In real usage you would run:
#
# .. code-block:: bash
#
#     python scripts/export_emg2pose_bids.py \
#         --src ~/data/emg2pose_dataset_mini --out ~/data/emg2pose-bids

root = Path(tempfile.mkdtemp()) / "emg2pose-bids"
ch_dir = root / "sub-893" / "ses-20220101" / "emg"
ch_dir.mkdir(parents=True)

sfreq = 1000
n_times = 5 * sfreq
t = np.arange(n_times) / sfreq
emg = 0.6 * np.sin(2 * np.pi * 40 * t) + 0.1 * np.random.randn(16, n_times)
angles = np.tile(np.sin(2 * np.pi * 1 * t), (20, 1)) * 0.5
data = np.concatenate([emg, angles]).astype(np.float64) * 1e-6

info = mne.create_info(
    [f"emg{i + 1}" for i in range(16)] + [f"ja{i}" for i in range(20)],
    sfreq,
    ["emg"] * 16 + ["misc"] * 20,
)
mne.export.export_raw(
    ch_dir / "sub-893_ses-20220101_task-fist-right_emg.vhdr",
    mne.io.RawArray(data, info, verbose="ERROR"),
    fmt="brainvision",
    overwrite=True,
    verbose="ERROR",
)

###############################################################################
# Sidecars carry the source-release metadata; unknown keys are kept so
# future upstream columns appear in descriptions automatically.

(ch_dir / "sub-893_ses-20220101_task-fist-right_emg.json").write_text(json.dumps({
    "SamplingFrequency": 2000.0,
    "stage": "fist",
    "side": "right",
    "split": "train",
    "moving_hand": "right",
    "held_out_user": False,
}))
(root / "participants.tsv").write_text("participant_id\thandedness\nsub-893\tR\n")

###############################################################################
# Load it like any other braindecode dataset and inspect the generalized
# description fields.

dataset = EMG2Pose(root)
print(dataset.records[["subject", "session", "task", "stage", "side", "split", "handedness"]])

###############################################################################
# Static preview of a few EMG channels and joint angles.

raw = dataset.datasets[0].raw.copy().crop(tmax=2.0).pick(["emg1", "emg8", "ja0", "ja10"])
fig = raw.plot(scalings="auto", show=False, block=False)
plt.show()

###############################################################################
# Interactive preview: ``plot`` embeds the eegdash-viewer over a
# localhost server (kernel and browser on the same machine). With a
# ``*_desc-pose.json`` skeleton sidecar next to the recording, the hand
# panel animates with the time cursor (toggle with ``p``). On remote
# kernels, point ``viewer_url``/``data_url`` at hosted deployments to
# skip the local server entirely.
#
# .. code-block:: python
#
#     dataset.plot(index=0)                                  # local
#     dataset.plot(viewer_url="https://viewer.eegdash.org",  # hosted
#                  data_url="https://data.eegdash.org/emg2pose")
