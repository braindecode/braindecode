# type: ignore
""".. _load-pretrained-models:

Loading and Adapting Pretrained Foundation Models
=================================================

.. raw:: html

   <div style="display: flex; align-items: center; justify-content: center;
               gap: 20px; margin-bottom: 24px; padding: 18px 24px;
               background: linear-gradient(135deg, #f0f7ff 0%, #fdf6ec 100%);
               border-radius: 10px; border: 1px solid #e0e8f0;">
     <img src="../../_static/braindecode_symbol.png"
          style="height: 55px;" alt="Braindecode">
     <span style="font-size: 26px; color: #9ca3af; font-weight: 300;">+</span>
     <img src="../../_static/hf-logo-with-title.png"
          style="height: 55px;" alt="Hugging Face">
   </div>

All braindecode models can load and save weights on the
`Hugging Face Hub <https://huggingface.co/braindecode>`_ via
``from_pretrained`` and ``push_to_hub``.  For **foundation models**
we additionally provide curated pretrained checkpoints with mapped
weights, so you can fine-tune them out of the box.

.. important::

   The curation and standardization of the pretrained foundation model
   weights available on the Braindecode Hugging Face organization was
   carried out as part of the
   `OpenEEG-Bench <https://huggingface.co/spaces/braindecode/OpenEEGBench>`_
   benchmark [2]_.
   If you use these pretrained weights, please cite the paper.

.. code-block:: bash

   pip install braindecode[hub]

This tutorial shows how to load pretrained EEG foundation models,
adapt them to new tasks, extract features, and save/restore full model
configurations, using a **unified API** inspired by the
`Hugging Face transformers <https://huggingface.co/docs/transformers>`_
library.

.. contents:: This example covers:
   :local:
   :depth: 2
"""

# Authors: Bruno Aristimunha <b.aristimunha@gmail.com>
#
# License: BSD (3-clause)

import json
import os
import warnings

import torch
from huggingface_hub import login

warnings.simplefilter("ignore")

hf_token = os.environ.get("HF_TOKEN")
if hf_token:
    login(token=hf_token)

######################################################################
# Loading a pretrained model
# --------------------------
#
# All braindecode foundation models support the ``from_pretrained``
# method, which downloads model weights and configuration from the
# Hugging Face Hub.
#
# We start with BENDR [1]_ as an example:

from braindecode.models import BENDR

model = BENDR.from_pretrained("braindecode/braindecode-bendr", n_outputs=2)
print(f"Loaded BENDR with n_outputs={model.n_outputs}")

######################################################################
# The loaded model is ready for inference:

x = torch.randn(2, 20, 768)  # (batch, n_chans, n_times)
model.eval()
with torch.no_grad():
    out = model(x)
print(f"Output shape: {out.shape}")

######################################################################
# Adapting to a new task
# ----------------------
#
# Changing the number of outputs
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# When fine-tuning a pretrained model on a dataset with a different number
# of classes, pass ``n_outputs`` directly to ``from_pretrained``. The
# backbone weights are loaded and the classification head is automatically
# rebuilt:

model = BENDR.from_pretrained("braindecode/braindecode-bendr", n_outputs=4)
print(f"n_outputs after loading: {model.n_outputs}")

with torch.no_grad():
    out = model(x)
print(f"Output shape with 4 classes: {out.shape}")

######################################################################
# You can also swap the head at any time using ``reset_head``:

model.reset_head(10)
print(f"n_outputs after reset_head: {model.n_outputs}")

with torch.no_grad():
    out = model(x)
print(f"Output shape after reset: {out.shape}")

######################################################################
# Extracting features
# -------------------
#
# All foundation models support ``return_features=True`` in their
# ``forward()`` method. This returns a dictionary with:
#
# - ``"features"`` -- encoder embeddings before the classification head
# - ``"cls_token"`` -- the CLS token (if the model has one, otherwise ``None``)

model.eval()
with torch.no_grad():
    out = model(x, return_features=True)

print(f"Type: {type(out)}")
print(f"Features shape: {out['features'].shape}")
print(f"CLS token: {out['cls_token']}")

######################################################################
# .. tip::
#
#    This is useful for **transfer learning**: freeze the backbone and
#    train only a new head on the extracted features.
#    See :ref:`finetune-foundation-model` for a complete example.

######################################################################
# Saving and restoring configurations
# ------------------------------------
#
# ``get_config`` returns a JSON-serializable dictionary of **all**
# ``__init__`` parameters (not just the 6 EEG-specific ones). This
# includes model-specific hyperparameters like ``encoder_h``,
# ``drop_prob``, ``activation``, etc.

config = model.get_config()
print(json.dumps({k: v for k, v in config.items() if k != "chs_info"}, indent=2))

######################################################################
# You can reconstruct the model (without weights) using ``from_config``:

model_copy = BENDR.from_config(config)
print(f"Reconstructed: n_outputs={model_copy.n_outputs}")

######################################################################
# When pushing to the Hub, the full config is saved automatically in
# ``config.json`` alongside the model weights:
#
# .. code-block:: python
#
#     model.push_to_hub("username/my-bendr-model")
#     # Later:
#     model = BENDR.from_pretrained("username/my-bendr-model")

######################################################################
# Unified API across foundation models
# -------------------------------------
#
# The same API works across **all** foundation models:
#
# .. list-table::
#    :header-rows: 1
#    :widths: 25 15 15 20 15
#
#    * - Model
#      - ``from_pretrained``
#      - ``reset_head``
#      - ``return_features``
#      - ``get_config``
#    * - :class:`~braindecode.models.BENDR`
#      - |check|
#      - |check|
#      - |check|
#      - |check|
#    * - :class:`~braindecode.models.BIOT`
#      - |check|
#      - |check|
#      - |check|
#      - |check|
#    * - :class:`~braindecode.models.CBraMod`
#      - |check|
#      - |check|
#      - |check|
#      - |check|
#    * - :class:`~braindecode.models.EEGPT`
#      - |check|
#      - |check|
#      - |check|
#      - |check|
#    * - :class:`~braindecode.models.Labram`
#      - |check|
#      - |check|
#      - |check| (+ ``cls_token``)
#      - |check|
#    * - :class:`~braindecode.models.LUNA`
#      - |check|
#      - |check|
#      - |check|
#      - |check|
#    * - :class:`~braindecode.models.REVE`
#      - |check|
#      - |check|
#      - |check|
#      - |check|
#    * - :class:`~braindecode.models.SignalJEPA` variants
#      - |check|
#      - |check|
#      - |check|
#      - |check|
#
# .. |check| unicode:: 0x2714
#
# The feature shapes differ between models (reflecting their
# architecture), but the API is always the same.

######################################################################
# Available pretrained weights
# ----------------------------
#
# .. raw:: html
#
#    <div style="display: flex; align-items: center; gap: 10px;
#                margin-bottom: 12px;">
#      <img src="../../_static/hf-logo-with-title.png"
#           style="height: 32px;" alt="Hugging Face">
#    </div>
#
# The figure below shows all available pretrained checkpoints ranked by
# parameter count, with colors indicating the hosting organization.
# Parameter counts are read directly from the Hub.

import matplotlib.pyplot as plt
import numpy as np

from braindecode.models import (
    BENDR,
    BIOT,
    EEGPT,
    LUNA,
    REVE,
    CBraMod,
    Labram,
    SignalJEPA,
)

# (display_name, model_class, from_pretrained kwargs, org)
checkpoints = [
    (
        "BENDR",
        BENDR,
        dict(
            pretrained_model_name_or_path="braindecode/braindecode-bendr", n_outputs=2
        ),
        "braindecode",
    ),
    (
        "BIOT (16ch)",
        BIOT,
        dict(pretrained_model_name_or_path="braindecode/biot-pretrained-prest-16chs"),
        "braindecode",
    ),
    (
        "BIOT (18ch)",
        BIOT,
        dict(
            pretrained_model_name_or_path="braindecode/biot-pretrained-shhs-prest-18chs"
        ),
        "braindecode",
    ),
    (
        "CBraMod",
        CBraMod,
        dict(
            pretrained_model_name_or_path="braindecode/cbramod-pretrained",
            n_outputs=2,
            n_chans=22,
            n_times=1000,
            sfreq=250,
        ),
        "braindecode",
    ),
    (
        "EEGPT",
        EEGPT,
        dict(
            pretrained_model_name_or_path="braindecode/eegpt-pretrained",
            n_chans=62,
            chan_proj_type="none",
        ),
        "braindecode",
    ),
    (
        "Labram",
        Labram,
        dict(
            pretrained_model_name_or_path="braindecode/labram-pretrained", n_chans=128
        ),
        "braindecode",
    ),
    (
        "SignalJEPA",
        SignalJEPA,
        dict(
            pretrained_model_name_or_path="braindecode/SignalJEPA-pretrained",
            n_outputs=2,
            n_chans=19,
            input_window_seconds=5,
            sfreq=256,
        ),
        "braindecode",
    ),
    (
        "REVE (base)",
        REVE,
        dict(
            pretrained_model_name_or_path="brain-bzh/reve-base",
            n_outputs=2,
            n_chans=64,
            n_times=512,
            sfreq=256,
        ),
        "brain-bzh",
    ),
    (
        "REVE (large)",
        REVE,
        dict(
            pretrained_model_name_or_path="brain-bzh/reve-large",
            n_outputs=2,
            n_chans=64,
            n_times=512,
            sfreq=256,
        ),
        "brain-bzh",
    ),
    (
        "LUNA (base)",
        LUNA,
        dict(
            pretrained_model_name_or_path="PulpBio/LUNA",
            filename="LUNA_base.safetensors",
            n_outputs=2,
            n_chans=22,
            n_times=1000,
            embed_dim=64,
            num_queries=4,
            depth=8,
        ),
        "PulpBio",
    ),
    (
        "LUNA (large)",
        LUNA,
        dict(
            pretrained_model_name_or_path="PulpBio/LUNA",
            filename="LUNA_large.safetensors",
            n_outputs=2,
            n_chans=22,
            n_times=1000,
            embed_dim=96,
            num_queries=6,
            depth=10,
        ),
        "PulpBio",
    ),
    (
        "LUNA (huge)",
        LUNA,
        dict(
            pretrained_model_name_or_path="PulpBio/LUNA",
            filename="LUNA_huge.safetensors",
            n_outputs=2,
            n_chans=22,
            n_times=1000,
            embed_dim=128,
            num_queries=8,
            depth=24,
        ),
        "PulpBio",
    ),
]

# Skip gated models when no HF token is available (e.g. fork PRs)
if not hf_token:
    checkpoints = [(d, c, k, o) for d, c, k, o in checkpoints if o != "brain-bzh"]

names, params_m, orgs = [], [], []
for display, cls, kwargs, org in checkpoints:
    mdl = cls.from_pretrained(**kwargs)
    n_params = sum(
        p.numel()
        for p in mdl.parameters()
        if not isinstance(p, torch.nn.UninitializedParameter)
    )
    names.append(display)
    params_m.append(n_params / 1e6)
    orgs.append(org)
    print(f"  {display:15s}  {n_params / 1e6:8.1f}M params")

params_m = np.array(params_m)

# Sort by parameter count (ascending) for horizontal bar chart
order = np.argsort(params_m)
names = [names[i] for i in order]
params_m = params_m[order]
orgs = [orgs[i] for i in order]

from matplotlib.ticker import FuncFormatter

# -- palette ----------------------------------------------------------
org_palette = {
    "braindecode": "#2D6A9F",
    "brain-bzh": "#C04E3E",
    "PulpBio": "#4A8B6F",
}
org_markers = {"braindecode": "o", "brain-bzh": "D", "PulpBio": "s"}
colors = [org_palette[o] for o in orgs]
markers = [org_markers[o] for o in orgs]

# -- figure setup ------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.5, 5.0), dpi=120)
fig.patch.set_facecolor("#FAFAFA")
ax.set_facecolor("#FAFAFA")

y_pos = np.arange(len(names))

# Horizontal reference lines (subtle, behind everything)
for y in y_pos:
    ax.axhline(y, color="#E8E8E8", linewidth=0.6, zorder=0)

# Stem lines from left edge to dot
ax.hlines(y_pos, 0.5, params_m, color="#D0D0D0", linewidth=0.9, zorder=1)

# Dots (different shapes per org for accessibility)
for i in range(len(names)):
    ax.scatter(
        params_m[i],
        y_pos[i],
        color=colors[i],
        s=70,
        zorder=3,
        marker=markers[i],
        edgecolors="white",
        linewidths=0.8,
    )

# Value labels always to the right of the dot
for i, pm in enumerate(params_m):
    ax.text(
        pm * 1.12,
        y_pos[i],
        f"{pm:.1f}M",
        ha="left",
        va="center",
        fontsize=7.8,
        color="#444444",
        fontweight="bold",
        zorder=4,
    )

# Y-axis
ax.set_yticks(y_pos)
ax.set_yticklabels(names, fontsize=9, color="#333333", fontweight="medium")

# X-axis: log scale with human-readable labels
ax.set_xscale("log")
ax.set_xlim(1.5, params_m.max() * 2.5)
ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.0f}M"))
ax.set_xticks([3, 10, 30, 100, 300])
ax.tick_params(axis="x", colors="#999999", labelsize=8, length=0, pad=4)
ax.tick_params(axis="y", length=0)

# Title + subtitle
ax.set_title(
    "EEG Foundation Models",
    fontsize=13,
    fontweight="bold",
    color="#1a1a1a",
    loc="left",
    pad=22,
)
ax.text(
    0.0,
    1.02,
    "Parameter count of curated checkpoints on Hugging Face Hub",
    transform=ax.transAxes,
    fontsize=8.5,
    color="#777777",
    va="bottom",
)

# Organization legend with shape + color
for org_name in org_palette:
    ax.scatter(
        [],
        [],
        c=org_palette[org_name],
        s=40,
        marker=org_markers[org_name],
        label=org_name,
        edgecolors="white",
    )
legend = ax.legend(
    fontsize=7.5,
    loc="lower right",
    frameon=True,
    framealpha=0.95,
    edgecolor="#E0E0E0",
    handletextpad=0.4,
    borderpad=0.8,
    labelspacing=0.7,
)
legend.get_frame().set_facecolor("#FAFAFA")

# Clean spines
for spine in ax.spines.values():
    spine.set_visible(False)

# Minimal grid on x only
ax.xaxis.grid(True, color="#E8E8E8", linewidth=0.5, which="major")
ax.yaxis.grid(False)
ax.set_axisbelow(True)

fig.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

######################################################################
# Braindecode organization
# ~~~~~~~~~~~~~~~~~~~~~~~~
#
# The `braindecode <https://huggingface.co/braindecode>`_ organization on
# Hugging Face re-hosts the official pretrained weights.
# All models below follow the same one-line loading pattern:
#
# .. code-block:: python
#
#    model = Model.from_pretrained("<repo-id>", n_outputs=...)
#
# .. list-table::
#    :header-rows: 1
#    :widths: 25 45 30
#
#    * - Model
#      - Hub Repository
#      - Details
#    * - :class:`~braindecode.models.BENDR`
#      - ``braindecode/braindecode-bendr``
#      - 20 channels
#    * - :class:`~braindecode.models.BIOT`
#      - ``braindecode/biot-pretrained-prest-16chs``
#      - 16 ch, PREST
#    * - :class:`~braindecode.models.BIOT`
#      - ``braindecode/biot-pretrained-shhs-prest-18chs``
#      - 18 ch, SHHS + PREST
#    * - :class:`~braindecode.models.BIOT`
#      - ``braindecode/biot-pretrained-six-datasets-18chs``
#      - 18 ch, 6 datasets
#    * - :class:`~braindecode.models.CBraMod`
#      - ``braindecode/cbramod-pretrained``
#      - channel-agnostic
#    * - :class:`~braindecode.models.EEGPT`
#      - ``braindecode/eegpt-pretrained``
#      - 62 ch, 250 Hz
#    * - :class:`~braindecode.models.Labram`
#      - ``braindecode/labram-pretrained``
#      - 128 channels
#    * - :class:`~braindecode.models.SignalJEPA`
#      - ``braindecode/SignalJEPA-pretrained``
#      - 19 channels
#    * - :class:`~braindecode.models.SignalJEPA_Contextual`
#      - ``braindecode/SignalJEPA-Contextual-pretrained``
#      - 19 channels
#    * - :class:`~braindecode.models.SignalJEPA_PostLocal`
#      - ``braindecode/SignalJEPA-PostLocal-pretrained``
#      - 19 channels
#    * - :class:`~braindecode.models.SignalJEPA_PreLocal`
#      - ``braindecode/SignalJEPA-PreLocal-pretrained``
#      - 19 channels
#
# External organizations
# ~~~~~~~~~~~~~~~~~~~~~~
#
# Some pretrained weights are hosted by the original model authors:
#
# .. list-table::
#    :header-rows: 1
#    :widths: 20 20 35 25
#
#    * - Model
#      - Organization
#      - Hub Repository
#      - Details
#    * - :class:`~braindecode.models.REVE`
#      - `brain-bzh <https://huggingface.co/brain-bzh>`_
#      - ``brain-bzh/reve-base``
#      - 69M params
#    * - :class:`~braindecode.models.REVE`
#      - `brain-bzh <https://huggingface.co/brain-bzh>`_
#      - ``brain-bzh/reve-large``
#      - 400M params
#    * - :class:`~braindecode.models.LUNA`
#      - `PulpBio <https://huggingface.co/PulpBio>`_
#      - ``PulpBio/LUNA``
#      - base / large / huge
#
# .. note::
#
#    **Loading LUNA**: This repo stores multiple weight variants in a
#    single repository. Use the ``filename`` parameter to select one:
#
#    .. code-block:: python
#
#       from braindecode.models import LUNA
#
#       model = LUNA.from_pretrained(
#           "PulpBio/LUNA",
#           filename="LUNA_base.safetensors",
#           n_outputs=2, n_chans=22, n_times=1000,
#           embed_dim=64, num_queries=4, depth=8,
#       )
#
#    Available files: ``LUNA_base.safetensors`` (7M),
#    ``LUNA_large.safetensors`` (43M), ``LUNA_huge.safetensors`` (311M).

######################################################################
# References
# ----------
#
# .. [1] Kostas, D., Aroca-Ouellette, S., and Bhatt, F. (2021).
#    BENDR: Using Transformers and a Contrastive Self-Supervised Learning
#    Task to Learn From Massive Amounts of EEG Data.
#    Frontiers in Human Neuroscience, 15.
#
# .. [2] Guetschel, P., Aristimunha, B., Truong, D., Kokate, K.,
#    Tangermann, M., and Delorme, A. (2026).
#    Toward OpenEEG-Bench: A Live Community-Driven Benchmark for EEG
#    Foundation Models.
#    In Proceedings of the 34th European Signal Processing Conference
#    (EUSIPCO 2026), Bruges, Belgium.
