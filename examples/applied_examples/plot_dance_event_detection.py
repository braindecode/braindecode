r""".. _dance-event-detection:

DANCE: asynchronous event detection on real P300 EEG
=======================================================

braindecode defines brain decoding as :math:`f_\theta: X \to y` (see
:ref:`the brain decode problem <models>`), and is explicit that **the definition of** :math:`y`
**is broad**. Most tutorials in this repository instantiate that mapping at a
single point in the space it allows: :math:`X` is a window already cut and
aligned to a known onset (a motor-imagery cue, a sleep-epoch boundary, a P300
flash), and :math:`y` is one categorical label for the whole window.

DANCE (this tutorial) instantiates the same :math:`f_\theta: X \to y` at a
different point: :math:`X` is one long, unaligned recording, and :math:`y` is
the set of ``(start, end, class)`` events :math:`f_\theta` must locate and
classify itself. The definition did not change -- only the shape of :math:`y`
did.

.. image:: /_static/dance/xy_shape_of_y.svg
   :alt: Same f theta of X arrow y, different shape of y -- onset-informed
       decoding gives one label per aligned window; DANCE gives a set of
       (start, end, class) events on a continuous recording.

This tutorial shows how to use the :class:`~braindecode.models.DANCE` model for
**event detection** on continuous, real EEG recordings, with plain PyTorch (no
skorch, no Lightning). We reproduce (at reduced scale) the P300 setting from
the original paper [1]_, using the Brain Invaders BI2014a dataset [2]_ loaded
through :class:`~braindecode.datasets.MOABBDataset`.

.. topic:: A new decoding paradigm: asynchronous event detection

    The dominant practice in EEG decoding benchmarks is to classify windows
    that are already time-locked to a known event onset (e.g. "this 1 s window,
    which we know starts exactly at a flash, is a target or non-target"). This
    *onset-informed* paradigm is convenient, but unrealistic outside controlled
    experiments: in real-world or naturalistic monitoring, the precise onset of
    an event is generally not known in advance.

    DANCE [1]_ reframes decoding as a **set-prediction problem**, directly on
    raw, unaligned signals: given a long window, the model predicts a *set* of
    ``(start, end, class)`` events, without ever being told where an event
    starts. This is exactly the DETR [3]_ recipe (object detection in images)
    transposed to time series, with a Perceiver [4]_ module bridging the
    variable input duration and the fixed-size event-query decoder. On the
    Brain Invaders BI2013/BI2014a/BI2014b P300 datasets, DANCE matches the
    accuracy of onset-informed models *without ever using the flash onset*.

The pipeline is:

1. load a few subjects of continuous, real P300 EEG (BI2014a) with
   :class:`~braindecode.datasets.MOABBDataset`,
2. preprocess with the exact minimal recipe from the paper: band-pass
   0.1-100 Hz, resample to 128 Hz, per-channel robust scaling clamped to
   ``[-16, 16]``,
3. cut it into fixed-length windows with
   :func:`~braindecode.preprocessing.create_fixed_length_windows` (``W = 32``
   s, matching the paper's BI2014a configuration),
4. turn each window's stimulus annotations (``Target``/``NonTarget`` flashes)
   into a DANCE target dict (normalized ``start``/``end`` in ``[0, 1]`` plus a
   dense per-token class map),
5. train :class:`~braindecode.models.DANCE` with
   :class:`~braindecode.training.DanceLoss` on a cross-subject split (train on
   some subjects, evaluate on held-out ones -- the paper's protocol), and
6. evaluate with the event-level :func:`~braindecode.training.f1_event` metric
   and a per-token macro F1 on the dense head
   (:func:`sklearn.metrics.f1_score`).

DANCE exposes two entry points: ``model.forward(x)`` returns the dense
per-token logits ``(B, num_latents, n_outputs)``, while ``model.detect(x)``
returns the event set ``{class, start, end, dense}``. Training and evaluation
below use ``detect`` (the loss needs the query set); a downstream user who only
wants a per-token prediction can call ``forward`` directly.

.. note::
    DANCE follows a DETR-style **class-0 = background / no-object** convention:
    real event classes are ``1 .. n_outputs - 1`` and class ``0`` is reserved
    for "no event". Here class ``1`` is a non-target flash and class ``2`` a
    target flash; the target builder and both F1 metrics honour this
    convention automatically.

.. note::
    To keep the docs build fast, this tutorial uses only 3 subjects and 2
    training epochs, far short of the paper's full protocol (all subjects,
    100 epochs with early stopping). Expect **low** F1 scores below; see the
    :ref:`Conclusion <dance-event-detection-conclusion>` for what changes at
    full scale.

.. contents:: This example covers:
   :local:
   :depth: 2

"""  # noqa: D205, E501

# Authors: Bruno Aristimunha <b.aristimunha@gmail.com>
#
# License: MIT

random_state = 0

######################################################################
# Loading and preprocessing the dataset
# -------------------------------------
#
# Loading the raw recordings
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# BI2014a is a Brain Invaders P300 (oddball) dataset: subjects watch a
# stream of flashes, most of which are "non-target" and a few "target". We
# load 3 subjects: 2 for training, 1 held out for evaluation, following the
# paper's **cross-subject** splitting protocol.

from braindecode.datasets import MOABBDataset

subject_ids = [1, 2, 3]
dataset = MOABBDataset(dataset_name="BI2014a", subject_ids=subject_ids)

######################################################################
# Preprocessing
# ~~~~~~~~~~~~~
#
# We reproduce the exact minimal preprocessing recipe of the DANCE paper
# [1]_: band-pass filter between 0.1 and 100 Hz, resample to a common
# 128 Hz, then a per-channel, per-session robust scaling (subtract the
# median, divide by the interquartile range) clamped to ``[-16, 16]``. We
# reach for :func:`sklearn.preprocessing.robust_scale` rather than
# hand-rolling the median/IQR arithmetic -- ``axis=1`` scales each channel
# (row) using its own median and interquartile range computed across time
# (columns), which is exactly the per-channel recipe above.

import numpy as np
from sklearn.preprocessing import robust_scale

from braindecode.preprocessing import Preprocessor, preprocess


def robust_scale_clamp(data):
    """Per-channel median/IQR scaling, clamped to [-16, 16] (DANCE recipe)."""
    return np.clip(robust_scale(data, axis=1), -16, 16)


SFREQ = 128.0
preprocessors = [
    Preprocessor("pick_types", eeg=True, stim=False),
    Preprocessor("filter", l_freq=0.1, h_freq=100.0),
    Preprocessor("resample", sfreq=SFREQ),
    Preprocessor(robust_scale_clamp, apply_on_array=True),
]
preprocess(dataset, preprocessors)

######################################################################
# Cut fixed-length windows and read the flash events
# -----------------------------------------------------
#
# DANCE consumes long, fixed-length windows rather than per-trial epochs, so
# we use :func:`~braindecode.preprocessing.create_fixed_length_windows`
# instead of :func:`~braindecode.preprocessing.create_windows_from_events`.
# We use ``W = 32`` s and up to 150 events per window, matching the paper's
# BI2014a configuration (mean 45.8 events/window in the full dataset).

from braindecode.preprocessing import create_fixed_length_windows

WINDOW_S, N_CLASSES, NUM_LATENTS, MAX_EVENTS = 32.0, 3, 256, 150
WINDOW_SAMPLES = int(WINDOW_S * SFREQ)

windows_ds = create_fixed_length_windows(
    dataset,
    window_size_samples=WINDOW_SAMPLES,
    window_stride_samples=WINDOW_SAMPLES,
    drop_last_window=True,
    preload=True,
    use_mne_epochs=False,  # guarantees EEGWindowsDataset -> (X, y, crop_inds)
)


def bi_annotations_to_events(raw):
    """Read ``(start_s, end_s, class_int)`` flash events from a BI* raw.

    Class ``1`` = non-target flash, class ``2`` = target flash (class ``0``
    is reserved for background/no-event, per the CLASS-0 CONTRACT).
    """
    label_to_class = {"NonTarget": 1, "Target": 2}
    events = []
    for ann in raw.annotations:
        cls = label_to_class.get(str(ann["description"]))
        if cls is None:
            continue
        events.append((float(ann["onset"]), float(ann["onset"] + ann["duration"]), cls))
    return events


######################################################################
# Map each window's annotations to DANCE targets
# -------------------------------------------------
#
# This is the code that builds the :math:`y` from the right-hand panel of
# the figure above. For one window :math:`X`, the paper [1]_ defines the
# target as a set of events
#
# .. math::
#
#     y = \{e_i\}_{i=1}^N, \qquad e_i = (b_i, c_i),
#
# where :math:`b_i = (t_{\mathrm{start}}, t_{\mathrm{end}}) \in [0, 1]^2` are
# the event's boundaries normalized to the window duration -- exactly the
# colored spans in the figure -- and :math:`c_i \in \{0, \dots, K\}` is its
# class: :math:`c_i = 0` is the shared **background / no-object** class (the
# unlabeled gray span in the figure), and real classes are :math:`1, \dots,
# K` (here :math:`K = 2`: non-target and target flashes).
#
# Two consequences of this definition matter for ``dance_target_builder``
# below:
#
# 1. **Padding is indistinguishable from background.**
#    :class:`~braindecode.training.DanceLoss`'s matcher expects a fixed-size
#    set of ``max_events`` slots per window; any slot with no real event
#    defaults to :math:`c_i = 0`, the same value that also denotes
#    "no object" -- there is only one id for both, by design, and it never
#    collides because a real annotation is never given class ``0``.
# 2. **The dense head needs a per-token version of the same set.** DANCE
#    also trains a per-timestep classifier over ``num_latents`` latent
#    tokens; we rasterize each event's :math:`[t_{\mathrm{start}},
#    t_{\mathrm{end}})` span onto that token grid, so every idle token
#    defaults to the same background class ``0``.
#
# This is exactly the recipe braindecode's own upstream reference -- the
# ``dance/example/data.py`` MOABB bridge in `facebookresearch/dance
# <https://github.com/facebookresearch/dance>`_ -- uses for this dataset:
# class ``0`` for padding/background, ``1`` for non-target, ``2`` for
# target, and events zero-padded to ``max_events = 150``.

import torch


def dance_target_builder(
    annotations, window_onset, window_duration, max_events, num_latents
):
    """Build one window's DANCE target: the event set y and its dense grid."""
    start = torch.zeros(max_events)
    end = torch.zeros(max_events)
    cls = torch.zeros(max_events, dtype=torch.long)  # cls[i] = 0 -> background/padding
    w0, wd = window_onset, window_duration
    kept = 0
    for s, e, c in annotations:
        # clip the annotation to the window and normalize to [0, 1]
        s_c, e_c = max(s, w0), min(e, w0 + wd)
        if e_c <= s_c or int(c) == 0 or kept >= max_events:
            continue  # outside the window, background, or out of query slots
        start[kept] = (s_c - w0) / wd
        end[kept] = (e_c - w0) / wd
        cls[kept] = int(c)
        kept += 1

    # rasterize the kept events onto the num_latents-token grid: dense[t] is
    # the class of whichever event (if any) covers token t
    dense = torch.zeros(num_latents, dtype=torch.long)
    s_tok = (start * num_latents).clamp(0, num_latents).long()
    e_tok = (end * num_latents).clamp(0, num_latents).long()
    for i in range(kept):
        a, b = int(s_tok[i]), int(e_tok[i])
        if a < b:
            dense[a:b] = int(cls[i])
    return {"start": start, "end": end, "class": cls, "dense": dense}


def dance_collate(batch):
    """Stack ``[(eeg, target_dict), ...]`` into the batched dict schema
    :class:`~braindecode.training.DanceLoss` expects."""
    eeg = torch.stack([b[0] for b in batch])
    out = {"eeg": eeg}
    for key in ("start", "end", "class", "dense"):
        out[key] = torch.stack([b[1][key] for b in batch])
    return out


raw_events = {
    ds.description["subject"]: bi_annotations_to_events(ds.raw)
    for ds in windows_ds.datasets
}
metadata = windows_ds.get_metadata()

samples, sample_subjects = [], []
for i in range(len(windows_ds)):
    x, _, crop_inds = windows_ds[i]
    eeg = torch.as_tensor(np.asarray(x), dtype=torch.float32)
    subject = int(metadata.iloc[i]["subject"])
    window_onset = float(crop_inds[1]) / SFREQ  # i_start_in_trial / sfreq
    target = dance_target_builder(
        raw_events[subject],
        window_onset=window_onset,
        window_duration=WINDOW_S,
        max_events=MAX_EVENTS,
        num_latents=NUM_LATENTS,
    )
    samples.append((eeg, target))
    sample_subjects.append(subject)

######################################################################
# Splitting into train and test sets (cross-subject)
# -----------------------------------------------------
#
# We hold out the last subject entirely for evaluation, mirroring the
# paper's cross-subject splitting protocol (train and test subjects never
# overlap).

from torch.utils.data import DataLoader

sample_subjects = np.asarray(sample_subjects)
test_subject = subject_ids[-1]
train_idx = np.flatnonzero(sample_subjects != test_subject)
test_idx = np.flatnonzero(sample_subjects == test_subject)
train_samples = [samples[i] for i in train_idx]
test_samples = [samples[i] for i in test_idx]
print(f"{len(train_samples)} train windows, {len(test_samples)} test windows")

train_loader = DataLoader(
    train_samples,
    batch_size=len(train_samples),
    shuffle=True,
    collate_fn=dance_collate,
)
test_loader = DataLoader(
    test_samples,
    batch_size=len(test_samples),
    collate_fn=dance_collate,
)

######################################################################
# Create the model and the criterion
# --------------------------------------
#

from braindecode.models import DANCE
from braindecode.training import DanceLoss, f1_event
from braindecode.util import set_random_seeds

cuda = torch.cuda.is_available()
device = "cuda" if cuda else "cpu"
set_random_seeds(seed=random_state, cuda=cuda)

chs_info = windows_ds.datasets[0].raw.info["chs"]
model = DANCE(
    n_outputs=N_CLASSES,
    n_chans=len(chs_info),
    chs_info=chs_info,
    n_times=WINDOW_SAMPLES,
    sfreq=SFREQ,
    input_window_seconds=WINDOW_S,
).to(device)
criterion = DanceLoss(num_latents=NUM_LATENTS)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

######################################################################
# Train the model
# -----------------
#
# ``detect()`` returns the query set + dense map that :class:`DanceLoss`
# needs; ``forward()`` alone would only give the dense per-token logits.
#
# .. warning::
#    Kept at 2 epochs, 3 subjects so the docs build stays fast -- this is
#    **not** a converged model. The paper trains on the full dataset (71
#    subjects) for up to 100 epochs with early stopping; see the
#    :ref:`Conclusion <dance-event-detection-conclusion>`.

n_epochs = 2
model.train()
for epoch in range(n_epochs):
    epoch_loss = 0.0
    for batch in train_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()
        out = model.detect(batch["eeg"])
        loss, details = criterion(out, batch, duration=WINDOW_S)
        loss.backward()
        optimizer.step()
        epoch_loss += float(loss)
    print(f"Epoch {epoch + 1}/{n_epochs}: loss={epoch_loss / len(train_loader):.3f}")

######################################################################
# Evaluate on the held-out subject
# ------------------------------------
#
# :func:`~braindecode.training.f1_event` scores the decoded event set (IoU >
# 0.5 + class match); the per-token macro F1 scores the dense head against
# the dense target with :func:`sklearn.metrics.f1_score`.


def detections_to_events(detections, duration):
    """Decode a DANCE ``detect()`` output into per-window event tuples.

    Each query becomes ``(start_s, end_s, class, confidence)`` in seconds
    within the window; queries whose argmax class is ``0``
    (background/no-object) are dropped, following the CLASS-0 CONTRACT.
    """
    probs = torch.softmax(detections["class"], dim=-1)  # (B, Q, K+1)
    confidence, label = probs.max(dim=-1)  # (B, Q), one softmax call for the batch
    start = detections["start"] * duration
    end = detections["end"] * duration
    events = []
    for bi in range(label.shape[0]):
        keep = label[bi] != 0
        events.append(
            list(
                zip(
                    start[bi, keep].tolist(),
                    end[bi, keep].tolist(),
                    label[bi, keep].tolist(),
                    confidence[bi, keep].tolist(),
                )
            )
        )
    return events


from sklearn.metrics import f1_score

model.eval()
ev_f1s = []
dense_preds, dense_targets = [], []
with torch.no_grad():
    for batch in test_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model.detect(batch["eeg"])
        pred_events = detections_to_events(out, duration=WINDOW_S)
        for bi in range(batch["eeg"].shape[0]):
            # ground-truth events of this window, in seconds within the window
            gt = [
                (float(s) * WINDOW_S, float(e) * WINDOW_S, int(c))
                for s, e, c in zip(
                    batch["start"][bi], batch["end"][bi], batch["class"][bi]
                )
                if int(c) != 0
            ]
            preds = [(s, e, c) for (s, e, c, _conf) in pred_events[bi]]
            ev_f1s.append(f1_event(preds, gt, iou_threshold=0.5))
        # dense head (B, T, n_outputs) vs dense target (B, T), one token per prediction
        dense_preds.append(out["dense"].argmax(-1).reshape(-1).cpu())
        dense_targets.append(batch["dense"].reshape(-1).cpu())

dense_preds = torch.cat(dense_preds).numpy()
dense_targets = torch.cat(dense_targets).numpy()
sample_f1 = f1_score(
    dense_targets, dense_preds, labels=list(range(N_CLASSES)), average="macro"
)
print(
    f"Held-out subject {test_subject}: "
    f"F1-event={np.mean(ev_f1s):.3f}  F1-sample={sample_f1:.3f}"
)

######################################################################
# Predicting with the official pretrained checkpoint
# --------------------------------------------------
#
# .. note::
#    As of this writing, braindecode has **not yet published an official
#    pretrained checkpoint** for DANCE on the `Hugging Face Hub
#    <https://huggingface.co/braindecode>`_ (unlike, e.g., the sleep-staging
#    tutorials). The cell below follows the same loading pattern used
#    elsewhere in the gallery so it starts working automatically once a
#    checkpoint is released; until then it falls back to the model trained
#    above and predicts on one held-out window so the cell still runs
#    end-to-end. As with any deep model, **training for longer closes the
#    gap**: the 2-epoch, 3-subject model above is far from converged, while
#    the paper's full protocol (all subjects, up to 100 epochs with early
#    stopping) is what a released checkpoint would reflect.

import warnings

repo_id = "braindecode/plot_dance_event_detection"
try:
    from huggingface_hub import hf_hub_download

    model.load_state_dict(
        torch.load(hf_hub_download(repo_id, "model.pt"), map_location=device)
    )
    print(f"Loaded the official pretrained checkpoint from {repo_id}.")
except Exception as exc:
    warnings.warn(
        f"Could not load a pretrained checkpoint from {repo_id} ({exc}); "
        "predicting with the locally trained short-run model instead.",
        stacklevel=2,
    )

model.eval()
with torch.no_grad():
    example = test_samples[0][0].unsqueeze(0).to(device)
    out = model.detect(example)
predicted = detections_to_events(out, duration=WINDOW_S)[0]
print(f"{len(predicted)} events predicted on the first held-out window:")
for s, e, label, conf in sorted(predicted, key=lambda ev: ev[0])[:10]:
    print(f"  [{s:5.2f}s, {e:5.2f}s]  class={label}  confidence={conf:.2f}")

######################################################################
# .. _dance-event-detection-conclusion:
#
# Conclusion
# ----------
#
# This tutorial trained :class:`~braindecode.models.DANCE` on real,
# continuous P300 EEG, detecting target and non-target flashes directly from
# unaligned windows -- without ever being told where a flash starts. At this
# reduced scale (3 subjects, 2 epochs) the reported F1 scores are low and
# should **not** be compared to the paper's numbers.
#
# At full scale (BI2014a, all subjects, up to 100 epochs with early stopping
# and a OneCycle learning rate schedule, following the paper's protocol),
# DANCE matches the accuracy of onset-informed models on this dataset --
# *without* ever using the flash onset -- and establishes a new state of the
# art on the harder task of seizure monitoring (Temple University Seizure
# corpus), where onsets are unknown by construction. This is the core claim
# of the asynchronous decoding paradigm: a single architecture that performs
# on par with onset-informed baselines while requiring no onset information
# at all.
#
# References
# ----------
#
# .. [1] Lévy, J., Banville, H., Rapin, J., King, J.-R., Moreau, T., &
#        d'Ascoli, S. (2026). DANCE: Detect and Classify Events in EEG.
#        arXiv:2605.10688.
# .. [2] Korczowski, L., Cederhout, M., Andreev, A., Cattan, G., Rodrigues,
#        P. L. C., Gautheret, V., & Congedo, M. (2019). Brain Invaders
#        calibration-less P300-based BCI with modulation of flash duration
#        dataset (bi2014a). GIPSA-lab.
# .. [3] Carion, N., Massa, F., Synnaeve, G., Usunier, N., Kirillov, A., &
#        Zagoruyko, S. (2020). End-to-end object detection with
#        transformers. ECCV. arXiv:2005.12872.
# .. [4] Jaegle, A., Gimeno, F., Brock, A., Vinyals, O., Zisserman, A., &
#        Carreira, J. (2021). Perceiver: General perception with iterative
#        attention. ICML.
