""".. _transfer-learning-pathology:

Transfer learning across EEG pathology datasets
================================================

Transfer learning reuses representations learned from a *source* domain to
initialize a model for a related *target* domain. It can be useful when the
target dataset is small, but differences in amplifiers, montages, sampling
rates, populations, and recording protocols can cause domain shift.

This tutorial follows the TUAB-to-NMT pathology-classification direction from
[1]_. A :class:`~braindecode.models.Deep4Net` is first trained on source-domain
recordings. Its parameters initialize a second model, which is then fine-tuned
on target-domain recordings. We also train the target model from scratch as a
control.

The full TUAB and NMT corpora are large, and TUAB access requires registration.
For a fast, network-free example, the executable part below uses synthetic MNE
recordings with the same channel and class contracts. The generated scores only
illustrate the workflow; they are not estimates of clinical performance or a
reproduction of [1]_.

.. contents:: This example covers:
   :local:
   :depth: 2
"""

# Authors: Mohammad Javad Darvishi Bayazi <mj.darvishi92@gmail.com>
#          Bruno Aristimunha <b.aristimunha@gmail.com>
#
# License: BSD (3-clause)

import mne
import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import scale as standard_scale
from skorch.callbacks import Callback
from skorch.helper import predefined_split

from braindecode import EEGClassifier
from braindecode.datasets import BaseConcatDataset, RawDataset
from braindecode.models import Deep4Net
from braindecode.preprocessing import (
    Preprocessor,
    create_fixed_length_windows,
    preprocess,
)
from braindecode.util import set_random_seeds

######################################################################
# Design the experiment before loading data
# -----------------------------------------
#
# The source and target tasks must have the same input and output contracts.
# Here, both use 21 ordered EEG channels sampled at 100 Hz, fixed-length
# windows, and the labels ``0 = normal`` and ``1 = pathological``. Matching the
# number of channels is not enough: their names and order must also match.
#
# The data flow is:
#
# .. code-block:: text
#
#    source train --fit--> source model --state_dict--> target model
#          |                                      target train --fit--^
#    source validation                         target validation
#                                                        |
#                                      held-out target test (final evaluation)
#
# Validation data may guide hyperparameters and stopping. Test data must not be
# passed to :func:`~skorch.helper.predefined_split`, a scheduler, or any other
# model-selection step. When participants have multiple recordings, split by
# participant *before* creating windows so that a participant cannot occur in
# more than one partition.

seed = 20240812
cuda = torch.cuda.is_available()
device = "cuda" if cuda else "cpu"
set_random_seeds(seed=seed, cuda=cuda)
mne.set_log_level("ERROR")

sfreq = 100
window_size_samples = 600  # Six seconds; kept short so the example runs quickly.
channel_names = [
    "Fp1",
    "Fp2",
    "F7",
    "F3",
    "Fz",
    "F4",
    "F8",
    "T3",
    "C3",
    "Cz",
    "C4",
    "T4",
    "T5",
    "P3",
    "Pz",
    "P4",
    "T6",
    "O1",
    "O2",
    "A1",
    "A2",
]


######################################################################
# Adapting the workflow to TUAB and NMT
# -------------------------------------
#
# Braindecode provides :class:`~braindecode.datasets.TUHAbnormal` and
# :class:`~braindecode.datasets.NMT`. TUAB must be obtained from the `official
# TUH portal <https://isip.piconepress.com/projects/nedc/html/tuh_eeg/>`_ under
# its access terms. NMT can be obtained from its `Zenodo record
# <https://zenodo.org/records/10909103>`_. Once the data are present locally,
# the public loaders are used as follows:
#
# .. code-block:: python
#
#    from braindecode.datasets import NMT, TUHAbnormal
#
#    source_recordings = TUHAbnormal(
#        path="/path/to/tuh_eeg_abnormal/v3.0.1/edf",
#        target_name="pathological",
#    )
#    target_recordings = NMT(
#        path="/path/to/nmt_scalp_eeg_dataset",
#        target_name="pathological",
#    )
#
# Before training, apply a single documented preprocessing contract to both
# datasets. The study in [1]_ used TUAB 3.0.0; the loader call above shows the
# currently supported TUAB 3.0.1 layout, so it is not an exact data-version
# reproduction. Its protocol selected 21 common electrodes, removed the first
# minute, retained at most 20 minutes, resampled to 100 Hz, clipped extreme
# amplitudes, and normalized the signals. In a reproduction, also preserve the
# datasets' official train/test assignments, derive validation participants only
# from the training assignment, and fit any population-level normalization on
# training data only. See the `companion implementation
# <https://github.com/javadbayazi/APD_EEG>`_ for the study-specific workflow.
#
# The next helper produces small synthetic source and target domains. Each
# ``RawDataset`` represents a distinct participant recording, so the split
# labels below also act as participant-level splits. The target domain has a
# small frequency shift and slow drift to mimic a domain change.

info = mne.create_info(channel_names, sfreq=sfreq, ch_types="eeg")


def make_synthetic_domain(domain, split_sizes, random_state):
    """Create balanced, labeled recordings for one synthetic domain."""
    rng = np.random.default_rng(random_state)
    recordings = []
    times = np.arange(window_size_samples) / sfreq

    for split, recordings_per_class in split_sizes.items():
        for pathological in (0, 1):
            for recording_index in range(recordings_per_class):
                data = rng.normal(
                    scale=0.25e-6,
                    size=(len(channel_names), window_size_samples),
                )
                frequency = 10.0 if pathological == 0 else 6.0
                if domain == "target":
                    frequency += 0.35
                phase = rng.uniform(0, 2 * np.pi)
                oscillation = np.sin(2 * np.pi * frequency * times + phase)
                data[:10] += 1.2e-6 * oscillation
                if domain == "target":
                    data += 0.05e-6 * np.sin(2 * np.pi * times)

                raw = mne.io.RawArray(data, info.copy(), verbose=False)
                description = {
                    "pathological": pathological,
                    "split": split,
                    "domain": domain,
                    "recording_id": (
                        f"{domain}-{split}-{pathological}-{recording_index}"
                    ),
                }
                recordings.append(
                    RawDataset(
                        raw,
                        description=description,
                        target_name="pathological",
                    )
                )

    return BaseConcatDataset(recordings)


source_recordings = make_synthetic_domain(
    domain="source",
    split_sizes={"train": 10, "valid": 3},
    random_state=seed,
)
target_recordings = make_synthetic_domain(
    domain="target",
    split_sizes={"train": 5, "valid": 3, "test": 4},
    random_state=seed + 1,
)

# Apply the same recording-wise transform to both domains. It has no fitted
# population state, so processing a test recording cannot leak information from
# the training population.
recording_wise_standardization = [Preprocessor(standard_scale, channel_wise=True)]
preprocess(source_recordings, recording_wise_standardization, n_jobs=1)
preprocess(target_recordings, recording_wise_standardization, n_jobs=1)

print("Source recordings by split and class:")
print(source_recordings.description.groupby(["split", "pathological"]).size())
print("\nTarget recordings by split and class:")
print(target_recordings.description.groupby(["split", "pathological"]).size())


######################################################################
# Split recordings, then create windows
# -------------------------------------
#
# We split first and window second. This ordering prevents windows from the
# same recording from leaking across train, validation, and test partitions.
# Real TUAB data can contain several sessions for one participant, so those
# sessions must additionally be grouped by participant.


def window_each_split(recordings):
    """Create one fixed-length window per synthetic recording."""
    return {
        split: create_fixed_length_windows(
            split_recordings,
            window_size_samples=window_size_samples,
            window_stride_samples=window_size_samples,
            drop_last_window=True,
            preload=True,
            verbose="error",
        )
        for split, split_recordings in recordings.split("split").items()
    }


source_windows = window_each_split(source_recordings)
target_windows = window_each_split(target_recordings)

target_ids = {
    split: set(windows.description["recording_id"])
    for split, windows in target_windows.items()
}
assert target_ids["train"].isdisjoint(target_ids["valid"])
assert target_ids["train"].isdisjoint(target_ids["test"])
assert target_ids["valid"].isdisjoint(target_ids["test"])

print("\nOne target-domain window and its metadata:")
sample, target, window_index = target_windows["train"][0]
print(f"{sample.shape=}, {target=}, {window_index=}")


######################################################################
# Configure Deep4Net and its training wrapper
# --------------------------------------------
#
# The original study used 60-second cropped training and the full Deep4Net
# configuration. Here we use six-second windows, fewer filters, and only a few
# epochs solely to keep the documentation example fast. These are demonstration
# choices, not recommended pathology-detection hyperparameters.

source_epochs = 5
target_epochs = 1


def make_classifier(validation_windows, max_epochs, callbacks=None):
    """Build a classifier with a fixed validation split."""
    return EEGClassifier(
        Deep4Net,
        module__n_chans=len(channel_names),
        module__n_outputs=2,
        module__n_times=window_size_samples,
        module__n_filters_time=4,
        module__n_filters_spat=4,
        module__n_filters_2=8,
        module__n_filters_3=8,
        module__n_filters_4=16,
        module__drop_prob=0.25,
        criterion=torch.nn.CrossEntropyLoss,
        optimizer=torch.optim.AdamW,
        optimizer__lr=0.005,
        batch_size=8,
        max_epochs=max_epochs,
        train_split=predefined_split(validation_windows),
        iterator_train__shuffle=True,
        iterator_train__drop_last=False,
        classes=[0, 1],
        callbacks=callbacks,
        device=device,
        verbose=0,
    )


######################################################################
# Pre-train on the source domain
# ------------------------------
#
# Only source training windows update the model. The predefined split evaluates
# source validation windows after each epoch.

source_clf = make_classifier(source_windows["valid"], max_epochs=source_epochs)
_ = source_clf.fit(source_windows["train"], y=None)
print(source_clf)

# A state dictionary is portable and avoids serializing the entire classifier.
# Clone tensors onto the CPU so this in-memory checkpoint is independent of the
# optimizer and works even if source training ran on a GPU.
source_state = {
    name: value.detach().cpu().clone()
    for name, value in source_clf.module_.state_dict().items()
}


######################################################################
# Train the target-domain control from scratch
# --------------------------------------------
#
# A transfer result is only interpretable relative to a target-domain control
# with the same architecture, training data, validation data, and optimization
# budget. We reset the random seed before both target runs so their minibatch and
# dropout streams are reproducible.

target_seed = seed + 2
set_random_seeds(seed=target_seed, cuda=cuda)
scratch_clf = make_classifier(
    target_windows["valid"],
    max_epochs=target_epochs,
)
_ = scratch_clf.fit(target_windows["train"], y=None)


######################################################################
# Load source weights and fine-tune on the target domain
# ------------------------------------------------------
#
# The source and target models have identical input geometry and class heads, so
# every tensor must match. ``strict=True`` turns a missing or unexpected key into
# an error instead of silently training a partly initialized network.
#
# :class:`~braindecode.EEGClassifier` infers signal properties when ``fit``
# starts, which may initialize its module. We therefore load the source state in
# ``on_train_begin``: after initialization but before the first optimization
# step. Loading it earlier into ``module_`` could be undone by initialization.


class LoadModelState(Callback):
    """Load a state dictionary after module initialization."""

    def __init__(self, state_dict):
        self.state_dict = state_dict

    def on_train_begin(self, net, **kwargs):
        net.module_.load_state_dict(self.state_dict, strict=True)


set_random_seeds(seed=target_seed, cuda=cuda)
fine_tune_clf = make_classifier(
    target_windows["valid"],
    max_epochs=target_epochs,
    callbacks=[("load_source_state", LoadModelState(source_state))],
)
_ = fine_tune_clf.fit(target_windows["train"], y=None)

# We fine-tune every layer, as in the main comparison in [1]_. With much less
# target data, another experiment could freeze the backbone and train only the
# current Deep4Net classifier head:
#
# .. code-block:: python
#
#    for name, parameter in fine_tune_clf.module_.named_parameters():
#        parameter.requires_grad = name.startswith("final_layer.")


######################################################################
# Evaluate once on the held-out target test set
# ------------------------------------------------
#
# Balanced accuracy gives equal weight to normal and pathological recall. Both
# classifiers are fully trained before the test labels are read. On real data,
# repeat training across documented random seeds and report uncertainty rather
# than selecting the seed with the best test score.

test_metadata = target_windows["test"].get_metadata()
y_test = test_metadata["target"].to_numpy(dtype=np.int64)
scratch_score = balanced_accuracy_score(
    y_test,
    scratch_clf.predict(target_windows["test"]),
)
fine_tune_score = balanced_accuracy_score(
    y_test,
    fine_tune_clf.predict(target_windows["test"]),
)

print(f"Target test balanced accuracy from scratch: {scratch_score:.3f}")
print(f"Target test balanced accuracy after transfer: {fine_tune_score:.3f}")


######################################################################
# Conclusion
# ----------
#
# Transfer learning is a controlled initialization experiment, not simply
# loading a checkpoint. A defensible comparison requires compatible input and
# label contracts, participant-disjoint partitions, validation data that come
# only from training participants, a target-from-scratch control, strict weight
# loading, and one final evaluation on untouched target test data. The synthetic
# example establishes those mechanics; reproducing [1]_ additionally requires
# the authors' full preprocessing, cropped-training, and evaluation protocol.
#
# References
# ----------
#
# .. [1] Darvishi-Bayazi, M. J., Ghaemi, M. S., Lesort, T., Arefin, M. R.,
#    Faubert, J., & Rish, I. (2024). Amplifying pathological detection in EEG
#    signaling pathways through cross-dataset transfer learning. *Computers in
#    Biology and Medicine*, 169, 107893.
#    https://doi.org/10.1016/j.compbiomed.2023.107893
# .. [2] Obeid, I., & Picone, J. (2016). The Temple University Hospital EEG
#    Data Corpus. *Frontiers in Neuroscience*, 10, 196.
#    https://doi.org/10.3389/fnins.2016.00196
# .. [3] Khan, H. A., Ul Ain, R., Kamboh, A. M., Butt, H. T., Shafait, S.,
#    Alamgir, W., Stricker, D., & Shafait, F. (2022). The NMT scalp EEG
#    dataset: an open-source annotated dataset of healthy and pathological EEG
#    recordings for predictive modeling. *Frontiers in Neuroscience*, 15,
#    755817. https://doi.org/10.3389/fnins.2021.755817
# .. [4] Schirrmeister, R. T., Springenberg, J. T., Fiederer, L. D. J., et al.
#    (2017). Deep learning with convolutional neural networks for EEG decoding
#    and visualization. *Human Brain Mapping*, 38(11), 5391--5420.
#    https://doi.org/10.1002/hbm.23730
