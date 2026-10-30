""".. _pretraining-overlap:

Checking whether an evaluation dataset was in a foundation model's pretraining data
====================================================================================

Foundation models such as :class:`~braindecode.models.Labram` or
:class:`~braindecode.models.CBraMod` are pretrained without labels on large EEG
collections, then fine-tuned on a downstream task. A downstream result is usually
reported as *subject-independent* because the test subjects are excluded from
fine-tuning. That says nothing about pretraining: if the evaluation recordings were
part of the pretraining corpus, the encoder has already seen the test subjects'
signals, and the result is not held out from the model as a whole.

This tutorial shows how to check this before running an evaluation, with
:func:`~braindecode.models.pretraining.check_pretraining_overlap`, and how to read
its answer.

.. contents:: This example covers:
   :local:
   :depth: 2

"""

# Authors: Yiheng Li <yihengli23@gmail.com>
#
# License: BSD (3-clause)

import warnings

import pandas as pd

from braindecode.models.pretraining import (
    PRETRAINING_CORPORA,
    check_pretraining_overlap,
)

######################################################################
# Checking one model and one dataset
# ----------------------------------
#
# Suppose you want to fine-tune LaBraM on the PhysioNet motor imagery dataset
# (``PhysionetMI`` in MOABB). The check returns ``"in_corpus"`` and warns, because
# PhysioNet (EEGMMIDB) is one of the datasets listed in LaBraM's pretraining corpus.

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    status = check_pretraining_overlap("Labram", "PhysionetMI")
print(status)
print(caught[0].message)

######################################################################
# The same model on Cho2017, which is not in LaBraM's listed corpus, returns
# ``"held_out"``. You can also pass a model class, a model instance, or a MOABB
# dataset object instead of names.

print(check_pretraining_overlap("Labram", "Cho2017"))

######################################################################
# Reading the answer
# ------------------
#
# The function returns one of four values:
#
# - ``"in_corpus"``: the dataset is listed in the model's pretraining data.
# - ``"same_collection"``: the dataset is a sub-corpus of the TUH EEG Corpus (for
#   example TUAB or TUSZ) and the model was pretrained on TUH data. Even when the
#   authors removed the exact evaluation recordings, the encoder has seen the same
#   hospital archive.
# - ``"held_out"``: no overlap with the recorded corpus.
# - ``"unknown"``: no corpus is recorded for this model.
#
# ``"held_out"`` is only as reliable as the corpus list behind it. Some papers do not
# itemise every pretraining dataset; those entries are marked ``complete=False``, and
# for them a ``"held_out"`` answer means "not among the datasets the paper names".
# The ``source`` field says where each list comes from.

print(PRETRAINING_CORPORA["Labram"].complete)
print(PRETRAINING_CORPORA["Labram"].source)

######################################################################
# A table for several models and benchmarks
# -----------------------------------------
#
# Before choosing an evaluation dataset, it helps to see the whole picture. Here we
# check a few common benchmarks against every model with a recorded corpus.

datasets = ["PhysionetMI", "BNCI2014_001", "Cho2017", "Lee2019_MI", "TUAB", "SEED"]
table = pd.DataFrame(
    {
        model: [check_pretraining_overlap(model, d, warn=False) for d in datasets]
        for model in PRETRAINING_CORPORA
    },
    index=datasets,
).T
table["corpus list complete"] = [PRETRAINING_CORPORA[m].complete for m in table.index]
print(table.to_string())

######################################################################
# What to do with the answer
# --------------------------
#
# Overlap does not make a result wrong. Pretraining is self-supervised, so no labels
# leak, and authors sometimes remove the evaluation recordings before pretraining.
# But a reader needs to know which case they are looking at. Some practical options:
#
# - Prefer an evaluation dataset that is ``"held_out"`` for the model, or report
#   results on both an in-corpus and a held-out dataset, so the difference is
#   visible.
# - When comparing two foundation models, check both: a dataset can be in one
#   model's corpus and held out for the other, which makes a direct comparison
#   uneven.
# - Keep the test subjects out of fine-tuning as usual (for example by splitting
#   the windows dataset on its ``"subject"`` description column); the overlap check
#   is about pretraining and does not replace a subject-wise split.
# - State the check in the paper or report: the model, the dataset, the status, and
#   whether the corpus list is complete.
#
# References
# ----------
#
# .. [1] Jiang, W., Zhao, L., & Lu, B. (2024). Large brain model for learning generic
#    representations with tremendous EEG data in BCI. ICLR 2024. arXiv:2405.18765.
# .. [2] Wang, J. et al. (2025). CBraMod: A criss-cross brain foundation model for EEG
#    decoding. ICLR 2025. arXiv:2412.07236.
