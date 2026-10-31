""".. _pretraining-overlap:

Which data did a pretrained checkpoint see? Reading the model card
===================================================================

Foundation models such as :class:`~braindecode.models.Labram`,
:class:`~braindecode.models.CBraMod` or :class:`~braindecode.models.BIOT` are
pretrained without labels on large EEG collections, then fine-tuned on a
downstream task. A downstream result is usually reported as
*subject-independent* because the test subjects are excluded from fine-tuning.
That says nothing about pretraining: if the evaluation recordings were part of
the pretraining corpus, the encoder has already seen the test subjects' signals,
and the result is not held out from the model as a whole.

The pretraining data belongs to a *checkpoint*, not to a model class: the three
BIOT checkpoints on the Hub were trained on different data. braindecode therefore
keeps this information where Hugging Face keeps it, in the ``datasets:`` field of
the checkpoint's model card. :meth:`~braindecode.models.EEGModuleMixin.from_pretrained`
reads that card and exposes it as ``model.card_data``, and
:meth:`~braindecode.models.EEGModuleMixin.push_to_hub` writes it when you share a
model of your own.

.. contents:: This example covers:
   :local:
   :depth: 2

"""

# Authors: Yiheng Li <yihengli23@gmail.com>
#          Bruno Aristimunha <b.aristimunha@gmail.com>
#
# License: BSD (3-clause)

import tempfile

from braindecode.models import EEGNet

######################################################################
# Recording the training data when you save or share a model
# ----------------------------------------------------------
#
# Pass ``datasets`` (and ``base_model`` for a fine-tuned checkpoint) through
# ``model_card_kwargs``. They are written to the YAML header of the model card,
# which the Hub indexes and which travels with the weights. Use the Hub dataset id
# where one exists (for example an OpenNeuro set mirrored by EEGDash), otherwise the
# MOABB or braindecode dataset class name.

model = EEGNet(n_chans=22, n_times=1000, n_outputs=4)
tmp_dir = tempfile.mkdtemp()
model.save_pretrained(
    tmp_dir,
    model_card_kwargs={"datasets": ["BNCI2014_001", "Schirrmeister2017"]},
)

######################################################################
# Reading it back
# ---------------
#
# ``from_pretrained`` reads the card next to the weights (locally or from the Hub,
# with the same ``revision`` and cache options) and sets ``card_data``. The card is
# informational: if there is none, the weights still load and ``card_data`` is
# ``None``.

restored = EEGNet.from_pretrained(tmp_dir)
print(restored.card_data.datasets)

######################################################################
# Checking a planned evaluation against it
# ----------------------------------------
#
# The overlap check is then a set intersection between the card and the dataset
# you plan to evaluate on. Here ``BNCI2014_001`` was in the training data and
# ``Cho2017`` was not.

seen = set(restored.card_data.datasets or [])
for evaluation_dataset in ["BNCI2014_001", "Cho2017"]:
    status = "in training data" if evaluation_dataset in seen else "held out"
    print(f"{evaluation_dataset}: {status}")

######################################################################
# Two things to keep in mind when reading a card:
#
# - **Collections.** The TUH EEG Corpus (``TUEG``) contains TUAB, TUEV, TUSL, TUAR,
#   TUSZ and TUEP. A checkpoint whose card lists ``TUEG`` has seen the same hospital
#   archive as those benchmarks even when the authors removed the exact evaluation
#   recordings, so treat them as *same collection* rather than held out.
# - **Completeness.** Some papers do not itemise every pretraining dataset (private
#   or self-collected recordings, long OpenNeuro lists given by accession only). The
#   card text says so; for such checkpoints "not listed" means "not among the
#   datasets the authors name", not "held out".
#
# For the checkpoints hosted under ``braindecode/`` on the Hub, the ``datasets:``
# fields are being filled from a page-cited audit of each model's paper; until a
# card carries the field, ``card_data.datasets`` is ``None`` and the Hub page of the
# checkpoint documents what is known.
#
# What to do with the answer
# --------------------------
#
# Overlap does not make a result wrong. Pretraining is self-supervised, so no labels
# leak, and authors sometimes remove the evaluation recordings before pretraining.
# But a reader needs to know which case they are looking at:
#
# - Prefer an evaluation dataset that the checkpoint has not seen, or report results
#   on both an in-corpus and a held-out dataset, so the difference is visible.
# - When comparing two checkpoints, check both: a dataset can be in one card and
#   absent from the other, which makes a direct comparison uneven.
# - Keep the test subjects out of fine-tuning as usual (for example by splitting the
#   windows dataset on its ``"subject"`` description column); the card is about
#   pretraining and does not replace a subject-wise split.
# - State it in the paper or report: the checkpoint, the dataset, and whether the
#   card's list is complete.
#
# References
# ----------
#
# .. [1] Jiang, W., Zhao, L., & Lu, B. (2024). Large brain model for learning generic
#    representations with tremendous EEG data in BCI. ICLR 2024. arXiv:2405.18765.
# .. [2] Wang, J. et al. (2025). CBraMod: A criss-cross brain foundation model for EEG
#    decoding. ICLR 2025. arXiv:2412.07236.
# .. [3] Yang, C., Westover, M. B., & Sun, J. (2023). BIOT: Biosignal transformer for
#    cross-data learning in the wild. NeurIPS 2023. arXiv:2305.10351.
