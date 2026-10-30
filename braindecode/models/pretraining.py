"""Pretraining corpora of the foundation models in braindecode, and an overlap check.

A downstream result is only "subject-independent" for a foundation model if the
encoder has not already seen the evaluation recordings during self-supervised
pretraining. This module records, for each pretrained model, the datasets listed
in its paper, so that :func:`check_pretraining_overlap` can warn when an
evaluation dataset is part of (``"in_corpus"``) or a sub-corpus of the same
collection as (``"same_collection"``) the model's pretraining data.

Lists are copied from each paper; see ``source``. ``complete=False`` means the
paper does not itemise every pretraining dataset, so a missing match is not proof
that a dataset is held out.
"""

# Authors: Yiheng Li <yihengli23@gmail.com>
#
# License: BSD (3-clause)

from __future__ import annotations

import warnings
from dataclasses import dataclass

__all__ = ["PRETRAINING_CORPORA", "PretrainingCorpus", "check_pretraining_overlap"]

# Sub-corpora of the TUH EEG Corpus (TUEG). A model pretrained on TUEG has seen
# the same hospital archive as these benchmarks, even when the authors remove
# the exact evaluation recordings.
_TUH_FAMILY = frozenset({"TUEG", "TUAB", "TUEV", "TUSL", "TUAR", "TUSZ", "TUEP"})

# MOABB / braindecode dataset names -> the names used in the corpora below.
_ALIASES = {
    "physionetmi": "EEGMMIDB",
    "eegbci": "EEGMMIDB",
    "eegmmidb": "EEGMMIDB",
    "bnci2014_001": "BCIC-IV-2a",
    "bnci2014001": "BCIC-IV-2a",
    "bnci2014_004": "BCIC-IV-2b",
    "bnci2014004": "BCIC-IV-2b",
    "schirrmeister2017": "HGD",
    "cho2017": "Cho2017",
    "lee2019_mi": "Lee2019",
    "lee2019": "Lee2019",
    "weibo2014": "Weibo2014",
    "siena": "Siena",
    "sienascalp": "Siena",
    "seed": "SEED",
    "seediv": "SEED-IV",
    "seed_iv": "SEED-IV",
    "tuh": "TUEG",
    "tueg": "TUEG",
    "tuhabnormal": "TUAB",
    "tuab": "TUAB",
    "tuev": "TUEV",
    "tusl": "TUSL",
    "tuar": "TUAR",
    "tusz": "TUSZ",
    "tdbrain": "TDBrain",
}


@dataclass(frozen=True)
class PretrainingCorpus:
    datasets: frozenset
    source: str
    complete: bool = True


PRETRAINING_CORPORA = {
    "BENDR": PretrainingCorpus(
        frozenset({"TUEG"}), "Kostas et al. 2021, arXiv:2101.12037"
    ),
    "BIOT": PretrainingCorpus(
        frozenset({"SHHS", "PREST"}), "Yang et al. 2023, arXiv:2305.10351"
    ),
    "Labram": PretrainingCorpus(
        frozenset(
            {
                "BCIC-IV-1",
                "Emobrain",
                "GAL",
                "KaggleERN",
                "EEGMMIDB",
                "Trujillo2020",
                "Trujillo2017",
                "SEED",
                "SEED-IV",
                "SEED-GER",
                "SEED-FRA",
                "Siena",
                "SPIS",
                "BrainInvaders",
                "TUAR",
                "TUEP",
                "TUSZ",
                "TUSL",
            }
        ),
        "Jiang et al. 2024, arXiv:2405.18765, Appendix D (plus unlisted self-collected EEG)",
        complete=False,
    ),
    "EEGPT": PretrainingCorpus(
        frozenset({"EEGMMIDB", "HGD", "TSU-SSVEP", "SEED", "M3CV"}),
        "Wang et al. 2024, NeurIPS",
    ),
    "CBraMod": PretrainingCorpus(
        frozenset({"TUEG"}), "Wang et al. 2025, arXiv:2412.07236"
    ),
    "LUNA": PretrainingCorpus(frozenset({"TUEG", "Siena"}), "LUNA, arXiv:2510.22257"),
    "REVE": PretrainingCorpus(
        frozenset(
            {
                "TUEG",
                "Siena",
                "BCIC-IV-2b",
                "HGD",
                "Cho2017",
                "Lee2019",
                "Weibo2014",
                "TDBrain",
            }
        ),
        "El Ouahidi et al. 2025, arXiv:2510.21585 (92 datasets; only those named are listed here)",
        complete=False,
    ),
    "CSBrain": PretrainingCorpus(frozenset({"TUEG"}), "CSBrain, arXiv:2506.23075"),
    "CodeBrain": PretrainingCorpus(frozenset({"TUEG"}), "CodeBrain, arXiv:2506.09110"),
    "BrainOmni": PretrainingCorpus(
        frozenset({"HBN-EEG", "PEARL-Neuro", "Features-EEG", "Go-Nogo", "MusicEEG"}),
        "BrainOmni, arXiv:2505.18185 (EEG part; MEG sets omitted)",
        complete=False,
    ),
    "EEGDINO": PretrainingCorpus(frozenset({"TUEG"}), "EEG-DINO, MICCAI 2025"),
}


def _canonical(name: str) -> str:
    key = name.lower().replace("-", "").replace(" ", "")
    return _ALIASES.get(key, _ALIASES.get(name.lower(), name))


def check_pretraining_overlap(model, dataset, warn: bool = True) -> str:
    """Tell whether ``dataset`` overlaps the pretraining data of ``model``.

    Parameters
    ----------
    model : str | type | torch.nn.Module
        A braindecode model class, instance, or class name (e.g. ``"Labram"``).
    dataset : str | object
        A dataset name (e.g. ``"PhysionetMI"``, ``"Cho2017"``, ``"TUAB"``) or a
        MOABB/braindecode dataset object, whose class name is used.
    warn : bool
        Emit a ``UserWarning`` when the result is not ``"held_out"``.

    Returns
    -------
    str
        ``"in_corpus"``, ``"same_collection"`` (a TUH sub-corpus while the model
        was pretrained on TUH data), ``"held_out"``, or ``"unknown"`` when the
        model has no recorded corpus. ``"held_out"`` from an incomplete corpus
        list is only as good as that list.
    """
    model_name = (
        model
        if isinstance(model, str)
        else getattr(model, "__name__", type(model).__name__)
    )
    data_name = dataset if isinstance(dataset, str) else type(dataset).__name__
    corpus = PRETRAINING_CORPORA.get(model_name)
    if corpus is None:
        return "unknown"
    canon = _canonical(data_name)
    if canon in corpus.datasets:
        status = "in_corpus"
    elif canon in _TUH_FAMILY and corpus.datasets & _TUH_FAMILY:
        status = "same_collection"
    else:
        status = "held_out"
    if warn and status != "held_out":
        warnings.warn(
            f"{data_name} is {status.replace('_', ' ')} for {model_name}'s pretraining data "
            f"({corpus.source}); results on it are not held out from pretraining.",
            UserWarning,
            stacklevel=2,
        )
    return status
