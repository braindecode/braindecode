# Authors: Yiheng Li <yihengli23@gmail.com>
#
# License: BSD (3-clause)

import pytest

from braindecode.models import REVE, Labram
from braindecode.models.pretraining import (
    PRETRAINING_CORPORA,
    check_pretraining_overlap,
)


class PhysionetMI:  # stand-in with the MOABB class name
    pass


def test_in_corpus_warns():
    with pytest.warns(UserWarning, match="in corpus"):
        assert check_pretraining_overlap(Labram, PhysionetMI()) == "in_corpus"


def test_same_tuh_collection():
    with pytest.warns(UserWarning, match="same collection"):
        assert check_pretraining_overlap("CBraMod", "TUAB") == "same_collection"


def test_held_out_and_unknown(recwarn):
    assert check_pretraining_overlap("Labram", "Cho2017") == "held_out"
    assert check_pretraining_overlap("EEGNet", "Cho2017") == "unknown"
    assert len(recwarn) == 0


def test_aliases():
    assert check_pretraining_overlap(REVE, "BNCI2014_004", warn=False) == "in_corpus"
    assert check_pretraining_overlap("REVE", "cho2017", warn=False) == "in_corpus"


def test_keys_are_braindecode_models():
    import braindecode.models as m

    assert all(hasattr(m, name) for name in PRETRAINING_CORPORA)
