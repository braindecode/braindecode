# Authors: Bruno Aristimunha <b.aristimunha@gmail.com>
#
# License: BSD (3-clause)
import numpy as np
import pytest

from braindecode.modules.channels import (
    match_names,
    nearest_vocabulary,
    resolve_montage,
)

# standard_1005 positions (metres), copied by hand from MNE so the tests do
# not derive their expectations from the code under test.
CZ = (0.0004009, -0.009167, 0.100244)
FZ = (0.0003122, 0.058512, 0.066462)


def _ch(name, loc=None, kind="eeg"):
    ch = {"ch_name": name, "kind": kind}
    if loc is not None:
        ch["loc"] = np.r_[np.asarray(loc, float), np.zeros(9)]
    return ch


def test_alias_keeps_exact_names():
    # Exact match first: T3 and T7 in one vocabulary stay distinct.
    assert match_names(["T3", "T7"], ["T3", "T7", "Cz"]).tolist() == [0, 1]
    # Alias only when no exact name exists, in both directions.
    assert match_names(["T3"], ["T7"]).tolist() == [0]
    assert match_names(["t7"], ["T3"]).tolist() == [0]
    assert match_names(["cz", "Xx"], ["Fz", "CZ"]).tolist() == [1, -1]


def test_fill_positions_from_standard_1005():
    m = resolve_montage([_ch("Cz"), _ch("Fz", loc=(0.01, 0.02, 0.03))])
    np.testing.assert_allclose(m.positions[0], CZ, atol=1e-6)
    np.testing.assert_allclose(m.positions[1], (0.01, 0.02, 0.03))
    assert m.has_position.tolist() == [False, True]
    assert m.canon == ("cz", "fz")


def test_all_zero_loc_counts_as_missing_and_unknown_name_stays_nan():
    m = resolve_montage([_ch("Fz", loc=(0, 0, 0)), _ch("E1")])
    np.testing.assert_allclose(m.positions[0], FZ, atol=1e-6)
    assert np.isnan(m.positions[1]).all()
    assert not m.has_position.any()
    m2 = resolve_montage([_ch("Cz")], fill_positions=False)
    assert np.isnan(m2.positions).all()


@pytest.mark.parametrize("loc", [None, (0.01, 0.0, 0.05)])
@pytest.mark.parametrize("kind", ["eog", 202])  # FIFF EOG code
def test_non_eeg_rejected_on_every_path(loc, kind):
    chs = [_ch("Cz"), _ch("EOG1", loc=loc, kind=kind)]
    with pytest.raises(ValueError, match="EOG1"):
        resolve_montage(chs)


def test_non_eeg_dropped_when_asked():
    chs = [_ch("STI"), _ch("Cz"), _ch("ECG", kind="ecg"), _ch("Pz", kind=2)]
    chs[0]["kind"] = 3  # FIFF stim
    m = resolve_montage(chs, drop_non_eeg=True)
    assert m.names == ("Cz", "Pz")
    assert m.picks.tolist() == [1, 3]
    assert m.n_input == 4


def test_duplicate_names_rejected():
    with pytest.raises(ValueError, match="(?i)duplicate.*cz"):
        resolve_montage([_ch("Cz"), _ch("Fz"), _ch("CZ")])


def test_key_is_stable():
    a = resolve_montage([_ch("Cz", loc=CZ), _ch("Fz", loc=FZ)])
    # Sub-0.1 mm jitter, a fresh list and loc given vs filled from the same
    # standard position: same key.
    b = resolve_montage(
        [_ch("Cz", loc=np.add(CZ, 1e-6)), _ch("Fz")],
    )
    assert a.key == b.key
    moved = resolve_montage([_ch("Cz", loc=np.add(CZ, 1e-3)), _ch("Fz")])
    assert moved.key != a.key
    renamed = resolve_montage([_ch("Fz", loc=CZ), _ch("Cz", loc=FZ)])
    assert renamed.key != a.key


def test_nearest_vocabulary_within_15_mm():
    vocab = np.array([CZ, FZ])
    pos = np.array([np.add(CZ, (0.01, 0, 0)), np.add(FZ, (0.02, 0, 0))])
    pos = np.vstack([pos, np.full(3, np.nan)])
    assert nearest_vocabulary(pos, vocab).tolist() == [0, -1, -1]
    assert nearest_vocabulary(pos, vocab, max_mm=25.0).tolist() == [0, 1, -1]


def test_intracranial_kinds_accepted_only_when_asked():
    seeg = [_ch("LA1", loc=[0.02, 0.01, 0.03], kind="seeg"), _ch("Cz")]
    seeg.append(_ch("LA2", loc=[0.02, 0.012, 0.03], kind=802))  # FIFF sEEG
    with pytest.raises(ValueError, match="not EEG"):
        resolve_montage(seeg)
    m = resolve_montage(seeg, kinds=("eeg", "seeg"))
    assert m.names == ("LA1", "Cz", "LA2")
    assert m.kinds == ("seeg", "eeg", "seeg")
    # A kind outside ``kinds`` still raises, or is dropped when asked.
    mixed = seeg + [_ch("EOG", kind="eog")]
    with pytest.raises(ValueError, match="EOG.*not one of"):
        resolve_montage(mixed, kinds=("eeg", "seeg"))
    assert resolve_montage(mixed, kinds=("eeg", "seeg"), drop_non_eeg=True).names == (
        "LA1",
        "Cz",
        "LA2",
    )
    # EEG-only montages keep their key; kinds enter it otherwise.
    eeg = [_ch("Cz"), _ch("Pz")]
    assert resolve_montage(eeg).key == resolve_montage(eeg, kinds=("eeg", "seeg")).key
    relabelled = [dict(ch, kind="eeg") for ch in seeg]
    assert resolve_montage(relabelled).key != m.key


def test_kinds_must_be_electrode_kinds():
    with pytest.raises(ValueError, match="subset"):
        resolve_montage([_ch("Cz")], kinds=("eeg", "eog"))
    with pytest.raises(ValueError, match="subset"):
        resolve_montage([_ch("Cz")], kinds=())
