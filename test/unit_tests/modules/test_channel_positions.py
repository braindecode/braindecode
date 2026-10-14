# Authors: Bruno Aristimunha <b.aristimunha@gmail.com>
#
# License: BSD (3-clause)
"""Channel positions from a name -> position bank (REVE's, offline copy)."""

import json
import warnings
from pathlib import Path

import mne
import numpy as np
import pytest

from braindecode.modules import resolve_channel_positions
from braindecode.util import resolve_montage_name

DATA = Path(__file__).parent / "data"
# REVE's released bank (brain-bzh/reve-positions, positions.json), copied so
# that no test needs the network.
BANK = {
    k: np.array(v, float)
    for k, v in json.loads((DATA / "reve_positions.json").read_text()).items()
}
TEN_TWENTY = "Fp1 Fp2 F7 F3 Fz F4 F8 T7 C3 Cz C4 T8 P7 P3 Pz P4 P8 O1 O2".split()
# Bank entries more than 1 mm from MNE's standard_1005 position of the same
# name (see test_bank_agrees_with_mne_standard_1005).
OFF_STANDARD = {}


def names(ns):
    return [{"ch_name": n, "kind": "eeg", "loc": np.zeros(12)} for n in ns]


def from_montage(montage, ns=None):
    """``raw.info["chs"]`` after ``set_montage``: ``loc`` in the head frame."""
    m = mne.channels.make_standard_montage(resolve_montage_name(montage))
    info = mne.create_info(ns or m.ch_names, 200.0, "eeg")
    info.set_montage(m)
    return info["chs"]


def at(name, xyz):
    return {"ch_name": name, "kind": "eeg", "loc": np.r_[xyz, np.zeros(9)]}


def resolve(chs, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no warning unless a test expects one
        return resolve_channel_positions(chs, BANK, **kw)


def test_bank_agrees_with_mne_standard_1005():
    """Every bank name of standard_1005 (any case) sits at MNE's position."""
    std = mne.channels.make_standard_montage(resolve_montage_name("standard_1005"))
    ref = {k.lower(): np.asarray(v) for k, v in std.get_positions()["ch_pos"].items()}
    common = [k for k in BANK if k.lower() in ref]
    assert len(common) > 150
    off = {
        k: round(float(np.linalg.norm(BANK[k] - ref[k.lower()]) * 1e3), 1)
        for k in common
        if np.linalg.norm(BANK[k] - ref[k.lower()]) > 1e-3
    }
    assert off == OFF_STANDARD


def test_exact_names_give_the_bank_rows():
    ns = [*TEN_TWENTY, "A1", "T3", "CZ", "E1", "biosemi128_A1"]
    out = resolve(names(ns))
    np.testing.assert_array_equal(out.positions, np.stack([BANK[n] for n in ns]))
    np.testing.assert_array_equal(out.index, np.arange(len(ns)))
    assert {p["rule"] for p in out.provenance} == {"exact"}
    assert [p["name"] for p in out.provenance] == ns


def test_head_frame_loc_does_not_move_exact_names():
    """``raw.set_montage`` gives ``loc`` 46-57 mm from the bank's frame: the
    registration absorbs it, so named channels keep the bank rows."""
    chs = from_montage("standard_1005", TEN_TWENTY)
    out = resolve(chs)
    np.testing.assert_array_equal(
        out.positions, np.stack([BANK[n] for n in TEN_TWENTY])
    )
    assert {p["rule"] for p in out.provenance} == {"exact"}
    assert max(p["distance"] for p in out.provenance) < 1e-3


def test_loc_only_channel_is_registered_onto_the_bank():
    chs = from_montage("standard_1005", [*TEN_TWENTY, "AFz"])
    chs[-1] = at("X1", chs[-1]["loc"][:3])  # AFz's site under an unknown name
    out = resolve(chs)
    assert out.provenance[-1]["rule"] == "loc" and out.provenance[-1]["name"] is None
    assert np.linalg.norm(out.positions[-1] - BANK["AFz"]) < 1e-3


def test_biosemi128_a1_a2_are_not_the_bank_earlobes():
    """REVE's ``A1``/``A2`` are 10-20 earlobes and ``C1``-``C6`` 10-10 sites;
    BioSemi-128 uses those names near the vertex (``biosemi128_A1``)."""
    chs = from_montage("biosemi128")
    clash = ["A1", "A2", "C1", "C2", "C3", "C4", "C5", "C6"]
    with pytest.warns(UserWarning, match="BioSemi A1/A2"):
        out = resolve_channel_positions(chs, BANK)
    rules = {p["ch_name"]: p["rule"] for p in out.provenance}
    assert sorted(n for n, r in rules.items() if r == "loc") == sorted(clash)
    for n in clash:
        i = [c["ch_name"] for c in chs].index(n)
        assert np.linalg.norm(out.positions[i] - BANK[f"biosemi128_{n}"]) < 1e-3
        assert np.linalg.norm(out.positions[i] - BANK[n]) > 0.03
    # montage-qualified names: every channel, no warning
    out = resolve(chs, montage="biosemi128")
    assert {p["rule"] for p in out.provenance} == {"montage"}
    np.testing.assert_array_equal(
        out.positions, np.stack([BANK[f"biosemi128_{c['ch_name']}"] for c in chs])
    )
    # names only: nothing to check them against, the checkpoint's lookup
    assert resolve(names(["A1"])).positions[0].tolist() == BANK["A1"].tolist()
    assert (
        resolve(names(["A1"]), montage="biosemi128").positions[0].tolist()
        == BANK["biosemi128_A1"].tolist()
    )


def test_few_named_locs_keep_the_bank_names():
    """Under 4 channels with a name and a ``loc``: no frame to check them in,
    the checkpoint's lookup (BioSemi users pass ``montage``)."""
    out = resolve(from_montage("biosemi128", ["A1", "A2", "A3"]))
    assert {p["rule"] for p in out.provenance} == {"exact"}
    np.testing.assert_array_equal(
        out.positions, np.stack([BANK[n] for n in ("A1", "A2", "A3")])
    )


def test_montage_reusing_bank_names_takes_its_loc():
    """GSN-HydroCel-256's E-names are other sites than REVE's (129-channel) ones."""
    chs = from_montage("GSN-HydroCel-256")
    with pytest.warns(UserWarning, match="more than 30 mm"):
        out = resolve_channel_positions(chs, BANK)
    rules = [p["rule"] for p in out.provenance]
    assert rules.count("loc") > 0.9 * len(chs)


def test_unnamed_montage_is_registered_on_the_template():
    """No bank name at all: MNE's standard_1005 registers the head frame."""
    chs = from_montage("standard_1005", TEN_TWENTY)
    chs = [at(f"X{i}", c["loc"][:3]) for i, c in enumerate(chs)]
    out = resolve(chs)
    assert {p["rule"] for p in out.provenance} == {"loc"}
    ref = np.stack([BANK[n] for n in TEN_TWENTY])
    assert np.linalg.norm(out.positions - ref, axis=1).max() < 1e-3


def test_gsn_hydrocel_129_e_names():
    chs = from_montage("GSN-HydroCel-129")
    out = resolve(chs)
    assert {p["rule"] for p in out.provenance} == {"exact"}
    np.testing.assert_array_equal(
        out.positions, np.stack([BANK[c["ch_name"]] for c in chs])
    )


def test_bipolar_midpoint():
    out = resolve(names(["F3-M2", "F3\u2013M2", "EEG Fp1-T3", "Cz"]))
    mid = (BANK["F3"] + BANK["M2"]) / 2
    np.testing.assert_array_equal(out.positions[0], mid)
    np.testing.assert_array_equal(out.positions[1], mid)
    np.testing.assert_array_equal(out.positions[2], (BANK["Fp1"] + BANK["T3"]) / 2)
    assert [p["rule"] for p in out.provenance] == ["bipolar"] * 3 + ["exact"]
    with pytest.raises(ValueError, match="F3-X9"):  # both halves or nothing
        resolve(names(["F3-X9"]))


def test_case_prefix_suffix_and_aliases():
    out = resolve(names(["fp1", "EEG FP2-REF", "EEG Cz-LE", "o1-AR", "fpz"]))
    assert [p["name"] for p in out.provenance] == ["FP1", "FP2", "Cz", "O1", "Fpz"]
    assert {p["rule"] for p in out.provenance} == {"normalised"}
    bank = {"T7": [-0.08, 0.0, 0.0], "P7": [-0.07, -0.04, 0.0], "Cz": [0, 0, 0.1]}
    bank.update(POO7=[-0.03, -0.09, 0.0])
    out = resolve_channel_positions(
        names(["T3", "t5", "EEG T5-REF", "VREF", "Cb1"]), bank
    )
    assert [p["name"] for p in out.provenance] == ["T7", "P7", "P7", "Cz", "POO7"]


def test_on_missing_error_suggests_but_never_places():
    with pytest.raises(ValueError, match=r"suggestions only.*'Fpz'"):
        resolve(names(["Cz", "Fpzz"]))
    with pytest.raises(ValueError, match="on_missing must be"):
        resolve(names(["Cz"]), on_missing="closer")


def test_on_missing_drop():
    with pytest.warns(UserWarning, match="dropped"):
        out = resolve_channel_positions(
            names(["Cz", "Fpzz", "Pz"]), BANK, on_missing="drop"
        )
    np.testing.assert_array_equal(out.index, [0, 2])
    np.testing.assert_array_equal(out.positions, np.stack([BANK["Cz"], BANK["Pz"]]))
    assert [p["rule"] for p in out.provenance] == ["exact", "dropped", "exact"]


def test_on_missing_nearest_snaps_positioned_channels_only():
    chs = from_montage("standard_1005", TEN_TWENTY)
    afz = from_montage("standard_1005", ["AFz"])[0]["loc"][:3]
    chs.append(at("X1", afz + [0.002, 0.0, 0.0]))
    out = resolve(chs, on_missing="nearest")
    p = out.provenance[-1]
    assert p["rule"] == "loc" and p["name"] == "AFz" and 1e-3 < p["distance"] < 3e-3
    np.testing.assert_array_equal(out.positions[-1], BANK["AFz"])
    # far from every entry: snapped, with a warning
    chs[-1] = at("X1", afz + [0.0, 0.05, 0.05])
    with pytest.warns(UserWarning, match="more than 15 mm away"):
        resolve_channel_positions(chs, BANK, on_missing="nearest")
    # no position at all: still never placed
    with pytest.raises(ValueError, match="on_missing='drop'"):
        resolve(names(["Cz", "Fpzz"]), on_missing="nearest")


def test_unknown_montage_qualifier_raises():
    with pytest.raises(ValueError, match="biosemi128"):
        resolve(names(["A1"]), montage="biosemi64")
