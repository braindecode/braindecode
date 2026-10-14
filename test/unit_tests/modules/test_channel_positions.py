"""resolve_channel_positions and its use in REVE."""

import json

import numpy as np
import pytest
import torch

from braindecode.models import REVE
from braindecode.modules import resolve_channel_positions

BANK = {
    "Cz": [0.0, 0.0, 0.1],
    "T7": [-0.08, 0.0, 0.0],
    "F3": [-0.05, 0.05, 0.05],
    "M2": [0.08, -0.03, -0.05],
    "A1": [-0.09, -0.02, -0.07],  # 10-20 earlobe
    "biosemi128_A1": [0.0, 0.0, 0.095],  # BioSemi A1, near the vertex
}


def chs(*names, loc=None):
    return [{"ch_name": n, "loc": loc} for n in names]


def test_names_aliases_montage_and_bipolar():
    pos, kept = resolve_channel_positions(chs("Cz", "cz", "EEG T3-REF", "F3-M2"), BANK)
    assert kept == [0, 1, 2, 3]
    np.testing.assert_array_equal(pos[:3], [BANK["Cz"], BANK["Cz"], BANK["T7"]])
    np.testing.assert_allclose(pos[3], (np.array(BANK["F3"]) + BANK["M2"]) / 2)
    pos, _ = resolve_channel_positions(chs("A1"), BANK, montage="biosemi128")
    np.testing.assert_array_equal(pos[0], BANK["biosemi128_A1"])


def test_on_missing():
    with pytest.raises(ValueError, match="X1"):
        resolve_channel_positions(chs("Cz", "X1"), BANK)
    with pytest.warns(UserWarning, match="X1"):
        _, kept = resolve_channel_positions(chs("Cz", "X1"), BANK, on_missing="drop")
    assert kept == [0]
    near = [{"ch_name": "E7", "loc": np.r_[-0.079, 0.001, 0.0, np.zeros(9)]}]
    pos, _ = resolve_channel_positions(near, BANK, on_missing="nearest")
    np.testing.assert_array_equal(pos[0], BANK["T7"])  # never placed without a loc
    with pytest.raises(ValueError, match="X1"):
        resolve_channel_positions(chs("X1"), BANK, on_missing="nearest")


@pytest.fixture
def reve_bank(tmp_path, monkeypatch):
    (tmp_path / "reve_positions.json").write_text(json.dumps(BANK))
    monkeypatch.setenv("REVE_POSITIONS_PATH", str(tmp_path))


def small_reve(chs_info, **kw):
    kwargs = dict(n_outputs=2, n_times=380, embed_dim=32, depth=1, heads=2, head_dim=16)
    return REVE(chs_info=chs_info, **kwargs, **kw).eval()


def test_reve_exact_names_keep_the_bank_lookup(reve_bank):
    model = small_reve(chs("Cz", "F3", "A1"))
    bank = model._position_bank
    assert torch.equal(model.default_pos, bank(["Cz", "F3", "A1"]))


def test_reve_drop_reads_the_kept_channels(reve_bank):
    torch.manual_seed(0)
    with pytest.warns(UserWarning, match="X1"):
        dropping = small_reve(chs("Cz", "X1", "F3"), on_missing="drop")
    known = small_reve(chs("Cz", "F3"))
    known.load_state_dict(dropping.state_dict())
    x = torch.randn(2, 3, 380)
    with torch.no_grad():
        torch.testing.assert_close(dropping(x), known(x[:, [0, 2]]))
    with pytest.raises(ValueError, match="X1"):
        bank = dropping._position_bank
        bank(["Cz", "X1"])
