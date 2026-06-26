# Authors: Bhargav Kowshik <bhargav.kowshik@gmail.com>
#
# License: BSD-3

"""Test that every model has a valid Modality entry in summary.csv."""

import csv
from pathlib import Path

import pytest

from braindecode.models.util import models_dict

# Controlled vocabulary for the "Modality" column of summary.csv. Keep this in
# sync with the column definition in docs/models/models_table.rst. Add a new
# token here (and document it) before using it in the CSV.
ALLOWED_MODALITIES = {"EEG", "MEG", "iEEG", "ECoG", "sEMG"}


def load_model_modalities():
    """Load model modalities from summary.csv (model name -> list of modalities)."""
    csv_path = Path(__file__).parents[3] / "braindecode" / "models" / "summary.csv"

    model_modalities = {}
    with open(csv_path, "r", encoding="utf-8-sig") as f:  # utf-8-sig handles BOM
        reader = csv.DictReader(f)
        for row in reader:
            raw = row.get("Modality", "") or ""
            modalities = [m.strip() for m in raw.split(",") if m.strip()]
            model_modalities[row["Model"]] = modalities

    return model_modalities


MODEL_MODALITIES = load_model_modalities()


@pytest.mark.parametrize("model_name", sorted(models_dict.keys()))
def test_model_has_valid_modality(model_name):
    """Every model has at least one modality, all from the allowed vocabulary."""
    modalities = MODEL_MODALITIES.get(model_name, [])

    assert len(modalities) > 0, (
        f"{model_name} has no Modality defined in summary.csv. "
        f"Add one of {sorted(ALLOWED_MODALITIES)}."
    )

    for modality in modalities:
        assert modality in ALLOWED_MODALITIES, (
            f"{model_name} has unknown modality '{modality}'. "
            f"Allowed values are {sorted(ALLOWED_MODALITIES)}. "
            f"If this is a new modality, add it to ALLOWED_MODALITIES and "
            f"document it in docs/models/models_table.rst."
        )
