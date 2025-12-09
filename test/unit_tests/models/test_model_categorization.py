# Authors: Bruno Aristimunha <b.aristimunha@gmail.com>
#
# License: BSD-3

"""Test that all models have categorization badges in their docstrings."""

import csv
from pathlib import Path

import pytest

from braindecode.models.util import models_dict

# Mapping of categorization names to badge formats
CATEGORIZATION_BADGES = {
    "Convolution": ":bdg-success:`Convolution`",
    "Recurrent": ":bdg-secondary:`Recurrent`",
    "Attention/Transformer": ":bdg-info:`Attention/Transformer`",
    "Filterbank": ":bdg-primary:`Filterbank`",
    "FilterBank": ":bdg-primary:`Filterbank`",  # Handle both cases
    "Interpretability": ":bdg-warning:`Interpretability`",
    "Foundation Model": ":bdg-danger:`Foundation Model`",
    "Channel": ":bdg-dark-line:`Channel`",
}


def load_model_categorizations():
    """Load model categorizations from summary.csv."""
    csv_path = Path(__file__).parents[3] / "braindecode" / "models" / "summary.csv"

    model_categorizations = {}
    with open(csv_path, "r", encoding="utf-8-sig") as f:  # utf-8-sig handles BOM
        reader = csv.DictReader(f)
        for row in reader:
            model_name = row["Model"]
            categorization = row["Categorization"]
            # Split by comma and strip whitespace
            categories = [cat.strip() for cat in categorization.split(",")]
            model_categorizations[model_name] = categories

    return model_categorizations


# Load the categorizations
MODEL_CATEGORIZATIONS = load_model_categorizations()


@pytest.mark.parametrize("model_name", sorted(models_dict.keys()))
def test_model_has_categorization_badges(model_name):
    """Test that each model has the correct categorization badges in its docstring."""
    model_class = models_dict[model_name]
    docstring = model_class.__doc__

    assert docstring is not None, f"{model_name} has no docstring"

    # Get expected categorizations from CSV
    expected_categories = MODEL_CATEGORIZATIONS.get(model_name, [])

    assert len(expected_categories) > 0, (
        f"{model_name} has no categorization defined in summary.csv"
    )

    # Check that each expected badge is present in the docstring
    for category in expected_categories:
        expected_badge = CATEGORIZATION_BADGES.get(category)

        assert expected_badge is not None, (
            f"Unknown category '{category}' for {model_name}. "
            f"Please add it to CATEGORIZATION_BADGES mapping."
        )

        assert expected_badge in docstring, (
            f"{model_name} is missing categorization badge: {expected_badge}\n"
            f"Expected categories: {expected_categories}\n"
            f"Docstring start: {docstring[:200]}"
        )


def test_all_models_in_summary_csv():
    """Test that all models in __init__.py are covered in summary.csv."""
    models_in_csv = set(MODEL_CATEGORIZATIONS.keys())
    models_in_code = set(models_dict.keys())

    # Models in code but not in CSV
    missing_in_csv = models_in_code - models_in_csv
    if missing_in_csv:
        pytest.fail(
            f"Models defined in code but missing from summary.csv: {missing_in_csv}"
        )

    # Models in CSV but not in code (just a warning via message)
    missing_in_code = models_in_csv - models_in_code
    if missing_in_code:
        print(
            f"Warning: Models in summary.csv but not in test: {missing_in_code}"
        )


def test_badge_format_consistency():
    """Test that all badges follow the expected format."""
    for model_name, model_class in models_dict.items():
        docstring = model_class.__doc__
        if docstring is None:
            continue

        # Find all badge-like patterns in the docstring
        import re
        badge_pattern = r':bdg-\w+:`[^`]+`'
        found_badges = re.findall(badge_pattern, docstring)

        # Check that all found badges are valid
        valid_badges = set(CATEGORIZATION_BADGES.values())
        for badge in found_badges:
            # Only check categorization badges (ignore other potential badges)
            if any(cat_name.lower() in badge.lower() for cat_name in [
                "convolution", "recurrent", "attention", "transformer", "filterbank",
                "interpretability", "foundation model", "channel"
            ]):
                assert badge in valid_badges, (
                    f"{model_name} has invalid badge format: {badge}\n"
                    f"Valid badges are: {valid_badges}"
                )
