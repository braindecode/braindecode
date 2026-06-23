import numpy as np
from matplotlib import colormaps


def test_relative_positioning_sleep_stage_colors_are_distinct():
    """Regression guard for the colormap sampling used in the SSL tutorial."""
    colors = colormaps["viridis"](np.linspace(0, 1, 5))

    assert np.unique(colors, axis=0).shape[0] == 5
