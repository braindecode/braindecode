import numpy as np
from matplotlib import colormaps


def test_colormap_linspace_sampling_produces_distinct_colors():
    """Regression guard for the colormap sampling used in the SSL tutorial."""
    colors = colormaps["viridis"](np.linspace(0, 1, 5))

    assert np.unique(colors, axis=0).shape[0] == 5
