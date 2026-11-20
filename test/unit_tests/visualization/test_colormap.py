import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps

TUTORIAL = (
    Path(__file__).resolve().parents[3]
    / "examples/advanced_training/plot_relative_positioning.py"
)
# sphinx-gallery starts a text cell at a line of 20 or more "#".
CELL_SEPARATOR = re.compile(r"^#{20,}$", re.MULTILINE)


def test_relative_positioning_uses_full_sleep_stage_palette():
    """The tutorial's PCA plot spans viridis across the five sleep stages."""
    # The PCA plot is the last code cell; only the conclusion text follows it.
    plot = CELL_SEPARATOR.split(TUTORIAL.read_text(encoding="utf-8"))[-2]
    # The cell starts from the feature vectors and sleep-stage labels.
    namespace = {"np": np, "plt": plt, "data": {"test": (np.eye(5), np.arange(5))}}
    exec(compile(plot, str(TUTORIAL), "exec"), namespace)  # nosec B102
    fig = namespace["fig"]
    handles, labels = fig.axes[0].get_legend_handles_labels()
    colors = [handle.get_facecolor()[0, :3] for handle in handles]
    plt.close(fig)
    assert labels == ["W", "N1", "N2", "N3", "R"]
    expected = colormaps["viridis"].resampled(5).colors
    np.testing.assert_allclose(colors, np.asarray(expected)[:, :3])
