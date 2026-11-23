import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb

TUTORIAL = (
    Path(__file__).resolve().parents[3]
    / "examples/advanced_training/plot_relative_positioning.py"
)
# sphinx-gallery starts a text cell at a line of 20 or more "#".
CELL_SEPARATOR = re.compile(r"^#{20,}$", re.MULTILINE)


def test_relative_positioning_labels_each_sleep_stage_in_its_own_color():
    """The tutorial's PCA plot draws the five sleep stages in distinct colors,
    each labelled on the plot in its color."""
    # The PCA plot is the last code cell; only the conclusion text follows it.
    plot = CELL_SEPARATOR.split(TUTORIAL.read_text(encoding="utf-8"))[-2]
    # The cell starts from the feature vectors and sleep-stage labels.
    namespace = {"np": np, "plt": plt, "data": {"test": (np.eye(5), np.arange(5))}}
    exec(compile(plot, str(TUTORIAL), "exec"), namespace)  # nosec B102
    ax = namespace["fig"].axes[0]
    colors = [tuple(c.get_facecolor()[0, :3]) for c in ax.collections]
    labels = {t.get_text(): to_rgb(t.get_color()) for t in ax.texts}
    plt.close(namespace["fig"])
    assert list(labels) == ["Wake", "N1", "N2", "N3", "REM"]
    assert len(set(colors)) == 5
    assert list(labels.values()) == colors
