# Authors: Robin Schirrmeister <robintibor@gmail.com>
#
# License: BSD (3-clause)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


def plot_confusion_matrix(confusion_mat,
                          class_names=None, figsize=None,
                          colormap=cm.bwr,
                          textcolor='black', vmin=None, vmax=None,
                          fontweight='normal',
                          rotate_row_labels=90,
                          rotate_col_labels=0,
                          with_f1_score=False,
                          norm_axes=(0, 1),
                          rotate_precision=False,
                          class_names_fontsize=12):
    """

    Generates a confusion matrix with additional precision and sensitivity metrics as in [1]_.

    Parameters
    ----------
    confusion_mat: 2d numpy array
        A confusion matrix, e.g. sklearn confusion matrix:
        https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
    class_names: array, optional
        List of classes/targets.
    figsize: tuple, optional
        Size of the generated confusion matrix figure.
    colormap: matplotlib cm colormap, optional
    textcolor: str, optional
        Color  of the text in the figure.
    vmin, vmax: float, optional
        The data range that the colormap covers.
    fontweight: str, optional
        Weight of the font in the figure:
        [ 'normal' | 'bold' | 'heavy' | 'light' | 'ultrabold' | 'ultralight']
    rotate_row_labels: int, optional
        The rotation angle of the row labels
    rotate_col_labels: int, optional
        The rotation angle of the column labels
    with_f1_score: bool, optional
    norm_axes: tuple, optional
    rotate_precision: bool, optional
    class_names_fontsize: int, optional

    Returns
    -------
    fig: matplotlib figure

    References
    ----------

    .. [1] Schirrmeister, R. T., Springenberg, J. T., Fiederer, L. D. J.,
       Glasstetter, M., Eggensperger, K., Tangermann, M., Hutter, F. & Ball, T. (2017).
       Deep learning with convolutional neural networks for EEG decoding and
       visualization.
       Human Brain Mapping , Aug. 2017. Online: http://dx.doi.org/10.1002/hbm.23730
    """
    # transpose to get confusion matrix same way as matlab
    confusion_mat = confusion_mat.T
    n_classes = confusion_mat.shape[0]
    if class_names is None:
        class_names = [str(i_class + 1) for i_class in range(n_classes)]

    # norm by all targets
    normed_conf_mat = confusion_mat / np.float32(np.sum(confusion_mat,
                                                        axis=norm_axes, keepdims=True))

    fig = plt.figure(figsize=figsize)
    plt.clf()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    if vmin is None:
        vmin = np.min(normed_conf_mat)
    if vmax is None:
        vmax = np.max(normed_conf_mat)

    # see http://stackoverflow.com/a/31397438/1469195
    # brighten so that black text remains readable
    # used alpha=0.6 before
    def _brighten(x, ):
        brightened_x = 1 - ((1 - np.array(x)) * 0.4)
        return brightened_x

    brightened_cmap = _cmap_map(_brighten, colormap)  # colormap #
    ax.imshow(np.array(normed_conf_mat), cmap=brightened_cmap,
              interpolation='nearest', vmin=vmin, vmax=vmax)

    # make space for precision and sensitivity
    plt.xlim(-0.5, normed_conf_mat.shape[0] + 0.5)
    plt.ylim(normed_conf_mat.shape[1] + 0.5, -0.5)
    width = len(confusion_mat)
    height = len(confusion_mat[0])
    for x in range(width):
        for y in range(height):
            if x == y:
                this_font_weight = 'bold'
            else:
                this_font_weight = fontweight
            annotate_str = "{:d}".format(confusion_mat[x][y])
            annotate_str += "\n"
            ax.annotate(annotate_str.format(confusion_mat[x][y]),
                        xy=(y, x),
                        horizontalalignment='center',
                        verticalalignment='center', fontsize=12,
                        color=textcolor,
                        fontweight=this_font_weight)
            if x != y or (not with_f1_score):
                ax.annotate(
                    "\n\n{:4.1f}%".format(
                        normed_conf_mat[x][y] * 100),
                    xy=(y, x),
                    horizontalalignment='center',
                    verticalalignment='center', fontsize=10,
                    color=textcolor,
                    fontweight=this_font_weight)
            else:
                assert x == y
                precision = confusion_mat[x][x] / float(np.sum(
                    confusion_mat[x, :]))
                sensitivity = confusion_mat[x][x] / float(np.sum(
                    confusion_mat[:, y]))
                f1_score = 2 * precision * sensitivity / (precision + sensitivity)

                ax.annotate("\n{:4.1f}%\n{:4.1f}% (F)".format(
                    (confusion_mat[x][y] / float(np.sum(confusion_mat))) * 100,
                    f1_score * 100),
                    xy=(y, x + 0.1),
                    horizontalalignment='center',
                    verticalalignment='center', fontsize=10,
                    color=textcolor,
                    fontweight=this_font_weight)

    # Add values for target correctness etc.
    for x in range(width):
        y = len(confusion_mat)
        if float(np.sum(confusion_mat[x, :])) == 0:
            annotate_str = "-"
        else:
            correctness = confusion_mat[x][x] / float(np.sum(confusion_mat[x, :]))
            annotate_str = ""
            annotate_str += "\n{:5.2f}%".format(correctness * 100)
        ax.annotate(annotate_str,
                    xy=(y, x),
                    horizontalalignment='center',
                    verticalalignment='center', fontsize=12)

    for y in range(height):
        x = len(confusion_mat)
        if float(np.sum(confusion_mat[:, y])) == 0:
            annotate_str = "-"
        else:
            correctness = confusion_mat[y][y] / float(np.sum(confusion_mat[:, y]))
            annotate_str = ""
            annotate_str += "\n{:5.2f}%".format(correctness * 100)
        ax.annotate(annotate_str,
                    xy=(y, x),
                    horizontalalignment='center',
                    verticalalignment='center', fontsize=12)

    overall_correctness = np.sum(np.diag(confusion_mat)) / np.sum(confusion_mat).astype(float)
    ax.annotate("{:5.2f}%".format(overall_correctness * 100),
                xy=(len(confusion_mat), len(confusion_mat)),
                horizontalalignment='center',
                verticalalignment='center', fontsize=12,
                fontweight='bold')

    plt.xticks(range(width), class_names, fontsize=class_names_fontsize,
               rotation=rotate_col_labels)
    plt.yticks(np.arange(0, height), class_names,
               va='center',
               fontsize=class_names_fontsize, rotation=rotate_row_labels)
    plt.grid(False)
    plt.ylabel('Predictions', fontsize=15)
    plt.xlabel('Targets', fontsize=15)

    # n classes is also shape of matrix/size
    ax.text(-1.2, n_classes + 0.2, "Recall", ha='center', va='center',
            fontsize=13)
    if rotate_precision:
        rotation = 90
        x_pos = -1.1
        va = 'center'
    else:
        rotation = 0
        x_pos = -0.8
        va = 'top'
    ax.text(n_classes, x_pos, "Precision", ha='center', va=va,
            rotation=rotation,  # 270,
            fontsize=13)

    return fig


# see http://stackoverflow.com/a/31397438/1469195
def _cmap_map(function, cmap, name='colormap_mod', N=None, gamma=None):
    """
    Modify a colormap using `function` which must operate on 3-element
    arrays of [r, g, b] values.

    You may specify the number of colors, `N`, and the opacity, `gamma`,
    value of the returned colormap. These values default to the ones in
    the input `cmap`.

    You may also specify a `name` for the colormap, so that it can be
    loaded using plt.get_cmap(name).
    """
    from matplotlib.colors import LinearSegmentedColormap as lsc
    if N is None:
        N = cmap.N
    if gamma is None:
        gamma = cmap._gamma
    cdict = cmap._segmentdata
    # Cast the steps into lists:
    step_dict = {key: list(map(lambda x: x[0], cdict[key])) for key in cdict}
    # Now get the unique steps (first column of the arrays):
    step_dicts = np.array(list(step_dict.values()))
    step_list = np.unique(step_dicts)
    # 'y0', 'y1' are as defined in LinearSegmentedColormap docstring:
    y0 = cmap(step_list)[:, :3]
    y1 = y0.copy()[:, :3]
    # Go back to catch the discontinuities, and place them into y0, y1
    for iclr, key in enumerate(['red', 'green', 'blue']):
        for istp, step in enumerate(step_list):
            try:
                ind = step_dict[key].index(step)
            except ValueError:
                # This step is not in this color
                continue
            y0[istp, iclr] = cdict[key][ind][1]
            y1[istp, iclr] = cdict[key][ind][2]
    # Map the colors to their new values:
    y0 = np.array(list(map(function, y0)))
    y1 = np.array(list(map(function, y1)))
    # Build the new colormap (overwriting step_dict):
    for iclr, clr in enumerate(['red', 'green', 'blue']):
        step_dict[clr] = np.vstack((step_list, y0[:, iclr], y1[:, iclr])).T
    # Remove alpha, otherwise crashes...
    step_dict.pop('alpha', None)
    return lsc(name, step_dict, N=N, gamma=gamma)
