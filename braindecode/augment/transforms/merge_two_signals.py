# Authors: Simon Freyburger
#
# License: BSD-3

import numpy as np


def merge_two_signals(datum, params):

    signal = datum.X
    y = datum.y
    train_sample = datum.ds
    label_index_dict = datum.required_variables["label_index_dict"]
    other_signal = np.zeros(datum.X.shape)
    for label in y.keys():
        # y[label] is the proportion of the label in y
        other_signal_index = np.random.choice(label_index_dict[label])
        other_signal += y[label] * train_sample[other_signal_index][0]

    datum.X = (1 - params["magnitude"]) * \
        signal + params["magnitude"] * other_signal

    return datum


def init_label_index_dict(dataset, transform_list):
    """Create a dictionnary, with as key the labels available in the
        multi-classification process, and as value for a given label
        all indexes of data that corresponds to its label.
        """
    subset_unaug_indices = list(range(len(dataset)))
    subset_unaug_labels = [dataset[indice][1]
                           for indice in subset_unaug_indices]
    list_labels = list(set(subset_unaug_labels))
    label_index_dict = {}
    for label in list_labels:
        label_index_dict[label] = []
    for i in range(len(subset_unaug_indices)):
        label_index_dict[subset_unaug_labels[i]].append(
            subset_unaug_indices[i])
    return label_index_dict


MERGE_TWO_SIGNALS_REQUIRED_VARIABLES = {
    "label_index_dict": init_label_index_dict}
