# Authors: Simon Freyburger
#
# License: BSD-3

import numpy as np


def merge_two_signals(datum, magnitude):

    signal = datum.X
    y = datum.y
    train_sample = datum.train_sample
    label_index_dict = datum.label_index_dict
    other_signal_index = np.random.choice(label_index_dict[y])
    other_signal = train_sample.get_unaugmented_data(other_signal_index)[0]
    final_signal = (1 - magnitude) * \
        signal + magnitude * other_signal
    datum.X = final_signal

    return datum


def init_label_index_dict(dataset):
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
    return(label_index_dict)


MERGE_TWO_SIGNALS_REQUIRED_VARIABLES = {
    "label_index_dict": init_label_index_dict}
