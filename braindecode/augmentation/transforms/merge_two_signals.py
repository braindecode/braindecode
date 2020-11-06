import numpy as np


def merge_two_signals(datum, magnitude):

    signal = datum.X
    y = datum.y
    train_sample = datum.train_sample
    label_index_dict = datum.label_index_dict
    other_signal_index = np.random.choice(label_index_dict[y])
    other_signal = train_sample.get_raw_data(other_signal_index)[0]
    final_signal = (1 - magnitude) * \
        signal + magnitude * other_signal
    datum.X = final_signal

    return datum
