import numpy as np

from braindecode.datasets.signal_target import apply_to_X_y


def concatenate_sets(set_a, set_b):
    return apply_to_X_y(lambda a,b: np.concatenate((a,b), axis=0),
                        set_a, set_b)


def split_into_two_sets(dataset, first_set_fraction=None, n_boundary=None):
    assert (first_set_fraction is None) != (n_boundary is None), (
        "Supply either first_set_fraction or n_boundary")
    if n_boundary is None:
        n_boundary = int(round(len(dataset.X) * first_set_fraction))
    first_set = apply_to_X_y(lambda a: a[:n_boundary], dataset)
    second_set = apply_to_X_y(lambda a: a[n_boundary:], dataset)
    return first_set, second_set
