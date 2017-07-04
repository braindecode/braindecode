import numpy as np

from braindecode.datautil.signal_target import apply_to_X_y, SignalAndTarget


def concatenate_sets(sets):
    concatenated_set = sets[0]
    for s in sets[1:]:
        concatenated_set = concatenate_two_sets(concatenated_set, s)
    return concatenated_set


def concatenate_two_sets(set_a, set_b):
    if hasattr(set_a.X, 'ndim') and hasattr(set_b.X, 'ndim'):
        new_X = np.concatenate((set_a.X, set_b.X), axis=0)
    else:
        if hasattr(set_a.X, 'ndim'):
            set_a.X = set_a.X.tolist()
        if hasattr(set_b.X, 'ndim'):
            set_b.X = set_b.X.tolist()
        new_X = set_a.X + set_b.X
    new_y = np.concatenate((set_a.y, set_b.y), axis=0)
    return SignalAndTarget(new_X, new_y)


def split_into_two_sets(dataset, first_set_fraction=None, n_boundary=None):
    assert (first_set_fraction is None) != (n_boundary is None), (
        "Supply either first_set_fraction or n_boundary")
    if n_boundary is None:
        n_boundary = int(round(len(dataset.X) * first_set_fraction))
    first_set = apply_to_X_y(lambda a: a[:n_boundary], dataset)
    second_set = apply_to_X_y(lambda a: a[n_boundary:], dataset)
    return first_set, second_set
