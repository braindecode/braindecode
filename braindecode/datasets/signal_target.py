class SignalAndTarget(object):
    def __init__(self, X, y):
        assert len(X) == len(y)
        self.X = X
        self.y = y


def apply_to_X_y(fn, *sets):
    X = fn(*[s.X for s in sets])
    y = fn(*[s.y for s in sets])
    return SignalAndTarget(X,y)


def split_into_two_sets(dataset, fraction=None, n_boundary=None):
    assert fraction is not None or n_boundary is not None
    if n_boundary is None:
        n_boundary = int(round(len(dataset.X) * fraction))
    first_set = apply_to_X_y(lambda a: a[:n_boundary], dataset)
    second_set = apply_to_X_y(lambda a: a[n_boundary:], dataset)
    return first_set, second_set
