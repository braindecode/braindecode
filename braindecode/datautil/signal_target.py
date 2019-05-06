class SignalAndTarget(object):
    """
    Simple data container class.

    Parameters
    ----------
    X: 3darray or list of 2darrays
        The input signal per trial.
    y: 1darray or list
        Labels for each trial.
    """

    def __init__(self, X, y):
        assert len(X) == len(y)
        self.X = X
        self.y = y


def apply_to_X_y(fn, *sets):
    """
    Apply a function to all `X` and `y` attributes of all given sets.
    
    Applies function to list of X arrays and to list of y arrays separately.
    
    Parameters
    ----------
    fn: function
        Function to apply
    sets: :class:`.SignalAndTarget` objects

    Returns
    -------
    result_set: :class:`.SignalAndTarget`
        Dataset with X and y as the result of the
        application of the function.
    """
    X = fn(*[s.X for s in sets])
    y = fn(*[s.y for s in sets])
    return SignalAndTarget(X, y)
