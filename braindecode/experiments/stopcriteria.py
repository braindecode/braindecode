import numpy as np


class MaxEpochs(object):
    """
    Stop when given number of epochs reached:
    
    Parameters
    ----------
    max_epochs: int
    """

    def __init__(self, max_epochs):
        self.max_epochs = max_epochs

    def should_stop(self, epochs_df):
        # Keep in mind  epoch 0 without training is also part of dataframe
        return len(epochs_df) - 1 >= self.max_epochs


class Or(object):
    """
    Stop when one of the given stop criteria is triggered.

    Parameters
    ----------
    stop_criteria: iterable of stop criteria objects
    """

    def __init__(self, stop_criteria):
        self.stop_criteria = stop_criteria
        self.triggered = dict([(s, False) for s in stop_criteria])

    def should_stop(self, epochs_df):
        # Update dictionary of which criterion was triggered ...
        for s in self.stop_criteria:
            self.triggered[s] = s.should_stop(epochs_df)
        # Then check if any of them was triggered.
        return np.any(list(self.triggered.values()))

    def was_triggered(self, criterion):
        """
        Return if given criterion was triggered in the last call to should stop.
        
        Parameters
        ----------
        criterion: stop criterion

        Returns
        -------
        triggered: bool

        """
        return self.triggered[criterion]


class And(object):
    """
    Stop when all of the given stop criteria are triggered.

    Parameters
    ----------
    stop_criteria: iterable of stop criteria objects
    """

    def __init__(self, stop_criteria):
        self.stop_criteria = stop_criteria

    def should_stop(self, epochs_df):
        # Update dictionary of which criterion was triggered ...
        for s in self.stop_criteria:
            self.triggered[s] = s.should_stop(epochs_df)
        # Then check if all of them were triggered.
        return np.all(list(self.triggered.values()))

    def was_triggered(self, criterion):
        """
        Return if given criterion was triggered in the last call to should stop.

        Parameters
        ----------
        criterion: stop criterion

        Returns
        -------
        triggered: bool

        """
        return self.triggered[criterion]


class NoDecrease(object):
    """ Stops if there is no decrease on a given monitor channel
    for given number of epochs.

    Parameters
    ----------
    column_name: str
        Name of column to monitor for decrease.
    num_epochs: str
        Number of epochs to wait before stopping when there is no decrease.
    min_decrease: float, optional
        Minimum relative decrease that counts as a decrease. E.g. 0.1 means
        only 10% decreases count as a decrease and reset the counter.
    """

    def __init__(self, column_name, num_epochs, min_decrease=1e-6):
        self.column_name = column_name
        self.num_epochs = num_epochs
        self.min_decrease = min_decrease
        self.best_epoch = 0
        self.lowest_val = float("inf")

    def should_stop(self, epochs_df):
        # -1 due to doing one monitor at start of training
        i_epoch = len(epochs_df) - 1
        current_val = float(epochs_df[self.column_name].iloc[-1])
        if current_val < ((1 - self.min_decrease) * self.lowest_val):
            self.best_epoch = i_epoch
            self.lowest_val = current_val

        return (i_epoch - self.best_epoch) >= self.num_epochs


class ColumnBelow:
    """
    Stops if the given column is below the given value.

    Parameters
    ----------
    column_name: str
        Name of column to monitor. 
    target_value: float
        When column decreases below this value, criterion will say to stop.
    """

    def __init__(self, column_name, target_value):
        self.column_name = column_name
        self.target_value = target_value

    def should_stop(self, epochs_df):
        # -1 due to doing one monitor at start of training
        current_val = float(epochs_df[self.column_name].iloc[-1])
        return current_val < self.target_value
