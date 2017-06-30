import numpy as np


class MaxEpochs(object):
    def __init__(self, max_epochs):
        self.max_epochs = max_epochs

    def should_stop(self, epochs_df):
        # Keep in mind  epoch 0 without training is also part of dataframe
        return len(epochs_df) - 1 >= self.max_epochs


class Or(object):
    def __init__(self, stop_criteria):
        self.stop_criteria = stop_criteria

    def should_stop(self, epochs_df):
        return np.any([s.should_stop(epochs_df)
                       for s in self.stop_criteria])


class NoDecrease(object):
    """ Stops if there is no decrease on a given monitor channel
    for given number of epochs."""

    def __init__(self, column_name, num_epochs, min_decrease=1e-6):
        self.column_name = column_name
        self.num_epochs = num_epochs
        self.min_decrease = min_decrease
        self.best_epoch = 0
        self.lowest_val = float('inf')

    def should_stop(self, epochs_df):
        # -1 due to doing one monitor at start of training
        i_epoch = len(epochs_df) - 1
        current_val = float(epochs_df[self.column_name].iloc[-1])
        if current_val < ((1 - self.min_decrease) * self.lowest_val):
            self.best_epoch = i_epoch
            self.lowest_val = current_val

        return (i_epoch - self.best_epoch) >= self.num_epochs


class ColumnBelow():
    """ Stops if the given column is below the given value."""
    def __init__(self, column_name, target_value):
        self.column_name = column_name
        self.target_value = target_value

    def should_stop(self, epochs_df):
        # -1 due to doing one monitor at start of training
        current_val = float(epochs_df[self.column_name].iloc[-1])
        return current_val < self.target_value
