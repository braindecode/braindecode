from abc import ABC, abstractmethod


class LazyDataset(ABC):
    """ Class implementing an abstract lazy data set. Custom lazy data sets
    have to override file_paths, X and y as well as the load_lazy function to
    load trials or crops. """

    def __init__(self):
        self.file_paths = "Not implemented: a list of all file paths"
        self.X = (
            "Not implemented: a list of empty ndarrays with number of "
            "samples as second dimension"
        )
        self.y = "Not implemented: a list of all targets"

    @abstractmethod
    def load_lazy(self, path, start_i, stop_i):
        """ Loading procedure that gets a file path, start and stop indices.
        Is supposed to return a trial / crop together with its target

        Parameters
        ----------
        path: str
            file path
        start_i: int
            start index of signal crop
        stop_i: int
            stop index of signal crop
        """
        raise NotImplementedError

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        """ Returns a two-tuple of example, label """
        try:
            idx, start_i, stop_i = idx
        except (TypeError, ValueError):
            start_i = 0
            stop_i = None
        file_path = self.file_paths[idx]
        x = self.load_lazy(file_path, start_i, stop_i)

        if x.ndim == 2:
            x = x[:, :, None]
        return x, self.y[idx]
