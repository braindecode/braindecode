from braindecode.datasets.croppedxy import CroppedXyDataset


class TrainTestSplit(object):
    """
    Class to perform splitting on a dataset with just X,y attributes.

    TODO: try make this work without supplying input time length
    and n_preds_per_trial, they are only passed to CroppedDatasetXy

    Parameters
    ----------
    train_size: int or float
        Train size in number of trials or fraction of trials
    input_time_length: int
        Input time length aka supercrop size in number of samples.
    n_preds_per_input:
        Number of predictions per supercrop (=> will be supercrop stride)
        in number of samples.
    """
    def __init__(
            self, train_size, input_time_length, n_preds_per_input):
        assert isinstance(train_size, (int, float))
        self.train_size = train_size
        self.input_time_length = input_time_length
        self.n_preds_per_input = n_preds_per_input

    def __call__(self, dataset, y, **kwargs):
        # can we directly use this https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
        # or stick to same API
        if isinstance(self.train_size, int):
            n_train_samples = self.train_size
        else:
            n_train_samples = int(self.train_size * len(dataset))
        X, y = dataset.X, dataset.y
        return (
            CroppedXyDataset(
                X[:n_train_samples],
                y[:n_train_samples],
                input_time_length=self.input_time_length,
                n_preds_per_input=self.n_preds_per_input,
            ),
            CroppedXyDataset(
                X[n_train_samples:],
                y[n_train_samples:],
                input_time_length=self.input_time_length,
                n_preds_per_input=self.n_preds_per_input,
            ),
        )
