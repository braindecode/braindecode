import numpy as np
import time


class MisclassMonitor(object):
    """
    Monitor the examplewise misclassification rate.
    
    Parameters
    ----------
    col_suffix: str, optional
        Name of the column in the monitoring output.
    threshold_for_binary_case: bool, optional
        In case of binary classification with only one output prediction
        per target, define the threshold for separating the classes, i.e.
        0.5 for sigmoid outputs, or np.log(0.5) for log sigmoid outputs
    """

    def __init__(self, col_suffix="misclass", threshold_for_binary_case=None):
        self.col_suffix = col_suffix
        self.threshold_for_binary_case = threshold_for_binary_case

    def monitor_epoch(self,):
        return

    def monitor_set(
        self, setname, all_preds, all_losses, all_batch_sizes, all_targets, dataset
    ):
        all_pred_labels = []
        all_target_labels = []
        for i_batch in range(len(all_batch_sizes)):
            preds = all_preds[i_batch]
            # preds could be examples x classes x time
            # or just
            # examples x classes
            # make sure not to remove first dimension if it only has size one
            if preds.ndim > 1:
                only_one_row = preds.shape[0] == 1

                pred_labels = np.argmax(preds, axis=1).squeeze()
                # add first dimension again if needed
                if only_one_row:
                    pred_labels = pred_labels[None]
            else:
                assert self.threshold_for_binary_case is not None, (
                    "In case of only one output, please supply the "
                    "threshold_for_binary_case parameter"
                )
                # binary classification case... assume logits
                pred_labels = np.int32(preds > self.threshold_for_binary_case)
            # now examples x time or examples
            all_pred_labels.extend(pred_labels)
            targets = all_targets[i_batch]
            if targets.ndim > pred_labels.ndim:
                # targets may be one-hot-encoded
                targets = np.argmax(targets, axis=1)
            elif targets.ndim < pred_labels.ndim:
                # targets may not have time dimension,
                # in that case just repeat targets on time dimension
                extra_dim = pred_labels.ndim - 1
                targets = np.repeat(
                    np.expand_dims(targets, extra_dim),
                    pred_labels.shape[extra_dim],
                    extra_dim,
                )
            assert targets.shape == pred_labels.shape
            all_target_labels.extend(targets)
        all_pred_labels = np.array(all_pred_labels)
        all_target_labels = np.array(all_target_labels)
        assert all_pred_labels.shape == all_target_labels.shape

        misclass = 1 - np.mean(all_target_labels == all_pred_labels)
        column_name = "{:s}_{:s}".format(setname, self.col_suffix)
        return {column_name: float(misclass)}


def compute_pred_labels_from_trial_preds(all_preds, threshold_for_binary_case=None):
    all_pred_labels = []
    for i_batch in range(len(all_preds)):
        preds = all_preds[i_batch]
        # preds could be examples x classes x time
        # or just
        # examples x classes
        # make sure not to remove first dimension if it only has size one
        if preds.ndim > 1:
            only_one_row = preds.shape[0] == 1

            pred_labels = np.argmax(preds, axis=1).squeeze()
            # add first dimension again if needed
            if only_one_row:
                pred_labels = pred_labels[None]
        else:
            assert threshold_for_binary_case is not None, (
                "In case of only one output, please supply the "
                "threshold_for_binary_case parameter"
            )
            # binary classification case... assume logits
            pred_labels = np.int32(preds > threshold_for_binary_case)
        # now examples x time or examples
        all_pred_labels.extend(pred_labels)
    all_pred_labels = np.array(all_pred_labels)
    return all_pred_labels


class AveragePerClassMisclassMonitor(object):
    """
    Compute average of misclasses per class,
    useful if classes are highly imbalanced.
    
    Parameters
    ----------
    col_suffix: str
        Name of the column in the monitoring output.
    """

    def __init__(self, col_suffix="misclass"):
        self.col_suffix = col_suffix

    def monitor_epoch(self,):
        return

    def monitor_set(
        self, setname, all_preds, all_losses, all_batch_sizes, all_targets, dataset
    ):
        all_pred_labels = []
        all_target_labels = []
        for i_batch in range(len(all_batch_sizes)):
            preds = all_preds[i_batch]
            # preds could be examples x classes x time
            # or just
            # examples x classes
            # make sure not to remove first dimension if it only has size one
            only_one_row = preds.shape[0] == 1
            n_classes = preds.shape[1]
            pred_labels = np.argmax(preds, axis=1).squeeze()
            # add first dimension again if needed
            if only_one_row:
                pred_labels = pred_labels[None]
            # now examples x time or examples
            all_pred_labels.extend(pred_labels)
            targets = all_targets[i_batch]
            if targets.ndim > pred_labels.ndim:
                # targets may be one-hot-encoded
                targets = np.argmax(targets, axis=1)
            elif targets.ndim < pred_labels.ndim:
                # targets may not have time dimension,
                # in that case just repeat targets on time dimension
                extra_dim = pred_labels.ndim - 1
                targets = np.repeat(
                    np.expand_dims(targets, extra_dim),
                    pred_labels.shape[extra_dim],
                    extra_dim,
                )
            assert targets.shape == pred_labels.shape
            all_target_labels.extend(targets)
        all_pred_labels = np.array(all_pred_labels)
        all_target_labels = np.array(all_target_labels)
        assert all_pred_labels.shape == all_target_labels.shape
        acc_per_class = []
        for i_class in range(n_classes):
            mask = all_target_labels == i_class
            acc = np.mean(all_pred_labels[mask] == all_target_labels[mask])
            acc_per_class.append(acc)
        misclass = 1 - np.mean(acc_per_class)
        column_name = "{:s}_{:s}".format(setname, self.col_suffix)
        return {column_name: float(misclass)}


class LossMonitor(object):
    """
    Monitor the examplewise loss.
    """

    def monitor_epoch(self,):
        return

    def monitor_set(
        self, setname, all_preds, all_losses, all_batch_sizes, all_targets, dataset
    ):
        batch_weights = np.array(all_batch_sizes) / float(np.sum(all_batch_sizes))
        loss_per_batch = [np.mean(loss) for loss in all_losses]
        mean_loss = np.sum(batch_weights * loss_per_batch)
        column_name = "{:s}_loss".format(setname)
        return {column_name: mean_loss}


class CroppedTrialMisclassMonitor(object):
    """
    Compute trialwise misclasses from predictions for crops.
    
    Parameters
    ----------
    input_time_length: int
        Temporal length of one input to the model.
    """

    def __init__(self, input_time_length=None):
        self.input_time_length = input_time_length

    def monitor_epoch(self,):
        return

    def monitor_set(
        self, setname, all_preds, all_losses, all_batch_sizes, all_targets, dataset
    ):
        """Assuming one hot encoding for now"""
        assert self.input_time_length is not None, "Need to know input time length..."
        # First case that each trial only has a single label
        if not hasattr(dataset.y[0], "__len__"):
            all_pred_labels = compute_trial_labels_from_crop_preds(
                all_preds, self.input_time_length, dataset.X
            )
            assert all_pred_labels.shape == dataset.y.shape
            all_trial_labels = dataset.y
        else:
            all_trial_labels, all_pred_labels = self._compute_trial_pred_labels_from_cnt_y(
                dataset, all_preds
            )
        assert all_pred_labels.shape == all_trial_labels.shape
        misclass = 1 - np.mean(all_pred_labels == all_trial_labels)
        column_name = "{:s}_misclass".format(setname)
        return {column_name: float(misclass)}

    def _compute_pred_labels(self, dataset, all_preds):
        preds_per_trial = compute_preds_per_trial_from_crops(
            all_preds, self.input_time_length, dataset.X
        )
        all_pred_labels = [np.argmax(np.mean(p, axis=1)) for p in preds_per_trial]
        all_pred_labels = np.array(all_pred_labels)
        assert all_pred_labels.shape == dataset.y.shape
        return all_pred_labels

    def _compute_trial_pred_labels_from_cnt_y(self, dataset, all_preds):
        # Todo: please test this
        # we only want the preds that are for the same labels as the last label in y
        # (there might be parts of other class-data at start, for trialwise misclass we assume
        # they are contained in other trials at the end...)
        preds_per_trial = compute_preds_per_trial_from_crops(
            all_preds, self.input_time_length, dataset.X
        )
        trial_labels = []
        trial_pred_labels = []
        for trial_pred, trial_y in zip(preds_per_trial, dataset.y):
            # first cut to the part actually having predictions
            trial_y = trial_y[-trial_pred.shape[1] :]
            wanted_class = trial_y[-1]
            trial_labels.append(wanted_class)
            # extract the first marker different from the wanted class
            # by starting from the back of the trial
            i_last_sample = np.flatnonzero(trial_y[::-1] != wanted_class)
            if len(i_last_sample) > 0:
                i_last_sample = i_last_sample[0]
                # remember last sample is now from back
                trial_pred = trial_pred[:, -i_last_sample:]
            trial_pred_label = np.argmax(np.mean(trial_pred, axis=1))
            trial_pred_labels.append(trial_pred_label)
        trial_labels = np.array(trial_labels)
        trial_pred_labels = np.array(trial_pred_labels)
        return trial_labels, trial_pred_labels


def compute_trial_labels_from_crop_preds(all_preds, input_time_length, X):
    """
    Compute predicted trial labels from arrays of crop predictions

    Parameters
    ----------
    all_preds: list of 2darrays (classes x time)
        All predictions for the crops.
    input_time_length: int
        Temporal length of one input to the model.
    X: ndarray
        Input tensor the crops were taken from.

    Returns
    -------
    pred_labels_per_trial: 1darray
        Predicted label for each trial.

    """

    preds_per_trial = compute_preds_per_trial_from_crops(
        all_preds, input_time_length, X
    )
    pred_labels_per_trial = [np.argmax(np.mean(p, axis=1)) for p in preds_per_trial]
    pred_labels_per_trial = np.array(pred_labels_per_trial)
    return pred_labels_per_trial


def compute_preds_per_trial_from_crops(all_preds, input_time_length, X):
    """
    Compute predictions per trial from predictions for crops.
    
    Parameters
    ----------
    all_preds: list of 2darrays (classes x time)
        All predictions for the crops. 
    input_time_length: int
        Temporal length of one input to the model.
    X: ndarray
        Input tensor the crops were taken from.
    
    Returns
    -------
    preds_per_trial: list of 2darrays (classes x time)
        Predictions for each trial, without overlapping predictions.

    """
    n_preds_per_input = all_preds[0].shape[2]
    n_receptive_field = input_time_length - n_preds_per_input + 1
    n_preds_per_trial = [trial.shape[1] - n_receptive_field + 1 for trial in X]
    preds_per_trial = compute_preds_per_trial_from_n_preds_per_trial(
        all_preds, n_preds_per_trial
    )
    return preds_per_trial


def compute_preds_per_trial_from_n_preds_per_trial(all_preds, n_preds_per_trial):
    """
    Compute predictions per trial from predictions for crops.

    Parameters
    ----------
    all_preds: list of 2darrays (classes x time)
        All predictions for the crops. 
    input_time_length: int
        Temporal length of one input to the model.
    n_preds_per_trial: list of int
        Number of predictions for each trial.
    Returns
    -------
    preds_per_trial: list of 2darrays (classes x time)
        Predictions for each trial, without overlapping predictions.

    """
    # all_preds_arr has shape forward_passes x classes x time
    all_preds_arr = np.concatenate(all_preds, axis=0)
    preds_per_trial = []
    i_pred_block = 0
    for i_trial in range(len(n_preds_per_trial)):
        n_needed_preds = n_preds_per_trial[i_trial]
        preds_this_trial = []
        while n_needed_preds > 0:
            # - n_needed_preds: only has an effect
            # in case there are more samples than we actually still need
            # in the block.
            # That can happen since final block of a trial can overlap
            # with block before so we can have some redundant preds.
            pred_samples = all_preds_arr[i_pred_block, :, -n_needed_preds:]
            preds_this_trial.append(pred_samples)
            n_needed_preds -= pred_samples.shape[1]
            i_pred_block += 1

        preds_this_trial = np.concatenate(preds_this_trial, axis=1)
        preds_per_trial.append(preds_this_trial)
    assert i_pred_block == len(all_preds_arr), (
        "Expect that all prediction forward passes are needed, "
        "used {:d}, existing {:d}".format(i_pred_block, len(all_preds_arr))
    )
    return preds_per_trial


class RuntimeMonitor(object):
    """
    Monitor the runtime of each epoch.
    
    First epoch will have runtime 0.
    """

    def __init__(self):
        self.last_call_time = None

    def monitor_epoch(self,):
        cur_time = time.time()
        if self.last_call_time is None:
            # just in case of first call
            self.last_call_time = cur_time
        epoch_runtime = cur_time - self.last_call_time
        self.last_call_time = cur_time
        return {"runtime": epoch_runtime}

    def monitor_set(
        self, setname, all_preds, all_losses, all_batch_sizes, all_targets, dataset
    ):
        return {}
