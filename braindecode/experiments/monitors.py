import numpy as np
import time


class MisclassMonitor(object):
    def __init__(self, exponentiate_preds=False, col_suffix='misclass'):
        self.col_suffix = col_suffix
        self.exponentiate_preds = exponentiate_preds

    def monitor_epoch(self, ):
        return

    def monitor_set(self, setname, all_preds, all_losses,
                    all_batch_sizes, all_targets, dataset):
        all_pred_labels = []
        all_target_labels = []
        for i_batch in range(len(all_batch_sizes)):
            preds = all_preds[i_batch]
            if True:
                preds = np.exp(preds)
            pred_labels = np.argmax(preds, axis=1)
            all_pred_labels.extend(pred_labels)
            targets = all_targets[i_batch]
            # targets may be one-hot-encoded or not
            if targets.ndim >= pred_labels.ndim:
                targets = np.argmax(targets, axis=1)
            assert targets.shape == pred_labels.shape
            all_target_labels.extend(targets)
        all_pred_labels = np.array(all_pred_labels)
        all_target_labels = np.array(all_target_labels)
        assert all_pred_labels.shape == all_target_labels.shape

        misclass = 1 - np.mean(all_target_labels == all_pred_labels)
        column_name = "{:s}_{:s}".format(setname, self.col_suffix)
        return {column_name: float(misclass)}


class LossMonitor(object):
    def monitor_epoch(self,):
        return

    def monitor_set(self, setname, all_preds, all_losses,
                    all_batch_sizes, all_targets, dataset):
        batch_weights = np.array(all_batch_sizes)/ np.sum(all_batch_sizes)
        loss_per_batch = [np.mean(loss) for loss in all_losses]
        mean_loss = np.sum(batch_weights * loss_per_batch)
        column_name = "{:s}_loss".format(setname)
        return {column_name: mean_loss}


class CroppedTrialMisclassMonitor(object):
    def __init__(self, input_time_length=None):
        self.input_time_length = input_time_length

    def monitor_epoch(self,):
        return

    def monitor_set(self, setname, all_preds, all_losses,
                    all_batch_sizes, all_targets, dataset):
        """Assuming one hot encoding for now"""
        assert self.input_time_length is not None, "Need to know input time length..."
        all_pred_labels = self.compute_pred_labels(dataset, all_preds)
        assert all_pred_labels.shape == dataset.y.shape
        misclass = 1 - np.mean(all_pred_labels == dataset.y)
        column_name = "{:s}_misclass".format(setname)
        return {column_name: float(misclass)}

    def compute_pred_labels(self, dataset, all_preds,):
        n_preds_per_input = all_preds[0].shape[2]
        n_receptive_field = self.input_time_length - n_preds_per_input + 1

        n_preds_per_trial = [trial.shape[1] - n_receptive_field + 1
                        for trial in dataset.X]
        preds_per_trial = compute_preds_per_trial_from_n_preds_per_trial(
            all_preds, n_preds_per_trial)
        all_pred_labels = [np.argmax(np.mean(p, axis=1))
                           for p in preds_per_trial]

        all_pred_labels = np.array(all_pred_labels)
        assert all_pred_labels.shape == dataset.y.shape
        return all_pred_labels


def compute_preds_per_trial_for_set(all_preds, input_time_length,
                                        dataset, ):
        n_preds_per_input = all_preds[0].shape[2]
        n_receptive_field = input_time_length - n_preds_per_input + 1
        n_preds_per_trial = [trial.shape[1] - n_receptive_field + 1
                             for trial in dataset.X]
        preds_per_trial = compute_preds_per_trial_from_n_preds_per_trial(
            all_preds, n_preds_per_trial)
        return preds_per_trial


def compute_preds_per_trial_from_n_preds_per_trial(
        all_preds, n_preds_per_trial):
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
            pred_samples = all_preds_arr[i_pred_block,:,
                           -n_needed_preds:]
            preds_this_trial.append(pred_samples)
            n_needed_preds -= pred_samples.shape[1]
            i_pred_block += 1

        preds_this_trial = np.concatenate(preds_this_trial, axis=1)
        preds_per_trial.append(preds_this_trial)
    assert i_pred_block == len(all_preds_arr), (
        "Expect that all prediction forward passes are needed, "
        "used {:d}, existing {:d}".format(
            i_pred_block, len(all_preds_arr)))
    return preds_per_trial


class RuntimeMonitor(object):
    def __init__(self):
        self.last_call_time = None

    def monitor_epoch(self,):
        cur_time = time.time()
        if self.last_call_time is None:
            # just in case of first call
            self.last_call_time = cur_time
        epoch_runtime = cur_time - self.last_call_time
        self.last_call_time = cur_time
        return {'runtime': epoch_runtime}

    def monitor_set(self, setname, all_preds, all_losses,
                    all_batch_sizes, all_targets, dataset):
        return {}
