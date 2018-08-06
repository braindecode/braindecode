import time

import numpy as np
from numpy.random import RandomState
import torch as th

from braindecode.experiments.monitors import LossMonitor, MisclassMonitor, \
    RuntimeMonitor, CroppedTrialMisclassMonitor, \
    compute_trial_labels_from_crop_preds, compute_pred_labels_from_trial_preds
from braindecode.experiments.stopcriteria import MaxEpochs
from braindecode.datautil.iterators import BalancedBatchSizeIterator, \
    CropsFromTrialsIterator
from braindecode.experiments.experiment import Experiment
from braindecode.datautil.signal_target import SignalAndTarget
from braindecode.models.util import to_dense_prediction_model
from braindecode.torch_ext.schedulers import CosineAnnealing, ScheduledOptimizer
from braindecode.torch_ext.util import np_to_var, var_to_np


def find_optimizer(optimizer_name):
    optim_found = False
    for name in th.optim.__dict__.keys():
        if name.lower() == optimizer_name.lower():
            optimizer = th.optim.__dict__[name]
            optim_found = True
            break
    if not optim_found:
        raise ValueError("Unknown optimizer {:s}".format(optimizer))
    return \
        optimizer


class BaseModel(object):
    def cuda(self):
        self._ensure_network_exists()
        assert not self.compiled,\
            ("Call cuda before compiling model, otherwise optimization will not work")
        self.network = self.network.cuda()
        self.cuda = True
        return self

    def parameters(self):
        self._ensure_network_exists()
        return self.network.parameters()

    def _ensure_network_exists(self):
        if not hasattr(self, 'network'):
            self.network = self.create_network()
            self.cuda = False
            self.compiled = False

    def compile(self, loss, optimizer, metrics,  cropped=False, seed=0):
        self.loss = loss
        self._ensure_network_exists()
        if cropped:
            to_dense_prediction_model(self.network)
        if not hasattr(optimizer, 'step'):
            optimizer_class = find_optimizer(optimizer)
            optimizer = optimizer_class(self.network.parameters())
        self.optimizer = optimizer
        self.metrics = metrics
        self.seed_rng = RandomState(seed)
        self.cropped = cropped
        self.compiled = True

    def fit(self, train_X, train_y, epochs, batch_size, input_time_length=None,
            validation_data=None, model_constraint=None,
            remember_best_column=None, scheduler=None):
        """
        
        
        Parameters
        ----------
        train_X
        train_y
        epochs
        batch_size
        input_time_length
        validation_data
        model_constraint
        remember_best_column
        scheduler

        Returns
        -------

        """
        if not self.compiled:
            raise ValueError("Compile the model first by calling model.compile(loss, optimizer, metrics)")


        if self.cropped and input_time_length is None:
            raise ValueError("In cropped mode, need to specify input_time_length,"
                             "which is the number of timesteps that will be pushed through"
                             "the network in a single pass.")
        if self.cropped:
            test_input = np_to_var(train_X[0:1], dtype=np.float32)
            while len(test_input.size()) < 4:
                test_input = test_input.unsqueeze(-1)
            if self.cuda:
                    test_input = test_input.cuda()
            out = self.network(test_input)
            n_preds_per_input = out.cpu().data.numpy().shape[2]
            self.iterator = CropsFromTrialsIterator(
                batch_size=batch_size, input_time_length=input_time_length,
                n_preds_per_input=n_preds_per_input,
                seed=self.seed_rng.randint(0, 4294967295))
        else:
            self.iterator = BalancedBatchSizeIterator(batch_size=batch_size, seed=self.seed_rng.randint(0, 4294967295))
        stop_criterion = MaxEpochs(epochs - 1)# -1 since we dont print 0 epoch, which matters for this stop criterion
        train_set = SignalAndTarget(train_X, train_y)
        optimizer = self.optimizer
        if scheduler is not None:
            assert scheduler == 'cosine'
            n_updates_per_epoch = sum(
                [1 for _ in self.iterator.get_batches(train_set, shuffle=True)])
            n_updates_per_period = n_updates_per_epoch * epochs
            if scheduler == 'cosine':
                scheduler = CosineAnnealing(n_updates_per_period)
            schedule_weight_decay = False
            if optimizer.__class__.__name__ == 'AdamW':
                schedule_weight_decay = True
            optimizer = ScheduledOptimizer(scheduler, self.optimizer,
                                           schedule_weight_decay=schedule_weight_decay)
        loss_function = self.loss
        if self.cropped:
            loss_function = lambda outputs, targets:\
                self.loss(th.mean(outputs, dim=2), targets)
        if validation_data is not None:
            valid_set = SignalAndTarget(validation_data[0], validation_data[1])
        else:
            valid_set = None
        test_set = None
        if self.cropped:
            monitor_dict = {'acc': lambda :
            CroppedTrialMisclassMonitor(input_time_length)}
        else:
            monitor_dict = {'acc': MisclassMonitor}
        self.monitors = [LossMonitor()]
        extra_monitors = [monitor_dict[m]() for m in self.metrics]
        self.monitors += extra_monitors
        self.monitors += [RuntimeMonitor()]
        exp = Experiment(self.network, train_set, valid_set, test_set,
                         iterator=self.iterator,
                         loss_function=loss_function, optimizer=optimizer,
                         model_constraint=model_constraint,
                         monitors=self.monitors,
                         stop_criterion=stop_criterion,
                         remember_best_column=remember_best_column,
                         run_after_early_stop=False, cuda=self.cuda, log_0_epoch=False,
                         do_early_stop=(remember_best_column is not None))
        exp.run()
        self.epochs_df = exp.epochs_df
        return exp

    def evaluate(self, X,y):
        stop_criterion = MaxEpochs(0)
        train_set = SignalAndTarget(X, y)
        model_constraint = None
        valid_set = None
        test_set = None
        loss_function = self.loss
        if self.cropped:
            loss_function = lambda outputs, targets: \
                self.loss(th.mean(outputs, dim=2), targets)

        # reset runtime monitor if exists...
        for monitor in self.monitors:
            if hasattr(monitor, 'last_call_time'):
                monitor.last_call_time = time.time()
        exp = Experiment(self.network, train_set, valid_set, test_set,
                         iterator=self.iterator,
                         loss_function=loss_function, optimizer=self.optimizer,
                         model_constraint=model_constraint,
                         monitors=self.monitors,
                         stop_criterion=stop_criterion,
                         remember_best_column=None,
                         run_after_early_stop=False, cuda=self.cuda,
                         log_0_epoch=False,
                         do_early_stop=False)

        exp.monitor_epoch({'train': train_set})

        result_dict = dict([(key.replace('train_', ''), val)
                            for key, val in
                            dict(exp.epochs_df.iloc[0]).items()])
        return result_dict

    def predict(self, X, threshold_for_binary_case=None):
        all_preds = []
        for b_X, _ in self.iterator.get_batches(SignalAndTarget(X, X), False):
            all_preds.append(var_to_np(self.network(np_to_var(b_X))))
        if self.cropped:
            pred_labels = compute_trial_labels_from_crop_preds(
                all_preds, self.iterator.input_time_length, X)
        else:
            pred_labels = compute_pred_labels_from_trial_preds(
                all_preds, threshold_for_binary_case)
        return pred_labels
