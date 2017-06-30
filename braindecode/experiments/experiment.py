from collections import OrderedDict
import logging
from copy import deepcopy
import pandas as pd
import numpy as np

from braindecode.splitters import concatenate_sets
from braindecode.experiments.stopcriteria import MaxEpochs, ColumnBelow, Or
from braindecode.torchext.util import np_to_var, set_random_seeds

log = logging.getLogger(__name__)


class RememberBest(object):
    def __init__(self, column_name):
        self.column_name = column_name
        self.best_epoch = 0
        self.lowest_val = float('inf')
        self.model_state_dict = None
        self.optimizer_state_dict = None

    def remember_epoch(self, epochs_df, model, optimizer):
        i_epoch = len(epochs_df) - 1
        current_val = float(epochs_df[self.column_name].iloc[-1])
        if current_val <= self.lowest_val:
            self.best_epoch = i_epoch
            self.lowest_val = current_val
            self.model_state_dict = deepcopy(model.state_dict())
            self.optimizer_state_dict = deepcopy(optimizer.state_dict())
            log.info("New best {:s}: {:5f}".format(self.column_name,
                                                   current_val))
            log.info("")

    def reset_to_best_model(self, epochs_df, model, optimizer):
        # Remove epochs past the best one from epochs dataframe
        epochs_df.drop(range(self.best_epoch+1, len(epochs_df)), inplace=True)
        model.load_state_dict(self.model_state_dict)
        optimizer.load_state_dict(self.optimizer_state_dict)


class Experiment(object):
    def __init__(self, model, train_set, valid_set, test_set,
                 iterator, loss_function, optimizer, model_constraint,
                 monitors, stop_criterion, remember_best_column,
                 run_after_early_stop,
                 batch_modifier=None, cuda=True):
        self.model = model
        self.datasets = OrderedDict(
            (('train', train_set), ('valid', valid_set), ('test', test_set)))
        self.iterator = iterator
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.model_constraint = model_constraint
        self.monitors = monitors
        self.stop_criterion = stop_criterion
        self.remember_best_column = remember_best_column
        self.run_after_early_stop = run_after_early_stop
        self.batch_modifier = batch_modifier
        self.cuda = cuda
        self.epochs_df = pd.DataFrame()
        self.before_stop_df = None
        self.rememberer = None

    def run(self):
        self.setup_training()
        log.info("Run until first stop...")
        self.run_until_early_stop()
        # always setup for second stop, in order to get best model
        # even if not running after early stop...
        log.info("Setup for second stop...")
        self.setup_after_stop_training()
        if self.run_after_early_stop:
            log.info("Run until second stop...")
            self.run_until_second_stop()

    def setup_training(self):
        # reset remember best extension in case you rerun some experiment
        self.rememberer = RememberBest(self.remember_best_column)
        self.epochs_df = pd.DataFrame()
        set_random_seeds(seed=2382938, cuda=self.cuda)
        if self.cuda:
            self.model.cuda()

    def run_until_early_stop(self):
        self.run_until_stop(self.datasets, remember_best=True)

    def run_until_second_stop(self):
        datasets = self.datasets
        datasets['train'] = concatenate_sets(datasets['train'],
                                             datasets['valid'])

        # Todo: actually keep remembering and in case of twice number of epochs
        # reset to best model again (check if valid loss not below train loss)
        self.run_until_stop(datasets, remember_best=False)

    def run_until_stop(self, datasets, remember_best):
        self.monitor_epoch(datasets)
        self.print_epoch()
        if remember_best:
            self.rememberer.remember_epoch(self.epochs_df, self.model,
                                           self.optimizer)

        self.iterator.reset_rng()
        while not self.stop_criterion.should_stop(self.epochs_df):
            self.run_one_epoch(datasets, remember_best)

    def run_one_epoch(self, datasets, remember_best):
        batch_generator = self.iterator.get_batches(datasets['train'],
                                                    shuffle=True)
        # TODO, add timing again?
        for inputs, targets in batch_generator:
            if self.batch_modifier is not None:
                inputs, targets = self.batch_modifier.process(inputs,
                                                              targets)
            # could happen that batch modifier has removed all inputs...
            if len(inputs) > 0:
                self.train_batch(inputs, targets)

        self.monitor_epoch(datasets)
        self.print_epoch()
        if remember_best:
            self.rememberer.remember_epoch(self.epochs_df, self.model,
                                           self.optimizer)

    def monitor_epoch(self, datasets):
        result_dicts_per_monitor = OrderedDict()
        for m in self.monitors:
            result_dicts_per_monitor[m] = OrderedDict()
        for m in self.monitors:
            result_dict = m.monitor_epoch()
            if result_dict is not None:
                result_dicts_per_monitor[m].update(result_dict)
        for setname in datasets:
            assert setname in ['train', 'valid', 'test']
            dataset = datasets[setname]
            all_preds = []
            all_losses = []
            all_batch_sizes = []
            all_targets = []
            for batch in self.iterator.get_batches(dataset, shuffle=False):
                preds, loss = self.eval_on_batch(batch[0], batch[1])
                all_preds.append(preds)
                all_losses.append(loss)
                all_batch_sizes.append(len(batch[0]))
                all_targets.append(batch[1])

            for m in self.monitors:
                result_dict = m.monitor_set(setname, all_preds, all_losses,
                                            all_batch_sizes, all_targets,
                                            dataset)
                if result_dict is not None:
                    result_dicts_per_monitor[m].update(result_dict)
        row_dict = OrderedDict()
        for m in self.monitors:
            row_dict.update(result_dicts_per_monitor[m])
        self.epochs_df = self.epochs_df.append(row_dict, ignore_index=True)
        assert set(self.epochs_df.columns) == set(row_dict.keys())
        self.epochs_df = self.epochs_df[list(row_dict.keys())]

    def print_epoch(self):
        # -1 due to doing one monitor at start of training
        i_epoch = len(self.epochs_df) - 1
        log.info("Epoch {:d}".format(i_epoch))
        last_row = self.epochs_df.iloc[-1]
        for key, val in last_row.iteritems():
            log.info("{:25s} {:.5f}".format(key, val))
        log.info("")

    def eval_on_batch(self, inputs, targets):
        self.model.eval()
        input_vars = np_to_var(inputs)
        target_vars = np_to_var(targets)
        if self.cuda:
            input_vars = input_vars.cuda()
            target_vars = target_vars.cuda()
        outputs = self.model(input_vars)
        loss = self.loss_function(outputs, target_vars)
        outputs = outputs.cpu().data.numpy()
        loss = loss.cpu().data.numpy()
        return outputs, loss

    def train_batch(self, inputs, targets):
        self.model.train()
        input_vars = np_to_var(inputs)
        target_vars = np_to_var(targets)
        if self.cuda:
            input_vars = input_vars.cuda()
            target_vars = target_vars.cuda()
        self.optimizer.zero_grad()
        outputs = self.model(input_vars)
        loss = self.loss_function(outputs, target_vars)
        loss.backward()
        self.optimizer.step()
        if self.model_constraint is not None:
            self.model_constraint.apply(self.model)

    def setup_after_stop_training(self):
        # also remember old monitor chans, will be put back into
        # monitor chans after experiment finished
        self.before_stop_df = deepcopy(self.epochs_df)
        self.rememberer.reset_to_best_model(self.epochs_df, self.model,
                                            self.optimizer)
        loss_to_reach = float(self.epochs_df['train_loss'].iloc[-1])
        self.stop_criterion = Or(stop_criteria=[
            MaxEpochs(max_epochs=self.rememberer.best_epoch * 2),
            ColumnBelow(column_name='valid_loss', target_value=loss_to_reach)])
        log.info("Train loss to reach {:.5f}".format(loss_to_reach))
