import torch as th
from numpy.random import RandomState
from braindecode.experiments.monitors import LossMonitor, MisclassMonitor, RuntimeMonitor
from braindecode.experiments.stopcriteria import MaxEpochs
from braindecode.datautil.iterators import BalancedBatchSizeIterator
from braindecode.experiments.experiment import Experiment
from braindecode.datautil.signal_target import SignalAndTarget


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
    def __init__(self):
        self.compiled = False

    def compile(self, loss, optimizer, metrics, seed=0):
        self.loss = loss
        self.network = self.create_network()
        if not hasattr(optimizer, 'step'):
            optimizer_class = find_optimizer(optimizer)
            optimizer = optimizer_class(self.network.parameters())
        self.optimizer = optimizer
        monitor_dict = {'acc': MisclassMonitor}
        self.monitors = [LossMonitor()]
        extra_monitors = [monitor_dict[m]() for m in metrics]
        self.monitors += extra_monitors
        self.monitors += [RuntimeMonitor()]
        self.seed_rng = RandomState(seed)
        self.compiled = True

    def fit(self, train_X, train_y, epochs, batch_size, validation_data=None, model_constraint=None,
            remember_best_column=None):
        if not self.compiled:
            raise ValueError("Compile the model first by calling model.compile(loss, optimizer, metrics")
        iterator = BalancedBatchSizeIterator(batch_size=batch_size, seed=self.seed_rng.randint(0, 4294967295))
        stop_criterion = MaxEpochs(epochs - 1)# -1 since we dont print 0 epoch, which matters for this stop criterion
        train_set = SignalAndTarget(train_X, train_y)
        if validation_data is not None:
            valid_set = SignalAndTarget(validation_data[0], validation_data[1])
        else:
            valid_set = None
        test_set = None
        exp = Experiment(self.network, train_set, valid_set, test_set, iterator=iterator,
                         loss_function=self.loss, optimizer=self.optimizer,
                         model_constraint=model_constraint,
                         monitors=self.monitors,
                         stop_criterion=stop_criterion,
                         remember_best_column=remember_best_column,
                         run_after_early_stop=False, cuda=True, print_0_epoch=False,
                         do_early_stop=(remember_best_column is not None))
        exp.run()
        return exp

    def evaluate(self, X,y, batch_size=32):
        # Create a dummy experiment for the evaluation
        iterator = BalancedBatchSizeIterator(batch_size=batch_size,
                                             seed=0) # seed irrelevant
        stop_criterion = MaxEpochs(0)
        train_set = SignalAndTarget(X, y)
        model_constraint = None
        valid_set = None
        test_set = None

        exp = Experiment(self.network, train_set, valid_set, test_set,
                         iterator=iterator,
                         loss_function=self.loss, optimizer=self.optimizer,
                         model_constraint=model_constraint,
                         monitors=self.monitors,
                         stop_criterion=stop_criterion,
                         remember_best_column=None,
                         run_after_early_stop=False, cuda=True,
                         print_0_epoch=False,
                         do_early_stop=False)

        exp.monitor_epoch({'train': train_set})

        result_dict = dict([(key.replace('train_', ''), val)
                            for key, val in
                            dict(exp.epochs_df.iloc[0]).items()])
        return result_dict