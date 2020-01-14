"""
Blah blah blah
==============

"""

import logging
import os.path
import time
from collections import OrderedDict
import sys

import numpy as np
import torch.nn.functional as F
from torch import optim
import torch as th

from braindecode.models.deep4 import Deep4Net
from braindecode.models.util import to_dense_prediction_model
from braindecode.datasets.bcic_iv_2a import BCICompetition4Set2A
from braindecode.experiments.experiment import Experiment
from braindecode.experiments.monitors import LossMonitor, MisclassMonitor, \
    RuntimeMonitor, CroppedTrialMisclassMonitor
from braindecode.experiments.stopcriteria import MaxEpochs, NoDecrease, Or
from braindecode.datautil.iterators import CropsFromTrialsIterator
from braindecode.models.shallow_fbcsp import ShallowFBCSPNet
from braindecode.datautil.splitters import split_into_two_sets
from braindecode.torch_ext.constraints import MaxNormDefaultConstraint
from braindecode.torch_ext.util import set_random_seeds, np_to_var
from braindecode.mne_ext.signalproc import mne_apply
from braindecode.datautil.signalproc import (bandpass_cnt,
                                             exponential_running_standardize)
from braindecode.datautil.trial_segment import create_signal_target_from_raw_mne

# moabb specific
from mne.io.base import concatenate_raws
from mne import find_events
# moabb BCI competition IV 2a dataset
# requires global 'MOABB_TOOLBOX' variabel path to moabb toolbox
try:
    sys.path.append(os.environ['MOABB_TOOLBOX_PY'])
    from moabb.datasets import BNCI2014001
except ImportError:  # moabb not found
    pass
except ModuleNotFoundError:  # coloredlogs or yaml not found
    pass


log = logging.getLogger(__name__)


def run_exp(subject_id, low_cut_hz, model, cuda):
    ival = [-500, 4000]
    input_time_length = 1000
    max_epochs = 800
    max_increase_epochs = 80
    batch_size = 60
    high_cut_hz = 38
    factor_new = 1e-3
    init_block_size = 1000
    valid_set_fraction = 0.2


    # load data from moabb
    data = BNCI2014001().get_data([subject_id])[subject_id]

    train_cnt = [data['session_T'][run] for run in data['session_T']]
    train_cnt = concatenate_raws(train_cnt)
    train_cnt.info['events'] = find_events(train_cnt)

    test_cnt = [data['session_E'][run] for run in data['session_E']]
    test_cnt = concatenate_raws(test_cnt)
    test_cnt.info['events'] = find_events(test_cnt)

    # Preprocessing

    train_cnt = train_cnt.drop_channels(['EOG1', 'EOG2', 'EOG3', 'stim'])
    assert len(train_cnt.ch_names) == 22
    # lets convert to millvolt for numerical stability of next operations
    train_cnt = mne_apply(lambda a: a * 1e6, train_cnt)
    train_cnt = mne_apply(
        lambda a: bandpass_cnt(a, low_cut_hz, high_cut_hz, train_cnt.info['sfreq'],
                               filt_order=3,
                               axis=1), train_cnt)
    train_cnt = mne_apply(
        lambda a: exponential_running_standardize(a.T, factor_new=factor_new,
                                                  init_block_size=init_block_size,
                                                  eps=1e-4).T,
        train_cnt)

    test_cnt = test_cnt.drop_channels(['EOG1', 'EOG2', 'EOG3', 'stim'])
    assert len(test_cnt.ch_names) == 22
    test_cnt = mne_apply(lambda a: a * 1e6, test_cnt)
    test_cnt = mne_apply(
        lambda a: bandpass_cnt(a, low_cut_hz, high_cut_hz, test_cnt.info['sfreq'],
                               filt_order=3,
                               axis=1), test_cnt)
    test_cnt = mne_apply(
        lambda a: exponential_running_standardize(a.T, factor_new=factor_new,
                                                  init_block_size=init_block_size,
                                                  eps=1e-4).T,
        test_cnt)

    marker_def = OrderedDict([('Left Hand', [1]), ('Right Hand', [2],),
                              ('Foot', [3]), ('Tongue', [4])])


    train_set = create_signal_target_from_raw_mne(train_cnt, marker_def, ival)
    test_set = create_signal_target_from_raw_mne(test_cnt, marker_def, ival)

    train_set, valid_set = split_into_two_sets(
        train_set, first_set_fraction=1-valid_set_fraction)

    set_random_seeds(seed=20190706, cuda=cuda)

    n_classes = 4
    n_chans = int(train_set.X.shape[1])
    if model == 'shallow':
        model = ShallowFBCSPNet(n_chans, n_classes, input_time_length=input_time_length,
                            final_conv_length=30).create_network()
    elif model == 'deep':
        model = Deep4Net(n_chans, n_classes, input_time_length=input_time_length,
                            final_conv_length=2).create_network()


    to_dense_prediction_model(model)
    if cuda:
        model.cuda()

    log.info("Model: \n{:s}".format(str(model)))
    dummy_input = np_to_var(train_set.X[:1, :, :, None])
    if cuda:
        dummy_input = dummy_input.cuda()
    out = model(dummy_input)

    n_preds_per_input = out.cpu().data.numpy().shape[2]

    optimizer = optim.Adam(model.parameters())

    iterator = CropsFromTrialsIterator(batch_size=batch_size,
                                       input_time_length=input_time_length,
                                       n_preds_per_input=n_preds_per_input)

    stop_criterion = Or([MaxEpochs(max_epochs),
                         NoDecrease('valid_misclass', max_increase_epochs)])

    monitors = [LossMonitor(), MisclassMonitor(col_suffix='sample_misclass'),
                CroppedTrialMisclassMonitor(
                    input_time_length=input_time_length), RuntimeMonitor()]

    model_constraint = MaxNormDefaultConstraint()

    loss_function = lambda preds, targets: F.nll_loss(
        th.mean(preds, dim=2, keepdim=False), targets)

    exp = Experiment(model, train_set, valid_set, test_set, iterator=iterator,
                     loss_function=loss_function, optimizer=optimizer,
                     model_constraint=model_constraint,
                     monitors=monitors,
                     stop_criterion=stop_criterion,
                     remember_best_column='valid_misclass',
                     run_after_early_stop=True, cuda=cuda)
    exp.run()
    return exp

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s %(levelname)s : %(message)s',
                        level=logging.DEBUG, stream=sys.stdout)
    subject_id = 2 # 1-9
    low_cut_hz = 4 # 0 or 4
    model = 'shallow' #'shallow' or 'deep'
    cuda = True
    exp = run_exp(subject_id, low_cut_hz, model, cuda)
    log.info("Last 10 epochs")
    log.info("\n" + str(exp.epochs_df.iloc[-10:]))
