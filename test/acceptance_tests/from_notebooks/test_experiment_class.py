def test_experiment_class():
    import mne
    from mne.io import concatenate_raws

    # 5,6,7,10,13,14 are codes for executed and imagined hands/feet
    subject_id = 1
    event_codes = [5, 6, 9, 10, 13, 14]

    # This will download the files if you don't have them yet,
    # and then return the paths to the files.
    physionet_paths = mne.datasets.eegbci.load_data(subject_id, event_codes)

    # Load each of the files
    parts = [mne.io.read_raw_edf(path, preload=True, stim_channel='auto',
                                 verbose='WARNING')
             for path in physionet_paths]

    # Concatenate them
    raw = concatenate_raws(parts)

    # Find the events in this dataset
    events, _ = mne.events_from_annotations(raw)

    # Use only EEG channels
    eeg_channel_inds = mne.pick_types(raw.info, meg=False, eeg=True, stim=False,
                                      eog=False,
                                      exclude='bads')

    # Extract trials, only using EEG channels
    epoched = mne.Epochs(raw, events, dict(hands=2, feet=3), tmin=1, tmax=4.1,
                         proj=False, picks=eeg_channel_inds,
                         baseline=None, preload=True)
    import numpy as np
    from braindecode.datautil.signal_target import SignalAndTarget
    from braindecode.datautil.splitters import split_into_two_sets
    # Convert data from volt to millivolt
    # Pytorch expects float32 for input and int64 for labels.
    X = (epoched.get_data() * 1e6).astype(np.float32)
    y = (epoched.events[:, 2] - 2).astype(np.int64)  # 2,3 -> 0,1

    train_set = SignalAndTarget(X[:60], y=y[:60])
    test_set = SignalAndTarget(X[60:], y=y[60:])

    train_set, valid_set = split_into_two_sets(train_set,
                                               first_set_fraction=0.8)
    from braindecode.models.shallow_fbcsp import ShallowFBCSPNet
    from torch import nn
    from braindecode.torch_ext.util import set_random_seeds
    from braindecode.models.util import to_dense_prediction_model

    # Set if you want to use GPU
    # You can also use torch.cuda.is_available() to determine if cuda is available on your machine.
    cuda = False
    set_random_seeds(seed=20170629, cuda=cuda)

    # This will determine how many crops are processed in parallel
    input_time_length = 450
    n_classes = 2
    in_chans = train_set.X.shape[1]
    # final_conv_length determines the size of the receptive field of the ConvNet
    model = ShallowFBCSPNet(in_chans=in_chans, n_classes=n_classes,
                            input_time_length=input_time_length,
                            final_conv_length=12).create_network()
    to_dense_prediction_model(model)

    if cuda:
        model.cuda()

    from torch import optim

    optimizer = optim.Adam(model.parameters())

    from braindecode.torch_ext.util import np_to_var
    # determine output size
    test_input = np_to_var(
        np.ones((2, in_chans, input_time_length, 1), dtype=np.float32))
    if cuda:
        test_input = test_input.cuda()
    out = model(test_input)
    n_preds_per_input = out.cpu().data.numpy().shape[2]
    print("{:d} predictions per input/trial".format(n_preds_per_input))

    from braindecode.experiments.experiment import Experiment
    from braindecode.datautil.iterators import CropsFromTrialsIterator
    from braindecode.experiments.monitors import RuntimeMonitor, LossMonitor, \
        CroppedTrialMisclassMonitor, MisclassMonitor
    from braindecode.experiments.stopcriteria import MaxEpochs
    import torch.nn.functional as F
    import torch as th
    from braindecode.torch_ext.modules import Expression
    # Iterator is used to iterate over datasets both for training
    # and evaluation
    iterator = CropsFromTrialsIterator(batch_size=32,
                                       input_time_length=input_time_length,
                                       n_preds_per_input=n_preds_per_input)

    # Loss function takes predictions as they come out of the network and the targets
    # and returns a loss
    loss_function = lambda preds, targets: F.nll_loss(
        th.mean(preds, dim=2, keepdim=False), targets)

    # Could be used to apply some constraint on the models, then should be object
    # with apply method that accepts a module
    model_constraint = None
    # Monitors log the training progress
    monitors = [LossMonitor(), MisclassMonitor(col_suffix='sample_misclass'),
                CroppedTrialMisclassMonitor(input_time_length),
                RuntimeMonitor(), ]
    # Stop criterion determines when the first stop happens
    stop_criterion = MaxEpochs(4)
    exp = Experiment(model, train_set, valid_set, test_set, iterator,
                     loss_function, optimizer, model_constraint,
                     monitors, stop_criterion,
                     remember_best_column='valid_misclass',
                     run_after_early_stop=True, batch_modifier=None, cuda=cuda)

    # need to setup python logging before to be able to see anything
    import logging
    import sys
    logging.basicConfig(format='%(asctime)s %(levelname)s : %(message)s',
                        level=logging.DEBUG, stream=sys.stdout)
    exp.run()

    import pandas as pd
    from io import StringIO
    compare_df = pd.read_csv(StringIO(
        'train_loss,valid_loss,test_loss,train_sample_misclass,valid_sample_misclass,'
        'test_sample_misclass,train_misclass,valid_misclass,test_misclass\n'
        '14.167170524597168,13.910758018493652,15.945781707763672,0.5,0.5,'
        '0.5333333333333333,0.5,0.5,0.5333333333333333\n'
        '1.1735659837722778,1.4342904090881348,1.8664429187774658,0.4629567736185384,'
        '0.5120320855614973,0.5336007130124778,0.5,0.5,0.5333333333333333\n'
        '1.3168460130691528,1.60431969165802,1.9181344509124756,0.49298128342245995,'
        '0.5109180035650625,0.531729055258467,0.5,0.5,0.5333333333333333\n'
        '0.8465543389320374,1.280307412147522,1.439755916595459,0.4413435828877005,'
        '0.5461229946524064,0.5283422459893048,0.47916666666666663,0.5,'
        '0.5333333333333333\n0.6977059841156006,1.1762590408325195,1.2779350280761719,'
        '0.40290775401069523,0.588903743315508,0.5307486631016043,0.5,0.5,0.5\n'
        '0.7934166193008423,1.1762590408325195,1.2779350280761719,0.4401069518716577,'
        '0.588903743315508,0.5307486631016043,0.5,0.5,0.5\n0.5982189178466797,'
        '0.8581563830375671,0.9598925113677979,0.32032085561497325,0.47660427807486627,'
        '0.4672905525846702,0.31666666666666665,0.5,0.4666666666666667\n0.5044312477111816,'
        '0.7133197784423828,0.8164243102073669,0.2591354723707665,0.45699643493761144,'
        '0.4393048128342246,0.16666666666666663,0.41666666666666663,0.43333333333333335\n'
        '0.4815250039100647,0.6736412644386292,0.8016976714134216,0.23413547237076648,'
        '0.39505347593582885,0.42932263814616756,0.15000000000000002,0.41666666666666663,0.5\n'))

    for col in compare_df:
        np.testing.assert_allclose(np.array(compare_df[col]),
                                   exp.epochs_df[col],
                                   rtol=1e-3, atol=1e-4)
