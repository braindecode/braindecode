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
    events = mne.find_events(raw, shortest_event=0, stim_channel='STI 014')

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
        u'train_loss,valid_loss,test_loss,train_sample_misclass,valid_sample_misclass,'
        'test_sample_misclass,train_misclass,valid_misclass,test_misclass\n'
        '0,0.8692976435025532,0.7483791708946228,0.6975634694099426,'
        '0.5389371657754011,0.47103386809269165,0.4425133689839572,'
        '0.6041666666666667,0.5,0.4\n1,2.3362590074539185,'
        '2.317707061767578,2.1407743096351624,0.4827874331550802,'
        '0.5,0.4666666666666667,0.5,0.5,0.4666666666666667\n'
        '2,0.5981490015983582,0.785034716129303,0.7005959153175354,'
        '0.3391822638146168,0.47994652406417115,0.41996434937611404,'
        '0.22916666666666663,0.41666666666666663,0.43333333333333335\n'
        '3,0.6355261653661728,0.785034716129303,'
        '0.7005959153175354,0.3673351158645276,0.47994652406417115,'
        '0.41996434937611404,0.2666666666666667,0.41666666666666663,'
        '0.43333333333333335\n4,0.625280424952507,'
        '0.802731990814209,0.7048938572406769,0.3367201426024955,'
        '0.43137254901960786,0.4229946524064171,0.3666666666666667,'
        '0.5833333333333333,0.33333333333333337\n'))

    for col in compare_df:
        np.testing.assert_allclose(np.array(compare_df[col]),
                                   exp.epochs_df[col],
                                   rtol=1e-4, atol=1e-5)