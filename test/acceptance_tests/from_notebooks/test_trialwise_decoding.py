def test_trialwise_decoding():
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

    # Convert data from volt to millivolt
    # Pytorch expects float32 for input and int64 for labels.
    X = (epoched.get_data() * 1e6).astype(np.float32)
    y = (epoched.events[:, 2] - 2).astype(np.int64)  # 2,3 -> 0,1

    from braindecode.datautil.signal_target import SignalAndTarget

    train_set = SignalAndTarget(X[:60], y=y[:60])
    test_set = SignalAndTarget(X[60:], y=y[60:])

    from braindecode.models.shallow_fbcsp import ShallowFBCSPNet
    from torch import nn
    from braindecode.torch_ext.util import set_random_seeds

    # Set if you want to use GPU
    # You can also use torch.cuda.is_available() to determine if cuda is available on your machine.
    cuda = False
    set_random_seeds(seed=20170629, cuda=cuda)
    n_classes = 2
    in_chans = train_set.X.shape[1]
    # final_conv_length = auto ensures we only get a single output in the time dimension
    model = ShallowFBCSPNet(in_chans=in_chans, n_classes=n_classes,
                            input_time_length=train_set.X.shape[2],
                            final_conv_length='auto').create_network()
    if cuda:
        model.cuda()

    from torch import optim

    optimizer = optim.Adam(model.parameters())

    from braindecode.torch_ext.util import np_to_var, var_to_np
    from braindecode.datautil.iterators import get_balanced_batches
    import torch.nn.functional as F
    from numpy.random import RandomState

    rng = RandomState((2017, 6, 30))
    losses = []
    accuracies = []
    for i_epoch in range(6):
        i_trials_in_batch = get_balanced_batches(len(train_set.X), rng,
                                                 shuffle=True,
                                                 batch_size=30)
        # Set model to training mode
        model.train()
        for i_trials in i_trials_in_batch:
            # Have to add empty fourth dimension to X
            batch_X = train_set.X[i_trials][:, :, :, None]
            batch_y = train_set.y[i_trials]
            net_in = np_to_var(batch_X)
            if cuda:
                net_in = net_in.cuda()
            net_target = np_to_var(batch_y)
            if cuda:
                net_target = net_target.cuda()
            # Remove gradients of last backward pass from all parameters
            optimizer.zero_grad()
            # Compute outputs of the network
            outputs = model(net_in)
            # Compute the loss
            loss = F.nll_loss(outputs, net_target)
            # Do the backpropagation
            loss.backward()
            # Update parameters with the optimizer
            optimizer.step()

        # Print some statistics each epoch
        model.eval()
        print("Epoch {:d}".format(i_epoch))
        for setname, dataset in (('Train', train_set), ('Test', test_set)):
            # Here, we will use the entire dataset at once, which is still possible
            # for such smaller datasets. Otherwise we would have to use batches.
            net_in = np_to_var(dataset.X[:, :, :, None])
            if cuda:
                net_in = net_in.cuda()
            net_target = np_to_var(dataset.y)
            if cuda:
                net_target = net_target.cuda()
            outputs = model(net_in)
            loss = F.nll_loss(outputs, net_target)
            losses.append(float(var_to_np(loss)))
            print("{:6s} Loss: {:.5f}".format(
                setname, float(var_to_np(loss))))
            predicted_labels = np.argmax(var_to_np(outputs), axis=1)
            accuracy = np.mean(dataset.y == predicted_labels)
            accuracies.append(accuracy * 100)
            print("{:6s} Accuracy: {:.1f}%".format(
                setname, accuracy * 100))

    np.testing.assert_allclose(
        np.array(losses),
        np.array([0.91796708,
                  1.2714895,
                  0.4999536,
                  0.94365239,
                  0.39268905,
                  0.89928466,
                  0.37648854,
                  0.8940345,
                  0.35774994,
                  0.86749417,
                  0.35080773,
                  0.80767328]),
        rtol=1e-4, atol=1e-5)

    np.testing.assert_allclose(
        np.array(accuracies),
        np.array(
            [55.,
             63.33333333,
             71.66666667,
             63.33333333,
             81.66666667,
             60.,
             78.33333333,
             63.33333333,
             83.33333333,
             66.66666667,
             80.,
             66.66666667]),
        rtol=1e-4, atol=1e-5)
