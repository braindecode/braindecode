from collections import namedtuple

import numpy as np

import torch as th
from torch import optim
import torch.nn.functional as F

import mne
from mne.io import concatenate_raws

from braindecode.util import var_to_np, np_to_var
from braindecode.models import ShallowFBCSPNet
from braindecode.models.util import to_dense_prediction_model
from braindecode.util import set_random_seeds
from braindecode.experiments.monitors import compute_preds_per_trial_from_crops
from braindecode.datautil.iterators import CropsFromTrialsIterator


def test_cropped_decoding():
    # 5,6,7,10,13,14 are codes for executed and imagined hands/feet
    subject_id = 1
    event_codes = [5, 6, 9, 10, 13, 14]

    # This will download the files if you don't have them yet,
    # and then return the paths to the files.
    physionet_paths = mne.datasets.eegbci.load_data(
        subject_id, event_codes, update_path=False)

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
    # Convert data from volt to millivolt
    # Pytorch expects float32 for input and int64 for labels.
    X = (epoched.get_data() * 1e6).astype(np.float32)
    y = (epoched.events[:, 2] - 2).astype(np.int64)  # 2,3 -> 0,1

    SignalAndTarget = namedtuple('SignalAndTarget', 'X y')

    train_set = SignalAndTarget(X[:60], y=y[:60])
    test_set = SignalAndTarget(X[60:], y=y[60:])

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

    optimizer = optim.Adam(model.parameters())
    # determine output size
    test_input = np_to_var(
        np.ones((2, in_chans, input_time_length, 1), dtype=np.float32))
    if cuda:
        test_input = test_input.cuda()
    out = model(test_input)
    n_preds_per_input = out.cpu().data.numpy().shape[2]
    print("{:d} predictions per input/trial".format(n_preds_per_input))

    iterator = CropsFromTrialsIterator(batch_size=32,
                                       input_time_length=input_time_length,
                                       n_preds_per_input=n_preds_per_input)

    losses = []
    accuracies = []
    for i_epoch in range(4):
        # Set model to training mode
        model.train()
        for batch_X, batch_y in iterator.get_batches(train_set, shuffle=False):
            net_in = np_to_var(batch_X)
            if cuda:
                net_in = net_in.cuda()
            net_target = np_to_var(batch_y)
            if cuda:
                net_target = net_target.cuda()
            # Remove gradients of last backward pass from all parameters
            optimizer.zero_grad()
            outputs = model(net_in)
            # Mean predictions across trial
            # Note that this will give identical gradients to computing
            # a per-prediction loss (at least for the combination of log softmax activation
            # and negative log likelihood loss which we are using here)
            outputs = th.mean(outputs, dim=2, keepdim=False)
            loss = F.nll_loss(outputs, net_target)
            loss.backward()
            optimizer.step()

        # Print some statistics each epoch
        model.eval()
        print("Epoch {:d}".format(i_epoch))
        for setname, dataset in (('Train', train_set), ('Test', test_set)):
            # Collect all predictions and losses
            all_preds = []
            all_losses = []
            batch_sizes = []
            for batch_X, batch_y in iterator.get_batches(dataset,
                                                         shuffle=False):
                net_in = np_to_var(batch_X)
                if cuda:
                    net_in = net_in.cuda()
                net_target = np_to_var(batch_y)
                if cuda:
                    net_target = net_target.cuda()
                outputs = model(net_in)
                all_preds.append(var_to_np(outputs))
                outputs = th.mean(outputs, dim=2, keepdim=False)
                loss = F.nll_loss(outputs, net_target)
                loss = float(var_to_np(loss))
                all_losses.append(loss)
                batch_sizes.append(len(batch_X))
            # Compute mean per-input loss
            loss = np.mean(np.array(all_losses) * np.array(batch_sizes) /
                           np.mean(batch_sizes))
            print("{:6s} Loss: {:.5f}".format(setname, loss))
            losses.append(loss)
            # Assign the predictions to the trials
            preds_per_trial = compute_preds_per_trial_from_crops(all_preds,
                                                                 input_time_length,
                                                                 dataset.X)
            # preds per trial are now trials x classes x timesteps/predictions
            # Now mean across timesteps for each trial to get per-trial predictions
            meaned_preds_per_trial = np.array(
                [np.mean(p, axis=1) for p in preds_per_trial])
            predicted_labels = np.argmax(meaned_preds_per_trial, axis=1)
            accuracy = np.mean(predicted_labels == dataset.y)
            accuracies.append(accuracy * 100)
            print("{:6s} Accuracy: {:.1f}%".format(
                setname, accuracy * 100))
    np.testing.assert_allclose(
        np.array(losses),
        np.array([1.31657708,
                  1.73548156,
                  1.02950428,
                  1.43932164,
                  0.78677772,
                  1.12382019,
                  0.55920881,
                  0.87277424]),
        rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(
        np.array(accuracies),
        np.array(
            [50.,
             46.66666667,
             50.,
             46.66666667,
             50.,
             46.66666667,
             66.66666667,
             50.]
        ),
        rtol=1e-4, atol=1e-5)
