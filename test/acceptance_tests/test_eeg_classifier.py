# Authors: Maciej Sliwowski
#          Robin Tibor Schirrmeister
#          Lukas Gemein
#
# License: BSD-3

import mne
import numpy as np
from mne.io import concatenate_raws
from skorch.helper import predefined_split
from torch import optim
from torch.nn.functional import nll_loss

from braindecode.classifier import EEGClassifier
from braindecode.datasets.xy import create_from_X_y
from braindecode.training.losses import CroppedLoss
from braindecode.models import ShallowFBCSPNet
from braindecode.models.util import to_dense_prediction_model
from braindecode.training.scoring import CroppedTrialEpochScoring
from braindecode.util import set_random_seeds, np_to_th


def assert_deep_allclose(expected, actual, *args, **kwargs):
    """
    Assert that two complex structures have almost equal contents.

    Compares lists, dicts and tuples recursively. Checks numeric values
    using test_case's :py:meth:`unittest.TestCase.assertAlmostEqual` and
    checks all other values with :py:meth:`unittest.TestCase.assertEqual`.
    Accepts additional positional and keyword arguments and pass those
    intact to assertAlmostEqual() (that's how you specify comparison
    precision).
    """
    is_root = "__trace" not in kwargs
    trace = kwargs.pop("__trace", "ROOT")
    try:
        if isinstance(expected, (int, float, complex)):
            np.testing.assert_allclose(expected, actual, *args, **kwargs)
        elif isinstance(expected, (list, tuple, np.ndarray)):
            assert len(expected) == len(actual)
            for index in range(len(expected)):
                v1, v2 = expected[index], actual[index]
                assert_deep_allclose(
                    v1, v2, __trace=repr(index), *args, **kwargs
                )
        elif isinstance(expected, dict):
            assert set(expected) == set(actual)
            for key in expected:
                assert_deep_allclose(
                    expected[key],
                    actual[key],
                    __trace=repr(key),
                    *args,
                    **kwargs
                )
        else:
            assert expected == actual
    except AssertionError as exc:
        exc.__dict__.setdefault("traces", []).append(trace)
        msg = (
            exc.message
            if hasattr(exc, "message")
            else exc.args[0]
            if exc.args
            else ""
        )
        if is_root:
            trace = " -> ".join(reversed(exc.traces))
            exc = AssertionError("%s\nTRACE: %s" % (msg, trace))
        raise exc


def test_eeg_classifier():
    # 5,6,7,10,13,14 are codes for executed and imagined hands/feet
    subject_id = 1
    event_codes = [5, 6, 9, 10, 13, 14]

    # This will download the files if you don't have them yet,
    # and then return the paths to the files.
    physionet_paths = mne.datasets.eegbci.load_data(
        subject_id, event_codes, update_path=False
    )

    # Load each of the files
    parts = [
        mne.io.read_raw_edf(
            path, preload=True, stim_channel="auto", verbose="WARNING"
        )
        for path in physionet_paths
    ]

    # Concatenate them
    raw = concatenate_raws(parts)

    # Find the events in this dataset
    events, _ = mne.events_from_annotations(raw)

    # Use only EEG channels
    eeg_channel_inds = mne.pick_types(
        raw.info, meg=False, eeg=True, stim=False, eog=False, exclude="bads"
    )

    # Extract trials, only using EEG channels
    epoched = mne.Epochs(
        raw,
        events,
        dict(hands=2, feet=3),
        tmin=1,
        tmax=4.1,
        proj=False,
        picks=eeg_channel_inds,
        baseline=None,
        preload=True,
    )

    # Convert data from volt to millivolt
    # Pytorch expects float32 for input and int64 for labels.
    X = (epoched.get_data() * 1e6).astype(np.float32)
    y = (epoched.events[:, 2] - 2).astype(np.int64)  # 2,3 -> 0,1

    # Set if you want to use GPU
    # You can also use torch.cuda.is_available() to determine if cuda is available on your machine.
    cuda = False
    set_random_seeds(seed=20170629, cuda=cuda)

    # This will determine how many crops are processed in parallel
    input_window_samples = 450
    n_classes = 2
    in_chans = X.shape[1]
    # final_conv_length determines the size of the receptive field of the ConvNet
    model = ShallowFBCSPNet(
        in_chans=in_chans,
        n_classes=n_classes,
        input_window_samples=input_window_samples,
        final_conv_length=12,
    )
    to_dense_prediction_model(model)

    if cuda:
        model.cuda()

    # determine output size
    test_input = np_to_th(
        np.ones((2, in_chans, input_window_samples, 1), dtype=np.float32)
    )
    if cuda:
        test_input = test_input.cuda()
    out = model(test_input)
    n_preds_per_input = out.cpu().data.numpy().shape[2]

    train_set = create_from_X_y(X[:48], y[:48],
                                drop_last_window=False,
                                sfreq=100,
                                window_size_samples=input_window_samples,
                                window_stride_samples=n_preds_per_input)

    valid_set = create_from_X_y(X[48:60], y[48:60],
                                drop_last_window=False,
                                sfreq=100,
                                window_size_samples=input_window_samples,
                                window_stride_samples=n_preds_per_input)

    cropped_cb_train = CroppedTrialEpochScoring(
        "accuracy",
        name="train_trial_accuracy",
        lower_is_better=False,
        on_train=True,
    )

    cropped_cb_valid = CroppedTrialEpochScoring(
        "accuracy",
        on_train=False,
        name="valid_trial_accuracy",
        lower_is_better=False,
    )

    clf = EEGClassifier(
        model,
        cropped=True,
        criterion=CroppedLoss,
        criterion__loss_function=nll_loss,
        optimizer=optim.Adam,
        train_split=predefined_split(valid_set),
        batch_size=32,
        callbacks=[
            ("train_trial_accuracy", cropped_cb_train),
            ("valid_trial_accuracy", cropped_cb_valid),
        ],
    )

    clf.fit(train_set, y=None, epochs=4)

    expected = [{'batches': [{'train_batch_size': 32, 'train_loss': 1.6639312505722046},
                             {'train_batch_size': 32, 'train_loss': 2.6161606311798096},
                             {'train_batch_size': 32, 'train_loss': 1.627132773399353},
                             {'valid_batch_size': 24, 'valid_loss': 0.9677614569664001}],
                 'epoch': 1,
                 'train_batch_count': 3,
                 'train_loss': 1.9690748850504558,
                 'train_loss_best': True,
                 'train_trial_accuracy': 0.4791666666666667,
                 'train_trial_accuracy_best': True,
                 'valid_batch_count': 1,
                 'valid_loss': 0.9677614569664001,
                 'valid_loss_best': True,
                 'valid_trial_accuracy': 0.5,
                 'valid_trial_accuracy_best': True},
                {'batches': [{'train_batch_size': 32, 'train_loss': 1.3829222917556763},
                             {'train_batch_size': 32, 'train_loss': 1.3123714923858643},
                             {'train_batch_size': 32, 'train_loss': 1.0109959840774536},
                             {'valid_batch_size': 24, 'valid_loss': 1.9435862302780151}],
                 'epoch': 2,
                 'train_batch_count': 3,
                 'train_loss': 1.2354299227396648,
                 'train_loss_best': True,
                 'train_trial_accuracy': 0.5,
                 'train_trial_accuracy_best': True,
                 'valid_batch_count': 1,
                 'valid_loss': 1.9435862302780151,
                 'valid_loss_best': False,
                 'valid_trial_accuracy': 0.5,
                 'valid_trial_accuracy_best': False},
                {'batches': [{'train_batch_size': 32, 'train_loss': 1.172208547592163},
                             {'train_batch_size': 32, 'train_loss': 0.8899562954902649},
                             {'train_batch_size': 32, 'train_loss': 1.0232216119766235},
                             {'valid_batch_size': 24, 'valid_loss': 0.9585554599761963}],
                 'epoch': 3,
                 'train_batch_count': 3,
                 'train_loss': 1.0284621516863506,
                 'train_loss_best': True,
                 'train_trial_accuracy': 0.5,
                 'train_trial_accuracy_best': False,
                 'valid_batch_count': 1,
                 'valid_loss': 0.9585554599761963,
                 'valid_loss_best': True,
                 'valid_trial_accuracy': 0.5,
                 'valid_trial_accuracy_best': False},
                {'batches': [{'train_batch_size': 32, 'train_loss': 0.9693693518638611},
                             {'train_batch_size': 32, 'train_loss': 0.900641918182373},
                             {'train_batch_size': 32, 'train_loss': 0.8839665651321411},
                             {'valid_batch_size': 24, 'valid_loss': 0.873468816280365}],
                 'epoch': 4,
                 'train_batch_count': 3,
                 'train_loss': 0.9179926117261251,
                 'train_loss_best': True,
                 'train_trial_accuracy': 0.625,
                 'train_trial_accuracy_best': True,
                 'valid_batch_count': 1,
                 'valid_loss': 0.873468816280365,
                 'valid_loss_best': True,
                 'valid_trial_accuracy': 0.4166666666666667,
                 'valid_trial_accuracy_best': False}]

    history_without_dur = [
        {k: v for k, v in h.items() if k != "dur"} for h in clf.history
    ]
    assert_deep_allclose(expected, history_without_dur, atol=1e-3, rtol=1e-3)
