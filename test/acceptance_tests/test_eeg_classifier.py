# Authors: Maciej Sliwowski
#          Robin Tibor Schirrmeister
#          Lukas Gemein
#
# License: BSD-3

import mne
import numpy as np
from mne.io import concatenate_raws
from torch import optim

from braindecode.classifier import EEGClassifier
from braindecode.datasets.croppedxy import CroppedXyDataset
from braindecode.datautil.splitters import TrainTestSplit
from braindecode.losses import CroppedNLLLoss
from braindecode.models import ShallowFBCSPNet
from braindecode.models.util import to_dense_prediction_model
from braindecode.scoring import CroppedTrialEpochScoring
from braindecode.util import set_random_seeds, np_to_var


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
    is_root = not "__trace" in kwargs
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
            err.message
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
    input_time_length = 450
    n_classes = 2
    in_chans = X.shape[1]
    # final_conv_length determines the size of the receptive field of the ConvNet
    model = ShallowFBCSPNet(
        in_chans=in_chans,
        n_classes=n_classes,
        input_time_length=input_time_length,
        final_conv_length=12,
    )
    to_dense_prediction_model(model)

    if cuda:
        model.cuda()

    # determine output size
    test_input = np_to_var(
        np.ones((2, in_chans, input_time_length, 1), dtype=np.float32)
    )
    if cuda:
        test_input = test_input.cuda()
    out = model(test_input)
    n_preds_per_input = out.cpu().data.numpy().shape[2]

    train_set = CroppedXyDataset(
        X[:60], y=y[:60], input_time_length=input_time_length,
        n_preds_per_input=n_preds_per_input)

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
        criterion=CroppedNLLLoss,
        optimizer=optim.Adam,
        train_split=TrainTestSplit(
            train_size=0.8,
            input_time_length=input_time_length,
            n_preds_per_input=n_preds_per_input,),
        batch_size=32,
        callbacks=[
            ("train_trial_accuracy", cropped_cb_train),
            ("valid_trial_accuracy", cropped_cb_valid),
        ],
    )

    clf.fit(train_set.X, train_set.y, epochs=4)

    expected = [
        {
            "batches": [
                {"train_loss": 2.0750882625579834, "train_batch_size": 32},
                {"train_loss": 3.09424090385437, "train_batch_size": 32},
                {"train_loss": 1.079931378364563, "train_batch_size": 32},
                {"valid_loss": 2.3208131790161133, "valid_batch_size": 24},
            ],
            "epoch": 1,
            "train_batch_count": 3,
            "valid_batch_count": 1,
            "train_loss": 2.083086848258972,
            "train_loss_best": True,
            "valid_loss": 2.3208131790161133,
            "valid_loss_best": True,
            "train_trial_accuracy": 0.5,
            "train_trial_accuracy_best": True,
            "valid_trial_accuracy": 0.5,
            "valid_trial_accuracy_best": True,
        },
        {
            "batches": [
                {"train_loss": 1.827332615852356, "train_batch_size": 32},
                {"train_loss": 1.4135494232177734, "train_batch_size": 32},
                {"train_loss": 1.1295170783996582, "train_batch_size": 32},
                {"valid_loss": 1.4291356801986694, "valid_batch_size": 24},
            ],
            "epoch": 2,
            "train_batch_count": 3,
            "valid_batch_count": 1,
            "train_loss": 1.4567997058232625,
            "train_loss_best": True,
            "valid_loss": 1.4291356801986694,
            "valid_loss_best": True,
            "train_trial_accuracy": 0.5,
            "train_trial_accuracy_best": False,
            "valid_trial_accuracy": 0.5,
            "valid_trial_accuracy_best": False,
        },
        {
            "batches": [
                {"train_loss": 1.1495535373687744, "train_batch_size": 32},
                {"train_loss": 2.356320381164551, "train_batch_size": 32},
                {"train_loss": 0.9548418521881104, "train_batch_size": 32},
                {"valid_loss": 2.248246908187866, "valid_batch_size": 24},
            ],
            "epoch": 3,
            "train_batch_count": 3,
            "valid_batch_count": 1,
            "train_loss": 1.4869052569071453,
            "train_loss_best": False,
            "valid_loss": 2.248246908187866,
            "valid_loss_best": False,
            "train_trial_accuracy": 0.5,
            "train_trial_accuracy_best": False,
            "valid_trial_accuracy": 0.5,
            "valid_trial_accuracy_best": False,
        },
        {
            "batches": [
                {"train_loss": 1.2157528400421143, "train_batch_size": 32},
                {"train_loss": 1.1182057857513428, "train_batch_size": 32},
                {"train_loss": 0.9163083434104919, "train_batch_size": 32},
                {"valid_loss": 0.9732739925384521, "valid_batch_size": 24},
            ],
            "epoch": 4,
            "train_batch_count": 3,
            "valid_batch_count": 1,
            "train_loss": 1.083422323067983,
            "train_loss_best": True,
            "valid_loss": 0.9732739925384521,
            "valid_loss_best": True,
            "train_trial_accuracy": 0.5,
            "train_trial_accuracy_best": False,
            "valid_trial_accuracy": 0.5,
            "valid_trial_accuracy_best": False,
        },
    ]

    history_without_dur = [
        {k: v for k, v in h.items() if k != "dur"} for h in clf.history
    ]
    assert_deep_allclose(history_without_dur, expected, atol=1e-3, rtol=1e-3)
