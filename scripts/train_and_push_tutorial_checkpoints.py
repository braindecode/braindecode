#!/usr/bin/env python

import argparse
import copy
import json
import sys
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
LOCAL_MOABB_ROOT = REPO_ROOT / "moabb"
if (LOCAL_MOABB_ROOT / "moabb" / "__init__.py").exists():
    sys.path.insert(0, str(LOCAL_MOABB_ROOT))
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import torch
from joblib import parallel_backend
from numpy import multiply
from sklearn.metrics import balanced_accuracy_score, r2_score
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.preprocessing import robust_scale
from sklearn.preprocessing import scale as standard_scale
from sklearn.utils import compute_class_weight
from skorch.callbacks import (
    EarlyStopping,
    EpochScoring,
    GradientNormClipping,
    LRScheduler,
)
from skorch.dataset import ValidSplit
from skorch.helper import SliceDataset, predefined_split
from torch import nn

from braindecode import EEGClassifier, EEGRegressor
from braindecode._tutorial_hub import (
    save_tutorial_checkpoint,
    tutorial_repo_id,
    upload_tutorial_artifacts,
)
from braindecode.augmentation import (
    AugmentedDataLoader,
    ChannelsDropout,
    FTSurrogate,
    IdentityTransform,
    SmoothTimeMask,
)
from braindecode.datasets import BCICompetitionIVDataset4, MOABBDataset, SleepPhysionet
from braindecode.datautil import infer_signal_properties
from braindecode.models import (
    AttnSleep,
    EEGNeX,
    ShallowFBCSPNet,
    SleepStagerChambon2018,
    USleep,
)
from braindecode.modules import TimeDistributed
from braindecode.preprocessing import (
    EEGPrep,
    Preprocessor,
    create_fixed_length_windows,
    create_windows_from_events,
    create_windows_from_target_channels,
    exponential_moving_standardize,
    preprocess,
)
from braindecode.samplers import SequenceSampler
from braindecode.training import (
    CroppedLoss,
    CroppedTimeSeriesEpochScoring,
    TimeSeriesLoss,
)
from braindecode.util import set_random_seeds

AVAILABLE_TUTORIALS = (
    "plot_bcic_iv_2a_moabb_trial",
    "plot_bcic_iv_2a_moabb_cropped",
    "plot_bcic_iv_2a_eegprep_cleaning",
    "bcic_iv_4_ecog_trial",
    "bcic_iv_4_ecog_cropped",
    "plot_sleep_staging_usleep",
    "plot_sleep_staging_eldele2021",
    "plot_sleep_staging_chambon2018",
    "plot_data_augmentation_search",
)


@dataclass
class TutorialArtifacts:
    clf: EEGClassifier
    repo_id: str
    metadata: dict


# Global wandb state, set by main() when --wandb is passed.
_WANDB_PROJECT: str | None = None


def _make_wandb_callback(tutorial_name: str, config: dict):
    """Create a WandbLogger callback if wandb is enabled, else return None."""
    if _WANDB_PROJECT is None:
        return None
    import wandb  # isort: skip
    from skorch.callbacks import WandbLogger  # isort: skip

    run = wandb.init(
        project=_WANDB_PROJECT,
        name=tutorial_name,
        config=config,
        reinit=True,
    )
    return ("wandb_logger", WandbLogger(run, save_model=False)), run


def _save_loss_curve(clf, output_dir: Path, tutorial_name: str):
    """Save a loss/metric curve plot as a PNG artifact."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    history = clf.history
    epochs = [h["epoch"] for h in history]

    has_bal_acc = "valid_bal_acc" in history[0]
    n_plots = 2 if has_bal_acc else 1
    fig, axes = plt.subplots(n_plots, 1, figsize=(8, 3.5 * n_plots), sharex=True)
    if n_plots == 1:
        axes = [axes]

    axes[0].plot(epochs, history[:, "train_loss"], "r-", label="Train")
    axes[0].plot(epochs, history[:, "valid_loss"], "b-", label="Valid")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[0].set_title(tutorial_name)

    if has_bal_acc:
        axes[1].plot(epochs, history[:, "train_bal_acc"], "r-", label="Train")
        axes[1].plot(epochs, history[:, "valid_bal_acc"], "b-", label="Valid")
        axes[1].set_ylabel("Balanced accuracy")
        axes[1].legend()
        axes[1].grid(alpha=0.3)

    axes[-1].set_xlabel("Epoch")
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / "loss_curve.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved loss curve to {plot_path}")
    return plot_path


SEARCH_RESULTS_FILENAME = "search_results.csv"


SLEEP_MAPPING = {
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3,
    "Sleep stage 4": 3,
    "Sleep stage R": 4,
}


def _common_preprocessing(
    dataset,
    *,
    use_eegprep: bool = False,
    n_jobs: int = 1,
):
    preprocessors = [
        Preprocessor("pick_types", eeg=True, meg=False, stim=False),
        Preprocessor(lambda data: multiply(data, 1e6)),
    ]
    if use_eegprep:
        preprocessors.append(
            EEGPrep(
                resample_to=128,
                bad_window_max_bad_channels=None,
            )
        )
    preprocessors.extend(
        [
            Preprocessor("filter", l_freq=4.0, h_freq=38.0),
            Preprocessor(
                exponential_moving_standardize,
                factor_new=1e-3,
                init_block_size=1000,
            ),
        ]
    )
    preprocess(dataset, preprocessors, n_jobs=n_jobs)


def _ecog_preprocessing(dataset, *, n_jobs: int = 1):
    preprocessors = [
        Preprocessor("pick_types", ecog=True, misc=True),
        Preprocessor(lambda x: x / 1e6, picks="ecog"),
        Preprocessor("filter", l_freq=1.0, h_freq=200.0),
        Preprocessor(
            exponential_moving_standardize,
            factor_new=1e-3,
            init_block_size=1000,
            picks="ecog",
        ),
    ]
    preprocess(dataset, preprocessors, n_jobs=n_jobs)


def _trialwise_windows(dataset):
    sfreq = dataset.datasets[0].raw.info["sfreq"]
    return create_windows_from_events(
        dataset,
        trial_start_offset_samples=int(-0.5 * sfreq),
        trial_stop_offset_samples=0,
        preload=True,
    )


def _training_callbacks(epochs: int, patience: int):
    return [
        "accuracy",
        ("lr_scheduler", LRScheduler("CosineAnnealingLR", T_max=max(1, epochs - 1))),
        ("early_stopping", EarlyStopping(patience=patience, load_best=True)),
    ]


def _device_and_seed(seed: int = 20200220):
    cuda = torch.cuda.is_available()
    mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    if cuda:
        torch.backends.cudnn.benchmark = True
    set_random_seeds(seed=seed, cuda=cuda)
    return cuda, "cuda" if cuda else "mps" if mps else "cpu"


def _pearson_r_score(net, dataset, y):
    preds = net.predict(dataset)
    corr_coeffs = [
        np.corrcoef(y[:, i], preds[:, i])[0, 1] for i in range(y.shape[1])
    ]
    return float(np.nanmean(corr_coeffs))


def _trialwise_shallow(
    tutorial_name: str,
    *,
    subject_id: int,
    epochs: int,
    patience: int,
) -> TutorialArtifacts:
    dataset = MOABBDataset(dataset_name="BNCI2014_001", subject_ids=[subject_id])
    _common_preprocessing(dataset, n_jobs=-1)
    windows_dataset = _trialwise_windows(dataset)
    splitted = windows_dataset.split("session")
    train_set = splitted["0train"]
    valid_set = splitted["1test"]
    sig_props = infer_signal_properties(train_set, mode="classification")

    _, device = _device_and_seed()
    model = ShallowFBCSPNet(
        n_chans=sig_props["n_chans"],
        n_outputs=sig_props["n_outputs"],
        n_times=sig_props["n_times"],
        final_conv_length="auto",
    )
    clf = EEGClassifier(
        model,
        criterion=torch.nn.CrossEntropyLoss,
        optimizer=torch.optim.AdamW,
        train_split=predefined_split(valid_set),
        optimizer__lr=0.0625 * 0.01,
        optimizer__weight_decay=0,
        batch_size=64,
        callbacks=_training_callbacks(epochs, patience),
        device=device,
        classes=list(range(sig_props["n_outputs"])),
    )
    clf.fit(train_set, y=None, epochs=epochs)
    metadata = {
        "best_valid_accuracy": max(clf.history[:, "valid_accuracy"]),
        "chance_level": 0.25,
        "display_metric_key": "best_valid_accuracy",
        "display_metric_name": "accuracy",
        "display_split_name": "held-out session",
        "epochs_ran": len(clf.history),
        "epochs_requested": epochs,
        "final_valid_accuracy": clf.history[-1, "valid_accuracy"],
        "patience": patience,
        "short_run_epochs": 4,
        "subject_id": subject_id,
        "tutorial": tutorial_name,
    }
    return TutorialArtifacts(
        clf=clf,
        repo_id=tutorial_repo_id(tutorial_name),
        metadata=metadata,
    )


def _ecog_trialwise(*, epochs: int, patience: int) -> TutorialArtifacts:
    tutorial_name = "bcic_iv_4_ecog_trial"
    dataset = BCICompetitionIVDataset4(subject_ids=[1])
    _ecog_preprocessing(dataset, n_jobs=1)
    windows_dataset = create_windows_from_target_channels(
        dataset, window_size_samples=1000, preload=False, last_target_only=True
    )
    windows_dataset.target_transform = lambda x: x[0:1]
    subsets = windows_dataset.split("session")
    train_set = subsets["train"]
    test_set = subsets["test"]
    idx_train, idx_valid = train_test_split(
        np.arange(len(train_set)),
        random_state=100,
        test_size=0.2,
        shuffle=False,
    )
    valid_set = torch.utils.data.Subset(train_set, idx_valid)
    train_set = torch.utils.data.Subset(train_set, idx_train)

    _, device = _device_and_seed()
    model = ShallowFBCSPNet(
        train_set[0][0].shape[0],
        train_set[0][1].shape[0],
        n_times=1000,
        final_conv_length="auto",
    )
    regressor = EEGRegressor(
        model,
        criterion=torch.nn.MSELoss,
        optimizer=torch.optim.AdamW,
        train_split=predefined_split(valid_set),
        optimizer__lr=0.0625 * 0.01,
        optimizer__weight_decay=0,
        batch_size=64,
        callbacks=[
            "r2",
            (
                "valid_pearson_r",
                EpochScoring(
                    _pearson_r_score,
                    lower_is_better=False,
                    on_train=False,
                    name="valid_pearson_r",
                ),
            ),
            (
                "train_pearson_r",
                EpochScoring(
                    _pearson_r_score,
                    lower_is_better=False,
                    on_train=True,
                    name="train_pearson_r",
                ),
            ),
            ("lr_scheduler", LRScheduler("CosineAnnealingLR", T_max=max(1, epochs - 1))),
            ("early_stopping", EarlyStopping(patience=patience, load_best=True)),
        ],
        device=device,
    )
    regressor.fit(train_set, y=None, epochs=epochs)
    preds_test = regressor.predict(test_set)
    y_test = np.stack([data[1] for data in test_set])
    corr_coeffs = [
        float(np.corrcoef(preds_test[:, dim], y_test[:, dim])[0, 1])
        for dim in range(y_test.shape[1])
    ]
    metadata = {
        "best_valid_pearson_r": float(max(regressor.history[:, "valid_pearson_r"])),
        "best_valid_r2": float(max(regressor.history[:, "valid_r2"])),
        "display_metric_as_percentage": False,
        "display_metric_key": "test_mean_pearson_r",
        "display_metric_name": "mean Pearson r",
        "display_split_name": "held-out test session",
        "epochs_ran": len(regressor.history),
        "epochs_requested": epochs,
        "final_valid_pearson_r": float(regressor.history[-1, "valid_pearson_r"]),
        "final_valid_r2": float(regressor.history[-1, "valid_r2"]),
        "patience": patience,
        "reference_recording_scope": "whole recording",
        "reference_uses_full_recordings": True,
        "short_run_epochs": 2,
        "subject_id": 1,
        "test_mean_pearson_r": float(np.nanmean(corr_coeffs)),
        "test_pearson_r_per_dim": corr_coeffs,
        "tutorial": tutorial_name,
    }
    return TutorialArtifacts(
        clf=regressor,
        repo_id=tutorial_repo_id(tutorial_name),
        metadata=metadata,
    )


def _pad_and_select_predictions(preds, y):
    preds = np.pad(
        preds,
        ((0, 0), (0, 0), (y.shape[2] - preds.shape[2], 0)),
        "constant",
        constant_values=0,
    )
    mask = ~np.isnan(y[0, 0, :])
    preds = np.squeeze(preds[..., mask], 0)
    y = np.squeeze(y[..., mask], 0)
    return y.T, preds.T


def _ecog_cropped(*, epochs: int, patience: int) -> TutorialArtifacts:
    tutorial_name = "bcic_iv_4_ecog_cropped"
    dataset = BCICompetitionIVDataset4(subject_ids=[1])
    dataset_split = dataset.split("session")
    train_set = dataset_split["train"]
    test_set = dataset_split["test"]

    train_duration_s = float(train_set.datasets[0].raw.times[-1])
    valid_tmin_s = 0.8 * train_duration_s
    valid_set = preprocess(
        copy.deepcopy(train_set),
        [Preprocessor("crop", tmin=valid_tmin_s, tmax=None)],
        n_jobs=1,
    )
    preprocess(
        train_set,
        [Preprocessor("crop", tmin=0, tmax=valid_tmin_s)],
        n_jobs=1,
    )
    _ecog_preprocessing(train_set, n_jobs=1)
    _ecog_preprocessing(valid_set, n_jobs=1)
    _ecog_preprocessing(test_set, n_jobs=1)

    _, device = _device_and_seed()
    n_times = 1000
    n_chans = train_set[0][0].shape[0] - 5
    model = ShallowFBCSPNet(
        n_chans,
        1,
        n_times=n_times,
        final_conv_length=2,
    )
    n_preds_per_input = model.get_output_shape()[2]
    train_windows = create_fixed_length_windows(
        train_set,
        start_offset_samples=0,
        stop_offset_samples=None,
        window_size_samples=n_times,
        window_stride_samples=n_preds_per_input,
        drop_last_window=False,
        targets_from="channels",
        last_target_only=False,
        preload=False,
    )
    valid_windows = create_fixed_length_windows(
        valid_set,
        start_offset_samples=0,
        stop_offset_samples=None,
        window_size_samples=n_times,
        window_stride_samples=n_preds_per_input,
        drop_last_window=False,
        targets_from="channels",
        last_target_only=False,
        preload=False,
    )
    test_windows = create_fixed_length_windows(
        test_set,
        start_offset_samples=0,
        stop_offset_samples=None,
        window_size_samples=n_times,
        window_stride_samples=n_preds_per_input,
        drop_last_window=False,
        targets_from="channels",
        last_target_only=False,
        preload=False,
    )
    train_windows.target_transform = lambda x: x[0:1]
    valid_windows.target_transform = lambda x: x[0:1]
    test_windows.target_transform = lambda x: x[0:1]

    regressor = EEGRegressor(
        model,
        cropped=True,
        aggregate_predictions=False,
        criterion=TimeSeriesLoss,
        criterion__loss_function=torch.nn.functional.mse_loss,
        optimizer=torch.optim.AdamW,
        train_split=predefined_split(valid_windows),
        optimizer__lr=0.0625 * 0.01,
        optimizer__weight_decay=0,
        iterator_train__shuffle=True,
        batch_size=27,
        callbacks=[
            ("lr_scheduler", LRScheduler("CosineAnnealingLR", T_max=max(1, epochs - 1))),
            (
                "r2_train",
                CroppedTimeSeriesEpochScoring(
                    r2_score,
                    lower_is_better=False,
                    on_train=True,
                    name="r2_train",
                ),
            ),
            (
                "r2_valid",
                CroppedTimeSeriesEpochScoring(
                    r2_score,
                    lower_is_better=False,
                    on_train=False,
                    name="r2_valid",
                ),
            ),
            ("early_stopping", EarlyStopping(patience=patience, load_best=True)),
        ],
        device=device,
    )
    regressor.fit(train_windows, y=None, epochs=epochs)
    preds_test, y_test = regressor.predict_trials(test_windows, return_targets=True)
    preds_test, y_test = _pad_and_select_predictions(preds_test, y_test)
    corr_coeffs = [
        float(np.corrcoef(preds_test[:, dim], y_test[:, dim])[0, 1])
        for dim in range(y_test.shape[1])
    ]
    metadata = {
        "best_valid_r2": float(max(regressor.history[:, "r2_valid"])),
        "display_metric_as_percentage": False,
        "display_metric_key": "test_mean_pearson_r",
        "display_metric_name": "mean Pearson r",
        "display_split_name": "held-out test session",
        "epochs_ran": len(regressor.history),
        "epochs_requested": epochs,
        "final_valid_r2": float(regressor.history[-1, "r2_valid"]),
        "patience": patience,
        "reference_recording_scope": "whole recording",
        "reference_uses_full_recordings": True,
        "short_run_epochs": 8,
        "subject_id": 1,
        "test_mean_pearson_r": float(np.nanmean(corr_coeffs)),
        "test_pearson_r_per_dim": corr_coeffs,
        "tutorial": tutorial_name,
        "valid_fraction": 0.2,
    }
    return TutorialArtifacts(
        clf=regressor,
        repo_id=tutorial_repo_id(tutorial_name),
        metadata=metadata,
    )


def _get_center_label(x):
    if isinstance(x, Integral):
        return x
    return x[np.ceil(len(x) / 2).astype(int)] if len(x) > 1 else x


def _sleep_physionet_windows(
    *,
    crop=None,
    picks=None,
    recording_ids=None,
    subject_ids=None,
    train_subject_ids=None,
    valid_subject_ids=None,
    raw_preprocessors: list[Preprocessor],
    window_preprocessors: list[Preprocessor] | None = None,
):
    if recording_ids is None:
        recording_ids = [2]
    if subject_ids is None:
        subject_ids = [0, 1]
    if train_subject_ids is None:
        train_subject_ids = subject_ids[::2]
    if valid_subject_ids is None:
        valid_subject_ids = subject_ids[1::2]
    dataset = SleepPhysionet(
        subject_ids=subject_ids,
        recording_ids=recording_ids,
        crop_wake_mins=30,
        crop=crop,
    )
    preprocess(dataset, raw_preprocessors, n_jobs=-1)
    windows_dataset = create_windows_from_events(
        dataset,
        trial_start_offset_samples=0,
        trial_stop_offset_samples=0,
        window_size_samples=30 * 100,
        window_stride_samples=30 * 100,
        picks=picks,
        preload=True,
        mapping=SLEEP_MAPPING,
    )
    if window_preprocessors:
        preprocess(windows_dataset, window_preprocessors, n_jobs=-1)

    splits = windows_dataset.split(
        dict(train=train_subject_ids, valid=valid_subject_ids)
    )
    train_set, valid_set = splits["train"], splits["valid"]
    train_sampler = SequenceSampler(
        train_set.get_metadata(), 3, 3, randomize=True
    )
    valid_sampler = SequenceSampler(valid_set.get_metadata(), 3, 3)
    return train_set, valid_set, train_sampler, valid_sampler


def _balanced_accuracy_multi(model, X, y):
    y_pred = model.predict(X)
    return balanced_accuracy_score(y.flatten(), y_pred.flatten())


def _sleep_usleep(*, epochs: int, patience: int) -> TutorialArtifacts:
    tutorial_name = "plot_sleep_staging_usleep"
    train_set, valid_set, train_sampler, valid_sampler = _sleep_physionet_windows(
        crop=(0, 30 * 400),
        raw_preprocessors=[Preprocessor(robust_scale, channel_wise=True)],
    )
    y_train = [train_set[idx][1][1] for idx in train_sampler]
    class_weights = compute_class_weight(
        "balanced", classes=np.unique(y_train), y=y_train
    )

    _, device = _device_and_seed(seed=31)
    n_classes = 5
    classes = list(range(n_classes))
    in_chans, input_size_samples = train_set[0][0].shape
    model = USleep(
        n_chans=in_chans,
        sfreq=100,
        depth=12,
        with_skip_connection=True,
        n_outputs=n_classes,
        n_times=input_size_samples,
    )
    if device != "cpu":
        model.to(device)

    callbacks = [
        (
            "train_bal_acc",
            EpochScoring(
                scoring=_balanced_accuracy_multi,
                on_train=True,
                name="train_bal_acc",
                lower_is_better=False,
            ),
        ),
        (
            "valid_bal_acc",
            EpochScoring(
                scoring=_balanced_accuracy_multi,
                on_train=False,
                name="valid_bal_acc",
                lower_is_better=False,
            ),
        ),
        ("early_stopping", EarlyStopping(patience=patience, load_best=True)),
    ]
    clf = EEGClassifier(
        model,
        criterion=torch.nn.CrossEntropyLoss,
        criterion__weight=torch.Tensor(class_weights).to(device),
        optimizer=torch.optim.Adam,
        iterator_train__shuffle=False,
        iterator_train__sampler=train_sampler,
        iterator_valid__sampler=valid_sampler,
        train_split=predefined_split(valid_set),
        optimizer__lr=1e-3,
        batch_size=32,
        callbacks=callbacks,
        device=device,
        classes=classes,
    )
    clf.set_params(callbacks__valid_acc=None)
    clf.fit(train_set, y=None, epochs=epochs)
    metadata = {
        "best_valid_bal_acc": max(clf.history[:, "valid_bal_acc"]),
        "chance_level": 0.20,
        "display_metric_key": "best_valid_bal_acc",
        "display_metric_name": "balanced accuracy",
        "display_split_name": "held-out recording",
        "epochs_ran": len(clf.history),
        "epochs_requested": epochs,
        "final_valid_bal_acc": clf.history[-1, "valid_bal_acc"],
        "patience": patience,
        "short_run_epochs": 3,
        "use_safetensors": False,
        "tutorial": tutorial_name,
    }
    return TutorialArtifacts(
        clf=clf,
        repo_id=tutorial_repo_id(tutorial_name),
        metadata=metadata,
    )


def _sleep_attnsleep(*, epochs: int, patience: int) -> TutorialArtifacts:
    tutorial_name = "plot_sleep_staging_eldele2021"
    train_set, valid_set, train_sampler, valid_sampler = _sleep_physionet_windows(
        picks="Fpz-Cz",
        recording_ids=[1, 2],
        subject_ids=[0, 1, 2, 3, 4],
        train_subject_ids=[0, 1, 2, 3],
        valid_subject_ids=[4],
        raw_preprocessors=[
            Preprocessor(lambda data: multiply(data, 1e6), apply_on_array=True),
            Preprocessor("filter", l_freq=None, h_freq=30),
        ],
        window_preprocessors=[Preprocessor(standard_scale, channel_wise=True)],
    )
    train_set.target_transform = _get_center_label
    valid_set.target_transform = _get_center_label
    y_train = [train_set[idx][1] for idx in train_sampler]
    class_weights = compute_class_weight(
        "balanced", classes=np.unique(y_train), y=y_train
    )

    _, device = _device_and_seed(seed=31)
    n_classes = 5
    feat_extractor = AttnSleep(
        sfreq=100,
        n_outputs=n_classes,
        n_times=train_set[0][0].shape[1],
        drop_prob=0.3,
        return_feats=True,
    )
    model = nn.Sequential(
        TimeDistributed(feat_extractor),
        nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Dropout(0.5),
            nn.Linear(feat_extractor.len_last_layer * 3, n_classes),
        ),
    )
    if device != "cpu":
        model.to(device)

    callbacks = [
        (
            "train_bal_acc",
            EpochScoring(
                scoring="balanced_accuracy",
                on_train=True,
                name="train_bal_acc",
                lower_is_better=False,
            ),
        ),
        (
            "valid_bal_acc",
            EpochScoring(
                scoring="balanced_accuracy",
                on_train=False,
                name="valid_bal_acc",
                lower_is_better=False,
            ),
        ),
        (
            "lr_scheduler",
            LRScheduler("CosineAnnealingLR", T_max=max(1, epochs - 1)),
        ),
        (
            "grad_clip",
            GradientNormClipping(gradient_clip_value=1.0),
        ),
        (
            "early_stopping",
            EarlyStopping(
                monitor="valid_bal_acc",
                lower_is_better=False,
                patience=patience,
                load_best=True,
            ),
        ),
    ]
    wandb_result = _make_wandb_callback(tutorial_name, config={
        "model": "AttnSleep",
        "sfreq": 100,
        "drop_prob": 0.3,
        "n_windows": 3,
        "lr": 1e-3,
        "weight_decay": 1e-3,
        "label_smoothing": 0.1,
        "batch_size": 32,
        "epochs": epochs,
        "patience": patience,
        "train_subjects": [0, 1, 2, 3],
        "valid_subjects": [4, 5],
        "recording_ids": [1, 2],
    })
    if wandb_result is not None:
        callbacks.append(wandb_result[0])
    clf = EEGClassifier(
        model,
        criterion=torch.nn.CrossEntropyLoss,
        criterion__weight=torch.Tensor(class_weights).to(device),
        criterion__label_smoothing=0.1,
        optimizer=torch.optim.Adam,
        iterator_train__shuffle=False,
        iterator_train__sampler=train_sampler,
        iterator_valid__sampler=valid_sampler,
        train_split=predefined_split(valid_set),
        optimizer__lr=1e-3,
        optimizer__weight_decay=1e-3,
        batch_size=32,
        callbacks=callbacks,
        device=device,
        classes=np.unique(y_train),
    )
    clf.fit(train_set, y=None, epochs=epochs)
    metadata = {
        "best_valid_bal_acc": max(clf.history[:, "valid_bal_acc"]),
        "chance_level": 0.20,
        "display_metric_key": "best_valid_bal_acc",
        "display_metric_name": "balanced accuracy",
        "display_split_name": "held-out recording",
        "epochs_ran": len(clf.history),
        "epochs_requested": epochs,
        "final_valid_bal_acc": clf.history[-1, "valid_bal_acc"],
        "patience": patience,
        "short_run_epochs": 3,
        "use_safetensors": False,
        "tutorial": tutorial_name,
    }
    if wandb_result is not None:
        import wandb
        metadata["wandb_run_url"] = wandb_result[1].get_url()
        wandb.finish()
    return TutorialArtifacts(
        clf=clf,
        repo_id=tutorial_repo_id(tutorial_name),
        metadata=metadata,
    )


def _sleep_chambon(*, epochs: int, patience: int) -> TutorialArtifacts:
    tutorial_name = "plot_sleep_staging_chambon2018"
    train_set, valid_set, train_sampler, valid_sampler = _sleep_physionet_windows(
        raw_preprocessors=[
            Preprocessor(lambda data: multiply(data, 1e6), apply_on_array=True),
            Preprocessor("filter", l_freq=None, h_freq=30),
        ],
        window_preprocessors=[Preprocessor(standard_scale, channel_wise=True)],
    )
    train_set.target_transform = _get_center_label
    valid_set.target_transform = _get_center_label
    y_train = [train_set[idx][1] for idx in train_sampler]
    class_weights = compute_class_weight(
        "balanced", classes=np.unique(y_train), y=y_train
    )

    _, device = _device_and_seed(seed=31)
    n_classes = 5
    n_channels, input_size_samples = train_set[0][0].shape
    feat_extractor = SleepStagerChambon2018(
        n_channels,
        100,
        n_outputs=n_classes,
        n_times=input_size_samples,
        return_feats=True,
    )
    model = nn.Sequential(
        TimeDistributed(feat_extractor),
        nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Dropout(0.5),
            nn.Linear(feat_extractor.len_last_layer * 3, n_classes),
        ),
    )
    if device != "cpu":
        model.to(device)

    callbacks = [
        (
            "train_bal_acc",
            EpochScoring(
                scoring="balanced_accuracy",
                on_train=True,
                name="train_bal_acc",
                lower_is_better=False,
            ),
        ),
        (
            "valid_bal_acc",
            EpochScoring(
                scoring="balanced_accuracy",
                on_train=False,
                name="valid_bal_acc",
                lower_is_better=False,
            ),
        ),
        ("early_stopping", EarlyStopping(patience=patience, load_best=True)),
    ]
    clf = EEGClassifier(
        model,
        criterion=torch.nn.CrossEntropyLoss,
        criterion__weight=torch.Tensor(class_weights).to(device),
        optimizer=torch.optim.Adam,
        iterator_train__shuffle=False,
        iterator_train__sampler=train_sampler,
        iterator_valid__sampler=valid_sampler,
        train_split=predefined_split(valid_set),
        optimizer__lr=1e-3,
        batch_size=32,
        callbacks=callbacks,
        device=device,
        classes=np.unique(y_train),
    )
    clf.fit(train_set, y=None, epochs=epochs)
    metadata = {
        "best_valid_bal_acc": max(clf.history[:, "valid_bal_acc"]),
        "chance_level": 0.20,
        "display_metric_key": "best_valid_bal_acc",
        "display_metric_name": "balanced accuracy",
        "display_split_name": "held-out recording",
        "epochs_ran": len(clf.history),
        "epochs_requested": epochs,
        "final_valid_bal_acc": clf.history[-1, "valid_bal_acc"],
        "patience": patience,
        "short_run_epochs": 10,
        "tutorial": tutorial_name,
    }
    return TutorialArtifacts(
        clf=clf,
        repo_id=tutorial_repo_id(tutorial_name),
        metadata=metadata,
    )


def _cropped_shallow(*, subject_id: int, epochs: int, patience: int) -> TutorialArtifacts:
    tutorial_name = "plot_bcic_iv_2a_moabb_cropped"
    dataset = MOABBDataset(dataset_name="BNCI2014_001", subject_ids=[subject_id])
    _common_preprocessing(dataset, n_jobs=-1)

    _, device = _device_and_seed()
    n_times = 1000
    model = ShallowFBCSPNet(22, 4, n_times=n_times, final_conv_length=30)
    model.to_dense_prediction_model()
    n_preds_per_input = model.get_output_shape()[2]

    sfreq = dataset.datasets[0].raw.info["sfreq"]
    windows_dataset = create_windows_from_events(
        dataset,
        trial_start_offset_samples=int(-0.5 * sfreq),
        trial_stop_offset_samples=0,
        window_size_samples=n_times,
        window_stride_samples=n_preds_per_input,
        drop_last_window=False,
        preload=True,
    )
    splitted = windows_dataset.split("session")
    train_set = splitted["0train"]
    valid_set = splitted["1test"]

    clf = EEGClassifier(
        model,
        cropped=True,
        criterion=CroppedLoss,
        criterion__loss_function=torch.nn.functional.cross_entropy,
        optimizer=torch.optim.AdamW,
        train_split=predefined_split(valid_set),
        optimizer__lr=0.0625 * 0.01,
        optimizer__weight_decay=0,
        iterator_train__shuffle=True,
        batch_size=64,
        callbacks=_training_callbacks(epochs, patience),
        device=device,
        classes=list(range(4)),
    )
    clf.fit(train_set, y=None, epochs=epochs)
    metadata = {
        "best_valid_accuracy": max(clf.history[:, "valid_accuracy"]),
        "chance_level": 0.25,
        "display_metric_key": "best_valid_accuracy",
        "display_metric_name": "accuracy",
        "display_split_name": "held-out session",
        "epochs_ran": len(clf.history),
        "epochs_requested": epochs,
        "final_valid_accuracy": clf.history[-1, "valid_accuracy"],
        "patience": patience,
        "short_run_epochs": 2,
        "subject_id": subject_id,
        "tutorial": tutorial_name,
    }
    return TutorialArtifacts(
        clf=clf,
        repo_id=tutorial_repo_id(tutorial_name),
        metadata=metadata,
    )


def _eegprep_eegnex(*, subject_id: int, epochs: int, patience: int) -> TutorialArtifacts:
    tutorial_name = "plot_bcic_iv_2a_eegprep_cleaning"
    dataset = MOABBDataset(dataset_name="BNCI2014_001", subject_ids=[subject_id])
    _common_preprocessing(dataset, use_eegprep=True, n_jobs=-1)
    windows_dataset = _trialwise_windows(dataset)
    splitted = windows_dataset.split("session")
    train_set = splitted["0train"]
    valid_set = splitted["1test"]

    _, device = _device_and_seed()
    model = EEGNeX(
        n_chans=train_set[0][0].shape[0],
        n_outputs=4,
        n_times=train_set[0][0].shape[1],
    )
    clf = EEGClassifier(
        model,
        criterion=torch.nn.CrossEntropyLoss,
        optimizer=torch.optim.AdamW,
        train_split=predefined_split(valid_set),
        optimizer__lr=1e-3,
        optimizer__weight_decay=0,
        batch_size=64,
        callbacks=_training_callbacks(epochs, patience),
        device=device,
        classes=list(range(4)),
    )
    clf.fit(train_set, y=None, epochs=epochs)
    metadata = {
        "best_valid_accuracy": max(clf.history[:, "valid_accuracy"]),
        "chance_level": 0.25,
        "display_metric_key": "best_valid_accuracy",
        "display_metric_name": "accuracy",
        "display_split_name": "held-out session",
        "epochs_ran": len(clf.history),
        "epochs_requested": epochs,
        "final_valid_accuracy": clf.history[-1, "valid_accuracy"],
        "patience": patience,
        "short_run_epochs": 4,
        "subject_id": subject_id,
        "tutorial": tutorial_name,
    }
    return TutorialArtifacts(
        clf=clf,
        repo_id=tutorial_repo_id(tutorial_name),
        metadata=metadata,
    )


def _make_search_candidate(
    transform,
    *,
    augmentation: str,
    magnitude: float,
    display_magnitude: float,
    axis_label: str,
    candidate_label: str,
    sort_order: int,
):
    transform._tutorial_candidate_label = candidate_label
    transform._tutorial_augmentation = augmentation
    transform._tutorial_magnitude = magnitude
    transform._tutorial_display_magnitude = display_magnitude
    transform._tutorial_axis_label = axis_label
    transform._tutorial_sort_order = sort_order
    return transform


def _data_augmentation_candidates(sfreq: float, seed: int):
    candidates = [
        _make_search_candidate(
            IdentityTransform(),
            augmentation="IdentityTransform",
            magnitude=0.0,
            display_magnitude=0.0,
            axis_label="Identity baseline",
            candidate_label="IdentityTransform()",
            sort_order=0,
        )
    ]

    for phase_noise in (0.1, 0.3, 0.5, 0.7, 0.9):
        candidates.append(
            _make_search_candidate(
                FTSurrogate(
                    probability=0.5,
                    phase_noise_magnitude=phase_noise,
                    random_state=seed,
                ),
                augmentation="FTSurrogate",
                magnitude=phase_noise,
                display_magnitude=phase_noise,
                axis_label="Phase noise magnitude",
                candidate_label=f"FTSurrogate(phase_noise_magnitude={phase_noise:.1f})",
                sort_order=1,
            )
        )

    for mask_len_samples in (100, 200, 300, 400, 500):
        candidates.append(
            _make_search_candidate(
                SmoothTimeMask(
                    probability=0.5,
                    mask_len_samples=mask_len_samples,
                    random_state=seed,
                ),
                augmentation="SmoothTimeMask",
                magnitude=mask_len_samples,
                display_magnitude=mask_len_samples / sfreq,
                axis_label="Mask length (s)",
                candidate_label=f"SmoothTimeMask(mask_len_samples={mask_len_samples})",
                sort_order=2,
            )
        )

    for p_drop in (0.2, 0.4, 0.6, 0.8, 1.0):
        candidates.append(
            _make_search_candidate(
                ChannelsDropout(probability=0.5, p_drop=p_drop, random_state=seed),
                augmentation="ChannelsDropout",
                magnitude=p_drop,
                display_magnitude=p_drop,
                axis_label="Drop probability",
                candidate_label=f"ChannelsDropout(p_drop={p_drop:.1f})",
                sort_order=3,
            )
        )

    return candidates


def _data_augmentation_search_table(cv_results: dict) -> pd.DataFrame:
    rows = []
    for index, params in enumerate(cv_results["params"]):
        transform = params["iterator_train__transforms"]
        rows.append(
            {
                "candidate_label": transform._tutorial_candidate_label,
                "augmentation": transform._tutorial_augmentation,
                "magnitude": transform._tutorial_magnitude,
                "display_magnitude": transform._tutorial_display_magnitude,
                "axis_label": transform._tutorial_axis_label,
                "sort_order": transform._tutorial_sort_order,
                "mean_training_accuracy": float(cv_results["mean_train_score"][index]),
                "std_training_accuracy": float(cv_results["std_train_score"][index]),
                "mean_validation_accuracy": float(cv_results["mean_test_score"][index]),
                "std_validation_accuracy": float(cv_results["std_test_score"][index]),
                "rank_validation_accuracy": int(cv_results["rank_test_score"][index]),
            }
        )

    search_results = pd.DataFrame(rows).sort_values(
        ["sort_order", "display_magnitude"]
    )
    identity_validation_score = float(
        search_results.loc[
            search_results["augmentation"] == "IdentityTransform",
            "mean_validation_accuracy",
        ].iloc[0]
    )
    identity_training_score = float(
        search_results.loc[
            search_results["augmentation"] == "IdentityTransform",
            "mean_training_accuracy",
        ].iloc[0]
    )
    search_results["relative_validation_improvement"] = (
        search_results["mean_validation_accuracy"] / identity_validation_score - 1
    )
    search_results["relative_training_improvement"] = (
        search_results["mean_training_accuracy"] / identity_training_score - 1
    )
    search_results["relative_validation_improvement_pct"] = (
        search_results["relative_validation_improvement"] * 100
    )
    search_results["relative_training_improvement_pct"] = (
        search_results["relative_training_improvement"] * 100
    )
    return search_results.reset_index(drop=True)


def _data_augmentation_search(*, subject_id: int, epochs: int, patience: int):
    tutorial_name = "plot_data_augmentation_search"
    dataset = MOABBDataset(dataset_name="BNCI2014_001", subject_ids=[subject_id])
    _common_preprocessing(dataset, n_jobs=-1)

    windows_dataset = _trialwise_windows(dataset)
    splitted = windows_dataset.split("session")
    train_set = splitted["0train"]
    eval_set = splitted["1test"]

    seed = 20200220
    sfreq = dataset.datasets[0].raw.info["sfreq"]
    search_candidates = _data_augmentation_candidates(sfreq, seed)

    _, device = _device_and_seed(seed=seed)
    model = ShallowFBCSPNet(
        n_chans=train_set[0][0].shape[0],
        n_outputs=4,
        n_times=train_set[0][0].shape[1],
        final_conv_length="auto",
    )
    if device != "cpu":
        model.to(device)

    clf = EEGClassifier(
        model,
        iterator_train=AugmentedDataLoader,
        iterator_train__transforms=[IdentityTransform()],
        criterion=torch.nn.CrossEntropyLoss,
        optimizer=torch.optim.AdamW,
        train_split=ValidSplit(0.2, stratified=True, random_state=seed),
        optimizer__lr=0.0625 * 0.01,
        optimizer__weight_decay=0,
        batch_size=64,
        callbacks=[
            "accuracy",
            ("lr_scheduler", LRScheduler("CosineAnnealingLR", T_max=max(1, epochs - 1))),
            ("early_stopping", EarlyStopping(patience=patience, load_best=True)),
        ],
        device=device,
        classes=list(range(4)),
    )
    clf.verbose = 0

    train_X = SliceDataset(train_set, idx=0)
    train_y = np.array(list(SliceDataset(train_set, idx=1)))
    search = GridSearchCV(
        estimator=clf,
        param_grid={"iterator_train__transforms": search_candidates},
        cv=KFold(n_splits=2, shuffle=True, random_state=seed),
        n_jobs=-1,
        return_train_score=True,
        scoring="accuracy",
        refit=True,
        verbose=1,
        error_score="raise",
    )
    with parallel_backend("threading", n_jobs=-1):
        search.fit(train_X, train_y, epochs=epochs)

    search_results = _data_augmentation_search_table(search.cv_results_)
    best_run = search_results.sort_values("mean_validation_accuracy", ascending=False).iloc[0]
    identity_validation_score = float(
        search_results.loc[
            search_results["augmentation"] == "IdentityTransform",
            "mean_validation_accuracy",
        ].iloc[0]
    )
    eval_accuracy = float(
        search.score(SliceDataset(eval_set, idx=0), SliceDataset(eval_set, idx=1))
    )
    metadata = {
        "best_augmentation": best_run["augmentation"],
        "best_candidate": best_run["candidate_label"],
        "best_magnitude": float(best_run["magnitude"]),
        "best_relative_validation_improvement": float(
            best_run["relative_validation_improvement"]
        ),
        "chance_level": 0.25,
        "cv_splits": 2,
        "display_metric_key": "eval_accuracy",
        "display_metric_name": "accuracy",
        "display_split_name": "held-out session",
        "epochs_requested": epochs,
        "eval_accuracy": eval_accuracy,
        "identity_validation_score": identity_validation_score,
        "patience": patience,
        "search_candidates": len(search_candidates),
        "search_magnitudes_per_augmentation": 5,
        "short_run_epochs": 2,
        "training_score": float(best_run["mean_training_accuracy"]),
        "tutorial": tutorial_name,
        "validation_score": float(best_run["mean_validation_accuracy"]),
    }
    return tutorial_repo_id(tutorial_name), metadata, search_results


def train_tutorial(
    tutorial_name: str,
    *,
    subject_id: int,
    epochs: int,
    patience: int,
) -> TutorialArtifacts:
    if tutorial_name == "plot_bcic_iv_2a_moabb_trial":
        return _trialwise_shallow(
            tutorial_name,
            subject_id=subject_id,
            epochs=epochs,
            patience=patience,
        )
    if tutorial_name == "plot_bcic_iv_2a_moabb_cropped":
        return _cropped_shallow(
            subject_id=subject_id,
            epochs=epochs,
            patience=patience,
    )
    if tutorial_name == "plot_bcic_iv_2a_eegprep_cleaning":
        return _eegprep_eegnex(
            subject_id=subject_id,
            epochs=epochs,
            patience=patience,
        )
    if tutorial_name == "bcic_iv_4_ecog_trial":
        return _ecog_trialwise(epochs=epochs, patience=patience)
    if tutorial_name == "bcic_iv_4_ecog_cropped":
        return _ecog_cropped(epochs=epochs, patience=patience)
    if tutorial_name == "plot_sleep_staging_usleep":
        return _sleep_usleep(epochs=epochs, patience=patience)
    if tutorial_name == "plot_sleep_staging_eldele2021":
        return _sleep_attnsleep(epochs=epochs, patience=patience)
    if tutorial_name == "plot_sleep_staging_chambon2018":
        return _sleep_chambon(epochs=epochs, patience=patience)
    raise ValueError(f"Unsupported tutorial: {tutorial_name}")


def _tutorial_example_path(tutorial_name: str) -> str:
    matches = list((REPO_ROOT / "examples").rglob(f"{tutorial_name}.py"))
    if not matches:
        return f"examples/{tutorial_name}.py"
    return str(matches[0].relative_to(REPO_ROOT))


def _build_readme(artifacts: TutorialArtifacts) -> str:
    tutorial = artifacts.metadata["tutorial"]
    example_path = _tutorial_example_path(tutorial)
    return (
        f"# {artifacts.repo_id.split('/')[-1]}\n\n"
        "Pretrained artifacts for the Braindecode tutorial "
        f"`{example_path}`.\n\n"
        "These files are meant to be loaded by the tutorial so the docs can "
        "show stable predictions without retraining the model from scratch.\n\n"
        "## Stored files\n\n"
        "- `params.safetensors`: classifier parameters\n"
        "- `history.json`: Skorch training history used by the tutorial plots\n"
        "- `metadata.json`: summary metrics for the stored checkpoint\n"
    )


def _build_search_readme(repo_id: str, tutorial_name: str) -> str:
    example_path = _tutorial_example_path(tutorial_name)
    return (
        f"# {repo_id.split('/')[-1]}\n\n"
        "Saved search results for the Braindecode tutorial "
        f"`{example_path}`.\n\n"
        "These files are meant to be loaded by the tutorial so the docs can "
        "plot the offline augmentation search without rerunning the full "
        "GridSearchCV procedure.\n\n"
        "## Stored files\n\n"
        f"- `{SEARCH_RESULTS_FILENAME}`: tidy cross-validation search summary\n"
        "- `metadata.json`: summary metrics for the saved search\n"
    )


def _save_search_artifacts(
    output_dir: Path,
    *,
    search_results: pd.DataFrame,
    metadata: dict,
    readme_text: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    search_results.to_csv(output_dir / SEARCH_RESULTS_FILENAME, index=False)
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    (output_dir / "README.md").write_text(readme_text)


def _parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train Braindecode tutorial checkpoints and optionally push them to "
            "Hugging Face Hub."
        )
    )
    parser.add_argument(
        "--tutorial",
        choices=("all",) + AVAILABLE_TUTORIALS,
        default="all",
        help="Tutorial checkpoint to train.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Maximum number of epochs to train.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Early stopping patience.",
    )
    parser.add_argument(
        "--subject-id",
        type=int,
        default=3,
        help="BCIC IV 2a subject id used by the tutorials.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("tutorial_artifacts"),
        help="Directory where the generated artifacts are written.",
    )
    parser.add_argument(
        "--push",
        action="store_true",
        help="Upload the generated artifacts to Hugging Face Hub.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create private Hugging Face repos when pushing.",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Log training metrics to Weights & Biases.",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="braindecode-tutorials",
        help="Weights & Biases project name.",
    )
    return parser.parse_args()


def main():
    args = _parse_args()

    global _WANDB_PROJECT
    if args.wandb:
        _WANDB_PROJECT = args.wandb_project

    tutorial_names = (
        AVAILABLE_TUTORIALS if args.tutorial == "all" else (args.tutorial,)
    )

    for tutorial_name in tutorial_names:
        print(f"Training {tutorial_name}...")
        if tutorial_name == "plot_data_augmentation_search":
            repo_id, metadata, search_results = _data_augmentation_search(
                subject_id=args.subject_id,
                epochs=args.epochs,
                patience=args.patience,
            )
            output_dir = args.output_root / tutorial_name
            _save_search_artifacts(
                output_dir,
                search_results=search_results,
                metadata=metadata,
                readme_text=_build_search_readme(repo_id, tutorial_name),
            )
            print(f"Saved artifacts to {output_dir}")

            if args.push:
                url = upload_tutorial_artifacts(
                    repo_id=repo_id,
                    artifact_dir=output_dir,
                    private=args.private,
                )
                print(f"Pushed artifacts to {url}")
            continue

        artifacts = train_tutorial(
            tutorial_name,
            subject_id=args.subject_id,
            epochs=args.epochs,
            patience=args.patience,
        )
        output_dir = args.output_root / tutorial_name
        use_safetensors = artifacts.metadata.get("use_safetensors", True)
        if tutorial_name == "plot_sleep_staging_eldele2021":
            use_safetensors = False
        save_tutorial_checkpoint(
            artifacts.clf,
            output_dir,
            metadata=artifacts.metadata,
            readme_text=_build_readme(artifacts),
            use_safetensors=use_safetensors,
        )
        _save_loss_curve(artifacts.clf, output_dir, tutorial_name)
        print(f"Saved artifacts to {output_dir}")

        if args.push:
            url = upload_tutorial_artifacts(
                repo_id=artifacts.repo_id,
                artifact_dir=output_dir,
                private=args.private,
            )
            print(f"Pushed artifacts to {url}")


if __name__ == "__main__":
    main()
