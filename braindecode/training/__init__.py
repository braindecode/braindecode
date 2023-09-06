"""
Functionality for skorch-based training.
"""
# flake8: noqa

from .losses import CroppedLoss, TimeSeriesLoss, mixup_criterion
from .scoring import (CroppedTimeSeriesEpochScoring, CroppedTrialEpochScoring,
                      PostEpochTrainScoring, predict_trials,
                      trial_preds_from_window_preds)
