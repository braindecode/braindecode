"""
Functionality for skorch-based training.
"""

from .losses import CroppedLoss, DanceLoss, TimeSeriesLoss, mixup_criterion
from .scoring import (
    CroppedTimeSeriesEpochScoring,
    CroppedTrialEpochScoring,
    PostEpochTrainScoring,
    predict_trials,
    trial_preds_from_window_preds,
)

__all__ = [
    "CroppedLoss",
    "DanceLoss",
    "mixup_criterion",
    "TimeSeriesLoss",
    "CroppedTrialEpochScoring",
    "PostEpochTrainScoring",
    "CroppedTimeSeriesEpochScoring",
    "trial_preds_from_window_preds",
    "predict_trials",
]
