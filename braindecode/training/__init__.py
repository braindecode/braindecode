"""
Functionality for skorch-based training.
"""

from .losses import CroppedLoss, DanceLoss, TimeSeriesLoss, mixup_criterion
from .scoring import (
    CroppedTimeSeriesEpochScoring,
    CroppedTrialEpochScoring,
    PostEpochTrainScoring,
    TrialPredictionChunk,
    f1_event,
    iter_trial_prediction_chunks,
    iter_trial_predictions,
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
    "iter_trial_predictions",
    "iter_trial_prediction_chunks",
    "TrialPredictionChunk",
    "f1_event",
]
