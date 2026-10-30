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
    "CroppedTimeSeriesEpochScoring",
    "CroppedTrialEpochScoring",
    "DanceLoss",
    "PostEpochTrainScoring",
    "TimeSeriesLoss",
    "TrialPredictionChunk",
    "f1_event",
    "iter_trial_prediction_chunks",
    "iter_trial_predictions",
    "mixup_criterion",
    "predict_trials",
    "trial_preds_from_window_preds",
]
