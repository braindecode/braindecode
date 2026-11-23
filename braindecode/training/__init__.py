"""
Functionality for skorch-based training.
"""

from .losses import CroppedLoss, DanceLoss, TimeSeriesLoss, info_nce, mixup_criterion
from .scoring import (
    CroppedTimeSeriesEpochScoring,
    CroppedTrialEpochScoring,
    PostEpochTrainScoring,
    TrialPredictionChunk,
    f1_event,
    iter_trial_prediction_chunks,
    iter_trial_predictions,
    predict_trials,
    szcore_event_scores,
    trial_preds_from_window_preds,
)

__all__ = [
    "CroppedLoss",
    "DanceLoss",
    "mixup_criterion",
    "info_nce",
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
    "szcore_event_scores",
]
