"""
Functionality for skorch-based training.
"""


from .losses import CroppedLoss, mixup_criterion, TimeSeriesLoss
from .scoring import (CroppedTrialEpochScoring, PostEpochTrainScoring,
                      CroppedTimeSeriesEpochScoring, trial_preds_from_window_preds, predict_trials)
