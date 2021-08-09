"""
Functionality for skorch-based training.
"""


from .losses import CroppedLoss, mixup_criterion
from .scoring import (CroppedTrialEpochScoring, PostEpochTrainScoring,
                      trial_preds_from_window_preds, predict_trials)
