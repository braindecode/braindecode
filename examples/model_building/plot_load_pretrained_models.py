# type: ignore
""".. _load-pretrained-models:

Loading and Adapting Pretrained Foundation Models
=================================================

This tutorial demonstrates how to load pretrained EEG foundation models from
the Hugging Face Hub, adapt them to new tasks, extract features, and
save/restore full model configurations.

Braindecode provides a consistent API across all foundation models
(:class:`~braindecode.models.BENDR`, :class:`~braindecode.models.BIOT`,
:class:`~braindecode.models.CBraMod`, :class:`~braindecode.models.EEGPT`,
:class:`~braindecode.models.Labram`, :class:`~braindecode.models.REVE`,
:class:`~braindecode.models.SignalJEPA` variants), inspired by the
`Hugging Face transformers <https://huggingface.co/docs/transformers>`_
library.

.. contents:: This example covers:
   :local:
   :depth: 2
"""

# Authors: Bruno Aristimunha <b.aristimunha@gmail.com>
#
# License: BSD (3-clause)

import json
import warnings

import torch

warnings.simplefilter("ignore")

######################################################################
# Loading a pretrained model
# --------------------------
#
# All braindecode models that inherit from
# :class:`~braindecode.models.base.EEGModuleMixin` support the
# ``from_pretrained`` method, which downloads the model weights and full
# configuration from the Hugging Face Hub.
#

from braindecode.models import BENDR

model = BENDR.from_pretrained("braindecode/braindecode-bendr")
print(f"Loaded BENDR with n_outputs={model.n_outputs}")

######################################################################
# The loaded model is ready for inference:

x = torch.randn(2, 20, 768)  # (batch, n_chans, n_times)
model.eval()
with torch.no_grad():
    out = model(x)
print(f"Output shape: {out.shape}")

######################################################################
# Adapting to a new task
# ----------------------
#
# Changing the number of outputs
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# When fine-tuning a pretrained model on a dataset with a different number
# of classes, pass ``n_outputs`` directly to ``from_pretrained``. The
# backbone weights are loaded and the classification head is automatically
# rebuilt:

model = BENDR.from_pretrained("braindecode/braindecode-bendr", n_outputs=4)
print(f"n_outputs after loading: {model.n_outputs}")

with torch.no_grad():
    out = model(x)
print(f"Output shape with 4 classes: {out.shape}")

######################################################################
# You can also swap the head at any time using ``reset_head``:

model.reset_head(10)
print(f"n_outputs after reset_head: {model.n_outputs}")

with torch.no_grad():
    out = model(x)
print(f"Output shape after reset: {out.shape}")

######################################################################
# Extracting features
# -------------------
#
# All foundation models support ``return_features=True`` in their
# ``forward()`` method.  This returns a consistent dictionary with
# ``"features"`` (the encoder embeddings before the classification head)
# and ``"cls_token"`` (the CLS token if the model has one, otherwise
# ``None``):

model.eval()
with torch.no_grad():
    out = model(x, return_features=True)

print(f"Type: {type(out)}")
print(f"Features shape: {out['features'].shape}")
print(f"CLS token: {out['cls_token']}")

######################################################################
# This is useful for transfer learning, where you freeze the backbone
# and train only a new head on the extracted features.

######################################################################
# Saving and restoring configurations
# ------------------------------------
#
# ``get_config`` returns a JSON-serializable dictionary of **all**
# ``__init__`` parameters (not just the 6 EEG-specific ones).  This
# includes model-specific hyperparameters like ``encoder_h``,
# ``drop_prob``, ``activation``, etc.

config = model.get_config()
print(json.dumps({k: v for k, v in config.items() if k != "chs_info"}, indent=2))

######################################################################
# You can save this to a JSON file and reconstruct the model later
# (without weights) using ``from_config``:

model_copy = BENDR.from_config(config)
print(f"Reconstructed: n_outputs={model_copy.n_outputs}")

######################################################################
# When pushing to the Hub, the full config is saved automatically in
# ``config.json`` alongside the model weights:
#
# .. code-block:: python
#
#     model.push_to_hub("username/my-bendr-model")
#     # Later:
#     model = BENDR.from_pretrained("username/my-bendr-model")

######################################################################
# Working with different foundation models
# -----------------------------------------
#
# The same API works across all foundation models.  Here is a summary:
#
# .. list-table::
#    :header-rows: 1
#    :widths: 20 20 20 20 20
#
#    * - Model
#      - ``from_pretrained``
#      - ``reset_head``
#      - ``return_features``
#      - ``get_config``
#    * - BENDR
#      - Yes
#      - Yes
#      - Yes
#      - Yes
#    * - BIOT
#      - Yes
#      - Yes
#      - Yes
#      - Yes
#    * - CBraMod
#      - Yes
#      - Yes
#      - Yes
#      - Yes
#    * - EEGPT
#      - Yes
#      - Yes
#      - Yes
#      - Yes
#    * - Labram
#      - Yes
#      - Yes
#      - Yes (+ ``cls_token``)
#      - Yes
#    * - REVE
#      - Yes
#      - Yes
#      - Yes
#      - Yes
#    * - SignalJEPA variants
#      - Yes
#      - Yes
#      - Yes
#      - Yes
#
# The feature shapes differ between models (reflecting their architecture),
# but the API is always the same.

######################################################################
# References
# ----------
#
# .. [1] Kostas, D., Aroca-Ouellette, S., and Bhatt, F. (2021).
#    BENDR: Using Transformers and a Contrastive Self-Supervised Learning
#    Task to Learn From Massive Amounts of EEG Data.
#    Frontiers in Human Neuroscience, 15.
