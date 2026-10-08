# Authors: lindicaphxag-tech
#
# License: BSD-3-Clause

"""Contrastive EEG-text representation learning.

Optional external text-encoder integration
------------------------------------------
This module does not itself distribute pretrained EEGCLIP weights.
For paired text descriptions, users may supply their own external
language encoder without making Transformers a Braindecode dependency::

    from transformers import AutoModel, AutoTokenizer
    from braindecode.models import EEGCLIP

    tokenizer = AutoTokenizer.from_pretrained("medicalai/ClinicalBERT")
    text_encoder = AutoModel.from_pretrained("medicalai/ClinicalBERT")
    model = EEGCLIP(
        n_chans=21,
        n_times=1000,
        n_outputs=64,
        text_encoder=text_encoder,
        text_embedding_dim=text_encoder.config.hidden_size,
    )
    tokens = tokenizer(reports, padding=True, return_tensors="pt")
    output = model.forward_paired(
        eeg_batch,
        tokens["input_ids"],
        attention_mask=tokens["attention_mask"],
    )
    loss = model.contrastive_loss(
        output["eeg_embeds"], output["text_embeds"]
    )
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from braindecode.models.base import EEGModuleMixin
from braindecode.models.deep4 import Deep4Net


class EEGCLIP(EEGModuleMixin, nn.Module, license="bsd-3-clause"):
    r"""Dual encoder for contrastive alignment of EEG and text [eegclip]_.

    :bdg-danger:`Foundation Model` :bdg-success:`Convolution`

    .. figure:: https://www.frontiersin.org/files/Articles/1625731/frobt-12-1625731-HTML/image_m/frobt-12-1625731-g001.jpg
        :align: center
        :alt: EEG-CLIP overview (N'dir et al., 2025, Fig. 1).

    EEG-CLIP learns a shared embedding space for paired EEG recordings and
    clinical text reports with a symmetric contrastive objective. The default
    EEG encoder is a dense-prediction :class:`~braindecode.models.Deep4Net`
    whose 128 log-softmax outputs per time step are projected and averaged
    over time, as in the authors' code. The text encoder is any
    :class:`torch.nn.Module` (e.g. a Hugging Face ClinicalBERT, kept as an
    optional user dependency); with ``text_encoder=None`` the text inputs are
    precomputed features.

    ``forward(X)`` returns the projected EEG embeddings only, so the model
    keeps braindecode's Tensor-valued contract. Train with
    :meth:`forward_paired` and :meth:`contrastive_loss`; score zero-shot with
    :meth:`compute_logits` against embedded candidate descriptions.

    Parameters
    ----------
    text_encoder : torch.nn.Module | None
        Text encoder called as ``text_encoder(text_inputs, **kwargs)``. It
        returns a tensor, or a tuple / Hugging Face output whose first element
        is the tensor. ``None`` treats ``text_inputs`` as precomputed features.
    text_embedding_dim : int
        Dimension of the pooled text features.
    eeg_encoder : torch.nn.Module | None
        EEG encoder returning ``(batch, features)`` or
        ``(batch, features, time)``. ``None`` builds the default Deep4Net.
    eeg_embedding_dim : int
        Feature dimension of the EEG encoder output.
    text_pooling : {"cls", "mean"}
        Pooling of token-sequence text outputs: first token, or the mean over
        tokens (masked when ``attention_mask`` is given).
    projection_layers : int
        Linear layers in each projection head; 3 as in the paper.
    activation : type[nn.Module]
        Activation of the non-final projection blocks.
    drop_prob : float
        Dropout probability of the non-final projection blocks.
    initial_temperature : float
        The logit scale starts at ``log(1 / initial_temperature)`` and, as in
        the authors' code, multiplies the similarities without ``exp``.

    Notes
    -----
    ``n_outputs`` is the shared embedding dimension, not a number of classes.
    The released ``modelsexample.ckpt`` predates the authors' current
    projection head and is not loadable. Models with a custom ``eeg_encoder``
    or ``text_encoder`` cannot be rebuilt from a config, so :meth:`get_config`
    and ``save_pretrained`` raise for them; save their ``state_dict``.

    The authors' repository has no license file; this implementation follows
    braindecode's BSD-3-Clause license, and its projection head mirrors the
    authors' five-line ``ProjectionHead``.

    Examples
    --------
    Train on paired EEG windows and precomputed text features::

        import torch
        from braindecode.models import EEGCLIP

        model = EEGCLIP(
            n_chans=21, n_times=1000, n_outputs=64, text_embedding_dim=768
        )
        eeg_windows = torch.randn(2, 21, 1000)
        text_features = torch.randn(2, 768)
        paired = model.forward_paired(eeg_windows, text_features)
        loss = model.contrastive_loss(
            paired["eeg_embeds"], paired["text_embeds"]
        )

    References
    ----------
    .. [eegclip] N'dir, T. C., Schirrmeister, R. T., & Ball, T. (2025).
       EEG-CLIP: Learning EEG representations from natural language descriptions.
       Frontiers in Robotics and AI, 12, 1625731.
       https://doi.org/10.3389/frobt.2025.1625731

    .. versionadded:: 1.9
    """

    def __init__(
        self,
        n_outputs=None,
        n_chans=None,
        n_times=None,
        text_encoder=None,
        text_embedding_dim=768,
        eeg_encoder=None,
        eeg_embedding_dim=128,
        text_pooling="cls",
        projection_layers=3,
        activation: type[nn.Module] = nn.ReLU,
        drop_prob=0.1,
        initial_temperature=0.07,
        chs_info=None,
        input_window_seconds=None,
        sfreq=None,
    ):
        super().__init__(
            n_outputs=n_outputs,
            n_chans=n_chans,
            chs_info=chs_info,
            n_times=n_times,
            input_window_seconds=input_window_seconds,
            sfreq=sfreq,
        )
        if text_pooling not in ("cls", "mean"):
            raise ValueError("text_pooling must be 'cls' or 'mean'.")
        self.text_pooling = text_pooling
        self.text_embedding_dim = text_embedding_dim
        self.eeg_embedding_dim = eeg_embedding_dim
        self.projection_layers = projection_layers
        self.activation = activation
        self.drop_prob = drop_prob
        self._custom_encoders = eeg_encoder is not None or text_encoder is not None

        if eeg_encoder is None:
            eeg_encoder = Deep4Net(
                n_chans=self.n_chans,
                n_times=self.n_times,
                n_outputs=eeg_embedding_dim,
                final_conv_length=2,
                stride_before_pool=True,
            )
            eeg_encoder.to_dense_prediction_model()
            # The authors' Deep4Net (braindecode <= 0.8) ends with a
            # log-softmax over its outputs; their projection sees log-probs.
            eeg_encoder.final_layer.add_module("log_softmax", nn.LogSoftmax(dim=1))
        self.eeg_encoder = eeg_encoder
        self.text_encoder = text_encoder

        self.text_projection = _projection_head(
            text_embedding_dim, self.n_outputs, projection_layers, activation, drop_prob
        )
        # EEG projection, named final_layer for braindecode's head conventions.
        self.final_layer = _projection_head(
            eeg_embedding_dim, self.n_outputs, projection_layers, activation, drop_prob
        )
        self.logit_scale = nn.Parameter(
            torch.tensor(math.log(1.0 / initial_temperature))
        )

    def forward(self, X):
        """Return projected EEG embeddings, shape ``(batch, n_outputs)``."""
        features = self.eeg_encoder(X)
        if features.ndim == 2:
            features = features.unsqueeze(-1)
        batch_size = features.shape[0]
        # Project every dense prediction, then average over time.
        per_step = features.transpose(1, 2).reshape(-1, features.shape[1])
        projected = self.final_layer(per_step)
        return projected.reshape(batch_size, -1, projected.shape[-1]).mean(dim=1)

    def encode_text(self, text_inputs, attention_mask=None, **text_kwargs):
        """Encode text tokens or features as shared-space projection vectors."""
        features = text_inputs
        if self.text_encoder is not None:
            if attention_mask is not None:
                text_kwargs["attention_mask"] = attention_mask
            features = self.text_encoder(text_inputs, **text_kwargs)
        if not isinstance(features, torch.Tensor):
            features = features[0]  # tuple or Hugging Face last_hidden_state
        if features.ndim == 3:
            if self.text_pooling == "cls":
                features = features[:, 0]
            elif attention_mask is None:
                features = features.mean(dim=1)
            else:
                mask = attention_mask.to(features.dtype).unsqueeze(-1)
                features = (features * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        return self.text_projection(features)

    def compute_logits(self, eeg_embeds, text_embeds):
        """Return EEG-to-text and text-to-EEG similarity logits.

        As in the authors' ``ClipLoss``, embeddings are not L2-normalized and
        the raw ``logit_scale`` multiplies the dot products.
        """
        logits_per_eeg = self.logit_scale * eeg_embeds @ text_embeds.T
        return logits_per_eeg, logits_per_eeg.T

    def contrastive_loss(self, eeg_embeds, text_embeds):
        """Symmetric cross-entropy over a batch of paired EEG/text embeddings."""
        logits_per_eeg, logits_per_text = self.compute_logits(eeg_embeds, text_embeds)
        labels = torch.arange(eeg_embeds.shape[0], device=eeg_embeds.device)
        return (
            F.cross_entropy(logits_per_eeg, labels)
            + F.cross_entropy(logits_per_text, labels)
        ) / 2

    def forward_paired(self, X, text_inputs, attention_mask=None, **text_kwargs):
        """Return paired EEG/text embeddings and bidirectional similarity logits."""
        eeg_embeds = self(X)
        text_embeds = self.encode_text(
            text_inputs, attention_mask=attention_mask, **text_kwargs
        )
        logits_per_eeg, logits_per_text = self.compute_logits(eeg_embeds, text_embeds)
        return {
            "eeg_embeds": eeg_embeds,
            "text_embeds": text_embeds,
            "logits_per_eeg": logits_per_eeg,
            "logits_per_text": logits_per_text,
        }

    def reset_head(self, n_outputs):
        """Rebuild both projection heads for a new shared embedding dimension."""
        self._set_n_outputs(n_outputs)
        for name, in_dim in (
            ("text_projection", self.text_embedding_dim),
            ("final_layer", self.eeg_embedding_dim),
        ):
            old_head = getattr(self, name)
            new_head = _projection_head(
                in_dim,
                n_outputs,
                self.projection_layers,
                self.activation,
                self.drop_prob,
            )
            new_head.to(next(old_head.parameters())).train(old_head.training)
            setattr(self, name, new_head)

    def get_config(self):
        """Return the config; custom encoders cannot be rebuilt from one."""
        if self._custom_encoders:
            raise ValueError(
                "EEGCLIP cannot serialize custom encoder architectures; build "
                "the encoders yourself and save the model state_dict."
            )
        return super().get_config()

    def _save_pretrained(self, save_directory):
        self.get_config()  # raises for custom encoders
        return super()._save_pretrained(save_directory)


def _projection_head(in_dim, out_dim, n_layers, activation, drop_prob):
    """Authors' ``ProjectionHead``: (Linear, BatchNorm, act, Dropout) blocks, then Linear.

    ``braindecode.modules.MLP`` uses LayerNorm/GELU, not BatchNorm, so it does not fit.
    """
    layers = []
    for _ in range(n_layers - 1):
        layers += [
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            activation(),
            nn.Dropout(drop_prob),
        ]
        in_dim = out_dim
    layers.append(nn.Linear(in_dim, out_dim))
    return nn.Sequential(*layers)
