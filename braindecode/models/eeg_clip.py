# Authors: lindicaphxag-tech
#
# License: BSD-3-Clause

"""Contrastive EEG-text representation learning."""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch
from torch import nn
from torch.nn import functional as F

from braindecode.models.base import EEGModuleMixin
from braindecode.models.deep4 import Deep4Net


class EEGCLIP(EEGModuleMixin, nn.Module):
    r"""Dual encoder for contrastive alignment of EEG and text.

    :bdg-danger:`Foundation Model` :bdg-success:`Convolution`

    EEG-CLIP learns a shared embedding space for paired EEG recordings and
    text descriptions using a symmetric contrastive objective [eegclip]_. The
    default EEG encoder is :class:`~braindecode.models.Deep4Net`. A text encoder
    can be supplied as any :class:`torch.nn.Module`; its outputs may be pooled
    vectors or token sequences. This keeps text-model dependencies optional and
    lets users provide a clinical language model or precomputed text embeddings.

    The model is trained with paired batches using :meth:`forward_paired` and
    :meth:`contrastive_loss`. Its ordinary ``forward(X)`` intentionally
    returns only projected EEG embeddings so the model preserves Braindecode's
    standard Tensor-valued inference contract (including skorch and TorchScript).
    Candidate text embeddings can be passed to :meth:`compute_logits` for
    zero-shot classification.

    Parameters
    ----------
    n_outputs : int
        Dimension of the shared EEG-text embedding space.
    n_chans : int
        Number of EEG channels.
    n_times : int
        Number of time samples in each input window.
    text_encoder : torch.nn.Module | None
        Optional text encoder. It must accept ``text_inputs`` and return a
        tensor, an object with ``last_hidden_state`` or ``pooler_output``, or a
        tuple whose first element is a tensor. If ``None``, ``text_inputs`` are
        treated as precomputed text features.
    text_embedding_dim : int
        Dimension of pooled text features before projection.
    eeg_encoder : torch.nn.Module | None
        Optional EEG encoder returning ``(batch, features)`` or
        ``(batch, features, time)``. Defaults to Deep4Net.
    eeg_embedding_dim : int
        Feature dimension emitted by the default EEG encoder, or expected from
        a custom EEG encoder.
    text_pooling : {"cls", "mean"}
        Pooling used for token-sequence text outputs. ``"cls"`` selects the
        first token; ``"mean"`` computes a masked mean when an attention mask
        is supplied.
    projection_layers : int
        Number of fully connected layers in each projection head. The default
        of three follows the architecture described in the EEG-CLIP paper.
    activation : type[nn.Module]
        Activation used in non-final projection blocks. Defaults to ReLU, as in
        the published EEG-CLIP architecture.
    drop_prob : float
        Dropout probability in non-final projection blocks.
    initial_temperature : float
        Initial temperature used to initialize the released EEG-CLIP logit-scale
        parameter as ``log(1 / temperature)``. For reference fidelity, the
        released implementation multiplies similarities by this learned raw
        parameter rather than exponentiating it.
    chs_info : list | None
        Channel information passed to :class:`~braindecode.models.EEGModuleMixin`.
    input_window_seconds : float | None
        Input duration passed to :class:`~braindecode.models.EEGModuleMixin`.
    sfreq : float | None
        Sampling frequency passed to :class:`~braindecode.models.EEGModuleMixin`.

    Notes
    -----
    ``n_outputs`` is the shared embedding dimension, not a number of diagnostic
    classes. The output of :meth:`compute_logits` is a matrix of EEG-to-text
    similarities; columns correspond to candidate text descriptions. Custom
    ``eeg_encoder`` or ``text_encoder`` modules are not included in Braindecode
    Hub configs; such models can use ``state_dict`` directly but cannot be saved
    with :meth:`get_config` or pushed to the Hub.

    Examples
    --------
    Use a Hugging Face text encoder without making Transformers a core
    dependency::

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

    Compare EEG windows against candidate descriptions with zero-shot logits::

        candidate_tokens = tokenizer(
            ["normal EEG", "EEG with epileptiform activity"],
            padding=True,
            return_tensors="pt",
        )
        eeg_embeds = model.encode_eeg(eeg_batch)
        text_embeds = model.encode_text(
            candidate_tokens["input_ids"],
            attention_mask=candidate_tokens["attention_mask"],
        )
        logits_per_eeg, _ = model.compute_logits(eeg_embeds, text_embeds)
        predicted_class = logits_per_eeg.argmax(dim=1)

    References
    ----------
    .. [eegclip] N'dir, T. C., Schirrmeister, R. T., & Ball, T. (2025).
       EEG-CLIP: Learning EEG representations from natural language descriptions.
       Frontiers in Robotics and AI, 12, 1625731.
       https://doi.org/10.3389/frobt.2025.1625731
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
        if text_pooling not in {"cls", "mean"}:
            raise ValueError("text_pooling must be 'cls' or 'mean'.")
        if not 0 < initial_temperature:
            raise ValueError("initial_temperature must be strictly positive.")
        if not 0 <= drop_prob < 1:
            raise ValueError("drop_prob must be in the interval [0, 1).")
        if projection_layers < 1:
            raise ValueError("projection_layers must be at least 1.")

        self.text_pooling = text_pooling
        self.text_embedding_dim = text_embedding_dim
        self.eeg_embedding_dim = eeg_embedding_dim
        self.projection_layers = projection_layers
        self.activation = activation
        self.drop_prob = drop_prob

        self._uses_default_eeg_encoder = eeg_encoder is None
        if eeg_encoder is None:
            eeg_encoder = Deep4Net(
                n_chans=self.n_chans,
                n_times=self.n_times,
                n_outputs=eeg_embedding_dim,
                final_conv_length=2,
                stride_before_pool=True,
            )
            eeg_encoder.to_dense_prediction_model()
        self.eeg_encoder = eeg_encoder
        self.text_encoder = text_encoder if text_encoder is not None else nn.Identity()

        self.text_projection = self._make_projection(
            text_embedding_dim,
            self.n_outputs,
            projection_layers,
            activation,
            drop_prob,
        )
        # Braindecode's integration checks (and skorch wrappers) expect the
        # EEG prediction head to be one of the final registered child modules.
        self.final_layer = self._make_projection(
            eeg_embedding_dim,
            self.n_outputs,
            projection_layers,
            activation,
            drop_prob,
        )
        self.logit_scale = nn.Parameter(
            torch.tensor(math.log(1.0 / initial_temperature), dtype=torch.float32)
        )

    @staticmethod
    def _make_projection(
        input_dim, output_dim, projection_layers, activation, drop_prob
    ):
        # The published EEG-CLIP architecture uses three fully connected
        # projection layers with ReLU activations. The authors' released
        # ProjectionHead additionally applies BatchNorm and dropout after each
        # non-final layer. Keeping the depth configurable also makes the
        # released two-layer configuration reproducible.
        if projection_layers == 1:
            return nn.Sequential(nn.Linear(input_dim, output_dim))

        layers = [
            nn.Linear(input_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            activation(),
            nn.Dropout(drop_prob),
        ]
        for _ in range(projection_layers - 2):
            layers.extend(
                [
                    nn.Linear(output_dim, output_dim),
                    nn.BatchNorm1d(output_dim),
                    activation(),
                    nn.Dropout(drop_prob),
                ]
            )
        layers.append(nn.Linear(output_dim, output_dim))
        return nn.Sequential(*layers)

    def encode_eeg(self, X):
        """Encode EEG windows as shared-space projection vectors.

        The standard forward path owns the EEG-only computation so it remains
        self-contained for Braindecode's plain-module/TorchScript integration.
        This helper is the eager multimodal API alias.
        """
        return self.forward(X)

    @staticmethod
    def _get_text_features(outputs):
        if isinstance(outputs, torch.Tensor):
            return outputs
        if hasattr(outputs, "last_hidden_state"):
            return outputs.last_hidden_state
        if isinstance(outputs, Mapping):
            if "last_hidden_state" in outputs:
                return outputs["last_hidden_state"]
            if outputs.get("pooler_output") is not None:
                return outputs["pooler_output"]
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            return outputs.pooler_output
        if isinstance(outputs, (tuple, list)) and outputs:
            return outputs[0]
        raise TypeError(
            "text_encoder must return a tensor, a model output with "
            "last_hidden_state/pooler_output, or a tuple containing a tensor."
        )

    def encode_text(self, text_inputs, attention_mask=None, **text_kwargs):
        """Encode text tokens or features as shared-space projection vectors."""
        if isinstance(self.text_encoder, nn.Identity):
            outputs = text_inputs
        else:
            if attention_mask is not None:
                text_kwargs["attention_mask"] = attention_mask
            outputs = self.text_encoder(text_inputs, **text_kwargs)
        features = self._get_text_features(outputs)
        if not isinstance(features, torch.Tensor):
            raise TypeError("The text encoder output must contain a torch.Tensor.")
        if features.ndim == 3:
            if self.text_pooling == "cls":
                features = features[:, 0]
            elif attention_mask is None:
                features = features.mean(dim=1)
            else:
                mask = attention_mask.to(device=features.device, dtype=features.dtype)
                if mask.ndim != 2 or mask.shape != features.shape[:2]:
                    raise ValueError(
                        "attention_mask must match the batch and token dimensions "
                        "of the text encoder output."
                    )
                mask = mask.unsqueeze(-1)
                features = (features * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        if features.ndim != 2:
            raise ValueError("Pooled text features must have shape (batch, features).")
        expected_dim = self.text_projection[0].in_features
        if features.shape[1] != expected_dim:
            raise ValueError(
                f"Text encoder returned {features.shape[1]} features; "
                f"text_embedding_dim={expected_dim} was configured."
            )
        return self.text_projection(features)

    def compute_logits(self, eeg_embeds, text_embeds):
        """Return EEG-to-text and text-to-EEG similarity logits."""
        if eeg_embeds.ndim != 2 or text_embeds.ndim != 2:
            raise ValueError("EEG and text embeddings must both be two-dimensional.")
        if eeg_embeds.shape[1] != text_embeds.shape[1]:
            raise ValueError("EEG and text embeddings must have the same dimension.")
        # Match the released EEG-CLIP implementation exactly: projection
        # vectors are not L2-normalized, and ClipLoss multiplies their dot
        # product by the learned raw logit_scale parameter. Although the
        # parameter is initialized as log(1 / 0.07), the released source does
        # not exponentiate it before computing logits.
        logits_per_eeg = self.logit_scale * eeg_embeds @ text_embeds.T
        return logits_per_eeg, logits_per_eeg.T

    def contrastive_loss(self, eeg_embeds, text_embeds):
        """Compute symmetric cross-entropy for paired EEG/text batches."""
        if eeg_embeds.shape[0] != text_embeds.shape[0]:
            raise ValueError("EEG and text batches must contain paired examples.")
        if eeg_embeds.shape[0] == 0:
            raise ValueError("EEG and text batches must not be empty.")
        logits_per_eeg, logits_per_text = self.compute_logits(eeg_embeds, text_embeds)
        labels = torch.arange(eeg_embeds.shape[0], device=eeg_embeds.device)
        return (
            F.cross_entropy(logits_per_eeg, labels)
            + F.cross_entropy(logits_per_text, labels)
        ) / 2

    def forward(self, X):
        """Return projected EEG embeddings.

        This Tensor-only path is self-contained because Braindecode's generic
        integration converts models to a plain ``nn.Module`` before scripting.
        Use :meth:`forward_paired` for multimodal training.
        """
        features = self.eeg_encoder(X)
        if features.ndim == 3:
            if features.shape[1] != self.eeg_embedding_dim:
                raise ValueError(
                    "EEG encoder feature dimension does not match eeg_embedding_dim."
                )
            batch_size = features.shape[0]
            n_predictions = features.shape[2]
            temporal_features = features.transpose(1, 2).reshape(
                batch_size * n_predictions, self.eeg_embedding_dim
            )
            projected = self.final_layer(temporal_features)
            projected = projected.reshape(batch_size, n_predictions, self.n_outputs)
            return projected.mean(dim=1)

        if features.ndim != 2:
            raise ValueError(
                "eeg_encoder output must have shape (batch, features) or "
                "(batch, features, time)."
            )
        if features.shape[1] != self.eeg_embedding_dim:
            raise ValueError(
                "EEG encoder feature dimension does not match eeg_embedding_dim."
            )
        return self.final_layer(features)

    def forward_paired(self, X, text_inputs, attention_mask=None, **text_kwargs):
        """Return paired EEG/text embeddings and bidirectional similarity logits."""
        eeg_embeds = self.encode_eeg(X)
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
        """Reset both projection heads to a new shared embedding dimension."""
        if n_outputs <= 0:
            raise ValueError(f"n_outputs must be positive; got {n_outputs}.")
        text_projection = self._make_projection(
            self.text_embedding_dim,
            n_outputs,
            self.projection_layers,
            self.activation,
            self.drop_prob,
        )
        final_layer = self._make_projection(
            self.eeg_embedding_dim,
            n_outputs,
            self.projection_layers,
            self.activation,
            self.drop_prob,
        )
        # Match each replacement submodule's mode to the module it replaces.
        # This preserves eval mode and intentional mixed modes such as
        # Monte-Carlo dropout when changing the shared embedding dimension.
        for old_head, new_head in (
            (self.text_projection, text_projection),
            (self.final_layer, final_layer),
        ):
            # Head replacement must not silently move a fine-tuned model back
            # to CPU/float32. Match the existing projection's device and dtype,
            # as other Braindecode reset_head implementations do.
            new_head.to(next(old_head.parameters()))
            for old_module, new_module in zip(old_head.modules(), new_head.modules()):
                new_module.training = old_module.training
        self.text_projection = text_projection
        self.final_layer = final_layer
        self._set_n_outputs(n_outputs)

    def get_config(self):
        """Return the model config when its text encoder can be reconstructed."""
        self._ensure_text_encoder_is_serializable()
        return super().get_config()

    def _save_pretrained(self, save_directory):
        """Save this model only when its text encoder is represented in config."""
        self._ensure_text_encoder_is_serializable()
        return super()._save_pretrained(save_directory)

    def _ensure_text_encoder_is_serializable(self):
        if (
            not isinstance(self.text_encoder, nn.Identity)
            or not self._uses_default_eeg_encoder
        ):
            raise ValueError(
                "EEGCLIP cannot serialize custom encoder architectures. "
                "Use precomputed text features (text_encoder=None) for Hub "
                "round-trips with the default Deep4Net EEG encoder, or save "
                "and restore the full model state_dict with your encoders "
                "constructed separately."
            )
