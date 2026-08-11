# Authors: Fashad Ahmed <Fashad-Ahmed@users.noreply.github.com>
#
# Code adapted from https://github.com/zou-group/sleepfm-clinical
#
# License: Creative Commons Attribution-NonCommercial 4.0 International
# This derivative is not covered by Braindecode's BSD-3 license.

"""SleepFM models for multimodal polysomnography."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping

import torch
from torch import nn

from braindecode.functional import sinusoidal_positional_encoding
from braindecode.models.base import EEGModuleMixin


def _unwrap_reference_state_dict(
    state_dict: Mapping[str, object],
) -> Mapping[str, torch.Tensor]:
    nested = state_dict.get("state_dict")
    if isinstance(nested, Mapping):
        state_dict = nested
    if not all(isinstance(value, torch.Tensor) for value in state_dict.values()):
        raise TypeError("SleepFM state dictionaries may only contain tensors.")
    return state_dict  # type: ignore[return-value]


def _strip_distributed_prefix(key: str) -> str:
    return key.removeprefix("module.")


def _map_reference_base_state_dict(
    state_dict: Mapping[str, object],
) -> OrderedDict[str, torch.Tensor]:
    """Map released SetTransformer keys to Braindecode keys."""
    mapped = OrderedDict()
    for key, value in _unwrap_reference_state_dict(state_dict).items():
        key = _strip_distributed_prefix(key)
        if key == "positional_encoding.pe":
            key = "positional_encoding"
        mapped[key] = value
    return mapped


def _map_reference_staging_state_dict(
    state_dict: Mapping[str, object],
) -> OrderedDict[str, torch.Tensor]:
    """Map released SleepEventLSTMClassifier keys to staging-head keys."""
    mapped = OrderedDict()
    for key, value in _unwrap_reference_state_dict(state_dict).items():
        key = _strip_distributed_prefix(key)
        if key == "positional_encoding.pe":
            key = "positional_encoding"
        elif key.startswith("fc."):
            key = key.replace("fc.", "final_layer.", 1)
        mapped[key] = value
    return mapped


class _SleepFMAttentionPooling(nn.Module):
    """Apply self-attention and a masked mean over an unordered set."""

    def __init__(
        self, input_dim: int, num_heads: int = 1, drop_prob: float = 0.1
    ) -> None:
        super().__init__()
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dropout=drop_prob,
            batch_first=True,
        )

    @staticmethod
    def _validate_mask(
        key_padding_mask: torch.Tensor | None,
        x: torch.Tensor,
        mask_name: str = "key_padding_mask",
    ) -> torch.Tensor | None:
        if key_padding_mask is None:
            return None
        expected_shape = x.shape[:2]
        if tuple(key_padding_mask.shape) != expected_shape:
            raise ValueError(
                f"{mask_name} must have shape "
                f"{tuple(expected_shape)}, got {tuple(key_padding_mask.shape)}."
            )
        if key_padding_mask.dtype != torch.bool:
            if not (
                torch.is_floating_point(key_padding_mask)
                or key_padding_mask.dtype
                in (
                    torch.uint8,
                    torch.int8,
                    torch.int16,
                    torch.int32,
                    torch.int64,
                )
            ):
                raise TypeError(f"{mask_name} must be boolean or contain 0/1.")
            if not torch.all((key_padding_mask == 0) | (key_padding_mask == 1)):
                raise ValueError(f"{mask_name} may only contain 0 and 1.")
            key_padding_mask = key_padding_mask.bool()
        if not torch.compiler.is_compiling() and key_padding_mask.all(dim=1).any():
            raise ValueError("Each sample must contain at least one valid channel.")
        return key_padding_mask

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Pool items from ``x`` with ``True`` marking padded items."""
        if x.ndim != 3:
            raise ValueError(
                "Attention pooling expects shape (batch, items, features), "
                f"got {tuple(x.shape)}."
            )
        key_padding_mask = self._validate_mask(key_padding_mask, x)
        all_masked = None
        if key_padding_mask is not None:
            all_masked = key_padding_mask.all(dim=1)
            key_padding_mask = key_padding_mask & ~all_masked.unsqueeze(1)
        if x.shape[1] == 1:
            output = x[:, 0]
            if all_masked is not None:
                output = output.masked_fill(all_masked.unsqueeze(1), 0)
            return output

        output = self.transformer_layer(
            x,
            src_key_padding_mask=key_padding_mask,
        )
        if key_padding_mask is None:
            return output.mean(dim=1)
        assert all_masked is not None
        valid = (~key_padding_mask).unsqueeze(-1).to(output.dtype)
        output = (output * valid).sum(dim=1) / valid.sum(dim=1)
        return output.masked_fill(all_masked.unsqueeze(1), 0)


class _SleepFMTokenizer(nn.Module):
    """Convert each signal channel into fixed-length patch embeddings."""

    def __init__(
        self,
        patch_size: int = 640,
        embed_dim: int = 128,
        activation: type[nn.Module] = nn.ELU,
    ) -> None:
        super().__init__()
        if patch_size < 64 or patch_size % 64:
            raise ValueError(
                "patch_size must be at least 64 and divisible by 64 for the "
                "six reference convolution blocks."
            )
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        layers: list[nn.Module] = []
        in_channels = 1
        for block_index, out_channels in enumerate((4, 8, 16, 32, 64, 128), 1):
            layers.extend(
                [
                    nn.Conv1d(
                        in_channels,
                        out_channels,
                        kernel_size=5,
                        stride=2,
                        padding=2,
                    ),
                    nn.BatchNorm1d(out_channels),
                    activation(),
                    nn.LayerNorm([out_channels, self.patch_size // (2**block_index)]),
                ]
            )
            in_channels = out_channels
        layers.extend(
            [
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Linear(128, self.embed_dim),
            ]
        )
        self.tokenizer = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Tokenize ``x`` with shape ``(batch, channels, time)``."""
        if x.ndim != 3:
            raise ValueError(
                "SleepFM expects input with shape (batch, channels, time), "
                f"got {tuple(x.shape)}."
            )
        batch, channels, n_times = x.shape
        n_patches = n_times // self.patch_size
        if n_patches == 0:
            raise ValueError("SleepFM requires at least one complete patch.")

        x = x[..., : n_patches * self.patch_size]
        x = x.reshape(batch, channels, n_patches, self.patch_size)
        x = x.reshape(batch * channels * n_patches, 1, self.patch_size)
        x = self.tokenizer(x)
        return x.reshape(batch, channels, n_patches, self.embed_dim)


class SleepFM(EEGModuleMixin, nn.Module):
    r"""Sleep foundation model for multimodal polysomnography.

    :bdg-danger:`Foundation Model` :bdg-info:`Attention/Transformer`
    :bdg-success:`Convolution`

    SleepFM [sleepfm2026]_ learns channel-agnostic representations from
    polysomnography (PSG), including brain activity signals (EEG and EOG),
    respiratory signals, ECG, and EMG. Every channel is divided into
    non-overlapping 5-second patches and embedded by a shared convolutional
    tokenizer. Attention pools the variable channel set, a Transformer models
    the patch sequence, and a second attention layer produces one
    trial-level representation.

    Input data must be resampled to 128 Hz before calling this model. With the
    reference ``patch_size=640``, trailing samples that do not form a complete
    5-second patch are discarded.

    Parameters
    ----------
    patch_size : int, default=640
        Number of samples in each non-overlapping input patch. It must be at
        least 64 and divisible by 64.
    embed_dim : int, default=128
        Token and Transformer embedding dimension.
    num_heads : int, default=8
        Number of heads in the temporal Transformer.
    num_layers : int, default=6
        Number of temporal Transformer encoder layers.
    pooling_heads : int, default=8
        Number of heads in channel and temporal attention pooling.
    drop_prob : float, default=0.3
        Dropout probability in attention and Transformer layers.
    max_seq_length : int, default=128
        Maximum number of patches accepted by the positional encoding.
    activation : type[nn.Module], default=nn.ELU
        Activation class used by the convolutional tokenizer. ``nn.ELU``
        matches the released checkpoint.

    Notes
    -----
    ``channel_mask`` passed to :meth:`forward` has shape
    ``(batch, n_chans)`` and uses ``True`` for missing or padded channels.
    Each sample must contain at least one valid channel.

    The paper's End-to-End PSG baseline trains a raw-signal tokenizer,
    channel pooling, and a bidirectional LSTM jointly from random
    initialization, then combines the PSG representation with age and sex.
    It is not equivalent to an unpretrained ``SleepFM``. The separate
    demographics baseline is a ``4 -> 32 -> n_outputs`` MLP using age, sex,
    BMI, and race/ethnicity. Both disease-prediction baselines require
    nonelectrophysiological covariates and are outside this model's API.

    The official implementation and weights are licensed under
    `CC BY-NC 4.0 <https://creativecommons.org/licenses/by-nc/4.0/>`_.
    This adapted implementation inherits those noncommercial terms.

    References
    ----------
    .. [sleepfm2026] Thapa, R., Kjaer, M. R., He, B., et al. (2026).
       A multimodal sleep foundation model for disease prediction.
       *Nature Medicine*, 32, 752–762.
       https://doi.org/10.1038/s41591-025-04133-4
    """

    def __init__(
        self,
        n_outputs=None,
        n_chans=None,
        chs_info=None,
        n_times=None,
        input_window_seconds=None,
        sfreq=None,
        patch_size: int = 640,
        embed_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 6,
        pooling_heads: int = 8,
        drop_prob: float = 0.3,
        max_seq_length: int = 128,
        activation: type[nn.Module] = nn.ELU,
    ) -> None:
        super().__init__(
            n_outputs=n_outputs,
            n_chans=n_chans,
            chs_info=chs_info,
            n_times=n_times,
            input_window_seconds=input_window_seconds,
            sfreq=sfreq,
        )
        if self.sfreq != 128:
            raise ValueError(
                f"SleepFM requires signals sampled at 128 Hz, got {self.sfreq:g} Hz."
            )
        if self.n_times < patch_size:
            raise ValueError(
                "SleepFM requires n_times to contain at least one complete patch."
            )
        n_patches = self.n_times // patch_size
        if n_patches > max_seq_length:
            raise ValueError(
                f"Input produces {n_patches} patches, which exceeds "
                f"max_seq_length={max_seq_length}."
            )
        for name, heads in (
            ("num_heads", num_heads),
            ("pooling_heads", pooling_heads),
        ):
            if embed_dim % heads:
                raise ValueError(f"embed_dim must be divisible by {name}.")

        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.pooling_heads = pooling_heads
        self.drop_prob = drop_prob
        self.max_seq_length = max_seq_length
        self.activation = activation

        self.patch_embedding = _SleepFMTokenizer(patch_size, embed_dim, activation)
        self.spatial_pooling = _SleepFMAttentionPooling(
            embed_dim, pooling_heads, drop_prob
        )
        self.register_buffer(
            "positional_encoding",
            sinusoidal_positional_encoding(max_seq_length, embed_dim).unsqueeze(0),
        )
        self.layer_norm = nn.LayerNorm(embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dropout=drop_prob,
            batch_first=True,
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )
        self.temporal_pooling = _SleepFMAttentionPooling(
            embed_dim, pooling_heads, drop_prob
        )
        self.final_layer = nn.Linear(embed_dim, self.n_outputs)

    @staticmethod
    def _prepare_channel_mask(
        channel_mask: torch.Tensor | None,
        x: torch.Tensor,
    ) -> torch.Tensor:
        expected_shape = x.shape[:2]
        if channel_mask is None:
            return torch.zeros(expected_shape, dtype=torch.bool, device=x.device)
        if tuple(channel_mask.shape) != expected_shape:
            raise ValueError(
                f"channel_mask must have shape {tuple(expected_shape)}, "
                f"got {tuple(channel_mask.shape)}."
            )
        if channel_mask.device != x.device:
            channel_mask = channel_mask.to(x.device)
        return _SleepFMAttentionPooling._validate_mask(
            channel_mask, x, mask_name="channel_mask"
        )  # type: ignore[return-value]

    def tokenize(self, x: torch.Tensor) -> torch.Tensor:
        """Return per-channel, per-patch embeddings."""
        return self.patch_embedding(x)

    def encode(
        self,
        x: torch.Tensor,
        channel_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return pooled and contextual SleepFM representations."""
        mask = self._prepare_channel_mask(channel_mask, x)
        tokens = self.tokenize(x)
        batch, channels, n_patches, embed_dim = tokens.shape

        tokens = tokens.permute(0, 2, 1, 3).reshape(
            batch * n_patches, channels, embed_dim
        )
        expanded_mask = (
            mask.unsqueeze(1)
            .expand(batch, n_patches, channels)
            .reshape(batch * n_patches, channels)
        )
        contextual_tokens = self.spatial_pooling(tokens, expanded_mask)
        contextual_tokens = contextual_tokens.reshape(batch, n_patches, embed_dim)
        contextual_tokens = contextual_tokens + self.positional_encoding[:, :n_patches]
        contextual_tokens = self.layer_norm(contextual_tokens)
        contextual_tokens = self.transformer_encoder(contextual_tokens)
        pooled = self.temporal_pooling(contextual_tokens)
        return pooled, contextual_tokens

    def forward(
        self,
        x: torch.Tensor,
        channel_mask: torch.Tensor | None = None,
        return_features: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor | None]:
        """Return trial logits or the pooled representation."""
        pooled, _ = self.encode(x, channel_mask)
        if return_features:
            return {"features": pooled, "cls_token": None}
        return self.final_layer(pooled)

    def reset_head(self, n_outputs: int):
        """Replace the trial-level output projection."""
        self._n_outputs = n_outputs
        self.final_layer = nn.Linear(self.embed_dim, n_outputs)
        return self

    def load_pretrained_backbone(
        self,
        state_dict: Mapping[str, object],
    ):
        """Load a released SleepFM base checkpoint, preserving this head."""
        mapped = _map_reference_base_state_dict(state_dict)
        incompatible = self.load_state_dict(mapped, strict=False)
        allowed_missing = {"final_layer.weight", "final_layer.bias"}
        unexpected_missing = set(incompatible.missing_keys) - allowed_missing
        if incompatible.unexpected_keys or unexpected_missing:
            raise RuntimeError(
                "SleepFM reference checkpoint is incompatible: "
                f"missing={sorted(unexpected_missing)}, "
                f"unexpected={sorted(incompatible.unexpected_keys)}."
            )
        return incompatible


class _SleepFMStagingHead(nn.Module):
    """Predict a sleep stage for every SleepFM patch embedding."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 4,
        num_layers: int = 1,
        pooling_heads: int = 4,
        drop_prob: float = 0.3,
        max_seq_length: int = 8196,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.spatial_pooling = _SleepFMAttentionPooling(
            embed_dim, pooling_heads, drop_prob
        )
        self.register_buffer(
            "positional_encoding",
            sinusoidal_positional_encoding(max_seq_length, embed_dim).unsqueeze(0),
        )
        self.layer_norm = nn.LayerNorm(embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dropout=drop_prob,
            batch_first=True,
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=embed_dim // 2,
            num_layers=num_layers,
            batch_first=True,
            dropout=drop_prob if num_layers > 1 else 0.0,
            bidirectional=True,
        )

    def forward(
        self,
        tokens: torch.Tensor,
        channel_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return contextual LSTM features for every signal patch."""
        batch, channels, n_patches, embed_dim = tokens.shape
        tokens = tokens.permute(0, 2, 1, 3).reshape(
            batch * n_patches, channels, embed_dim
        )
        expanded_mask = (
            channel_mask.unsqueeze(1)
            .expand(batch, n_patches, channels)
            .reshape(batch * n_patches, channels)
        )
        features = self.spatial_pooling(tokens, expanded_mask)
        features = features.reshape(batch, n_patches, embed_dim)
        features = features + self.positional_encoding[:, :n_patches]
        features = self.layer_norm(features)
        features = self.transformer_encoder(features)
        features, _ = self.lstm(features)
        return features


class SleepFMStager(EEGModuleMixin, nn.Module):
    r"""SleepFM tokenizer with the released token-wise sleep-staging head.

    :bdg-danger:`Foundation Model` :bdg-info:`Attention/Transformer`
    :bdg-success:`Convolution` :bdg-secondary:`Recurrent`

    This model composes the raw-signal tokenizer from :class:`SleepFM` with
    the authors' downstream sleep-staging architecture [sleepfm2026]_. Channel
    attention pools per-channel embeddings, a Transformer contextualizes the
    patch sequence, and a bidirectional LSTM emits one prediction for every
    5-second patch.

    The output shape is ``(batch, n_outputs, n_patches)``. For the released
    checkpoint, ``n_outputs=5`` corresponds to Wake, N1, N2, N3, and REM.
    Use a time-series target with one label per patch; this model is not a
    trial-level :class:`~braindecode.EEGClassifier` head.

    Parameters
    ----------
    patch_size : int, default=640
        Number of samples per patch at 128 Hz.
    embed_dim : int, default=128
        Token and recurrent feature dimension.
    staging_num_heads : int, default=4
        Number of heads in the staging Transformer.
    staging_num_layers : int, default=1
        Number of staging Transformer and bidirectional-LSTM layers.
    staging_pooling_heads : int, default=4
        Number of channel-pooling attention heads.
    drop_prob : float, default=0.3
        Dropout probability in the staging head.
    max_seq_length : int, default=8196
        Maximum number of 5-second patches.
    activation : type[nn.Module], default=nn.ELU
        Tokenizer activation class. Keep ``nn.ELU`` for official weights.

    Notes
    -----
    Load the released base checkpoint with
    :meth:`load_pretrained_backbone`, then load the released staging
    checkpoint with :meth:`load_pretrained_staging_head`. Both methods accept
    the dictionaries returned by :func:`torch.load`.

    The official implementation and weights are licensed under
    `CC BY-NC 4.0 <https://creativecommons.org/licenses/by-nc/4.0/>`_.

    References
    ----------
    .. [sleepfm2026] Thapa, R., Kjaer, M. R., He, B., et al. (2026).
       A multimodal sleep foundation model for disease prediction.
       *Nature Medicine*, 32, 752–762.
       https://doi.org/10.1038/s41591-025-04133-4
    """

    def __init__(
        self,
        n_outputs=None,
        n_chans=None,
        chs_info=None,
        n_times=None,
        input_window_seconds=None,
        sfreq=None,
        patch_size: int = 640,
        embed_dim: int = 128,
        staging_num_heads: int = 4,
        staging_num_layers: int = 1,
        staging_pooling_heads: int = 4,
        drop_prob: float = 0.3,
        max_seq_length: int = 8196,
        activation: type[nn.Module] = nn.ELU,
    ) -> None:
        super().__init__(
            n_outputs=n_outputs,
            n_chans=n_chans,
            chs_info=chs_info,
            n_times=n_times,
            input_window_seconds=input_window_seconds,
            sfreq=sfreq,
        )
        if self.sfreq != 128:
            raise ValueError(
                "SleepFMStager requires signals sampled at 128 Hz, "
                f"got {self.sfreq:g} Hz."
            )
        if self.n_times < patch_size:
            raise ValueError(
                "SleepFMStager requires n_times to contain at least one complete patch."
            )
        n_patches = self.n_times // patch_size
        if n_patches > max_seq_length:
            raise ValueError(
                f"Input produces {n_patches} patches, which exceeds "
                f"max_seq_length={max_seq_length}."
            )
        for name, heads in (
            ("staging_num_heads", staging_num_heads),
            ("staging_pooling_heads", staging_pooling_heads),
        ):
            if embed_dim % heads:
                raise ValueError(f"embed_dim must be divisible by {name}.")

        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.staging_num_heads = staging_num_heads
        self.staging_num_layers = staging_num_layers
        self.staging_pooling_heads = staging_pooling_heads
        self.drop_prob = drop_prob
        self.max_seq_length = max_seq_length
        self.activation = activation

        self.patch_embedding = _SleepFMTokenizer(patch_size, embed_dim, activation)
        self.staging_head = _SleepFMStagingHead(
            embed_dim=embed_dim,
            num_heads=staging_num_heads,
            num_layers=staging_num_layers,
            pooling_heads=staging_pooling_heads,
            drop_prob=drop_prob,
            max_seq_length=max_seq_length,
        )
        self.final_layer = nn.Linear(embed_dim, self.n_outputs)

    def forward(
        self,
        x: torch.Tensor,
        channel_mask: torch.Tensor | None = None,
        return_features: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor | None]:
        """Return patch-wise staging logits or contextual features."""
        mask = SleepFM._prepare_channel_mask(channel_mask, x)
        features = self.staging_head(self.patch_embedding(x), mask)
        if return_features:
            return {"features": features, "cls_token": None}
        logits = self.final_layer(features)
        return logits.transpose(1, 2)

    def reset_head(self, n_outputs: int):
        """Replace the token-wise sleep-staging output projection."""
        self._n_outputs = n_outputs
        self.final_layer = nn.Linear(self.embed_dim, n_outputs)
        return self

    def load_pretrained_backbone(
        self,
        state_dict: Mapping[str, object],
    ):
        """Load the tokenizer weights from a released SleepFM base checkpoint."""
        mapped = _map_reference_base_state_dict(state_dict)
        tokenizer_state = OrderedDict(
            (
                key.removeprefix("patch_embedding."),
                value,
            )
            for key, value in mapped.items()
            if key.startswith("patch_embedding.")
        )
        if not tokenizer_state:
            raise RuntimeError(
                "SleepFM reference checkpoint has no patch_embedding weights."
            )
        return self.patch_embedding.load_state_dict(tokenizer_state, strict=True)

    def load_pretrained_staging_head(
        self,
        state_dict: Mapping[str, object],
    ):
        """Load the released SleepFM sleep-staging head checkpoint."""
        mapped = _map_reference_staging_state_dict(state_dict)
        head_state = OrderedDict(
            (key, value)
            for key, value in mapped.items()
            if not key.startswith("final_layer.")
        )
        final_state = OrderedDict(
            (key.removeprefix("final_layer."), value)
            for key, value in mapped.items()
            if key.startswith("final_layer.")
        )
        incompatible = self.staging_head.load_state_dict(head_state, strict=True)
        self.final_layer.load_state_dict(final_state, strict=True)
        return incompatible
