# Authors: Fashad Ahmed <Fashad-Ahmed@users.noreply.github.com>
#
# Code adapted from https://github.com/zou-group/sleepfm-clinical
#
# License: Creative Commons Attribution-NonCommercial 4.0 International
# This derivative is not covered by Braindecode's BSD-3 license.

"""SleepFM models for multimodal polysomnography."""

from __future__ import annotations

import warnings

import torch
from einops import rearrange, repeat
from torch import nn

from braindecode.functional import sinusoidal_positional_encoding
from braindecode.models.base import EEGModuleMixin


def _validate_channel_mask(
    mask: torch.Tensor | None,
    x: torch.Tensor,
    mask_name: str = "channel_mask",
) -> torch.Tensor | None:
    """Return ``mask`` as a boolean tensor matching the first two axes of ``x``.

    A mask marks *padded* items with ``True``, following
    :class:`~torch.nn.TransformerEncoderLayer`. Integer and float masks holding
    only 0/1 are accepted and cast, because a caller building a mask from a
    dataframe or from ``numpy`` rarely has a boolean dtype at hand.

    Parameters
    ----------
    mask : torch.Tensor | None
        Mask of shape ``x.shape[:2]``, or ``None`` for "nothing is padded".
    x : torch.Tensor
        Tensor the mask applies to; only its first two axes are used.
    mask_name : str
        Name used in error messages, so the message quotes the argument the
        caller actually passed rather than an internal one.

    Returns
    -------
    torch.Tensor | None
        Boolean mask, or ``None`` if ``mask`` was ``None``.
    """
    if mask is None:
        return None
    expected_shape = x.shape[:2]
    if tuple(mask.shape) != expected_shape:
        raise ValueError(
            f"{mask_name} must have shape "
            f"{tuple(expected_shape)}, got {tuple(mask.shape)}."
        )
    if mask.dtype != torch.bool:
        if not (
            torch.is_floating_point(mask)
            or mask.dtype
            in (
                torch.uint8,
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
            )
        ):
            raise TypeError(f"{mask_name} must be boolean or contain 0/1.")
        if not torch.all((mask == 0) | (mask == 1)):
            raise ValueError(f"{mask_name} may only contain 0 and 1.")
        mask = mask.bool()
    # A fully masked sample has no channel to pool, so it is rejected eagerly.
    # Under torch.compile the test would specialise on tensor *values*, so it is
    # skipped there and the pooling zeroes those samples instead.
    if not torch.compiler.is_compiling() and mask.all(dim=1).any():
        raise ValueError("Each sample must contain at least one valid channel.")
    return mask


def _prepare_channel_mask(
    channel_mask: torch.Tensor | None,
    x: torch.Tensor,
) -> torch.Tensor:
    """Return a validated ``(batch, n_chans)`` channel mask on ``x``'s device.

    Unlike :func:`_validate_channel_mask` this never returns ``None``: a missing
    mask becomes an all-``False`` mask, so the downstream code has a single path
    and stays :func:`torch.compile`-friendly.
    """
    if channel_mask is None:
        return torch.zeros(x.shape[:2], dtype=torch.bool, device=x.device)
    if channel_mask.device != x.device:
        channel_mask = channel_mask.to(x.device)
    validated = _validate_channel_mask(channel_mask, x, mask_name="channel_mask")
    assert validated is not None  # channel_mask is not None here
    return validated


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

    The pretrained encoder is available from the braindecode mirror of the
    released checkpoint::

        model = SleepFM.from_pretrained(n_chans=4, n_times=3840, n_outputs=5)

    The trial-level head is *not* pretrained: the released checkpoint is a
    contrastive encoder, so ``final_layer`` starts from a random
    initialisation and must be fine-tuned.

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

    _HF_DEFAULT_REPO = "braindecode/SleepFM"

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
            warnings.warn(
                f"SleepFM was pretrained on signals sampled at 128 Hz, got "
                f"{self.sfreq:g} Hz. Resample the data to reuse the released "
                "weights; the patch length is a sample count, not a duration.",
                UserWarning,
                stacklevel=2,
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
        # The tokenizer owns the patch arithmetic, so the sequence length it
        # will produce is asked of it rather than recomputed here.
        n_patches = self.patch_embedding.n_patches(self.n_times)
        if n_patches > max_seq_length:
            raise ValueError(
                f"Input produces {n_patches} patches, which exceeds "
                f"max_seq_length={max_seq_length}."
            )
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

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        """Load the released SleepFM encoder from the braindecode mirror.

        ``pretrained_model_name_or_path`` defaults to ``"braindecode/SleepFM"``,
        a mirror of the upstream ``model_base/best.pt`` checkpoint whose keys
        were rewritten to this implementation's parameter names. Pass a
        different repo id or local path to override it. Only the encoder is
        pretrained: the released checkpoint is contrastive and carries no
        classification head, so ``final_layer`` is randomly initialised and must
        be fine-tuned.
        """
        if not args and kwargs.get("pretrained_model_name_or_path") is None:
            kwargs["pretrained_model_name_or_path"] = cls._HF_DEFAULT_REPO
        return super().from_pretrained(*args, **kwargs)

    def encode(
        self,
        x: torch.Tensor,
        channel_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r"""Return the pooled and the per-patch SleepFM representations.

        The tokenizer embeds every (channel, patch) pair independently, giving
        :math:`t_{b,c,p} \in \mathbb{R}^{D}`. Channels are then pooled *within*
        each patch, since which channels a recording carries is a property of
        the montage and not of the signal:

        .. math::
            z_{b,p} = \operatorname{MaskedMean}_{c}
                      \big(\operatorname{SelfAttn}(t_{b,:,p})\big),

        where the mean runs over the channels with ``channel_mask`` ``False``.
        The patch sequence is then contextualised, with :math:`\mathrm{PE}` the
        fixed sinusoidal encoding of Vaswani et al. (2017):

        .. math::
            h_{b} = \operatorname{Transformer}
                    \big(\operatorname{LN}(z_{b} + \mathrm{PE})\big),

        and pooled over time into one trial-level vector:

        .. math::
            g_{b} = \operatorname{MaskedMean}_{p}
                    \big(\operatorname{SelfAttn}(h_{b})\big).

        Parameters
        ----------
        x : torch.Tensor
            Input of shape ``(batch, n_chans, n_times)``.
        channel_mask : torch.Tensor | None
            ``(batch, n_chans)`` boolean mask, ``True`` for padded channels.

        Returns
        -------
        pooled : torch.Tensor
            Trial-level representation :math:`g`, shape ``(batch, embed_dim)``.
        contextual_tokens : torch.Tensor
            Per-patch representation :math:`h`, shape
            ``(batch, n_patches, embed_dim)``.
        """
        mask = _prepare_channel_mask(channel_mask, x)
        tokens = self.patch_embedding(x)
        n_patches = tokens.shape[2]

        # Channel pooling treats the channels of one patch as an unordered set,
        # so patches are folded into the batch axis and pooled independently.
        tokens = rearrange(tokens, "batch chans patch emb -> (batch patch) chans emb")
        expanded_mask = repeat(
            mask, "batch chans -> (batch patch) chans", patch=n_patches
        )
        contextual_tokens = self.spatial_pooling(tokens, expanded_mask)
        contextual_tokens = rearrange(
            contextual_tokens, "(batch patch) emb -> batch patch emb", patch=n_patches
        )

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
            # "cls_token" is braindecode's feature-return contract, not a
            # credential: SleepFM pools instead of using a class token.
            return {"features": pooled, "cls_token": None}  # nosec B105
        return self.final_layer(pooled)

    def reset_head(self, n_outputs: int):
        """Replace the trial-level output projection."""
        self._n_outputs = n_outputs
        self.final_layer = nn.Linear(self.embed_dim, n_outputs)
        return self


class SleepFMStager(EEGModuleMixin, nn.Module):
    r"""SleepFM tokenizer with the released token-wise sleep-staging head.

    :bdg-danger:`Foundation Model` :bdg-info:`Attention/Transformer`
    :bdg-success:`Convolution` :bdg-secondary:`Recurrent`

    This model composes the raw-signal tokenizer from :class:`SleepFM` with
    the authors' downstream sleep-staging architecture [sleepfm2026]_. Channel
    attention pools per-channel embeddings, a Transformer contextualizes the
    patch sequence, and a bidirectional LSTM emits one prediction for every
    5-second patch.

    Two properties of the paper's staging setup are visible in the API. First,
    the label rate is the *patch* rate, not the 30-second epoch rate of manual
    scoring: with the reference ``patch_size=640`` at 128 Hz, one 30-second
    epoch spans six predictions, and comparing against a scored hypnogram means
    aggregating them. Second, the head is recurrent and its positional encoding
    is sized for whole nights (``max_seq_length=8196`` patches, about 11 hours),
    so it is meant to see a long contiguous recording rather than shuffled
    windows -- the LSTM is what carries sleep-stage context across the night.

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
    Unlike :class:`SleepFM`, this model is pretrained end to end: the mirror
    merges the released tokenizer with the released staging head, so a single
    call returns a usable stager::

        model = SleepFMStager.from_pretrained(
            n_chans=4, n_times=3840, n_outputs=5
        )

    The official implementation and weights are licensed under
    `CC BY-NC 4.0 <https://creativecommons.org/licenses/by-nc/4.0/>`_.

    References
    ----------
    .. [sleepfm2026] Thapa, R., Kjaer, M. R., He, B., et al. (2026).
       A multimodal sleep foundation model for disease prediction.
       *Nature Medicine*, 32, 752–762.
       https://doi.org/10.1038/s41591-025-04133-4
    """

    _HF_DEFAULT_REPO = "braindecode/SleepFMStager"

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
            warnings.warn(
                f"SleepFMStager was pretrained on signals sampled at 128 Hz, "
                f"got {self.sfreq:g} Hz. Resample the data to reuse the "
                "released weights; the patch length is a sample count, not a "
                "duration.",
                UserWarning,
                stacklevel=2,
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
        # Same as SleepFM: the tokenizer is the single source of truth for how
        # many patches an input of ``n_times`` samples yields.
        n_patches = self.patch_embedding.n_patches(self.n_times)
        if n_patches > max_seq_length:
            raise ValueError(
                f"Input produces {n_patches} patches, which exceeds "
                f"max_seq_length={max_seq_length}."
            )
        self.staging_head = _SleepFMStagingHead(
            embed_dim=embed_dim,
            num_heads=staging_num_heads,
            num_layers=staging_num_layers,
            pooling_heads=staging_pooling_heads,
            drop_prob=drop_prob,
            max_seq_length=max_seq_length,
        )
        self.final_layer = nn.Linear(embed_dim, self.n_outputs)

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        """Load the released sleep stager from the braindecode mirror.

        ``pretrained_model_name_or_path`` defaults to
        ``"braindecode/SleepFMStager"``, which merges the tokenizer of the
        upstream ``model_base/best.pt`` with the upstream
        ``model_sleep_staging/best.pth`` head. The whole stager is therefore
        pretrained, its five-class output layer included; pass ``n_outputs``
        different from 5 to reinitialise that layer for another label set.
        """
        if not args and kwargs.get("pretrained_model_name_or_path") is None:
            kwargs["pretrained_model_name_or_path"] = cls._HF_DEFAULT_REPO
        return super().from_pretrained(*args, **kwargs)

    def forward(
        self,
        x: torch.Tensor,
        channel_mask: torch.Tensor | None = None,
        return_features: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor | None]:
        """Return patch-wise staging logits or contextual features."""
        mask = _prepare_channel_mask(channel_mask, x)
        # (batch, n_chans, n_times) -> (batch, n_chans, n_patches, embed_dim)
        tokens = self.patch_embedding(x)
        # Channel pooling and the recurrent context both happen in the head,
        # which returns one feature vector per patch.
        features = self.staging_head(tokens, mask)
        if return_features:
            # "cls_token" is braindecode's feature-return contract, not a
            # credential: the stager has no class token, it labels every patch.
            return {"features": features, "cls_token": None}  # nosec B105
        logits = self.final_layer(features)
        # braindecode's cropped/time-series convention puts the class axis
        # second, whereas the head emits (batch, n_patches, n_outputs).
        return rearrange(logits, "batch patch cls -> batch cls patch")

    def reset_head(self, n_outputs: int):
        """Replace the token-wise sleep-staging output projection."""
        self._n_outputs = n_outputs
        self.final_layer = nn.Linear(self.embed_dim, n_outputs)
        return self


class _SleepFMTokenizer(nn.Module):
    """Convert each signal channel into fixed-length patch embeddings.

    Six stride-2 convolution blocks reduce one ``patch_size``-sample patch to a
    single ``embed_dim`` vector, independently of the channel it came from --
    that channel-agnostic tokenizer is what lets SleepFM accept whatever
    montage a recording happens to carry.
    """

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

    def n_patches(self, n_times: int) -> int:
        """Return how many complete patches ``n_times`` samples contain."""
        n_patches = n_times // self.patch_size
        if n_patches == 0:
            raise ValueError(
                f"SleepFM requires at least one complete patch: got "
                f"n_times={n_times} for patch_size={self.patch_size}."
            )
        return n_patches

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Tokenize ``x`` with shape ``(batch, channels, time)``."""
        if x.ndim != 3:
            raise ValueError(
                "SleepFM expects input with shape (batch, channels, time), "
                f"got {tuple(x.shape)}."
            )
        batch, channels, n_times = x.shape
        n_patches = self.n_patches(n_times)

        # Trailing samples that do not fill a patch are dropped, then every
        # (sample, channel, patch) triple is embedded independently.
        x = x[..., : n_patches * self.patch_size]
        x = rearrange(
            x,
            "batch chans (patch time) -> (batch chans patch) 1 time",
            time=self.patch_size,
        )
        x = self.tokenizer(x)
        return rearrange(
            x,
            "(batch chans patch) emb -> batch chans patch emb",
            batch=batch,
            chans=channels,
        )


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
        key_padding_mask = _validate_channel_mask(
            key_padding_mask, x, mask_name="key_padding_mask"
        )
        all_masked = None
        if key_padding_mask is not None:
            # A fully masked sample would make attention and the mean undefined,
            # so its mask is neutralised here and its output zeroed at the end.
            all_masked = key_padding_mask.all(dim=1)
            key_padding_mask = key_padding_mask & ~all_masked.unsqueeze(1)
        # Upstream only takes this shortcut for a masked singleton. Without a
        # mask the Transformer layer still runs, including for temporal pooling
        # over a one-patch recording.
        if key_padding_mask is not None and x.shape[1] == 1:
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


class _SleepFMStagingHead(nn.Module):
    """Predict one sleep stage per SleepFM patch embedding.

    The head consumes the ``(batch, n_chans, n_patches, embed_dim)`` tokens of
    :class:`_SleepFMTokenizer` and returns one feature vector per patch. It
    reproduces the released downstream architecture, whose three stages answer
    three different questions:

    1. **Channel attention pooling** collapses the channel axis inside each
       patch, so a recording is described by what its channels agree on rather
       than by their order or count.
    2. **A Transformer encoder** over the patch sequence (with fixed sinusoidal
       positional encoding) provides local context within the night.
    3. **A bidirectional LSTM** carries longer-range context, which is what
       makes a stage decision depend on what came before and after it -- the
       transition structure a human scorer relies on.

    The classification layer itself lives in :class:`SleepFMStager`, so
    :meth:`braindecode.models.base.EEGModuleMixin.reset_head` can replace it
    without touching this head.
    """

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
        n_patches = tokens.shape[2]

        # Channels are pooled patch by patch, so patches join the batch axis.
        tokens = rearrange(tokens, "batch chans patch emb -> (batch patch) chans emb")
        expanded_mask = repeat(
            channel_mask, "batch chans -> (batch patch) chans", patch=n_patches
        )
        features = self.spatial_pooling(tokens, expanded_mask)
        features = rearrange(
            features, "(batch patch) emb -> batch patch emb", patch=n_patches
        )

        # The patch sequence is then read as a time series of the night.
        features = features + self.positional_encoding[:, :n_patches]
        features = self.layer_norm(features)
        features = self.transformer_encoder(features)
        features, _ = self.lstm(features)
        return features
