# Authors: Fashad Ahmed <Fashad-Ahmed@users.noreply.github.com>
#
# Code adapted from https://github.com/zou-group/sleepfm-clinical
#
# License: Creative Commons Attribution-NonCommercial 4.0 International
# This derivative is not covered by Braindecode's BSD-3 license.

"""SleepFM models for multimodal polysomnography."""

from __future__ import annotations

import warnings
from collections.abc import Hashable, Sequence
from pathlib import Path

import torch
from einops import rearrange, repeat
from torch import nn

from braindecode.functional import sinusoidal_positional_encoding
from braindecode.models.base import _HF_INSTALL_HINT, EEGModuleMixin, huggingface_hub

# The released staging configuration (``max_channels: 4``) always feeds the
# staging head four modality slots (BAS, RESP, EKG, EMG), padding missing ones.
_RELEASED_MODALITY_SLOTS = 4
# Length of the positional table of the released encoder (SleepFM's default).
_ENCODER_MAX_SEQ_LENGTH = 128


def _validate_channel_mask(
    mask: torch.Tensor | None,
    x: torch.Tensor,
    mask_name: str = "channel_mask",
    item_name: str = "channel",
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
    item_name : str
        What one entry of the second axis is (``"channel"`` or ``"patch"``),
        for the error raised on a fully masked sample.

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
        raise ValueError(f"Each sample must contain at least one valid {item_name}.")
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


def _prepare_temporal_mask(
    temporal_mask: torch.Tensor | None,
    x: torch.Tensor,
    n_patches: int,
) -> torch.Tensor | None:
    """Return a validated ``(batch, n_patches)`` patch mask, or ``None``."""
    if temporal_mask is None:
        return None
    if temporal_mask.device != x.device:
        temporal_mask = temporal_mask.to(x.device)
    # Only the first two axes of the reference tensor are compared.
    reference = x.new_empty((x.shape[0], n_patches))
    return _validate_channel_mask(
        temporal_mask, reference, mask_name="temporal_mask", item_name="patch"
    )


def _without_fully_masked_rows(mask: torch.Tensor) -> torch.Tensor:
    """Unmask the rows of ``mask`` that are masked everywhere.

    Attention over a fully masked row is undefined (every logit is minus
    infinity), so such a row is computed unmasked; callers discard its output
    afterwards.
    """
    return mask & ~mask.all(dim=1, keepdim=True)


def _temporal_transformer(
    embed_dim: int, num_heads: int, num_layers: int, drop_prob: float
) -> nn.TransformerEncoder:
    """Pre-norm Transformer encoder shared by the SleepFM encoder and head."""
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=embed_dim,
        nhead=num_heads,
        dropout=drop_prob,
        batch_first=True,
        norm_first=True,
    )
    return nn.TransformerEncoder(
        encoder_layer,
        num_layers=num_layers,
        enable_nested_tensor=False,
    )


class _SleepFMSequenceMixin:
    """Channel pooling and positional Transformer shared by the SleepFM blocks.

    The encoder of :class:`SleepFM`, the encoder inside :class:`SleepFMStager`
    and :class:`_SleepFMStagingHead` all pool a set of per-patch embeddings
    and then read the patch sequence with a Transformer. They name the modules
    alike (``spatial_pooling``, ``positional_encoding``, ``layer_norm``,
    ``transformer_encoder``), which is what keeps the released checkpoint keys
    valid, so the two steps are written once here.
    """

    spatial_pooling: _SleepFMAttentionPooling
    positional_encoding: torch.Tensor
    layer_norm: nn.LayerNorm
    transformer_encoder: nn.TransformerEncoder

    def _pool_channels(
        self, tokens: torch.Tensor, channel_mask: torch.Tensor
    ) -> torch.Tensor:
        """Pool ``(batch, set, patch, emb)`` tokens into ``(batch, patch, emb)``."""
        n_patches = tokens.shape[2]
        # The set of one patch is pooled on its own, so patches join the batch.
        tokens = rearrange(tokens, "batch chans patch emb -> (batch patch) chans emb")
        expanded_mask = repeat(
            channel_mask, "batch chans -> (batch patch) chans", patch=n_patches
        )
        pooled = self.spatial_pooling(tokens, expanded_mask)
        return rearrange(
            pooled, "(batch patch) emb -> batch patch emb", patch=n_patches
        )

    def _contextualize(
        self,
        tokens: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Add positions, normalise and run the Transformer over the patches."""
        tokens = tokens + self.positional_encoding[:, : tokens.shape[1]]
        tokens = self.layer_norm(tokens)
        return self.transformer_encoder(tokens, src_key_padding_mask=key_padding_mask)


class SleepFM(EEGModuleMixin, _SleepFMSequenceMixin, nn.Module, license="cc-by-nc-4.0"):
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
    Each sample must contain at least one valid channel. Masked channels never
    reach the output, in training mode included: they are left out of the
    tokenizer's batch normalization, so neither the batch statistics nor the
    running averages depend on what the padding contains. The official
    pretraining code normalizes its zero-padded channels together with the real
    ones; the two agree whenever no channel is masked, and always in eval mode.

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
        self.transformer_encoder = _temporal_transformer(
            embed_dim, num_heads, num_layers, drop_prob
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
        n_patches = self.patch_embedding.n_patches(x.shape[-1])
        # Masked channels are kept out of the tokenizer, BatchNorm included.
        # Without a mask nothing is left out, so the plain path is taken.
        patch_mask = None
        if channel_mask is not None:
            patch_mask = repeat(
                mask, "batch chans -> batch chans patch", patch=n_patches
            )
        tokens = self.patch_embedding(x, patch_mask)
        # Channel pooling treats the channels of one patch as an unordered set.
        contextual_tokens = self._contextualize(self._pool_channels(tokens, mask))
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
        self._set_n_outputs(n_outputs)
        self.final_layer = nn.Linear(self.embed_dim, n_outputs)
        return self


class SleepFMStager(
    EEGModuleMixin, _SleepFMSequenceMixin, nn.Module, license="cc-by-nc-4.0"
):
    r"""SleepFM encoder with the released token-wise sleep-staging head.

    :bdg-danger:`Foundation Model` :bdg-info:`Attention/Transformer`
    :bdg-success:`Convolution` :bdg-secondary:`Recurrent`

    This model reproduces the two-stage sleep-staging pipeline of
    [sleepfm2026]_ end to end, from raw signals to one prediction per
    5-second patch:

    1. **Encoder** (the released :class:`SleepFM` weights, frozen in the
       paper). Channels are grouped by modality with ``channel_modalities``.
       Each modality is cut into non-overlapping chunks of
       ``encoder_chunk_patches`` patches (5 minutes in the release), and each
       chunk goes through the tokenizer, channel attention pooling,
       positional encoding and the temporal Transformer on its own, so the
       positional encoding restarts at every chunk. This gives one embedding
       per patch and modality.
    2. **Staging head.** Attention pools the modality embeddings of every
       patch, a Transformer adds local context and a bidirectional LSTM
       carries context across the night.

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
    channel_modalities : sequence of hashable, optional
        Modality of every input channel, e.g.
        ``["BAS", "BAS", "RESP", "EKG", "EMG"]``. The release groups brain
        activity (EEG, EOG), respiratory, ECG and EMG channels into ``"BAS"``,
        ``"RESP"``, ``"EKG"`` and ``"EMG"``; any labels work, channels sharing
        a label are encoded together. Use strings or integers, so that the
        configuration can be saved with ``save_pretrained``. ``None`` puts
        every channel in a single modality.
    patch_size : int, default=640
        Number of samples per patch at 128 Hz.
    embed_dim : int, default=128
        Token and recurrent feature dimension.
    encoder_num_heads : int, default=8
        Number of heads in the encoder's temporal Transformer.
    encoder_num_layers : int, default=6
        Number of layers of the encoder's temporal Transformer.
    encoder_pooling_heads : int, default=8
        Number of heads of the encoder's channel attention pooling.
    encoder_drop_prob : float, default=0.0
        Dropout probability in the encoder. The release computes the staging
        embeddings with dropout disabled.
    encoder_chunk_patches : int, default=60
        Number of patches the encoder sees at once (60 patches of 5 seconds,
        i.e. 5 minutes, in the release). Values above 128 lengthen the
        encoder's positional table, which then no longer loads from the
        released weights.
    staging_num_heads : int, default=4
        Number of heads in the staging Transformer.
    staging_num_layers : int, default=1
        Number of staging Transformer and bidirectional-LSTM layers.
    staging_pooling_heads : int, default=4
        Number of modality-pooling attention heads.
    drop_prob : float, default=0.3
        Dropout probability in the staging head.
    max_seq_length : int, default=8196
        Maximum number of 5-second patches.
    activation : type[nn.Module], default=nn.ELU
        Tokenizer activation class. Keep ``nn.ELU`` for official weights.

    Notes
    -----
    :meth:`forward` takes two optional masks, both using ``True`` for padding:

    - ``channel_mask``, shape ``(batch, n_chans)``, marks missing or padded
      channels. A modality with no valid channel in a sample is left out of
      the modality pooling. The staging head always receives at least four
      modality slots, the missing ones masked, as in the released staging
      configuration.
    - ``temporal_mask``, shape ``(batch, n_patches)``, marks padded patches at
      the end of shorter recordings. As in the release, padded patches enter
      the staging head as zero embeddings masked out of its Transformer; the
      bidirectional LSTM still runs over them, so, as in the release, the
      valid predictions depend on how much padding there is (never on what it
      contains). Pad with whole encoder chunks to match the release exactly;
      padded patches inside a chunk are also masked out of the encoder.

    Masked channels and padded patches are left out of the tokenizer's batch
    normalization, so in training mode neither the batch statistics nor the
    running averages depend on the padding. For padded patches this is what
    the release does, since it never encodes padded chunks. For masked
    channels it is a deliberate deviation: the official pretraining code
    normalizes zero-padded channels together with the real ones, so in
    training its output depends on how many channels are padded. Both agree
    whenever no channel is masked, and always in eval mode.

    A trailing group of fewer than ``encoder_chunk_patches`` patches is
    encoded as a shorter chunk, so every patch gets a prediction; the official
    embedding script drops it.

    The model is pretrained end to end. :meth:`from_pretrained` reads it from
    the ``braindecode/SleepFMStager`` mirror, a copy of the released
    checkpoints; older revisions of that mirror hold only the tokenizer and
    the staging head, and the rest of the encoder is then read from the
    ``braindecode/SleepFM`` mirror::

        model = SleepFMStager.from_pretrained(
            n_chans=7,
            n_times=38400,
            n_outputs=5,
            channel_modalities=["BAS"] * 3 + ["RESP"] * 2 + ["EKG", "EMG"],
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
    # Older revisions of the stager mirror hold the tokenizer and the staging
    # head only; their other encoder weights are those of the released base
    # model.
    _HF_ENCODER_REPO = "braindecode/SleepFM"

    def __init__(
        self,
        n_outputs=None,
        n_chans=None,
        chs_info=None,
        n_times=None,
        input_window_seconds=None,
        sfreq=None,
        channel_modalities: Sequence[Hashable] | None = None,
        patch_size: int = 640,
        embed_dim: int = 128,
        encoder_num_heads: int = 8,
        encoder_num_layers: int = 6,
        encoder_pooling_heads: int = 8,
        encoder_drop_prob: float = 0.0,
        encoder_chunk_patches: int = 60,
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
            ("encoder_num_heads", encoder_num_heads),
            ("encoder_pooling_heads", encoder_pooling_heads),
            ("staging_num_heads", staging_num_heads),
            ("staging_pooling_heads", staging_pooling_heads),
        ):
            if embed_dim % heads:
                raise ValueError(f"embed_dim must be divisible by {name}.")
        if encoder_chunk_patches < 1:
            raise ValueError("encoder_chunk_patches must be at least 1.")

        if channel_modalities is None:
            modality_of_channel: list[Hashable] = [0] * self.n_chans
        else:
            modality_of_channel = list(channel_modalities)
            if len(modality_of_channel) != self.n_chans:
                raise ValueError(
                    f"channel_modalities must name one modality per channel: "
                    f"got {len(modality_of_channel)} for n_chans={self.n_chans}."
                )
        # Modalities are numbered in order of first appearance. Predictions do
        # not depend on this order, since the staging head pools them as a set;
        # in training, BatchNorm updates its running averages in this order.
        modalities = list(dict.fromkeys(modality_of_channel))
        channel_order = [
            index
            for modality in modalities
            for index, label in enumerate(modality_of_channel)
            if label == modality
        ]
        self._modality_sizes = [modality_of_channel.count(m) for m in modalities]
        self.register_buffer(
            "_channel_order", torch.tensor(channel_order), persistent=False
        )

        self.channel_modalities = channel_modalities
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.encoder_num_heads = encoder_num_heads
        self.encoder_num_layers = encoder_num_layers
        self.encoder_pooling_heads = encoder_pooling_heads
        self.encoder_drop_prob = encoder_drop_prob
        self.encoder_chunk_patches = encoder_chunk_patches
        self.staging_num_heads = staging_num_heads
        self.staging_num_layers = staging_num_layers
        self.staging_pooling_heads = staging_pooling_heads
        self.drop_prob = drop_prob
        self.max_seq_length = max_seq_length
        self.activation = activation

        # Encoder: same modules and parameter names as SleepFM, without its
        # trial-level temporal pooling.
        self.patch_embedding = _SleepFMTokenizer(patch_size, embed_dim, activation)
        # Same as SleepFM: the tokenizer is the single source of truth for how
        # many patches an input of ``n_times`` samples yields.
        n_patches = self.patch_embedding.n_patches(self.n_times)
        if n_patches > max_seq_length:
            raise ValueError(
                f"Input produces {n_patches} patches, which exceeds "
                f"max_seq_length={max_seq_length}."
            )
        self.spatial_pooling = _SleepFMAttentionPooling(
            embed_dim, encoder_pooling_heads, encoder_drop_prob
        )
        # Kept in the state dict, as in SleepFM: the released table differs
        # from a CPU recomputation in the last float32 bits, so it is loaded
        # with the checkpoint. Its 128 rows are those of the released encoder.
        n_positions = max(encoder_chunk_patches, _ENCODER_MAX_SEQ_LENGTH)
        self.register_buffer(
            "positional_encoding",
            sinusoidal_positional_encoding(n_positions, embed_dim).unsqueeze(0),
        )
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.transformer_encoder = _temporal_transformer(
            embed_dim, encoder_num_heads, encoder_num_layers, encoder_drop_prob
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
    def from_pretrained(
        cls,
        *args,
        encoder_model_name_or_path=None,
        encoder_revision=None,
        **kwargs,
    ):
        """Load the released sleep stager from the braindecode mirrors.

        ``pretrained_model_name_or_path`` defaults to
        ``"braindecode/SleepFMStager"``, which merges the encoder of the
        upstream ``model_base/best.pt`` with the upstream
        ``model_sleep_staging/best.pth`` head. The whole stager is pretrained,
        its five-class output layer included; pass ``n_outputs`` different
        from 5 to reinitialise that layer for another label set.

        Older revisions of that mirror hold only the tokenizer and the staging
        head, not the encoder's channel pooling and temporal Transformer.
        When a checkpoint lacks exactly those weights they are
        read from ``encoder_model_name_or_path`` (a repo id or a local
        directory; default ``"braindecode/SleepFM"``, the released base
        encoder), so the loaded model is complete, at ``encoder_revision``
        (a revision of that repo, not of the stager's). A checkpoint saved
        from a :class:`SleepFMStager` already holds them and is loaded as is.

        The released staging head was trained on four modalities encoded
        separately, so pass ``channel_modalities``; without it every channel
        is encoded as one modality and a warning is raised.
        """
        if not args and kwargs.get("pretrained_model_name_or_path") is None:
            kwargs["pretrained_model_name_or_path"] = cls._HF_DEFAULT_REPO
        model = super().from_pretrained(*args, **kwargs)
        # Warn whatever the checkpoint layout: a complete checkpoint loads the
        # same staging head, which expects the modalities encoded separately.
        if model.channel_modalities is None:
            warnings.warn(
                "Loading the released SleepFM stager without "
                "channel_modalities: every channel is encoded as a single "
                "modality, whereas the release encodes BAS, RESP, EKG and "
                "EMG channels separately. Pass channel_modalities to "
                "reproduce it.",
                UserWarning,
                stacklevel=2,
            )
        missing = model.__dict__.pop("_missing_encoder_keys", None)
        if missing:
            source = encoder_model_name_or_path or cls._HF_ENCODER_REPO
            state_dict = _read_safetensors(
                source,
                revision=encoder_revision,
                cache_dir=kwargs.get("cache_dir"),
                force_download=kwargs.get("force_download", False),
                local_files_only=kwargs.get("local_files_only", False),
                token=kwargs.get("token"),
            )
            absent = sorted(set(missing) - set(state_dict))
            if absent:
                raise RuntimeError(
                    f"{source} does not hold the SleepFM encoder weights "
                    f"{absent[:3]}{'...' if len(absent) > 3 else ''}."
                )
            model.load_state_dict({k: state_dict[k] for k in missing}, strict=False)
        return model

    @classmethod
    def _load_as_safetensor(cls, model, model_file, map_location, strict):
        from safetensors.torch import load_file

        state_dict = load_file(model_file, device=str(map_location))
        return cls._load_checkpoint(model, state_dict, strict)

    @classmethod
    def _load_as_pickle(cls, model, model_file, map_location, strict):
        state_dict = torch.load(
            model_file, map_location=torch.device(map_location), weights_only=True
        )
        model = cls._load_checkpoint(model, state_dict, strict)
        model.eval()
        return model

    @classmethod
    def _load_checkpoint(cls, model, state_dict, strict):
        """Load ``state_dict``, noting encoder weights left for the base mirror."""
        encoder_keys = {
            key
            for key in model.state_dict()
            if not key.startswith(("patch_embedding.", "staging_head.", "final_layer."))
        }
        missing = set(model.state_dict()) - set(state_dict)
        if missing and missing == encoder_keys:
            # Tokenizer + head checkpoint, as in older revisions of the
            # braindecode/SleepFMStager mirror: everything else must load, the
            # encoder comes afterwards.
            result = model.load_state_dict(state_dict, strict=False)
            if result.unexpected_keys:
                raise RuntimeError(
                    f"Unexpected keys in the checkpoint: {result.unexpected_keys}"
                )
            model._missing_encoder_keys = sorted(missing)
        else:
            model.load_state_dict(state_dict, strict=strict)
        return model

    def forward(
        self,
        x: torch.Tensor,
        channel_mask: torch.Tensor | None = None,
        return_features: bool = False,
        temporal_mask: torch.Tensor | None = None,
    ) -> torch.Tensor | dict[str, torch.Tensor | None]:
        """Return patch-wise staging logits or contextual features.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape ``(batch, n_chans, n_times)``.
        channel_mask : torch.Tensor | None
            ``(batch, n_chans)`` mask, ``True`` for padded channels.
        return_features : bool
            Return the per-patch LSTM features instead of the logits.
        temporal_mask : torch.Tensor | None
            ``(batch, n_patches)`` mask, ``True`` for padded patches.
        """
        mask = _prepare_channel_mask(channel_mask, x)
        n_patches = self.patch_embedding.n_patches(x.shape[-1])
        patch_mask = _prepare_temporal_mask(temporal_mask, x, n_patches)

        embeddings, missing_modalities = [], []
        for signal, modality_mask in zip(
            x.index_select(1, self._channel_order).split(self._modality_sizes, 1),
            mask.index_select(1, self._channel_order).split(self._modality_sizes, 1),
        ):
            embedding, missing = self._encode_modality(
                signal,
                modality_mask,
                patch_mask,
                mask_tokens=channel_mask is not None or patch_mask is not None,
            )
            embeddings.append(embedding)
            missing_modalities.append(missing)

        # Empty slots stand for the modalities the release pads to four.
        for _ in range(_RELEASED_MODALITY_SLOTS - len(embeddings)):
            embeddings.append(torch.zeros_like(embeddings[0]))
            missing_modalities.append(torch.ones_like(missing_modalities[0]))
        # (batch, n_modalities, n_patches, embed_dim)
        embeddings = torch.stack(embeddings, dim=1)
        modality_mask = torch.stack(missing_modalities, dim=1)

        features = self.staging_head(embeddings, modality_mask, patch_mask)
        if return_features:
            # "cls_token" is braindecode's feature-return contract, not a
            # credential: the stager has no class token, it labels every patch.
            return {"features": features, "cls_token": None}  # nosec B105
        logits = self.final_layer(features)
        # braindecode's cropped/time-series convention puts the class axis
        # second, whereas the head emits (batch, n_patches, n_outputs).
        return rearrange(logits, "batch patch cls -> batch cls patch")

    def _encode_modality(
        self,
        x: torch.Tensor,
        channel_mask: torch.Tensor,
        patch_mask: torch.Tensor | None,
        mask_tokens: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode one modality chunk by chunk, as the release's embedding step.

        Returns the ``(batch, n_patches, embed_dim)`` embeddings, zero where
        the modality is missing or the patch is padded, and the ``(batch,)``
        mask of samples without any valid channel of this modality.
        ``mask_tokens=False`` (nothing masked) skips the tokenizer's masked
        path, whose gather is slower in training.
        """
        n_patches = self.patch_embedding.n_patches(x.shape[-1])
        missing = channel_mask.all(dim=1)
        tokenizer_mask = None
        if mask_tokens:
            tokenizer_mask = repeat(
                channel_mask, "batch chans -> batch chans patch", patch=n_patches
            )
            if patch_mask is not None:
                tokenizer_mask = tokenizer_mask | patch_mask.unsqueeze(1)
        tokens = self.patch_embedding(x, tokenizer_mask)
        # A sample without this modality is pooled unmasked, then zeroed.
        tokens = self._pool_channels(tokens, _without_fully_masked_rows(channel_mask))

        # Chunks are encoded independently, so the positional encoding
        # restarts at each of them; a shorter trailing chunk is kept.
        chunk = self.encoder_chunk_patches
        n_full = n_patches // chunk
        encoded = []
        for start, stop, n_chunks in (
            (0, n_full * chunk, n_full),
            (n_full * chunk, n_patches, 1),
        ):
            if stop == start:
                continue
            key_padding_mask = None
            if patch_mask is not None:
                key_padding_mask = _without_fully_masked_rows(
                    rearrange(
                        patch_mask[:, start:stop],
                        "batch (chunk patch) -> (batch chunk) patch",
                        chunk=n_chunks,
                    )
                )
            segment = rearrange(
                tokens[:, start:stop],
                "batch (chunk patch) emb -> (batch chunk) patch emb",
                chunk=n_chunks,
            )
            segment = self._contextualize(segment, key_padding_mask)
            encoded.append(
                rearrange(
                    segment,
                    "(batch chunk) patch emb -> batch (chunk patch) emb",
                    chunk=n_chunks,
                )
            )
        embeddings = torch.cat(encoded, dim=1)

        # The release pads the staging input with zero embeddings.
        empty = repeat(missing, "batch -> batch patch", patch=n_patches)
        if patch_mask is not None:
            empty = empty | patch_mask
        return embeddings.masked_fill(empty.unsqueeze(-1), 0.0), missing

    def reset_head(self, n_outputs: int):
        """Replace the token-wise sleep-staging output projection."""
        self._set_n_outputs(n_outputs)
        self.final_layer = nn.Linear(self.embed_dim, n_outputs)
        return self


def _read_safetensors(
    name_or_path: str | Path,
    revision=None,
    cache_dir=None,
    force_download: bool = False,
    local_files_only: bool = False,
    token=None,
) -> dict[str, torch.Tensor]:
    """Read ``model.safetensors`` from a local directory or a Hub repo."""
    from safetensors.torch import load_file

    if Path(name_or_path).is_dir():
        return load_file(Path(name_or_path) / "model.safetensors")
    if huggingface_hub is False:
        raise ImportError(f"SleepFMStager.from_pretrained() {_HF_INSTALL_HINT}")
    path = huggingface_hub.hf_hub_download(
        repo_id=str(name_or_path),
        filename="model.safetensors",
        revision=revision,
        cache_dir=cache_dir,
        force_download=force_download,
        local_files_only=local_files_only,
        token=token,
    )
    return load_file(path)


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

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Tokenize ``x`` with shape ``(batch, channels, time)``.

        ``padding_mask`` of shape ``(batch, channels, n_patches)`` marks the
        patches to leave out with ``True``; their tokens are zero. Every other
        layer treats patches independently, but batch normalization pools them
        in training mode, so there the masked patches are not computed at all:
        otherwise their content would shift the statistics of the valid ones
        and the running averages.
        """
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
        if padding_mask is None:
            tokens = self.tokenizer(x)
        else:
            padded = rearrange(
                padding_mask, "batch chans patch -> (batch chans patch) 1"
            )
            if self.training:
                valid = ~padded[:, 0]
                tokens = x.new_zeros(x.shape[0], self.embed_dim)
                # An all-padded batch would give BatchNorm an empty input,
                # whose statistics are NaN, so it is skipped.
                if bool(valid.any()):
                    tokens = tokens.masked_scatter(
                        padded.logical_not(), self.tokenizer(x[valid])
                    )
            else:
                # Running statistics make every patch independent here. The
                # input is zeroed first, so a non-finite value in a masked
                # patch cannot reach the gradients either.
                x = x.masked_fill(padded.unsqueeze(-1), 0.0)
                tokens = self.tokenizer(x).masked_fill(padded, 0.0)
        return rearrange(
            tokens,
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


class _SleepFMStagingHead(_SleepFMSequenceMixin, nn.Module):
    """Predict one sleep stage per SleepFM patch embedding.

    The head consumes ``(batch, n_modalities, n_patches, embed_dim)`` encoder
    embeddings and returns one feature vector per patch. It reproduces the
    released downstream architecture, whose three stages answer three
    different questions:

    1. **Modality attention pooling** collapses the modality axis inside each
       patch, so a recording is described by what its modalities agree on
       rather than by their order or count.
    2. **A Transformer encoder** over the patch sequence (with fixed sinusoidal
       positional encoding) provides local context within the night; padded
       patches are masked out of its attention.
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
        self.transformer_encoder = _temporal_transformer(
            embed_dim, num_heads, num_layers, drop_prob
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
        temporal_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return contextual LSTM features for every signal patch.

        ``channel_mask`` is ``(batch, n_modalities)`` and ``temporal_mask``
        ``(batch, n_patches)``, both ``True`` for padding.
        """
        features = self._pool_channels(tokens, channel_mask)
        if temporal_mask is None:
            # The release always passes a (here all-valid) padding mask.
            temporal_mask = torch.zeros(
                features.shape[:2], dtype=torch.bool, device=features.device
            )
        # The patch sequence is then read as a time series of the night.
        features = self._contextualize(features, temporal_mask)
        features, _ = self.lstm(features)
        return features
