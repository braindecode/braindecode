# Authors: Fashad Ahmed <Fashad-Ahmed@users.noreply.github.com>
#
# License: CC-BY-NC-4.0
# Adapted from https://github.com/zou-group/sleepfm-clinical (CC BY-NC 4.0).

"""SleepFM models for multimodal polysomnography."""

from __future__ import annotations

import warnings
from collections.abc import Hashable, Sequence
from pathlib import Path

import torch
from einops import rearrange, repeat
from torch import nn

from braindecode.functional import sinusoidal_positional_encoding
from braindecode.models.base import EEGModuleMixin, huggingface_hub
from braindecode.modules import PatchTokenizer


class SleepFM(EEGModuleMixin, nn.Module, license="cc-by-nc-4.0"):
    r"""Sleep foundation model for multimodal polysomnography [sleepfm2026]_.

    :bdg-danger:`Foundation Model` :bdg-info:`Attention/Transformer`
    :bdg-success:`Convolution`

    Every channel is cut into non-overlapping 5-second patches (``patch_size``
    samples at 128 Hz; trailing samples are dropped) and embedded by a shared
    convolutional tokenizer. Attention pools the variable channel set of each
    patch, a Transformer models the patch sequence and a second attention
    layer pools it into one trial-level vector (see :meth:`encode`).

    Parameters
    ----------
    patch_size : int, default=640
        Samples per patch; at least 64 and divisible by 64.
    embed_dim : int, default=128
        Token and Transformer embedding dimension.
    num_heads : int, default=8
        Heads of the temporal Transformer.
    num_layers : int, default=6
        Layers of the temporal Transformer.
    pooling_heads : int, default=8
        Heads of the channel and temporal attention pooling.
    drop_prob : float, default=0.3
        Dropout probability of the attention and Transformer layers.
    max_seq_length : int, default=128
        Maximum number of patches (length of the positional table).
    activation : type[nn.Module], default=nn.ELU
        Tokenizer activation; ``nn.ELU`` matches the released weights.

    Notes
    -----
    ``channel_mask`` (``(batch, n_chans)``, ``True`` for missing channels)
    keeps masked channels out of every output, and out of the tokenizer's
    batch normalization in training. The official pretraining code normalizes
    zero-padded channels with the real ones; both agree when no channel is
    masked, and always in eval mode.

    The released encoder loads from the braindecode mirror; ``final_layer`` is
    not pretrained (the release is a contrastive encoder)::

        model = SleepFM.from_pretrained(n_chans=4, n_times=3840, n_outputs=5)

    `License <https://creativecommons.org/licenses/by-nc/4.0/>`_

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
        _check_input(self, patch_size, max_seq_length)
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        self.patch_embedding = _SleepFMTokenizer(
            patch_size, embed_dim, activation, self.n_times
        )
        _add_context_layers(
            self,
            embed_dim,
            num_heads,
            num_layers,
            pooling_heads,
            drop_prob,
            max_seq_length,
        )
        self.temporal_pooling = _SleepFMAttentionPooling(
            embed_dim, pooling_heads, drop_prob
        )
        self.final_layer = nn.Linear(embed_dim, self.n_outputs)

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        """Load the encoder; the repo defaults to ``"braindecode/SleepFM"``."""
        if not args and kwargs.get("pretrained_model_name_or_path") is None:
            kwargs["pretrained_model_name_or_path"] = "braindecode/SleepFM"
        return super().from_pretrained(*args, **kwargs)

    def encode(
        self,
        x: torch.Tensor,
        channel_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r"""Return the pooled and the per-patch representations.

        With :math:`t_{b,c,p}` the token of channel :math:`c` and patch
        :math:`p`, and masked means over the unmasked items:

        .. math::
            z_{b,p} = \operatorname{MaskedMean}_{c}
                      \big(\operatorname{SelfAttn}(t_{b,:,p})\big), \quad
            h_{b} = \operatorname{Transformer}
                    \big(\operatorname{LN}(z_{b} + \mathrm{PE})\big), \quad
            g_{b} = \operatorname{Mean}_{p}
                    \big(\operatorname{SelfAttn}(h_{b})\big).

        Parameters
        ----------
        x : torch.Tensor
            Input of shape ``(batch, n_chans, n_times)``.
        channel_mask : torch.Tensor | None
            ``(batch, n_chans)`` mask, ``True`` for missing channels.

        Returns
        -------
        pooled : torch.Tensor
            :math:`g`, shape ``(batch, embed_dim)``.
        contextual_tokens : torch.Tensor
            :math:`h`, shape ``(batch, n_patches, embed_dim)``.
        """
        mask = _check_mask(channel_mask, x.shape[:2], x, "channel_mask")
        patch_mask = None
        if mask is None:
            mask = torch.zeros(x.shape[:2], dtype=torch.bool, device=x.device)
        else:
            patch_mask = repeat(
                mask,
                "batch chans -> batch chans patch",
                patch=x.shape[-1] // self.patch_size,
            )
        tokens = self.patch_embedding(x, patch_mask)
        contextual_tokens = _contextualize(self, _pool_set(self, tokens, mask))
        return self.temporal_pooling(contextual_tokens), contextual_tokens

    def forward(
        self,
        x: torch.Tensor,
        channel_mask: torch.Tensor | None = None,
        return_features: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor | None]:
        """Return trial logits, or the pooled features."""
        pooled, _ = self.encode(x, channel_mask)
        if return_features:
            return {"features": pooled, "cls_token": None}  # nosec B105
        return self.final_layer(pooled)

    def reset_head(self, n_outputs: int):
        """Replace the trial-level output layer."""
        self._set_n_outputs(n_outputs)
        self.final_layer = nn.Linear(self.embed_dim, n_outputs)
        return self


class SleepFMStager(EEGModuleMixin, nn.Module, license="cc-by-nc-4.0"):
    r"""SleepFM encoder with the released patch-wise sleep-staging head [sleepfm2026]_.

    :bdg-danger:`Foundation Model` :bdg-info:`Attention/Transformer`
    :bdg-success:`Convolution` :bdg-secondary:`Recurrent`

    1. **Encoder** (:class:`SleepFM` without its trial pooling). Channels are
       grouped by ``channel_modalities``; each modality is encoded in
       independent chunks of ``encoder_chunk_patches`` patches (5 minutes in
       the release, positions restart at every chunk). A shorter trailing
       chunk is kept, whereas the official embedding script drops it.
    2. **Staging head.** Attention pools the modalities of every patch (always
       at least four slots, missing ones masked, as released), then a
       Transformer and a bidirectional LSTM run over the night.

    The output has shape ``(batch, n_outputs, n_patches)``: one prediction per
    5-second patch (Wake, N1, N2, N3, REM for the release), so a 30-second
    scored epoch spans six predictions.

    Parameters
    ----------
    channel_modalities : sequence of hashable, optional
        Modality label of every channel, e.g. ``["BAS", "RESP", "EKG", "EMG"]``
        as in the release. ``None`` puts every channel in one modality.
    patch_size : int, default=640
        Samples per patch.
    embed_dim : int, default=128
        Token and recurrent feature dimension.
    encoder_num_heads : int, default=8
        Heads of the encoder's temporal Transformer.
    encoder_num_layers : int, default=6
        Layers of the encoder's temporal Transformer.
    encoder_pooling_heads : int, default=8
        Heads of the encoder's channel attention pooling.
    encoder_drop_prob : float, default=0.0
        Encoder dropout; the release embeds with dropout disabled.
    encoder_chunk_patches : int, default=60
        Patches the encoder sees at once. Above 128 the encoder's positional
        table no longer loads from the released weights.
    staging_num_heads : int, default=4
        Heads of the staging Transformer.
    staging_num_layers : int, default=1
        Layers of the staging Transformer and of the bidirectional LSTM.
    staging_pooling_heads : int, default=4
        Heads of the modality attention pooling.
    drop_prob : float, default=0.3
        Dropout probability of the staging head.
    max_seq_length : int, default=8196
        Maximum number of patches.
    activation : type[nn.Module], default=nn.ELU
        Tokenizer activation; ``nn.ELU`` matches the released weights.

    Notes
    -----
    ``temporal_mask`` (``(batch, n_patches)``, ``True`` for padding) marks
    padded patches at the end of shorter recordings. As in the release they
    enter the head as zero embeddings masked out of its Transformer, but the
    LSTM still runs over them, so valid predictions depend on the amount of
    padding (never on its content). ``channel_mask`` works as in
    :class:`SleepFM`; a modality without a valid channel is masked.

    The released stager loads from the braindecode mirror::

        model = SleepFMStager.from_pretrained(
            n_chans=7,
            n_times=38400,
            n_outputs=5,
            channel_modalities=["BAS"] * 3 + ["RESP"] * 2 + ["EKG", "EMG"],
        )

    `License <https://creativecommons.org/licenses/by-nc/4.0/>`_

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
        _check_input(self, patch_size, max_seq_length)
        labels: list[Hashable] = [0] * self.n_chans
        if channel_modalities is not None:
            labels = list(channel_modalities)
            if len(labels) != self.n_chans:
                raise ValueError(
                    f"channel_modalities must name one modality per channel: "
                    f"got {len(labels)} for n_chans={self.n_chans}."
                )
        # Modalities in order of first appearance; the head pools them as a set.
        modalities = list(dict.fromkeys(labels))
        channel_order = [
            i for m in modalities for i, label in enumerate(labels) if label == m
        ]
        self._modality_sizes = [labels.count(m) for m in modalities]
        self.register_buffer(
            "_channel_order", torch.tensor(channel_order), persistent=False
        )
        self.channel_modalities = channel_modalities
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.encoder_chunk_patches = encoder_chunk_patches

        # Same modules and parameter names as SleepFM's encoder.
        self.patch_embedding = _SleepFMTokenizer(
            patch_size, embed_dim, activation, self.n_times
        )
        _add_context_layers(
            self,
            embed_dim,
            encoder_num_heads,
            encoder_num_layers,
            encoder_pooling_heads,
            encoder_drop_prob,
            max(encoder_chunk_patches, 128),  # the released table has 128 rows
        )
        self.staging_head = _SleepFMStagingHead(
            embed_dim,
            staging_num_heads,
            staging_num_layers,
            staging_pooling_heads,
            drop_prob,
            max_seq_length,
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
        """Load the released stager; the repo defaults to ``"braindecode/SleepFMStager"``.

        Older revisions of that mirror hold only the tokenizer and the head;
        the missing encoder weights are then read from
        ``encoder_model_name_or_path`` (repo id or local directory, default
        ``"braindecode/SleepFM"``) at ``encoder_revision``.
        """
        if not args and kwargs.get("pretrained_model_name_or_path") is None:
            kwargs["pretrained_model_name_or_path"] = "braindecode/SleepFMStager"
        model = super().from_pretrained(*args, **kwargs)
        if model.channel_modalities is None:
            warnings.warn(
                "The released stager encodes BAS, RESP, EKG and EMG channels "
                "separately; pass channel_modalities to reproduce it.",
                UserWarning,
                stacklevel=2,
            )
        missing = model.__dict__.pop("_missing_encoder_keys", None)
        if missing:
            from safetensors.torch import load_file

            source = str(encoder_model_name_or_path or "braindecode/SleepFM")
            path = Path(source) / "model.safetensors"
            if not path.parent.is_dir():
                path = huggingface_hub.hf_hub_download(
                    source,
                    "model.safetensors",
                    revision=encoder_revision,
                    cache_dir=kwargs.get("cache_dir"),
                    local_files_only=kwargs.get("local_files_only", False),
                    token=kwargs.get("token"),
                )
            state_dict = load_file(path)
            model.load_state_dict({k: state_dict[k] for k in missing}, strict=False)
        return model

    @classmethod
    def _load_as_safetensor(cls, model, model_file, map_location, strict):
        from safetensors.torch import load_file

        state_dict = load_file(model_file, device=str(map_location))
        head = ("patch_embedding.", "staging_head.", "final_layer.")
        model._missing_encoder_keys = [
            k
            for k in model.state_dict()
            if k not in state_dict and not k.startswith(head)
        ]
        model.load_state_dict(
            state_dict, strict=strict and not model._missing_encoder_keys
        )
        return model

    def forward(
        self,
        x: torch.Tensor,
        channel_mask: torch.Tensor | None = None,
        return_features: bool = False,
        temporal_mask: torch.Tensor | None = None,
    ) -> torch.Tensor | dict[str, torch.Tensor | None]:
        """Return patch-wise logits, or the per-patch LSTM features."""
        n_patches = x.shape[-1] // self.patch_size
        patch_mask = _check_mask(
            temporal_mask, (x.shape[0], n_patches), x, "temporal_mask"
        )
        mask = _check_mask(channel_mask, x.shape[:2], x, "channel_mask")
        mask_tokens = mask is not None or patch_mask is not None
        if mask is None:
            mask = torch.zeros(x.shape[:2], dtype=torch.bool, device=x.device)

        embeddings, missing_modalities = [], []
        for signal, modality_mask in zip(
            x.index_select(1, self._channel_order).split(self._modality_sizes, 1),
            mask.index_select(1, self._channel_order).split(self._modality_sizes, 1),
        ):
            embedding, missing = self._encode_modality(
                signal, modality_mask, patch_mask, mask_tokens
            )
            embeddings.append(embedding)
            missing_modalities.append(missing)
        # The released head (``max_channels: 4``) always gets four modality
        # slots; missing ones are empty and masked.
        for _ in range(4 - len(embeddings)):
            embeddings.append(torch.zeros_like(embeddings[0]))
            missing_modalities.append(torch.ones_like(missing_modalities[0]))
        features = self.staging_head(
            torch.stack(embeddings, dim=1),
            torch.stack(missing_modalities, dim=1),
            patch_mask,
        )
        if return_features:
            return {"features": features, "cls_token": None}  # nosec B105
        logits = self.final_layer(features)
        return rearrange(logits, "batch patch cls -> batch cls patch")

    def _encode_modality(self, x, channel_mask, patch_mask, mask_tokens):
        """Encode one modality chunk by chunk, as the release's embedding step.

        Returns ``(batch, n_patches, embed_dim)`` embeddings, zero where the
        modality is missing or the patch padded, and the ``(batch,)`` mask of
        samples without this modality.
        """
        n_patches = x.shape[-1] // self.patch_size
        missing = channel_mask.all(dim=1)
        tokenizer_mask = None
        if mask_tokens:  # without any mask the faster unmasked path is taken
            tokenizer_mask = repeat(
                channel_mask, "batch chans -> batch chans patch", patch=n_patches
            )
            if patch_mask is not None:
                tokenizer_mask = tokenizer_mask | patch_mask.unsqueeze(1)
        tokens = self.patch_embedding(x, tokenizer_mask)
        # A sample without this modality is pooled unmasked, then zeroed.
        unmasked = channel_mask & ~missing.unsqueeze(1)
        tokens = _pool_set(self, tokens, unmasked)

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
                key_padding_mask = rearrange(
                    patch_mask[:, start:stop],
                    "batch (chunk patch) -> (batch chunk) patch",
                    chunk=n_chunks,
                )
                # A fully padded chunk is computed unmasked, then zeroed.
                key_padding_mask = key_padding_mask & ~key_padding_mask.all(
                    dim=1, keepdim=True
                )
            segment = rearrange(
                tokens[:, start:stop],
                "batch (chunk patch) emb -> (batch chunk) patch emb",
                chunk=n_chunks,
            )
            segment = _contextualize(self, segment, key_padding_mask)
            encoded.append(
                rearrange(
                    segment,
                    "(batch chunk) patch emb -> batch (chunk patch) emb",
                    chunk=n_chunks,
                )
            )
        embeddings = torch.cat(encoded, dim=1)
        empty = repeat(missing, "batch -> batch patch", patch=n_patches)
        if patch_mask is not None:
            empty = empty | patch_mask
        return embeddings.masked_fill(empty.unsqueeze(-1), 0.0), missing

    def reset_head(self, n_outputs: int):
        """Replace the patch-wise output layer."""
        self._set_n_outputs(n_outputs)
        self.final_layer = nn.Linear(self.embed_dim, n_outputs)
        return self


class _SleepFMTokenizer(nn.Module):
    """Embed every ``patch_size``-sample patch of every channel independently.

    Patches are cut by :class:`~braindecode.modules.PatchTokenizer` (trailing
    samples cropped), then six stride-2 convolution blocks reduce each patch
    to one ``embed_dim`` vector.
    """

    def __init__(self, patch_size, embed_dim, activation, n_times) -> None:
        super().__init__()
        if patch_size < 64 or patch_size % 64:
            raise ValueError("patch_size must be at least 64 and divisible by 64.")
        self.embed_dim = embed_dim
        self.patchify = PatchTokenizer(patch_size, n_times, on_non_divisible="crop")
        layers: list[nn.Module] = []
        in_channels = 1
        for block_index, out_channels in enumerate((4, 8, 16, 32, 64, 128), 1):
            layers.extend(
                [
                    nn.Conv1d(in_channels, out_channels, 5, stride=2, padding=2),
                    nn.BatchNorm1d(out_channels),
                    activation(),
                    nn.LayerNorm([out_channels, patch_size // (2**block_index)]),
                ]
            )
            in_channels = out_channels
        layers.extend(
            [nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(128, embed_dim)]
        )
        self.tokenizer = nn.Sequential(*layers)

    def forward(self, x, padding_mask=None):
        """Tokenize ``(batch, chans, time)``; ``padding_mask`` is ``(batch, chans, n_patches)``.

        Masked tokens are zero. In training, masked patches are not computed
        at all, so they do not enter the batch-norm statistics.
        """
        batch, channels, _ = x.shape
        x = rearrange(
            self.patchify(x), "batch chans patch time -> (batch chans patch) 1 time"
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
                if bool(valid.any()):  # BatchNorm of an empty batch is NaN
                    tokens = tokens.masked_scatter(
                        padded.logical_not(), self.tokenizer(x[valid])
                    )
            else:
                # Zero the input too, so a non-finite masked value cannot
                # reach the gradients.
                x = x.masked_fill(padded.unsqueeze(-1), 0.0)
                tokens = self.tokenizer(x).masked_fill(padded, 0.0)
        return rearrange(
            tokens,
            "(batch chans patch) emb -> batch chans patch emb",
            batch=batch,
            chans=channels,
        )


class _SleepFMAttentionPooling(nn.Module):
    """Self-attention, then a masked mean over an unordered set."""

    def __init__(self, input_dim, num_heads, drop_prob) -> None:
        super().__init__()
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=input_dim, nhead=num_heads, dropout=drop_prob, batch_first=True
        )

    def forward(self, x, key_padding_mask=None):
        """Pool ``(batch, items, emb)``; ``True`` in the mask marks padding."""
        if key_padding_mask is None:
            return self.transformer_layer(x).mean(dim=1)
        # A fully masked set is pooled unmasked, then zeroed.
        all_masked = key_padding_mask.all(dim=1, keepdim=True)
        key_padding_mask = key_padding_mask & ~all_masked
        if x.shape[1] == 1:  # upstream skips attention for a masked singleton
            output = x[:, 0]
        else:
            output = self.transformer_layer(x, src_key_padding_mask=key_padding_mask)
            valid = (~key_padding_mask).unsqueeze(-1).to(output.dtype)
            output = (output * valid).sum(dim=1) / valid.sum(dim=1)
        return output.masked_fill(all_masked, 0)


class _SleepFMStagingHead(nn.Module):
    """Modality attention pooling, Transformer and bidirectional LSTM per patch."""

    def __init__(
        self, embed_dim, num_heads, num_layers, pooling_heads, drop_prob, max_seq_length
    ) -> None:
        super().__init__()
        _add_context_layers(
            self,
            embed_dim,
            num_heads,
            num_layers,
            pooling_heads,
            drop_prob,
            max_seq_length,
        )
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=embed_dim // 2,
            num_layers=num_layers,
            batch_first=True,
            dropout=drop_prob if num_layers > 1 else 0.0,
            bidirectional=True,
        )

    def forward(self, tokens, modality_mask, temporal_mask=None):
        """Map ``(batch, modalities, patches, emb)`` to ``(batch, patches, emb)``."""
        features = _pool_set(self, tokens, modality_mask)
        if temporal_mask is None:  # the release always passes a padding mask
            temporal_mask = torch.zeros(
                features.shape[:2], dtype=torch.bool, device=features.device
            )
        features = _contextualize(self, features, temporal_mask)
        features, _ = self.lstm(features)
        return features


def _check_input(model, patch_size, max_seq_length):
    """Warn on a sampling rate other than 128 Hz; check the patch count."""
    if model.sfreq != 128:
        warnings.warn(
            f"{type(model).__name__} was pretrained at 128 Hz, got "
            f"{model.sfreq:g} Hz; resample to reuse the released weights.",
            UserWarning,
            stacklevel=3,
        )
    n_patches = model.n_times // patch_size
    if not 0 < n_patches <= max_seq_length:
        raise ValueError(
            f"n_times={model.n_times} gives {n_patches} patches of {patch_size} "
            f"samples; between 1 and max_seq_length={max_seq_length} are needed."
        )


def _check_mask(mask, shape, x, name):
    """Return ``mask`` as a boolean tensor on ``x``'s device, or ``None``."""
    if mask is None:
        return None
    if tuple(mask.shape) != tuple(shape):
        raise ValueError(
            f"{name} must have shape {tuple(shape)}, got {tuple(mask.shape)}."
        )
    return mask.to(device=x.device, dtype=torch.bool)


def _add_context_layers(
    module, embed_dim, num_heads, num_layers, pooling_heads, drop_prob, n_positions
):
    """Add the set pooling, positional table, LayerNorm and pre-norm Transformer.

    SleepFM's encoder and the staging head share this stack and the parameter
    names of the released checkpoints. The positional table is a buffer: the
    released one differs from a CPU recomputation in the last float32 bits.
    """
    module.spatial_pooling = _SleepFMAttentionPooling(
        embed_dim, pooling_heads, drop_prob
    )
    module.register_buffer(
        "positional_encoding",
        sinusoidal_positional_encoding(n_positions, embed_dim).unsqueeze(0),
    )
    module.layer_norm = nn.LayerNorm(embed_dim)
    layer = nn.TransformerEncoderLayer(
        d_model=embed_dim,
        nhead=num_heads,
        dropout=drop_prob,
        batch_first=True,
        norm_first=True,
    )
    module.transformer_encoder = nn.TransformerEncoder(
        layer, num_layers=num_layers, enable_nested_tensor=False
    )


def _pool_set(module, tokens, mask):
    """Pool the set axis of ``(batch, set, patch, emb)`` within every patch."""
    n_patches = tokens.shape[2]
    tokens = rearrange(tokens, "batch chans patch emb -> (batch patch) chans emb")
    mask = repeat(mask, "batch chans -> (batch patch) chans", patch=n_patches)
    pooled = module.spatial_pooling(tokens, mask)
    return rearrange(pooled, "(batch patch) emb -> batch patch emb", patch=n_patches)


def _contextualize(module, tokens, mask=None):
    r"""Return :math:`\operatorname{Transformer}(\operatorname{LN}(z + \mathrm{PE}))`."""
    tokens = tokens + module.positional_encoding[:, : tokens.shape[1]]
    tokens = module.layer_norm(tokens)
    return module.transformer_encoder(tokens, src_key_padding_mask=mask)
