# Authors: Rukuang Huang, SungJun Cho and the OHBA Analysis Group (original implementation)
#          Bruno Aristimunha <b.aristimunha@gmail.com> (Braindecode adaptation)
#
# License: MIT
# Adapted from https://github.com/OHBA-analysis/MEG-GPT (MIT) and
# https://github.com/OHBA-analysis/EphysTokenizer (MIT).
"""MEG-GPT: a data-driven tokenizer and a time-attention GPT for source-space MEG."""

from __future__ import annotations

import math
from functools import partial
from typing import Dict, Optional, Sequence, Union

import torch
from einops.layers.torch import Rearrange
from torch import Tensor, nn

from braindecode.functional import (
    check_ch_mask,
    init_weights_,
    pad_or_crop_to_patches,
)
from braindecode.functional.patches import _check_on_non_divisible
from braindecode.models.base import EEGModuleMixin
from braindecode.modules import FeedForwardBlock


class MEGGPT(EEGModuleMixin, nn.Module, license="mit"):
    r"""MEG-GPT from Huang et al. (2026) [huang2026meggpt]_.

    :bdg-danger:`Foundation Model` :bdg-info:`Attention/Transformer`
    :bdg-secondary:`Recurrent`

    .. figure:: ../_static/model/meggpt_arch.png
       :align: center
       :alt: MEG-GPT architecture
       :width: 1000px

       MEG-GPT predicting the next token from token, channel, position and
       optional extra embeddings through a Transformer decoder, reproduced from
       [huang2026meggpt]_ (CC BY 4.0).

    .. versionadded:: 2.1

    .. rubric:: Architecture Overview

    With ``B`` the batch, ``C = n_chans``, ``D = embed_dim`` and ``S`` the number
    of segments (``n_times``, padded or cropped to a multiple of the segment
    length, divided by it):

    1. The EphysTokenizer encoder [ephystokenizer]_ turns each channel into a
       stream of discrete tokens, one per sample: a GRU runs over the channel,
       a linear layer and a layer norm give one logit per codebook entry, and
       the argmax is remapped to a token id through the released vocabulary.
       ``x (B, C, T) -> token ids (B, C, S * 100)``.
    2. The token stream is cut into segments of ``n_patches[0] * patch_lens[0]``
       tokens (100, i.e. 0.4 s at 250 Hz). Token, time and channel embeddings
       are summed. ``-> (B * S, 100, C, D)``.
    3. A stack of causal time-attention layers runs on every channel. In layer
       ``i`` the keys and values are pooled into ``n_patches[i]`` patches of
       ``patch_lens[i]`` steps and the output keeps the last steps only, so the
       sequence shrinks to ``n_last_steps`` outputs per segment.
       ``-> (B * S, n_last_steps, C, D)``, merged to
       ``(B, S * n_last_steps, C, D)``.
    4. The outputs are averaged over time and segments into
       ``(batch, n_chans, embed_dim)`` features (Section 2.4 of the paper),
       flattened and classified by a linear layer.
       ``-> (B, C, D) -> (B, C * D) -> (B, n_outputs)``.

    .. rubric:: Macro Components

    ``MEGGPT.tokenizer``
        **Operations:** per-channel GRU, linear layer, layer norm, argmax and
        vocabulary lookup.

        **Role:** discretise each sample of each channel. The argmax has no
        gradient, so the tokenizer is frozen (its parameters do not require
        gradients).

    ``MEGGPT.layers``
        **Operations:** pre-norm causal attention over time with patched keys
        and values (a linear layer mixes the heads and steps of a patch), a
        banded mask in layers with a ``patch_band``, and a feed-forward block.

        **Role:** model the temporal dependencies of each brain region.

    ``MEGGPT.final_layer``
        **Operations:** flatten and linear projection.

        **Role:** map the time-averaged features to class logits.

    .. rubric:: Temporal, Spatial, and Spectral Encoding

    - **Temporal:** the tokenizer GRU encodes the whole window in one pass (the
      reference uses 200-sample windows, with the same tokens up to 200
      samples); the GPT attends causally within independent segments of
      ``n_patches[0] * patch_lens[0]`` tokens.
    - **Spatial:** each parcel is tokenized and attended alone, with a learned
      channel embedding; only the flattened ``final_layer`` mixes parcels.
    - **Spectral:** implicit in the tokenizer GRU; there is no explicit
      transform.

    .. rubric:: Additional Mechanisms

    - **Input:** the 52 parcels of the Glasser 52 parcellation in the released
      order, since the channel embedding is indexed by position, sign-flip
      aligned (Section 2.5.3), at 250 Hz and z-scored over the recording
      (``tokenize_session(standardize=True)`` in the reference); the model
      neither normalises nor checks ``sfreq``. With ``on_non_divisible="pad"``,
      the feature mean skips the kept steps after the end of the input.
    - **Channel mask:** with ``return_features=True``, ``ch_mask`` (the mask of
      :func:`~braindecode.datasets.pad_channels_collate`, real channels first)
      gives each recording the features of its real parcels run alone, and
      zeros for padded channels. The flattened head is tied to ``n_chans``, so
      without ``return_features`` it raises a ``ValueError``.
    - **Token mask:** ``token_mask`` (True=hide) zeroes selected clean token
      embeddings before the causal decoder, without changing the frozen
      tokenizer or the attention masks; it is neither the released pretraining
      method nor a bidirectional masked-sample reconstruction objective.
      :meth:`encode_tokens` returns next-token targets for a downstream
      vocabulary head; the encoder installs none.
    - **Session embedding:** the released model adds one for its 621 Cam-CAN
      sessions; the paper discards it downstream (Section 2.5.4), so the class
      has none, the converted weights drop it, and the forward equals the
      released model called without session labels.
    - **Not ported:** only the released configuration, time attention with
      patching. The reference's optional channel and cross attention, unpatched
      keys, rotary or sinusoidal positions and batch/group norms are not
      implemented.

    .. rubric:: Pretrained Weights

    The released Cam-CAN MEG-GPT and EphysTokenizer weights, converted to this
    class's keys without the next-token head and the tokenizer's signal
    decoder, are at ``braindecode/meggpt-pretrained`` (revision ``5c49e27c``);
    the head is randomly initialized::

        from braindecode.models import MEGGPT
        model = MEGGPT.from_pretrained("braindecode/meggpt-pretrained", n_outputs=4)

    They expect the input described above: source-space MEG at 250 Hz, the 52
    parcels in the released order, sign-flip aligned and z-scored over the
    recording (Section 2.5.3). On the same token ids, the decoder outputs and
    the features equal those of the authors' model called without session
    labels (:gh:`1555`).

    Parameters
    ----------
    n_tokens : int, default=66
        Size of the GPT vocabulary (65 used codebook entries plus id 0 for the
        unused ones in the released vocabulary).
    n_codes : int, default=128
        Codebook size of the tokenizer.
    rnn_hidden_size : int, default=128
        Hidden size of the tokenizer GRU.
    temperature : float, default=0.1
        Divisor of the tokenizer's normalised logits before the argmax (the
        released EphysTokenizer value). It leaves the argmax unchanged except for
        float near-ties, so keep it to reproduce the released tokens.
    embed_dim : int, default=256
        Embedding and model dimension.
    num_heads : int, default=4
        Number of attention heads.
    feedforward_dim : int, default=1024
        Hidden width of the feed-forward blocks.
    n_patches : sequence of int, default=(100, 25, 20, 10, 8)
        Number of key/value patches in each layer.
    patch_lens : sequence of int, default=(1, 4, 4, 8, 8)
        Steps per key/value patch in each layer. Layer ``i`` reads
        ``n_patches[i] * patch_lens[i]`` steps and outputs as many steps as the
        next layer reads; ``n_patches[0] * patch_lens[0]`` is the segment length.
    n_last_steps : int, default=16
        Number of steps the last layer outputs per segment.
    patch_band : sequence of int or None, default=(8, None, None, None, None)
        Half-width, in patches, of the banded attention of each layer; ``None``
        for no band.
    activation : type[nn.Module], default=nn.GELU
        Activation of the feed-forward blocks.
    drop_prob : float, default=0.2
        Dropout after the embeddings, the attention and the feed-forward blocks.
    embedding_init_bound : float, default=0.05
        Bound ``b`` of the uniform initialisation ``U(-b, b)`` of the token
        embedding, as in the reference initializer.
    on_non_divisible : {"pad", "crop", "error"}, default="pad"
        How a window that is not a multiple of the segment length is handled:
        right zero-padded, cropped, or rejected.

    Input shape
    -----------
    ``(batch, n_chans, n_times)``.

    Output shape
    ------------
    By default ``(batch, n_outputs)`` logits. With ``return_features=True``,
    ``{"features": (batch, n_chans, embed_dim), "cls_token": None,
    "tokens": (batch, n_chans, n_segments * n_last_steps, embed_dim)}``, where
    ``"tokens"`` are the last layer's outputs before the mean over time.

    References
    ----------
    .. [huang2026meggpt] Huang, R., Cho, S., Gohil, C., Parker Jones, O., &
       Woolrich, M. (2026). MEG-GPT: A transformer-based foundation model for
       magnetoencephalography data. Imaging Neuroscience, 4, IMAG.a.1301.
       https://doi.org/10.1162/imag.a.1301 (preprint arXiv:2510.18080).
    .. [ephystokenizer] EphysTokenizer, released implementation:
       https://github.com/OHBA-analysis/EphysTokenizer

    `License <https://github.com/OHBA-analysis/MEG-GPT/blob/main/LICENSE>`_
    """

    _time_axis = "free"
    _channel_axis = "fixed"  # fixed: channel_embedding

    def __init__(
        self,
        # braindecode parameters
        n_outputs=None,
        n_chans=None,
        chs_info=None,
        n_times=None,
        input_window_seconds=None,
        sfreq=None,
        *,
        # model-specific parameters
        n_tokens: int = 66,
        n_codes: int = 128,
        rnn_hidden_size: int = 128,
        temperature: float = 0.1,
        embed_dim: int = 256,
        num_heads: int = 4,
        feedforward_dim: int = 1024,
        n_patches: Sequence[int] = (100, 25, 20, 10, 8),
        patch_lens: Sequence[int] = (1, 4, 4, 8, 8),
        n_last_steps: int = 16,
        patch_band: Sequence[Optional[int]] = (8, None, None, None, None),
        activation: type[nn.Module] = nn.GELU,
        drop_prob: float = 0.2,
        embedding_init_bound: float = 0.05,
        on_non_divisible: str = "pad",
    ):
        super().__init__(
            n_outputs=n_outputs,
            n_chans=n_chans,
            chs_info=chs_info,
            n_times=n_times,
            input_window_seconds=input_window_seconds,
            sfreq=sfreq,
        )
        del n_outputs, n_chans, chs_info, n_times, input_window_seconds, sfreq
        _check_on_non_divisible(on_non_divisible)
        self.segment_len = n_patches[0] * patch_lens[0]
        self.on_non_divisible = on_non_divisible
        self.n_last_steps = n_last_steps
        # Read again by reset_parameters, through _init_parameters.
        self.embedding_init_bound = embedding_init_bound

        self.tokenizer = _EphysTokenizer(
            n_codes, n_tokens, rnn_hidden_size, temperature
        )
        self.to_segments = Rearrange("b c (s t) -> (b s) t c", t=self.segment_len)
        self.from_segments = Rearrange("b s t c d -> b (s t) c d")
        self.to_tokens = Rearrange("b t c d -> b c t d")
        self.token_embedding = nn.Embedding(n_tokens, embed_dim)
        self.time_embedding = nn.Parameter(torch.empty(self.segment_len, embed_dim))
        self.channel_embedding = nn.Parameter(torch.empty(self.n_chans, embed_dim))
        self.drop = nn.Dropout(drop_prob)
        # Each layer outputs as many steps as the next one reads.
        n_outs = [n * p for n, p in zip(n_patches[1:], patch_lens[1:], strict=True)]
        self.layers = nn.ModuleList(
            _DecoderLayer(
                embed_dim,
                num_heads,
                feedforward_dim,
                n,
                p,
                n_out,
                band,
                activation,
                drop_prob,
            )
            for n, p, n_out, band in zip(
                n_patches, patch_lens, n_outs + [n_last_steps], patch_band, strict=True
            )
        )
        self.final_layer = nn.Sequential(
            nn.Flatten(), nn.Linear(self.n_chans * embed_dim, self.n_outputs)
        )
        self._n_times_step = self.segment_len
        self._snaps_in_prepare_inputs = False
        self._min_n_times = 1 if self.on_non_divisible == "pad" else self.segment_len
        self._init_parameters()

    def _accepts_any_n_chans(self, args: tuple, kwargs: dict) -> bool:
        if super()._accepts_any_n_chans(args, kwargs):
            return True
        return bool(self._call_arg(args, kwargs, "return_features"))

    def _init_parameters(self) -> None:
        # Reference initialisation (meg_gpt/optim/initializer.py): Xavier linear
        # layers with zero bias, small uniform embeddings.
        bound = self.embedding_init_bound
        uniform = partial(nn.init.uniform_, a=-bound, b=bound)
        rules = {nn.Linear: nn.init.xavier_uniform_, nn.Embedding: uniform}
        self.apply(partial(init_weights_, weights=rules))
        nn.init.xavier_uniform_(self.time_embedding)
        nn.init.xavier_uniform_(self.channel_embedding)

    def forward(
        self,
        x: Tensor,
        return_features: bool = False,
        ch_mask: Optional[Tensor] = None,
        token_mask: Optional[Tensor] = None,
    ) -> Union[Tensor, Dict[str, Optional[Tensor]]]:
        """``ch_mask``: optional ``(batch, n_chans)`` mask of real channels, see
        :func:`~braindecode.functional.check_ch_mask`. Needs
        ``return_features=True``; the features and tokens of padded channels
        are zero. ``token_mask`` is a boolean ``(batch, n_chans, n_times)``
        mask (True=hide): selected clean token embeddings are replaced by
        zeros before positions and causal context, not before tokenization.
        See :meth:`encode_tokens` for next-token targets and loss selection.
        """
        ch_mask = check_ch_mask(ch_mask, x)  # (batch, n_chans) or None
        if ch_mask is not None:
            if not return_features:
                raise ValueError(
                    "ch_mask needs return_features=True: the flattened head is "
                    "tied to the channels."
                )
            # Padding may hold NaN; each channel is tokenized alone.
            # (batch, n_chans, n_times), unchanged
            x = x.masked_fill(~ch_mask[:, :, None], 0.0)
        # (batch, n_chans, n_segments * segment_len) or None
        token_mask = self._prepare_token_mask(token_mask, x)
        n_times = x.shape[-1]
        # (batch, n_chans, n_times) -> (batch, n_chans, n_segments * segment_len)
        x = pad_or_crop_to_patches(x, self.segment_len, self.on_non_divisible)
        token_ids = self.tokenizer(x)  # (batch, n_chans, n_segments * segment_len)
        # (batch, n_segments * n_last_steps, n_chans, embed_dim)
        h = self._encode_prepared_tokens(token_ids, token_mask)
        features = self._pool_steps(
            h, x.shape[-1], n_times
        )  # (batch, n_chans, embed_dim)
        if return_features:
            # (batch, n_chans, n_segments * n_last_steps, embed_dim)
            tokens = self.to_tokens(h)
            if ch_mask is not None:
                # (batch, n_chans, embed_dim), unchanged
                features = features.masked_fill(~ch_mask[:, :, None], 0.0)
                # (batch, n_chans, n_segments * n_last_steps, embed_dim), unchanged
                tokens = tokens.masked_fill(~ch_mask[:, :, None, None], 0.0)
            out: Dict[str, Optional[Tensor]] = {
                "features": features,
                "cls_token": None,
                "tokens": tokens,
            }
            return out
        # (batch, n_chans, embed_dim) -> (batch, n_chans * embed_dim)
        # -> (batch, n_outputs)
        return self.final_layer(features)

    def _pool_steps(self, h: Tensor, n_prepared: int, n_times: int) -> Tensor:
        """Mean over the kept steps that sit on a real sample.

        ``h`` is ``(batch, n_segments * n_last_steps, n_chans, embed_dim)`` and
        the result ``(batch, n_chans, embed_dim)``. The last segment of a padded
        window may keep steps after the end of the input. The kept steps are in
        time order, so the real ones are a prefix; counting them from ints keeps
        forward free of host syncs.
        """
        first = self.segment_len - self.n_last_steps
        n_real = 0
        for start in range(0, n_prepared, self.segment_len):
            n_real += min(max(n_times - start - first, 0), self.n_last_steps)
        if n_real == 0:  # window shorter than the first kept step
            return h.mean(dim=1)  # (batch, n_chans, embed_dim)
        return h[:, :n_real].mean(dim=1)  # (batch, n_chans, embed_dim)

    def _prepare_token_mask(
        self, token_mask: Optional[Tensor], x: Tensor
    ) -> Optional[Tensor]:
        """Check ``token_mask`` and pad or crop it like the input.

        ``(batch, n_chans, n_times)`` bool -> ``(batch, n_chans, n_segments *
        segment_len)`` bool, False at the padded positions; ``None`` stays
        ``None``.
        """
        if token_mask is None:
            return None
        if token_mask.dtype != torch.bool or list(token_mask.shape) != list(x.shape):
            raise ValueError(
                "token_mask must be boolean with the input's (batch, n_chans, n_times) shape."
            )
        token_mask = token_mask.to(device=x.device)  # (batch, n_chans, n_times)
        # (batch, n_chans, n_segments * segment_len); new positions are False
        return pad_or_crop_to_patches(
            token_mask, self.segment_len, self.on_non_divisible
        )

    def _encode_prepared_tokens(
        self, token_ids: Tensor, token_mask: Optional[Tensor] = None
    ) -> Tensor:
        """Embed token ids and run the decoder layers.

        Steps: cut the ids into segments, embed them, zero the hidden embeddings,
        add the time and channel embeddings, apply dropout, run the layers and
        merge the segments. ``token_ids`` ``(batch, n_chans, n_segments *
        segment_len)`` -> ``(batch, n_segments * n_last_steps, n_chans,
        embed_dim)``.
        """
        batch_size, n_chans = token_ids.shape[0], token_ids.shape[1]
        if n_chans > self.channel_embedding.shape[0]:
            raise ValueError(
                f"MEGGPT has embeddings for {self.channel_embedding.shape[0]} channels "
                f"but got {n_chans}. Build the model for this montage."
            )
        # (batch, n_chans, n_segments * segment_len)
        # -> (batch * n_segments, segment_len, n_chans)
        token_ids = self.to_segments(token_ids)
        # (batch * n_segments, segment_len, n_chans, embed_dim)
        h = self.token_embedding(token_ids)
        if token_mask is not None:
            # (batch * n_segments, segment_len, n_chans, 1)
            hidden = self.to_segments(token_mask).unsqueeze(-1)
            # Fixed zero embedding, NOT vocabulary id0; positions stay intact.
            h = h.masked_fill(hidden, 0.0)  # same shape as h, before context
        # + time_embedding (segment_len, 1, embed_dim)
        h = h + self.time_embedding[:, None]
        # + channel_embedding (n_chans, embed_dim); a recording run alone may
        # have fewer channels than the model, its first parcels.
        h = h + self.channel_embedding[:n_chans]
        h = self.drop(h)  # (batch * n_segments, segment_len, n_chans, embed_dim)
        # layer i: (batch * n_segments, n_patches[i] * patch_lens[i], n_chans,
        # embed_dim) -> (batch * n_segments, steps read by layer i + 1, n_chans,
        # embed_dim); the last layer outputs n_last_steps steps
        for layer in self.layers:
            h = layer(h)
        # Unflatten first: the einops layer cannot take the batch size at call
        # time. (batch * n_segments, n_last_steps, n_chans, embed_dim)
        # -> (batch, n_segments, n_last_steps, n_chans, embed_dim)
        # -> (batch, n_segments * n_last_steps, n_chans, embed_dim)
        h = self.from_segments(h.unflatten(0, [batch_size, -1]))
        return h  # (batch, n_segments * n_last_steps, n_chans, embed_dim)

    @torch.jit.export
    def encode_tokens(
        self,
        token_ids: Tensor,
        token_mask: Optional[Tensor] = None,
        ch_mask: Optional[Tensor] = None,
        lengths: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """Encode cached token ids and return their shifted next-token targets.

        Parameters
        ----------
        token_ids : Tensor
            Clean int64 ids ``(batch, n_chans, n_times)``, e.g. from
            ``model.tokenizer(x)``.
        token_mask : Tensor or None
            Boolean, same shape, True hides that token's embedding.
        ch_mask : Tensor or None
            Boolean ``(batch, n_chans)``, True for real channels (real parcels
            first).
        lengths : Tensor or None
            Int64 ``(batch,)`` valid time counts, default ``n_times``; ids past
            them are zeroed before the embedding.

        Returns
        -------
        dict of str to Tensor
            ``features`` and ``tokens`` as in ``forward``, and ``token_ids``,
            the ids as encoded (invalid ones zeroed, padded or cropped).
            ``output_indices`` are the positions of the ids that ``tokens``
            output, the last ``n_last_steps`` of each segment.
            ``next_token_targets`` are the ids at ``output_indices + 1``,
            ``target_valid`` marks those on real channels and before
            ``lengths``, and ``prediction_mask`` also requires the target token
            to be hidden by ``token_mask``, if given. The three are
            ``(batch, n_chans, n_segments * n_last_steps)``; invalid targets are
            zero.
        """
        if token_ids.dim() != 3 or token_ids.dtype != torch.long:
            raise ValueError("token_ids must be int64 (batch, n_chans, n_times).")
        batch_size, n_chans, n_times = token_ids.shape
        ch_mask = check_ch_mask(ch_mask, token_ids)  # (batch, n_chans) or None
        if lengths is None:
            lengths = torch.full(
                (batch_size,), n_times, device=token_ids.device, dtype=torch.long
            )
        if lengths.dtype != torch.long or list(lengths.shape) != [batch_size]:
            raise ValueError("lengths must be int64 (batch,).")
        lengths = lengths.to(device=token_ids.device)  # (batch,)
        if not torch.jit.is_scripting():
            if bool(((lengths < 1) | (lengths > n_times)).any()):
                raise ValueError("lengths must be in [1, n_times].")
        # (n_times,), then (batch, n_chans, n_times)
        original_indices = torch.arange(n_times, device=token_ids.device)
        valid = (original_indices[None, None] < lengths[:, None, None]).expand_as(
            token_ids
        )
        if ch_mask is not None:
            valid = valid & ch_mask[:, :, None]  # (batch, n_chans, n_times)
        clean_ids = token_ids.masked_fill(~valid, 0)  # (batch, n_chans, n_times)
        # (batch, n_chans, n_segments * segment_len) or None
        prepared_mask = self._prepare_token_mask(token_mask, token_ids)
        prepared = pad_or_crop_to_patches(
            clean_ids, self.segment_len, self.on_non_divisible
        )
        # (batch, n_segments * n_last_steps, n_chans, embed_dim)
        h = self._encode_prepared_tokens(prepared, prepared_mask)
        # (batch, n_chans, embed_dim), as in forward
        features = self._pool_steps(h, prepared.shape[-1], n_times)
        tokens = self.to_tokens(
            h
        )  # (batch, n_chans, n_segments * n_last_steps, embed_dim)
        if ch_mask is not None:
            features = features.masked_fill(~ch_mask[:, :, None], 0.0)  # same shape
            tokens = tokens.masked_fill(~ch_mask[:, :, None, None], 0.0)  # same shape
        input_indices = torch.arange(prepared.shape[-1], device=token_ids.device)
        # (n_segments, segment_len) -> (n_segments, n_last_steps) -> (output_times,)
        output_indices = input_indices.unflatten(0, [-1, self.segment_len])[
            :, -self.n_last_steps :
        ].flatten()
        next_indices = output_indices + 1  # (output_times,)
        # Clamp only for safe gather; validity below excludes unavailable next tokens.
        safe_next = next_indices.clamp_max(n_times - 1)  # (output_times,)
        targets = clean_ids.index_select(
            -1, safe_next
        )  # (batch, n_chans, output_times)
        target_valid = (
            valid.index_select(-1, safe_next) & (next_indices < n_times)[None, None]
        )
        targets = targets.masked_fill(
            ~target_valid, 0
        )  # (batch, n_chans, output_times)
        prediction_mask = target_valid
        if token_mask is not None:
            # Shape/type checked by _prepare_token_mask; select TARGET positions.
            selected = token_mask.to(device=token_ids.device).index_select(
                -1, safe_next
            )
            prediction_mask = target_valid & selected  # (batch, n_chans, output_times)
        return {
            "features": features,
            "tokens": tokens,
            "token_ids": prepared,
            "output_indices": output_indices,
            "next_token_targets": targets,
            "target_valid": target_valid,
            "prediction_mask": prediction_mask,
        }

    def reset_head(self, n_outputs: int) -> None:
        self._set_n_outputs(n_outputs)
        self.final_layer[1] = nn.Linear(self.final_layer[1].in_features, n_outputs)


class _EphysTokenizer(nn.Module):
    """EphysTokenizer encoder: ``(batch, chans, time)`` floats to token ids.

    A GRU runs over each channel separately; the argmax of the normalised
    logits, divided by ``temperature`` beforehand, is mapped through ``vocab``
    (codebook entry to token id, frequency ranked; unused entries map to 0 in
    the released vocabulary).
    """

    def __init__(
        self, n_codes: int, n_tokens: int, rnn_hidden_size: int, temperature: float
    ):
        super().__init__()
        self.rnn = nn.GRU(1, rnn_hidden_size, batch_first=True)
        self.proj = nn.Linear(rnn_hidden_size, n_codes)
        self.norm = nn.LayerNorm(n_codes)
        self.to_series = Rearrange("b c t -> (b c) t 1")
        # Released scaling of the logits (EphysTokenizer): float near-ties break
        # as in the reference.
        self.temperature = temperature
        # Placeholder until the released vocabulary is loaded.
        self.register_buffer("vocab", torch.arange(n_codes) % n_tokens)
        self.requires_grad_(False)

    def forward(self, x: Tensor) -> Tensor:
        batch_size, n_chans = x.shape[0], x.shape[1]
        # (batch, n_chans, n_times) -> (batch * n_chans, n_times, 1)
        # -> (batch * n_chans, n_times, rnn_hidden_size)
        h, _ = self.rnn(self.to_series(x))
        # (batch * n_chans, n_times, n_codes)
        logits = self.norm(self.proj(h)) / self.temperature
        codes = logits.argmax(dim=-1)  # (batch * n_chans, n_times)
        # token ids (batch * n_chans, n_times) -> (batch, n_chans, n_times)
        return self.vocab[codes].unflatten(0, [batch_size, n_chans])


class _DecoderLayer(nn.Module):
    """Pre-norm patched time attention and feed-forward block.

    Steps: attention on the normed input, dropout, residual, then a normed
    feed-forward block with its residual. The attention outputs the last
    ``n_out`` steps; the residual keeps the same steps of the input.
    ``(batch, n_patches * patch_len, n_chans, embed_dim)`` ->
    ``(batch, n_out, n_chans, embed_dim)``, with ``batch`` counting segments.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        feedforward_dim: int,
        n_patches: int,
        patch_len: int,
        n_out: int,
        patch_band: Optional[int],
        activation: type[nn.Module],
        drop_prob: float,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = _PatchedTimeAttention(
            embed_dim, num_heads, n_patches, patch_len, n_out, patch_band
        )
        self.drop = nn.Dropout(drop_prob)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = FeedForwardBlock(
            emb_size=embed_dim,
            expansion=1,  # unused: hidden_features sets the width
            drop_p=0.0,  # no dropout after the activation, only after the output
            activation=activation,
            hidden_features=feedforward_dim,
            output_drop_p=drop_prob,
        )

    def forward(self, x: Tensor) -> Tensor:
        # x: (batch, n_patches * patch_len, n_chans, embed_dim)
        h = self.attn(self.norm1(x))  # (batch, n_out, n_chans, embed_dim)
        h = self.drop(h)  # (batch, n_out, n_chans, embed_dim)
        x = h + x[:, -h.shape[1] :]  # (batch, n_out, n_chans, embed_dim)
        return x + self.mlp(self.norm2(x))  # (batch, n_out, n_chans, embed_dim)


class _PatchedTimeAttention(nn.Module):
    """Causal attention over time, per channel, on ``(batch, time, chans, dim)``.

    Keys and values are pooled into patches by one linear layer over the
    ``num_heads * patch_len`` (head, step) pairs of a patch, the layout of the
    released weights, which no shared attention module has. Queries are the
    last ``n_out`` steps. A query that sees no patch outputs zeros, as in the
    reference.
    ``(batch, n_patches * patch_len, n_chans, embed_dim)`` ->
    ``(batch, n_out, n_chans, embed_dim)``, with ``batch`` counting segments.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        n_patches: int,
        patch_len: int,
        n_out: int,
        patch_band: Optional[int],
    ):
        super().__init__()
        self.n_out = n_out
        self.scale = math.sqrt(embed_dim // num_heads)
        self.to_patches = Rearrange(
            "b (p l) c (h k) -> b p c k (h l)", h=num_heads, l=patch_len
        )
        self.from_patches = Rearrange("b p c k h -> b p c (h k)")
        pool = num_heads * patch_len
        self.patch_k = nn.Linear(pool, num_heads) if patch_len > 1 else nn.Identity()
        self.patch_v = nn.Linear(pool, num_heads) if patch_len > 1 else nn.Identity()
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.split_heads = Rearrange("b t c (h k) -> b h c t k", h=num_heads)
        self.split_heads_t = Rearrange("b t c (h k) -> b h c k t", h=num_heads)
        self.merge_heads = Rearrange("b h c t k -> b t c (h k)")

        query = torch.arange(n_out)[:, None]
        patch = torch.arange(n_patches)
        # A patch is visible once its last step is at or before the query.
        visible = (patch + 1 - n_patches) * patch_len + n_out - 1 <= query
        if patch_band is not None:
            visible &= (patch - query // patch_len).abs() <= patch_band
        self.register_buffer("mask", ~visible, persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        # x: (batch, n_patches * patch_len, n_chans, embed_dim), batch counting
        # segments; head_dim = embed_dim // num_heads.
        # (batch, n_patches, n_chans, head_dim, num_heads * patch_len)
        patches = self.to_patches(x)
        # patch_k/v: -> (batch, n_patches, n_chans, head_dim, num_heads)
        # from_patches, k/v_proj: -> (batch, n_patches, n_chans, embed_dim)
        k = self.k_proj(self.from_patches(self.patch_k(patches)))
        v = self.v_proj(self.from_patches(self.patch_v(patches)))
        q = self.q_proj(x[:, -self.n_out :])  # (batch, n_out, n_chans, embed_dim)
        q = self.split_heads(q)  # (batch, num_heads, n_chans, n_out, head_dim)
        k = self.split_heads_t(k)  # (batch, num_heads, n_chans, head_dim, n_patches)
        v = self.split_heads(v)  # (batch, num_heads, n_chans, n_patches, head_dim)
        scores = q @ k / self.scale  # (batch, num_heads, n_chans, n_out, n_patches)
        # mask (n_out, n_patches): True where a patch is not visible to a query
        # (batch, num_heads, n_chans, n_out, n_patches), unchanged
        scores = scores.masked_fill(self.mask, float("-inf"))
        weights = scores.softmax(dim=-1)  # same shape as scores
        weights = weights.nan_to_num(nan=0.0)  # same shape as scores
        # (batch, num_heads, n_chans, n_out, head_dim)
        # -> (batch, n_out, n_chans, embed_dim)
        out = self.merge_heads(weights @ v)
        return self.out_proj(out)  # (batch, n_out, n_chans, embed_dim)
