# Authors: Bruno Aristimunha <b.aristimunha@gmail.com> (braindecode adaptation)
#
# License: BSD (3-clause)
#
# Adapted from the masked autoencoder module of the GeoNIRS authors,
# https://github.com/GeoNIRS/GeoNIRS (``model/`` at commit 81f8995), their
# implementation of Jung, Lee and An (2025), doi:10.1109/BCI65088.2025.10931275.
from __future__ import annotations

import math
from functools import partial
from typing import Literal

import torch
from einops.layers.torch import Rearrange, Reduce
from torch import nn

from braindecode.functional import (
    check_ch_mask,
    init_weights_,
    pad_or_crop_to_patches,
    patch_count,
    sinusoidal_positional_encoding,
)
from braindecode.models.base import EEGModuleMixin
from braindecode.models.util import fnirs_channel_order, fnirs_split_hbo_hbr


class JungMAE(EEGModuleMixin, nn.Module):
    r"""JungMAE from Jung et al. (2025), as implemented by the GeoNIRS authors.

    :bdg-info:`Attention/Transformer` :bdg-danger:`Foundation Model`

    .. versionadded:: 2.1

    Jung, Lee and An :footcite:`jung2025mae` pretrain a masked autoencoder on
    unlabelled fNIRS recordings, so that a decoder needs fewer labelled ones.
    Their paper (IEEE BCI 2025) came without code: this class is the encoder of
    the GeoNIRS authors' implementation, whose Jung-MAE baseline it is
    (:footcite:`li2026geonirs`, ``GeoNIRS/GeoNIRS`` at commit ``81f8995``), with
    a linear head; the pretraining decoder is not part of it.

    .. rubric:: Architecture Overview

    Every HbO/HbR pair (one source-detector channel) is a sequence of its own:
    its window is cut into 2 s patches, each patch becomes one token holding
    both chromophores, and a Transformer encodes the tokens of each pair. The
    same weights serve every pair, which is how one checkpoint handles montages
    of any size; the classifier then flattens the tokens of all the pairs. With
    :math:`B` the batch size, :math:`C` = ``n_chans // 2`` pairs, :math:`Q` =
    ``patch_size``, :math:`P` patches and :math:`D` = ``embed_dim``::

        x (B, 2C, n_times): HbO and HbR rows, NaN = missing sample
        1. HbO/HbR split, invalid samples to zero  -> (B, C, P*Q) each
        2. to_patches: [HbO_t, HbR_t, ...]         -> (B*C, P, 2Q)
        3. patch_embed + sinusoidal position       -> (B*C, P, D)
        4. encoder, invalid patches not attended   -> (B, C, P, D)   encode()
        5. flatten_layer, head_dropout             -> (B, C*P*D)
        6. final_layer                             -> (B, n_outputs)

    .. rubric:: Macro Components

    - ``JungMAE.patch_embed`` **Patch tokenization**

      *Operations.* A patch interleaves the :math:`Q` HbO samples :math:`o` and
      HbR samples :math:`r` of pair :math:`c`. One :class:`~torch.nn.Linear`
      maps it to a :math:`D`-wide token, and the fixed sinusoidal encoding of
      the patch index :math:`p` is added:

      .. math::

          \mathbf{z}_{c,p} = \mathbf{W}\,[o_{c,pQ},\, r_{c,pQ},\, \dots,\,
          o_{c,pQ+Q-1},\, r_{c,pQ+Q-1}] + \mathbf{b} + \mathrm{PE}(p).

      *Role.* One token per patch, with both chromophores of each time step.

    - ``JungMAE.encoder`` **Transformer encoder**

      *Operations.* Four pre-norm :class:`~torch.nn.TransformerEncoderLayer`
      blocks (8 heads, feed-forward width 768, GELU) and a final
      :class:`~torch.nn.LayerNorm`, over the :math:`P` tokens of one pair.
      Invalid patches are masked as attention keys; the reference packs the
      valid tokens instead, which gives the same features up to float32
      rounding.

      *Role.* Context across the time of a pair, with the same weights for
      every pair.

    - ``JungMAE.final_layer`` **Classification head**

      Flatten, :class:`~torch.nn.Dropout` and one :class:`~torch.nn.Linear`,
      the ``flatten_linear`` head of the GeoNIRS benchmark. The benchmark
      standardizes the flattened features with training-set statistics first;
      this head does not, so standardize the eval-mode ``return_features`` to
      reproduce it. The head is not pretrained, and its input size fixes
      ``n_chans`` and ``n_times``.

    .. rubric:: Temporal, Spatial, and Spectral Encoding

    - **Temporal.** Non-overlapping 2 s patches (10 samples at 5 Hz) with a
      fixed sinusoidal position. The weights were pretrained on 30 s windows.
    - **Spatial.** None. Pairs never attend to each other and the model sees
      no optode position. A pair is identified only by where its features sit
      in the head input, so the pairs must keep the order used in training.
    - **Spectral.** None: there is no frequency decomposition.
    - **Chromophores.** HbO and HbR share each token, sample by sample; HbT is
      not used. The rows must be the HbO and the HbR rows of the same pairs
      (see ``channel_order``).

    .. rubric:: Additional Mechanisms

    - **Missing data.** NaN marks a missing sample, and a pair whose rows are
      all NaN is a missing channel: the head spans every pair, so there is no
      ``ch_mask``. A sample is valid when both chromophores are finite, and a
      patch when at least ``minimum_patch_valid_fraction`` of its samples are
      (:math:`\lceil 0.8\,Q \rceil` by default). Invalid samples are set to
      zero, invalid patches are not attended and give zero tokens, and a pair
      without a valid patch gives zero features.

    .. rubric:: Pretrained Weights

    The encoder of ``jung_mae.pt``, trained by the GeoNIRS authors and
    converted to these parameter names, is on the Hugging Face Hub at
    ``braindecode/jungmae-pretrained``; the head is not pretrained. The
    weights expect the inputs of the GeoNIRS paper (Appendix B.1): HbO and HbR
    in µmol/L resampled to 5 Hz, each chromophore z-scored per recording over
    all its valid channels and time points. With them and the GeoNIRS
    benchmark's linear-probe loop, this class reproduces the paper's
    Jung-MAE-LP results (:gh:`1589`).

    .. code-block:: python

        from braindecode.models import JungMAE

        model = JungMAE.from_pretrained(
            "braindecode/jungmae-pretrained", n_chans=40, n_outputs=2, n_times=60
        )

    Parameters
    ----------
    patch_size : int, default=10
        Samples per patch (``patch_samples`` in the reference), 2 s at 5 Hz.
    embed_dim : int, default=192
        Width of the tokens (``encoder_dim`` in the reference).
    depth : int, default=4
        Number of Transformer layers (``encoder_layers`` in the reference).
    num_heads : int, default=8
        Attention heads per layer (``encoder_heads`` in the reference).
    mlp_ratio : float, default=4.0
        Feed-forward width over ``embed_dim``.
    drop_prob : float, default=0.1
        Dropout of the encoder layers (``dropout`` in the reference).
    head_drop_prob : float, default=0.1
        Dropout before ``final_layer`` (``head_dropout`` in the benchmark).
    minimum_patch_valid_fraction : float, default=0.8
        Fraction of valid samples a patch needs, in (0, 1].
    activation : type[nn.Module], default=nn.GELU
        Activation of the feed-forward layers.
    channel_order : {"stacked", "interleaved"} or None, default=None
        ``[HbO_1..HbO_C, HbR_1..HbR_C]`` or MNE's ``[HbO_1, HbR_1, ...]``;
        ``None`` reads it from ``chs_info`` (``"stacked"`` without it).
        ``n_chans`` must be even.
    on_non_divisible : {"error", "pad", "crop"}, default="error"
        ``n_times`` not a multiple of ``patch_size``: raise, as the reference
        does; pad with missing samples, to which the validity rule applies; or
        crop.

    References
    ----------
    .. footbibliography::
    """

    _time_axis = "fixed"  # fixed: final_layer.weight
    _channel_axis = "fixed"  # fixed: final_layer.weight

    def __init__(
        self,
        # Braindecode parameters
        n_outputs: int | None = None,
        n_chans: int | None = None,
        chs_info: list[dict] | None = None,
        n_times: int | None = None,
        input_window_seconds: float | None = None,
        sfreq: float | None = None,
        *,
        # Model-specific parameters
        patch_size: int = 10,
        embed_dim: int = 192,
        depth: int = 4,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        drop_prob: float = 0.1,
        head_drop_prob: float = 0.1,
        minimum_patch_valid_fraction: float = 0.8,
        activation: type[nn.Module] = nn.GELU,
        channel_order: Literal["stacked", "interleaved"] | None = None,
        on_non_divisible: Literal["error", "pad", "crop"] = "error",
    ) -> None:
        super().__init__(
            n_outputs=n_outputs,
            n_chans=n_chans,
            chs_info=chs_info,
            n_times=n_times,
            input_window_seconds=input_window_seconds,
            sfreq=sfreq,
        )
        del n_outputs, n_chans, chs_info, n_times, input_window_seconds, sfreq

        # Checkers
        # Layout of the HbO and HbR rows, read from chs_info when it is given
        self.channel_order = fnirs_channel_order(self._chs_info, channel_order)
        # A token holds the two chromophores of one source-detector pair
        if self.n_chans % 2 != 0:
            raise ValueError(
                "JungMAE needs HbO/HbR pairs, because a token holds both "
                f"chromophores of a pair: n_chans must be even, got {self.n_chans}."
            )
        # At zero a patch without data would be encoded; above one, none would
        if not 0 < minimum_patch_valid_fraction <= 1:
            raise ValueError(
                "minimum_patch_valid_fraction must be in (0, 1], "
                f"got {minimum_patch_valid_fraction}: it is the share of valid "
                "samples a patch needs to be encoded."
            )

        # Parameters
        # Q in the docstring: samples of each chromophore in one patch
        self.patch_size = patch_size
        # What to do with a window that is not a whole number of patches
        self.on_non_divisible = on_non_divisible
        # C pairs and P patches: with the token width, the input size of the
        # head. patch_count refuses a bad patch_size or on_non_divisible
        n_pairs = self.n_chans // 2
        n_patches = patch_count(self.n_times, patch_size, on_non_divisible)
        # Validity rule of the reference: a patch is encoded when at least
        # ceil(minimum_patch_valid_fraction * Q) of its samples are valid
        self.min_valid_samples = math.ceil(patch_size * minimum_patch_valid_fraction)

        # Layers
        # Following the nomenclature of the reference implementation

        # Patch tokenization. The window of one pair is cut into P patches, and
        # the Q HbO and Q HbR samples of a patch are interleaved,
        # [HbO_t, HbR_t, HbO_t+1, HbR_t+1, ...]: both chromophores share a token
        self.to_patches = Rearrange(
            "batch pairs (patches samples) chromophores"
            " -> (batch pairs) patches (samples chromophores)",
            samples=patch_size,
        )
        # Patch embedding: the linear projection of a patch to a token of width D
        self.patch_embed = nn.Linear(
            in_features=2 * patch_size,
            out_features=embed_dim,
        )

        # Patch validity. Number of valid samples in each patch, which encode
        # compares with min_valid_samples
        self.count_valid_samples = Reduce(
            "batch pairs (patches samples) -> batch pairs patches",
            reduction="sum",
            samples=patch_size,
        )
        # The encoder reads each pair as a sequence of its own, so the patch
        # validity is laid out like the tokens: one row per pair
        self.merge_pairs = Rearrange("batch pairs patches -> (batch pairs) patches")

        # Transformer encoder: pre-norm layers and a final LayerNorm, over the
        # P tokens of one pair. All the pairs go through the same weights, so
        # one checkpoint serves any number of channels
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=int(embed_dim * mlp_ratio),
                dropout=drop_prob,
                activation=activation(),
                batch_first=True,
                norm_first=True,
            ),
            num_layers=depth,
            norm=nn.LayerNorm(embed_dim),
            enable_nested_tensor=False,
        )

        # Classification head, the flatten_linear head of the GeoNIRS benchmark.
        # It is not pretrained: the checkpoint holds the encoder only

        # The tokens of all the pairs and patches, in one vector per window
        self.flatten_layer = nn.Flatten()
        # Dropout on that vector, before the classifier
        self.head_dropout = nn.Dropout(head_drop_prob)
        # Linear classifier over C * P * D features
        self.final_layer = nn.Linear(
            in_features=n_pairs * n_patches * embed_dim,
            out_features=self.n_outputs,
        )

        # Initialization of the reference
        self._init_parameters()
        self._n_times_step = self.patch_size
        self._snaps_in_prepare_inputs = False
        self._min_n_times = 1 if self.on_non_divisible == "pad" else self.patch_size

    def _accepts_any_n_chans(self, args: tuple, kwargs: dict) -> bool:
        return super()._accepts_any_n_chans(args, kwargs) or self._accepts_any_n_times(
            args, kwargs
        )

    def _accepts_any_n_times(self, args: tuple, kwargs: dict) -> bool:
        return bool(self._call_arg(args, kwargs, "return_features", False))

    def _init_parameters(self) -> None:
        """Xavier-uniform weights and zero biases of the Linear layers and the
        attention projections, as the reference. nn.TransformerEncoder copies
        one layer, so this draw also gives every layer its own weights."""
        self.apply(_init_weights)

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
        ch_mask: torch.Tensor | None = None,
    ) -> torch.Tensor | dict[str, torch.Tensor | None]:
        """Forward pass of the JungMAE model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor with shape (batch_size, n_chans, n_times). NaN marks
            a missing sample.
        return_features : bool, default=False
            If True, return the input of ``final_layer`` instead of the
            logits.
        ch_mask : torch.Tensor | None, default=None
            Pass by keyword. ``(batch_size, n_chans)`` mask with ``True`` for
            real channels, see :func:`~braindecode.functional.check_ch_mask`.
            Padded channels are read as missing samples (NaN).

        Returns
        -------
        torch.Tensor or dict
            Output tensor with shape (batch_size, n_outputs), or
            ``{"features": (batch_size, n_pairs * n_patches * embed_dim),
            "cls_token": None}`` with ``return_features=True``.
        """
        # Shapes use the symbols of the class docstring: C pairs, P patches of
        # Q samples, tokens of width D.

        # Padded channels become missing samples: (batch, 2C, n_times)
        ch_mask = check_ch_mask(ch_mask, x)
        if ch_mask is not None:
            x = x.masked_fill(~ch_mask.unsqueeze(-1), float("nan"))
        # A token per patch of each pair, zero where the patch is invalid:
        # (batch, 2C, n_times) -> (batch, C, P, D)
        x = self.encode(x)["foundation"]
        # Every token of the window in one vector, pair after pair, the input
        # of the benchmark's flatten_linear head: (batch, C * P * D)
        x = self.flatten_layer(x)
        # Dropout before the classifier: (batch, C * P * D)
        x = self.head_dropout(x)
        if return_features:
            return {"features": x, "cls_token": None}
        # Linear classifier: (batch, n_outputs)
        return self.final_layer(x)

    def encode(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Encode the patches of every HbO/HbR pair.

        The window is cut into whole patches. A sample is valid when both
        chromophores are finite, and a patch when enough of its samples are.
        Each patch of a pair, its invalid samples set to zero, becomes a token
        with its sinusoidal position; the Transformer encodes the tokens of
        each pair with the invalid patches hidden as keys, and the invalid
        tokens are zeroed.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor with shape (batch_size, n_chans, n_times). NaN marks
            a missing sample.

        Returns
        -------
        dict of torch.Tensor
            ``"foundation"``, the tokens with shape (batch_size, n_pairs,
            n_patches, embed_dim), zero at the invalid patches, and
            ``"valid"``, the boolean patch validity with shape (batch_size,
            n_pairs, n_patches). These are the names of the reference.
        """
        # Shapes use the symbols of the class docstring: C pairs, P patches of
        # Q samples, tokens of width D.
        n_chans = x.shape[1]
        if n_chans < 2 or n_chans % 2:
            raise ValueError(
                f"JungMAE needs complete HbO/HbR pairs, got {n_chans} channels."
            )

        n_times = x.shape[-1]
        # Whole patches; a partial last one is refused, padded or cropped:
        # (batch, 2C, P * Q)
        x = pad_or_crop_to_patches(x, self.patch_size, self.on_non_divisible)
        # The two chromophores of each pair: HbO and HbR, (batch, C, P * Q) each
        hbo, hbr = fnirs_split_hbo_hbr(x, self.channel_order)
        batch_size, n_pairs, _ = hbo.shape

        # A sample is valid when both chromophores are finite and it is not
        # padding: (batch, C, P * Q)
        valid_samples = torch.isfinite(hbo) & torch.isfinite(hbr)
        valid_samples[:, :, n_times:] = False
        # A patch is valid with at least min_valid_samples valid samples:
        # (batch, C, P)
        n_valid_samples = self.count_valid_samples(valid_samples)
        valid_patches = n_valid_samples >= self.min_valid_samples
        # The same validity with one row per pair, as the encoder reads the
        # tokens: (batch * C, P)
        valid_tokens = self.merge_pairs(valid_patches)
        invalid_tokens = ~valid_tokens

        # Invalid samples are zero before the projection, as in the
        # reference: (batch, C, P * Q) each
        hbo = torch.where(valid_samples, hbo, 0.0)
        hbr = torch.where(valid_samples, hbr, 0.0)
        # HbO and HbR interleaved sample by sample in each patch, each pair a
        # sequence of its own: (batch, C, P * Q, 2) -> (batch * C, P, 2 * Q)
        x = self.to_patches(torch.stack((hbo, hbr), dim=-1))
        # A token per patch, z = W patch + b: (batch * C, P, D)
        x = self.patch_embed(x)
        # Sinusoidal position PE(p) of the patch, computed in float32 at least
        # and cast, as the reference does in half precision: (batch * C, P, D)
        _, n_patches, embed_dim = x.shape
        positions_dtype = torch.promote_types(x.dtype, torch.float32)
        positions = sinusoidal_positional_encoding(
            n_patches, embed_dim, device=x.device, dtype=positions_dtype
        )
        x = x + positions.to(x.dtype)

        # Invalid patches are hidden as keys, except in a pair without any
        # valid patch, whose softmax over no key would be NaN: (batch * C, P)
        has_valid_token = valid_tokens.any(dim=-1, keepdim=True)
        key_padding_mask = invalid_tokens & has_valid_token
        # Transformer over the tokens of each pair, with the same weights for
        # every pair: (batch * C, P, D)
        x = self.encoder(x, src_key_padding_mask=key_padding_mask)
        # Invalid tokens are zero, so a pair without data gives zero features;
        # the trailing axis of the mask broadcasts over the D features:
        # (batch * C, P, D)
        x = x.masked_fill(invalid_tokens[..., None], 0.0)
        # The pairs back on their own axis: (batch, C, P, D)
        x = x.unflatten(0, [batch_size, n_pairs])
        return {"foundation": x, "valid": valid_patches}

    def reset_head(self, n_outputs: int) -> None:
        self._set_n_outputs(n_outputs)
        # A new classifier over the same C * P * D features, drawn as the
        # constructor draws it
        self.final_layer = nn.Linear(self.final_layer.in_features, n_outputs)
        _init_weights(self.final_layer)


# Xavier-uniform weights and zero biases of the Linear layers and of the
# attention input projections: the initialization of the reference
# (GeoNIRS/GeoNIRS@81f8995, model/)
_init_weights = partial(
    init_weights_,
    weights={(nn.Linear, nn.MultiheadAttention): nn.init.xavier_uniform_},
    norms=(),
)
