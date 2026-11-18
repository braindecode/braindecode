# Authors: Bruno Aristimunha <b.aristimunha@gmail.com> (braindecode adaptation)
#
# Adapted from the masked autoencoder module of the GeoNIRS authors,
# https://github.com/GeoNIRS/GeoNIRS (``model/`` at commit 81f8995), their
# implementation of Jung, Lee and An (2025), doi:10.1109/BCI65088.2025.10931275.
from __future__ import annotations

import math
from typing import Literal, Optional

import torch
from einops.layers.torch import Rearrange, Reduce
from torch import nn

from braindecode.functional import (
    pad_or_crop_to_patches,
    patch_count,
    sinusoidal_positional_encoding,
)
from braindecode.functional.patches import _check_on_non_divisible
from braindecode.models.base import EEGModuleMixin
from braindecode.models.util import fnirs_channel_order, fnirs_split_hbo_hbr


class JungMAE(EEGModuleMixin, nn.Module):
    r"""JungMAE from Jung et al. (2025), as implemented by the GeoNIRS authors.

    :bdg-info:`Attention/Transformer` :bdg-danger:`Foundation Model`

    .. versionadded:: 2.1

    Labelled fNIRS data are slow and costly to collect, and supervised
    decoders need a lot of them. Jung, Lee and An :footcite:`jung2025mae`
    "propose a novel foundation model for fNIRS data based on a
    self-supervised masked autoencoder framework", pretrained on unlabelled
    recordings. They report that it matches supervised training "while
    requiring only one-third of the labeled training data", and that "a larger
    masking size aligns with the low-frequency nature of fNIRS signals". Their
    paper (IEEE BCI 2025) is not open access and came without code, so what is
    said about it here comes from its abstract.

    This class ports the implementation and the ``jung_mae.pt`` weights of the
    GeoNIRS authors (``GeoNIRS/GeoNIRS`` at commit ``81f8995``), who use the
    model as a baseline and call it Jung-MAE (Tables 1, 14 and 15 of
    :footcite:`li2026geonirs`). Their paper describes it as "a paper-guided
    implementation of the multi-dataset fNIRS pretraining framework of Jung et
    al. (2025)" which, to "accommodate heterogeneous channel counts", "uses
    channel-independent encoding with shared parameters" (Section 3.1). The
    details below are read from the GeoNIRS code.

    .. rubric:: Architectural Overview

    Every HbO/HbR pair (one source-detector channel) is a sequence of its own.
    Its window is cut into 2 s patches, each patch becomes one token that
    holds both chromophores, and a small Transformer encodes the tokens of
    that pair. The same weights serve every pair, which is how one checkpoint
    handles montages of any size. The classifier then flattens the tokens of
    all the pairs. With :math:`B` the batch size, :math:`C` = ``n_chans // 2``
    pairs, :math:`Q` = ``patch_size``, :math:`P` patches and :math:`D` =
    ``embed_dim``, the model has the following sequence of steps::

        x (B, 2C, n_times): HbO and HbR rows, NaN = missing sample
        1. HbO/HbR split, invalid samples to zero  -> (B, C, P*Q) each
        2. to_patches: [HbO_t, HbR_t, ...]         -> (B*C, P, 2Q)
        3. patch_embed + sinusoidal position       -> (B*C, P, D)
        4. encoder, invalid patches not attended   -> (B, C, P, D)   encode()
        5. flatten_layer, head_dropout             -> (B, C*P*D)
        6. final_layer                             -> (B, n_outputs)

    .. rubric:: Macro Components

    - ``JungMAE.patch_embed`` **Patch tokenization**

      A patch interleaves the :math:`Q` HbO samples :math:`o` and HbR samples
      :math:`r` of pair :math:`c`. One :class:`~torch.nn.Linear` maps it to a
      :math:`D`-wide token, and the fixed sinusoidal encoding of the patch
      index :math:`p` is added:

      .. math::

          \mathbf{z}_{c,p} = \mathbf{W}\,[o_{c,pQ},\, r_{c,pQ},\, \dots,\,
          o_{c,pQ+Q-1},\, r_{c,pQ+Q-1}] + \mathbf{b} + \mathrm{PE}(p).

    - **Patch validity** (``minimum_patch_valid_fraction``)

      A sample is valid when both chromophores are finite and it is not in
      ``mask``. Invalid samples are set to zero, and a patch counts only when
      at least :math:`\lceil 0.8\,Q \rceil` of its samples are valid. The
      GeoNIRS paper applies the same 80 % rule to its own model (Appendix
      B.1.2).

    - ``JungMAE.encoder`` **Transformer encoder**

      Four pre-norm :class:`~torch.nn.TransformerEncoderLayer` blocks (8
      heads, feed-forward width 768, GELU) and a final
      :class:`~torch.nn.LayerNorm`, over the :math:`P` tokens of one pair.
      Invalid patches are not attended and their output is zero, so a pair
      without any valid patch gives zero features.

    - ``JungMAE.final_layer`` **Classification head**

      Flatten, :class:`~torch.nn.Dropout` and one :class:`~torch.nn.Linear`,
      the ``flatten_linear`` head of the GeoNIRS benchmark. It is not
      pretrained, and its input size fixes ``n_chans`` and ``n_times``.

    .. rubric:: Temporal, Spatial, and Chromophore Encoding

    - **Temporal.** Non-overlapping 2 s patches (10 samples at 5 Hz) with a
      fixed sinusoidal position. The GeoNIRS paper pretrains on 30 s windows
      (15 patches) and keeps each downstream task's 10 s or 12 s window
      (Appendix B.1.2).
    - **Spatial.** None. Pairs never attend to each other and the model sees
      no optode position. A pair is identified only by where its features sit
      in the head input, so the pairs must keep the order used in training.
    - **Chromophores.** HbO and HbR share each token, sample by sample; HbT
      is not used. The rows must be the HbO and the HbR rows of the same
      pairs (see ``channel_order``).

    .. rubric:: Pretraining

    In the reference, each pair keeps half of its valid patches visible,
    drawn at random, and hides the others. The encoder reads the visible
    tokens only. A lighter decoder (two layers of width 96, with a learned
    mask token) predicts every patch, and the loss is the mean squared error
    over the valid samples of the hidden patches. The decoder is stored in
    ``jung_mae.pt`` and is not part of this class.

    .. rubric:: Known Limitations

    - The model "does not model physical channel geometry" (GeoNIRS paper,
      Section 3.1): different channel counts "are supported without modeling
      physical channel coordinates or cross-layout correspondence" (Table 13).
    - It is "a paper-guided implementation rather than an exact reproduction
      of the original method" (GeoNIRS paper, Appendix G).

    .. rubric:: Pretrained Weights

    ``checkpoint/jung_mae.pt`` of ``GeoNIRS/GeoNIRS`` at ``81f8995`` was
    trained by the GeoNIRS authors and is not on Hugging Face. It
    holds the encoder (1,783,872 parameters), the decoder (244,436) and the
    model configuration, which equals the defaults of this class.

    .. important::
       :meth:`load_state_dict` takes the file as released. It drops the
       decoder and keeps the new ``final_layer``:

       .. code-block:: python

           import torch
           from braindecode.models import JungMAE

           payload = torch.load("jung_mae.pt", map_location="cpu", weights_only=True)
           model = JungMAE(n_chans=40, n_outputs=2, n_times=60, sfreq=5.0)
           model.load_state_dict(payload)

    The weights expect the inputs of the GeoNIRS paper (Appendix B.1): HbO and
    HbR in µmol/L resampled to 5 Hz, each chromophore z-scored per recording
    over all its valid channels and time points.

    .. rubric:: Differences from the Reference Implementation

    - **Input.** One tensor ``(batch, n_chans, n_times)``, with the HbO rows
      then the HbR rows, or MNE's interleaved order read from ``chs_info``.
      The reference takes ``hbo`` and ``hbr`` tensors with validity masks.
      Here NaN marks a missing sample (an all-NaN row is a missing channel),
      and the opt-in ``mask`` marks more.
    - **Parity.** The reference packs the valid tokens of each pair; this
      class masks the invalid ones as attention keys. With ``jung_mae.pt``
      the features equal those of the authors' module (max-abs 0.0) in eval
      mode under :func:`torch.no_grad`. With gradients on they differ by
      about 2e-6 at most (float32 rounding), and in training mode the
      dropout draws differ.
    - **Head.** The benchmark head standardizes the flattened features with
      training-set statistics before its Dropout; this head does not, so
      standardize the eval-mode ``return_features`` yourself to reproduce
      the benchmark. ``final_layer`` starts Xavier-uniform with a zero bias,
      as the other Linear layers of the reference.
    - **Additions.** ``on_non_divisible="pad"`` or ``"crop"`` (the reference
      refuses a partial patch) and ``mask``. ``ch_mask`` is refused, because
      ``final_layer`` spans every pair.

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
        ``[HbO_1..HbO_C, HbR_1..HbR_C]`` or MNE's ``[HbO_1, HbR_1, ...]``.
        ``None`` reads it from ``chs_info`` (``"stacked"`` without it).
        ``n_chans`` must be even.
    on_non_divisible : {"error", "pad", "crop"}, default="error"
        What to do when ``n_times`` is not a multiple of ``patch_size``:
        raise, as the reference does; pad with missing samples, to which the
        validity rule applies; or crop.

    Notes
    -----
    The GeoNIRS benchmark (Section 3.1 and Appendices B.5-B.6 of
    :footcite:`li2026geonirs`) evaluates eight held-out binary tasks with
    subject-level five-fold cross-validation and the seeds 810 and 811. Every
    method gets the same windows and folds and the same training: 30 epochs,
    AdamW, class-weighted cross-entropy and model selection by validation
    balanced accuracy. Jung-MAE-LP trains the head only, with a learning rate
    of 1e-4.

    This class with ``jung_mae.pt`` and the benchmark's linear-probe loop
    reproduces Jung-MAE-LP (mean over the five test folds and the two seeds):

    =====================  ========  =================  ==============
    Task                   Metric    GeoNIRS paper (%)  This class (%)
    =====================  ========  =================  ==============
    DS42 (action mode)     macro-F1  68.36              68.35
    DS42 (action mode)     AUROC     76.78              76.80
    DS19 (pain intensity)  macro-F1  54.23              54.95
    DS19 (pain intensity)  AUROC     58.39              58.63
    =====================  ========  =================  ==============

    References
    ----------
    .. footbibliography::
    """

    def __init__(
        self,
        # Braindecode parameters
        n_outputs: int | None = None,
        n_chans: int | None = None,
        chs_info=None,
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
        _check_on_non_divisible(on_non_divisible)
        # A patch holds patch_size samples of each chromophore
        if patch_size < 1:
            raise ValueError(
                f"patch_size must be positive, got {patch_size}: "
                "a patch needs at least one sample."
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
        # C pairs and P patches: with the token width, the input size of the head
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
        # Patch embedding: the linear projection of a patch to a token of width
        # D (W and b in the docstring equation)
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
        # Trailing axis, to zero every feature of the token of an invalid patch
        self.to_token_mask = Rearrange("rows patches -> rows patches 1")

        # Transformer encoder: pre-norm layers and a final LayerNorm, over the
        # P tokens of one pair. All the pairs go through the same weights (the
        # "channel-independent encoding with shared parameters" of the GeoNIRS
        # paper), so one checkpoint serves any number of channels
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
        # It is not pretrained: the checkpoint holds the encoder and the decoder

        # The tokens of all the pairs and patches, in one vector per window
        self.flatten_layer = Rearrange("batch ... -> batch (...)")
        # Dropout on that vector, before the classifier
        self.head_dropout = nn.Dropout(head_drop_prob)
        # Linear classifier over C * P * D features
        self.final_layer = nn.Linear(
            in_features=n_pairs * n_patches * embed_dim,
            out_features=self.n_outputs,
        )

        # Initialization of the reference
        self._init_parameters()

    def _init_parameters(self) -> None:
        """Xavier-uniform weights and zero biases, as the reference."""
        for module in self.modules():
            # Patch embedding, feed-forward layers and classifier
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
            # Query, key and value projections. nn.TransformerEncoder copies
            # one layer, so without this draw every layer would start equal
            elif isinstance(module, nn.MultiheadAttention):
                nn.init.xavier_uniform_(module.in_proj_weight)
                nn.init.zeros_(module.in_proj_bias)

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
        ch_mask: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
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
        ch_mask : None
            Not supported, because ``final_layer`` spans every pair. Mark a
            missing channel with NaN instead.
        mask : torch.Tensor of bool, optional
            Same shape as ``x``. True samples count as missing.

        Returns
        -------
        torch.Tensor or dict
            Output tensor with shape (batch_size, n_outputs), or
            ``{"features": (batch_size, n_pairs * n_patches * embed_dim),
            "cls_token": None}`` with ``return_features=True``.
        """
        # Shapes use the symbols of the class docstring: C pairs, P patches of
        # Q samples, tokens of width D.

        # The classifier spans every pair, so a padded channel cannot be left
        # out of it: a missing channel is given as NaN instead
        if ch_mask is not None:
            raise ValueError("JungMAE does not accept ch_mask; use NaN channels.")

        # Encoder: one token per patch of each HbO/HbR pair, zero where the
        # patch is invalid ("foundation" in the reference)
        # (batch, 2C, n_times) -> (batch, C, P, D)
        encoded = self.encode(x, mask)
        x = encoded["foundation"]

        # Input of the flatten_linear head of the benchmark: every token of
        # the window in one vector, pair after pair
        # (batch, C, P, D) -> (batch, C * P * D)
        x = self.flatten_layer(x)

        # Dropout on the features, before the classifier
        # (batch, C * P * D), unchanged
        x = self.head_dropout(x)
        if return_features:
            return {"features": x, "cls_token": None}

        # Linear classifier: the logits of the window
        # (batch, C * P * D) -> (batch, n_outputs)
        x = self.final_layer(x)
        return x

    def encode(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> dict[str, torch.Tensor]:
        """Encode the patches of every HbO/HbR pair.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor with shape (batch_size, n_chans, n_times). NaN marks
            a missing sample.
        mask : torch.Tensor of bool, optional
            Same shape as ``x``. True samples count as missing.

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

        # Samples hidden by ``mask`` are missing data, like NaN
        # (batch, 2C, n_times), unchanged
        if mask is not None:
            x = x.masked_fill(mask, float("nan"))
        n_times = x.shape[-1]

        # The window is cut into whole patches of Q samples: a partial last
        # patch is refused, padded or cropped (on_non_divisible)
        # (batch, 2C, n_times) -> (batch, 2C, P * Q)
        x = pad_or_crop_to_patches(x, self.patch_size, self.on_non_divisible)

        # The two chromophores of each source-detector pair
        # (batch, 2C, P * Q) -> HbO (batch, C, P * Q) and HbR (batch, C, P * Q)
        hbo, hbr = fnirs_split_hbo_hbr(x, self.channel_order)
        batch_size, n_pairs, _ = hbo.shape

        # Sample validity: a sample counts when both chromophores are finite.
        # The samples added by the padding do not count
        # -> (batch, C, P * Q), bool
        hbo_valid = torch.isfinite(hbo)
        hbr_valid = torch.isfinite(hbr)
        valid_samples = hbo_valid & hbr_valid
        valid_samples[:, :, n_times:] = False

        # Patch validity: a patch is encoded when at least
        # ceil(minimum_patch_valid_fraction * Q) of its samples are valid
        # (batch, C, P * Q) -> (batch, C, P), bool
        n_valid_samples = self.count_valid_samples(valid_samples)
        valid_patches = n_valid_samples >= self.min_valid_samples

        # The same validity with one row per pair, as the encoder reads the
        # tokens, and its complement
        # (batch, C, P) -> (batch * C, P), bool
        valid_tokens = self.merge_pairs(valid_patches)
        invalid_tokens = ~valid_tokens

        # Invalid samples are set to zero before the projection, as in the
        # reference
        # (batch, C, P * Q) each, unchanged
        hbo = torch.where(valid_samples, hbo, 0.0)
        hbr = torch.where(valid_samples, hbr, 0.0)

        # Patch tokenization: HbO and HbR are paired sample by sample ...
        # -> (batch, C, P * Q, 2)
        x = torch.stack((hbo, hbr), dim=-1)
        # ... and cut into patches [HbO_t, HbR_t, HbO_t+1, HbR_t+1, ...], with
        # each pair as a sequence of its own
        # (batch, C, P * Q, 2) -> (batch * C, P, 2 * Q)
        x = self.to_patches(x)

        # Patch embedding: linear projection of each patch to a token,
        # z = W patch + b
        # (batch * C, P, 2 * Q) -> (batch * C, P, D)
        x = self.patch_embed(x)

        # Fixed sinusoidal encoding PE(p) of the patch index p, added to the
        # tokens. It is computed in float32 at least and then cast, which is
        # what the reference does in half precision
        # (P, D), broadcast over the batch * C sequences
        _, n_patches, embed_dim = x.shape
        positions_dtype = torch.promote_types(x.dtype, torch.float32)
        positions = sinusoidal_positional_encoding(
            n_patches, embed_dim, device=x.device, dtype=positions_dtype
        )
        positions = positions.to(x.dtype)
        x = x + positions

        # Invalid patches are hidden from the attention as keys. A pair without
        # any valid patch keeps all its keys, because a softmax over no key is
        # NaN; its tokens are zeroed below
        # (batch * C, P), bool
        has_valid_token = valid_tokens.any(dim=-1, keepdim=True)
        key_padding_mask = invalid_tokens & has_valid_token

        # Transformer encoder over the P tokens of each pair, with the same
        # weights for every pair
        # (batch * C, P, D), unchanged
        x = self.encoder(x, src_key_padding_mask=key_padding_mask)

        # The token of an invalid patch is zero, so a pair without data gives
        # zero features
        # (batch * C, P, D), unchanged
        invalid_tokens = self.to_token_mask(invalid_tokens)
        x = x.masked_fill(invalid_tokens, 0.0)

        # The pairs back on their own axis
        # (batch * C, P, D) -> (batch, C, P, D)
        x = x.unflatten(0, [batch_size, n_pairs])
        return {"foundation": x, "valid": valid_patches}

    def reset_head(self, n_outputs: int) -> None:
        self._set_n_outputs(n_outputs)
        # A new classifier over the same C * P * D features
        self.final_layer = nn.Linear(self.final_layer.in_features, n_outputs)
        # Drawn as the constructor draws it (_init_parameters)
        nn.init.xavier_uniform_(self.final_layer.weight)
        nn.init.zeros_(self.final_layer.bias)

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        """Load a state dict, or the ``jung_mae.pt`` payload as released.

        The payload keeps the weights under ``"model"`` and has the
        pretraining decoder, which is dropped. A state dict without
        ``final_layer`` keeps the head of this model.
        """
        # The released file is a payload with the weights under "model"
        state_dict = state_dict.get("model", state_dict)
        # The pretraining decoder (its input projection, mask token, layers and
        # prediction layer) is not part of this class
        decoder_keys = ("decoder_embed.", "mask_token", "decoder.", "prediction.")
        state_dict = {
            key: value
            for key, value in state_dict.items()
            if not key.startswith(decoder_keys)
        }
        # The checkpoint has no classifier: the one of this model is kept, so
        # that a strict load of the encoder passes
        if not any(key.startswith("final_layer.") for key in state_dict):
            head = self.final_layer.state_dict(prefix="final_layer.")
            state_dict.update(head)
        return super().load_state_dict(state_dict, strict=strict, assign=assign)
