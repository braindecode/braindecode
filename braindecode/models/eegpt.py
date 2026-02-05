# Authors: Young Truong <dt.young112@gmail.com>
#          Kuntal Kokate <kukokate@ucsd.edu>
#
# License: BSD-3

import math
from functools import partial
from typing import Literal, Optional

import mne
import torch
from einops import rearrange, repeat
from torch import nn

from braindecode.models.base import EEGModuleMixin
from braindecode.modules import DropPath
from braindecode.modules.convolution import Conv1dWithConstraint
from braindecode.modules.linear import LinearWithConstraint


class EEGPT(EEGModuleMixin, nn.Module):
    r"""
    EEGPT: Pretrained Transformer for Universal and Reliable Representation of EEG Signals from Wang et al. (2024) [eegpt]_.

    :bdg-danger:`Foundation Model` :bdg-info:`Attention/Transformer`

    .. figure:: https://github.com/BINE022/EEGPT/raw/main/figures/EEGPT.jpg
        :align: center
        :alt: EEGPT Architecture
        :width: 1000px

        a) The EEGPT structure involves patching the input EEG signal as :math:`p_{i,j}` through masking
        (50% time and 80% channel patches), creating masked part :math:`\mathcal{M}` and unmasked part :math:`\bar{\mathcal{M}}`.
        b) Local spatio-temporal embedding maps patches to tokens.
        c) Use of dual self-supervised learning with Spatio-Temporal Representation Alignment and Mask-based Reconstruction.

    **EEGPT** is a pretrained transformer model designed for universal EEG feature extraction.
    It addresses challenges like low SNR and inter-subject variability by employing
    a dual self-supervised learning method that combines **Spatio-Temporal Representation Alignment**
    and **Mask-based Reconstruction** [eegpt]_.

    .. rubric:: Model Overview (Layer-by-layer)

    1. **Patch embedding** (``_PatchEmbed`` or ``_PatchNormEmbed``): split each channel into
       ``patch_size`` time patches and project to ``embed_dim``, yielding tokens with shape
       ``(batch, n_patches, n_chans, embed_dim)``.
    2. **Channel embedding** (``chan_embed``): add a learned embedding for each channel to preserve
       spatial identity before attention.
    3. **Transformer encoder blocks** (``_EEGTransformer.blocks``): for each patch group, append
       ``embed_num`` learned summary tokens and process the sequence with multi-head self-attention
       and MLP layers.
    4. **Summary extraction**: keep only the summary tokens, apply ``norm`` if set, and reshape back
       to ``(batch, n_patches, embed_num, embed_dim)``.
    5. **Task head** (``final_layer``): flatten summary tokens across patches and map to
       ``n_outputs``; if ``return_encoder_output=True``, return the encoder features instead.

    .. rubric:: Dual Self-Supervised Learning

    EEGPT moves beyond simple masked reconstruction by introducing a representation alignment objective.
    The pretraining loss :math:`\mathcal{L}` is the sum of alignment loss :math:`\mathcal{L}_A` and reconstruction loss :math:`\mathcal{L}_R`:

    .. math::
        \mathcal{L} = \mathcal{L}_A + \mathcal{L}_R

    1.  **Spatio-Temporal Representation Alignment:** (:math:`\mathcal{L}_A`)
        Aligns the predicted features of masked regions with global features extracted by a Momentum Encoder.
        This forces the model to learn semantic, high-level representations rather than just signal waveform details.

        .. math::
            \mathcal{L}_A = - \frac{1}{N} \sum_{j=1}^{N} ||pred_j - LN(menc_j)||_2^2

        where :math:`pred_j` is the predictor output and :math:`menc_j` is the momentum encoder output.

    2.  **Mask-based Reconstruction:** (:math:`\mathcal{L}_R`)
        Standard masked autoencoder objective to reconstruct the raw EEG patches, ensuring local temporal fidelity.

        .. math::
            \mathcal{L}_R = - \frac{1}{|\mathcal{M}|} \sum_{(i,j) \in \mathcal{M}} ||rec_{i,j} - LN(p_{i,j})||_2^2

        where :math:`rec_{i,j}` is the reconstructed patch and :math:`p_{i,j}` is the original patch.

    .. rubric:: Macro Components

    - `EEGPT.target_encoder` **(Universal Encoder)**
        - *Operations.* A hierarchical backbone that consists of **Local Spatio-Temporal Embedding** followed
          by a standard Transformer encoder [eegpt]_.
        - *Role.* Maps raw spatio-temporal EEG patches into a sequence of latent tokens :math:`z`.
    - `EEGPT.chans_id` **(Channel Identification)**
        - *Operations.* A buffer containing channel indices mapped from the standard channel names provided
          in ``chs_info`` [eegpt]_.
        - *Role.* Provides the spatial identity for each input channel, allowing the model to look up
          the correct channel embedding vector :math:`\varsigma_i`.
    - **Local Spatio-Temporal Embedding** (Input Processing)
        - *Operations.* The input signal :math:`X` is chunked into patches :math:`p_{i,j}`. Each patch
          is linearly projected and summed with a specific channel embedding:
          :math:`token_{i,j} = \text{Embed}(p_{i,j}) + \varsigma_i` [eegpt]_.
        - *Role.* Converts the 2D EEG grid (Channels :math:`\times` Time) into a unified sequence of tokens
          that preserves both channel identity and temporal order.

    .. rubric:: How the information is encoded temporally, spatially, and spectrally

    * **Temporal.**
      The model segments continuous EEG signals into small, non-overlapping patches (e.g., 250ms windows
      with ``patch_size=64``) [eegpt]_. This **Patching** mechanism captures short-term local temporal
      structure, while the subsequent Transformer encoder captures long-range temporal dependencies across
      the entire window.
    * **Spatial.**
      Unlike convolutional models that may rely on fixed spatial order, EEGPT uses **Channel Embeddings**
      :math:`\varsigma_i` [eegpt]_. Each channel's data is treated as a distinct sequence of tokens tagged
      with its spatial identity. This allows the model to flexibly handle different montages and
      missing channels by simply mapping channel names to their corresponding learnable embeddings.
    * **Spectral.**
      Spectral information is implicitly learned through the **Mask-based Reconstruction** objective
      (:math:`\mathcal{L}_R`) [eegpt]_. By forcing the model to reconstruct raw waveforms (including phase
      and amplitude) from masked inputs, the model learns to encode frequency-specific patterns necessary
      refines this by encouraging these spectral features to align with robust, high-level semantic representations.

    .. rubric:: Pretrained Weights

    Weights are available on `HuggingFace <https://huggingface.co/braindecode/eegpt-pretrained>`_.

    .. important::
       **Pre-trained Weights Available**

       This model has pre-trained weights available on the Hugging Face Hub.
       `Link here <https://huggingface.co/braindecode/eegpt-pretrained>`_.

       You can load them using:

       .. code-block:: python

           from braindecode.models import EEGPT

           # Load pre-trained model from Hugging Face Hub
           model = EEGPT.from_pretrained("braindecode/eegpt-pretrained")

       To push your own trained model to the Hub:

       .. code-block:: python

           # After training your model
           model.push_to_hub(
               repo_id="username/my-eegpt-model", commit_message="Upload trained EEGPT model"
           )

       Requires installing ``braindecode[hug]`` for Hub integration.

    .. rubric:: Usage

    The model can be initialized for specific downstream tasks (e.g., classification) by specifying
    `n_outputs`, `chs_info`, `n_times`.

    .. code-block:: python

        from braindecode.models import EEGPT

        model = EEGPT(
            n_chans=22,
            n_times=1000,
            chs_info=chs_info,
            n_outputs=4,  # For classification tasks
            patch_size=64,
            depth=8,
            embed_dim=512,
        )

        # Forward pass
        # Input shape: (batch_size, n_chans, n_times)
        y = model(x)

    Parameters
    ----------
    return_encoder_output : bool, default=False
        Whether to return the encoder output or the classifier output.
    patch_size : int, default=64
        Size of the patches for the transformer.
    patch_stride : int, default=32
        Stride of the patches for the transformer.
    embed_num : int, default=4
        Number of summary tokens used for the global representation.
    embed_dim : int, default=512
        Dimension of the embeddings.
    depth : int, default=8
        Number of transformer layers.
    num_heads : int, default=8
        Number of attention heads.
    mlp_ratio : float, default=4.0
        Ratio of the MLP hidden dimension to the embedding dimension.
    drop_prob : float, default=0.0
        Dropout probability.
    attn_drop_rate : float, default=0.0
        Attention dropout rate.
    drop_path_rate : float, default=0.0
        Drop path rate.
    init_std : float, default=0.02
        Standard deviation for weight initialization.
    qkv_bias : bool, default=True
        Whether to use bias in the QKV projection.
    norm_layer : torch.nn.Module, default=None
        Normalization layer. If None, defaults to ``nn.LayerNorm`` with epsilon ``layer_norm_eps``.
    layer_norm_eps : float, default=1e-6
        Epsilon value for the normalization layer.

    References
    ----------
    .. [eegpt] Wang, G., Liu, W., He, Y., Xu, C., Ma, L., & Li, H. (2024).
       EEGPT: Pretrained transformer for universal and reliable representation of eeg signals.
       Advances in Neural Information Processing Systems, 37, 39249-39280.
       Online: https://proceedings.neurips.cc/paper_files/paper/2024/file/4540d267eeec4e5dbd9dae9448f0b739-Paper-Conference.pdf

    Notes
    -----
    When loading pretrained weights from the original EEGPT checkpoint (e.g., for
    fine-tuning), you may encounter "unexpected keys" related to the `predictor`
    and `reconstructor` modules (e.g., `predictor.mask_token`, `reconstructor.time_embed`).
    These components are used only during the self-supervised pre-training phase
    (Masked Auto-Encoder) and are not part of this encoder-only model used for
    downstream tasks. It is safe to ignore them.
    """

    def __init__(
        self,
        # braindecode parameters
        n_outputs=None,
        n_chans=None,
        chs_info=None,
        n_times=None,
        input_window_seconds=None,
        sfreq=None,
        # model parameters
        patch_size: int = 64,
        patch_stride: int = 32,
        embed_num: int = 4,
        embed_dim: int = 512,
        depth: int = 8,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        drop_prob: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        init_std: float = 0.02,
        qkv_bias: bool = True,
        patch_module: Optional[nn.Module] = None,
        norm_layer: Optional[nn.Module] = None,
        layer_norm_eps: float = 1e-6,
        return_encoder_output: bool = False,
        # downstream finetuning parameters
        chan_proj_type: Literal[
            "conv1d_constraint", "linear", "none"
        ] = "conv1d_constraint",
        n_chans_target: int = 19,
        chan_conv_max_norm: float = 1.0,
        final_layer: type[nn.Module] | None = None,
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

        # model parameters
        self.return_encoder_output = return_encoder_output
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.embed_num = embed_num
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.drop_prob = drop_prob
        self.attn_drop_rate = attn_drop_rate
        self.drop_path_rate = drop_path_rate
        self.init_std = init_std
        self.qkv_bias = qkv_bias
        self.layer_norm_eps = layer_norm_eps
        self.norm_layer = norm_layer or partial(nn.LayerNorm, eps=layer_norm_eps)
        # set default patch module if not provided
        if patch_module is None:
            patch_module = _PatchEmbed
        # check if patch module is _PatchEmbed or _PatchNormEmbed
        if not issubclass(patch_module, _PatchEmbed):
            raise ValueError("patch_module must be a subclass of _PatchEmbed")
        else:
            self.patch_module = patch_module

        if final_layer is not None and return_encoder_output:
            raise ValueError(
                "return_encoder_output is not compatible with providing a final_layer which will be nn.Identity"
            )

        # Downstream finetuning config
        self.chan_proj_type = chan_proj_type
        self.n_chans_target = n_chans_target

        # Build channel projection (before encoder)
        if chan_proj_type != "none":
            self.chan_proj = _ChannelProjection(
                in_channels=self.n_chans,
                out_channels=n_chans_target,
                proj_type=chan_proj_type,
                max_norm=chan_conv_max_norm,
            )
            encoder_n_chans = n_chans_target
        else:
            self.chan_proj = nn.Identity()
            encoder_n_chans = self.n_chans

        self.target_encoder = _EEGTransformer(
            n_chans=encoder_n_chans,
            n_times=self.n_times,
            patch_size=self.patch_size,
            patch_stride=self.patch_stride,
            embed_num=self.embed_num,
            embed_dim=self.embed_dim,
            depth=self.depth,
            num_heads=self.num_heads,
            mlp_ratio=self.mlp_ratio,
            drop_rate=self.drop_prob,
            attn_drop_rate=self.attn_drop_rate,
            drop_path_rate=self.drop_path_rate,
            init_std=self.init_std,
            qkv_bias=self.qkv_bias,
            norm_layer=self.norm_layer,
            patch_module=self.patch_module,
        )

        # Prepare channel IDs
        if chan_proj_type != "none":
            # Use standard 19 channels when projecting
            self.channel_names = EEGPT_19_CHANNELS
        elif self._chs_info is not None:
            self.channel_names = [ch["ch_name"] for ch in self.chs_info]  # type: ignore
        else:
            self.channel_names = None  # type: ignore

        self.register_buffer(
            "chans_id", self.target_encoder.prepare_chan_ids(self.channel_names)
        )

        self.flattened_encoder_output_dim = (
            self.target_encoder.num_patches[1] * self.embed_num * self.embed_dim
        )

        # Build final layer (classification head)
        if return_encoder_output:
            self.final_layer = nn.Identity()
        elif final_layer is not None:
            # Use provided final_layer
            self.final_layer = nn.Sequential(
                nn.Flatten(1),
                final_layer(),
            )
        else:
            # Default: _LinearConstraintProbe (original EEGPT probe)
            self.final_layer = _LinearConstraintProbe(
                n_patches=self.target_encoder.num_patches[1],
                embed_num=self.embed_num,
                embed_dim=self.embed_dim,
                n_outputs=self.n_outputs,
            )

    @property
    def n_patches(self) -> int:
        """Number of temporal patches from encoder."""
        return self.target_encoder.num_patches[1]

    def get_probe_params(self) -> dict:
        """Get parameters needed to create a _LinearConstraintProbe.

        Returns
        -------
        dict
            Parameters dict with n_patches, embed_num, embed_dim.
        """
        return {
            "n_patches": self.n_patches,
            "embed_num": self.embed_num,
            "embed_dim": self.embed_dim,
        }

    def forward(self, x):
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            EEG data of shape (batch, n_chans, n_times).

        Returns
        -------
        torch.Tensor
            Model output. Shape depends on `n_outputs` and `return_encoder_output`.
        """
        # Channel projection (if configured)
        x = self.chan_proj(x)

        # z shape: (batch, n_patches, embed_num, embed_dim)
        z = self.target_encoder(x, self.chans_id)

        if self.return_encoder_output:
            return z

        # Pass to final_layer
        # _LinearConstraintProbe expects z in 4D (batch, n_patches, embed_num, embed_dim)
        # Default linear layer expects flattened input
        return self.final_layer(z)


# These channels correspond to a subset of the standard 10-20 system.
# The order matches the pre-trained weights and corresponds to
# mne.channels.make_standard_montage("standard_1020") (filtered).
_ALLOWED_CHANNELS = {
    "FP1",
    "FPZ",
    "FP2",
    "AF7",
    "AF3",
    "AF4",
    "AF8",
    "F7",
    "F5",
    "F3",
    "F1",
    "FZ",
    "F2",
    "F4",
    "F6",
    "F8",
    "FT7",
    "FC5",
    "FC3",
    "FC1",
    "FCZ",
    "FC2",
    "FC4",
    "FC6",
    "FT8",
    "T7",
    "C5",
    "C3",
    "C1",
    "CZ",
    "C2",
    "C4",
    "C6",
    "T8",
    "TP7",
    "CP5",
    "CP3",
    "CP1",
    "CPZ",
    "CP2",
    "CP4",
    "CP6",
    "TP8",
    "P7",
    "P5",
    "P3",
    "P1",
    "PZ",
    "P2",
    "P4",
    "P6",
    "P8",
    "PO7",
    "PO5",
    "PO3",
    "POZ",
    "PO4",
    "PO6",
    "PO8",
    "O1",
    "OZ",
    "O2",
}


def _get_eegpt_channels():
    montage = mne.channels.make_standard_montage("standard_1020")
    return [ch.upper() for ch in montage.ch_names if ch.upper() in _ALLOWED_CHANNELS]


EEGPT_CHANNELS = _get_eegpt_channels()

CHANNEL_DICT = {ch: i for i, ch in enumerate(EEGPT_CHANNELS)}

# Standard 19 channels used in original EEGPT linear probe
EEGPT_19_CHANNELS = [
    "FP1",
    "FP2",
    "F7",
    "F3",
    "FZ",
    "F4",
    "F8",
    "T7",
    "C3",
    "CZ",
    "C4",
    "T8",
    "P7",
    "P3",
    "PZ",
    "P4",
    "P8",
    "O1",
    "O2",
]


class _LinearConstraintProbe(nn.Module):
    """
    Original EEGPT two-stage probe with LinearWithConstraint layers.

    This replicates the exact probe architecture from the original EEGPT paper.

    Parameters
    ----------
    n_patches : int
        Number of temporal patches from encoder.
    embed_num : int
        Number of summary tokens (default 4 for EEGPT).
    embed_dim : int
        Embedding dimension (default 512 for EEGPT).
    n_outputs : int
        Number of output classes.
    probe_hidden_dim : int
        Hidden dimension between probe layers (default 16).
    dropout_p : float
        Dropout probability (default 0.5).
    probe1_max_norm : float
        Max norm for first LinearWithConstraint (default 1.0).
    probe2_max_norm : float
        Max norm for second LinearWithConstraint (default 0.25).
    """

    def __init__(
        self,
        n_patches: int,
        embed_num: int,
        embed_dim: int,
        n_outputs: int,
        probe_hidden_dim: int = 16,
        dropout_p: float = 0.5,
        probe1_max_norm: float = 1.0,
        probe2_max_norm: float = 0.25,
    ):
        super().__init__()
        self.probe1 = LinearWithConstraint(
            embed_num * embed_dim, probe_hidden_dim, max_norm=probe1_max_norm
        )
        self.dropout = nn.Dropout(p=dropout_p)
        self.probe2 = LinearWithConstraint(
            n_patches * probe_hidden_dim, n_outputs, max_norm=probe2_max_norm
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: (batch, n_patches, embed_num, embed_dim)
        h = z.flatten(2)  # (batch, n_patches, embed_num * embed_dim)
        h = self.probe1(self.dropout(h))  # (batch, n_patches, probe_hidden_dim)
        h = h.flatten(1)  # (batch, n_patches * probe_hidden_dim)
        return self.probe2(h)


class _ChannelProjection(nn.Module):
    """Channel projection layer to adapt input channels to target channels."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        proj_type: Literal["conv1d_constraint", "linear", "none"] = "none",
        max_norm: float = 1.0,
    ):
        super().__init__()
        self.proj_type = proj_type

        if proj_type == "none":
            if in_channels != out_channels:
                raise ValueError(
                    f"proj_type='none' requires in_channels ({in_channels}) == out_channels ({out_channels})"
                )
            self.proj = nn.Identity()
        elif proj_type == "conv1d_constraint":
            self.proj = Conv1dWithConstraint(
                in_channels, out_channels, kernel_size=1, max_norm=max_norm
            )
        elif proj_type == "linear":
            self.proj = nn.Linear(in_channels, out_channels)
        else:
            raise ValueError(f"Unknown proj_type: {proj_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.proj_type == "linear":
            # x: (batch, channels, time) -> (batch, time, channels) -> linear -> transpose back
            return self.proj(x.transpose(1, 2)).transpose(1, 2)
        return self.proj(x)


def _rotate_half(x):
    """Rotate half of the dimensions for RoPE.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor of shape (..., d).

    Returns
    -------
    torch.Tensor
        Rotated tensor of the same shape.
    """
    x = x.reshape((*x.shape[:-1], x.shape[-1] // 2, 2))
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return x.flatten(-2)


def _apply_rotary_emb(freqs, t, start_index=0, scale=1.0):
    """Apply rotary positional embeddings (RoPE) to input tensor.

    Parameters
    ----------
    freqs : torch.Tensor
        Frequency tensor for rotation.
    t : torch.Tensor
        Input tensor to rotate.
    start_index : int, default=0
        Starting index for rotation.
    scale : float, default=1.0
        Scaling factor.

    Returns
    -------
    torch.Tensor
        Tensor with rotary embeddings applied.
    """
    freqs = freqs.to(t)
    rot_dim = freqs.shape[-1]
    end_index = start_index + rot_dim
    if rot_dim > t.shape[-1]:
        raise ValueError(
            f"feature dimension {t.shape[-1]} is not of sufficient size "
            f"to rotate in all the positions {rot_dim}"
        )
    t_left, t_mid, t_right = (
        t[..., :start_index],
        t[..., start_index:end_index],
        t[..., end_index:],
    )
    t_mid = (t_mid * freqs.cos() * scale) + (_rotate_half(t_mid) * freqs.sin() * scale)
    return torch.cat((t_left, t_mid, t_right), dim=-1)


def _apply_mask(mask, x):
    r"""Apply mask to select specific patches from input tensor.

    The operation flattens the patch and channel dimensions, gathers the selected
    indices specified by the mask, and then optionally reshapes back if the mask
    is 2D.

    .. math::
       x_{flat} = \text{Flatten}(x, \text{dims}=(1, 2))
       x_{masked} = \text{Gather}(x_{flat}, \text{index}=mask)

    Parameters
    ----------
    mask : torch.Tensor
        Mask tensor containing indices of patches to keep.
        Can be 1D ``(n_masked_items,)`` or 2D ``(n_masked_patches, n_masked_chans)``.
    x : torch.Tensor
        Input tensor.
        Shape: ``(batch_size, n_patches, n_chans, embed_dim)``

    Returns
    -------
    torch.Tensor
        Masked tensor with selected patches.
        If mask is 2D, output shape is ``(batch_size, n_masked_patches, n_masked_chans, embed_dim)``.
        If mask is 1D, output shape is ``(batch_size, n_masked_items, embed_dim)``.
    """
    batch_size, n_patches, n_chans, embed_dim = x.shape

    # Flatten patches and channels: (b, n, c, d) -> (b, n*c, d)
    x_flat = rearrange(x, "b n c d -> b (n c) d")

    if len(mask.shape) == 2:
        n_masked_patches, n_masked_chans = mask.shape

        # Flatten mask: (mn, mc) -> (mn*mc)
        mask_flat = rearrange(mask, "mn mc -> (mn mc)")

        # Prepare indices for gathering: (1, mn*mc, 1) -> (b, mn*mc, d)
        mask_keep = repeat(
            mask_flat,
            "m -> b m d",
            b=batch_size,
            d=embed_dim,
        )

        # Gather selected patch-channel pairs
        masked_x_flat = torch.gather(x_flat, dim=1, index=mask_keep)

        # Reshape back to 2D structure: (b, mn*mc, d) -> (b, mn, mc, d)
        masked_x = rearrange(
            masked_x_flat,
            "b (mn mc) d -> b mn mc d",
            mn=n_masked_patches,
            mc=n_masked_chans,
        )
    else:
        # Mask is 1D: (n_masked_items,)

        # Prepare indices for gathering: (m, ) -> (b, m, d)
        mask_keep = repeat(
            mask,
            "m -> b m d",
            b=batch_size,
            d=embed_dim,
        )

        # Gather
        masked_x = torch.gather(x_flat, dim=1, index=mask_keep)

    return masked_x


def _apply_mask_t(mask_t, x):
    r"""Apply temporal mask to select specific patches.

    This function selects the temporal patches specified by the boolean mask.
    It operates on the sequence length dimension.

    The operation performed is effectively:

    .. math::
       x_{masked}[b, i, d] = x[b, \text{index}[b, i], d]

    where :math:`\text{index}` corresponds to the indices where `mask_t` is True.

    Parameters
    ----------
    mask_t : torch.Tensor
        Mask tensor containing temporal indices to keep.
        Shape: ``(n_masked_patches, )``
    x : torch.Tensor
        Input tensor.
        Shape: ``(batch_size, n_patches, embed_dim)``

    Returns
    -------
    torch.Tensor
        Masked tensor with selected temporal patches.
        Shape: ``(batch_size, n_masked_patches, embed_dim)``
    """
    batch_size, n_patches, embed_dim = x.shape

    mask_keep = repeat(
        mask_t,
        "n -> b n d",
        b=batch_size,
        d=embed_dim,
    )

    masked_x = torch.gather(x, dim=1, index=mask_keep)
    return masked_x


class _MLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) with GELU activation and dropout.

    This block consists of two linear layers with a GELU activation and dropout
    in between, and another dropout after the second linear layer.

    Parameters
    ----------
    in_features : int
        Number of input features.
    hidden_features : int, optional
        Number of hidden features. If None, defaults to `in_features`.
    out_features : int, optional
        Number of output features. If None, defaults to `in_features`.
    act_layer : nn.Module, default=nn.GELU
        Activation function.
    drop : float, default=0.0
        Dropout probability.
    """

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class _Attention(nn.Module):
    """
    Multi-head Self-Attention with optional RoPE and Causal Masking.

    This layer implements multi-head self-attention using PyTorch's
    scaled_dot_product_attention (Flash Attention) for efficiency.
    It supports Rotary Positional Embeddings (RoPE) and causal masking.

    Parameters
    ----------
    dim : int
        Input and output embedding dimension.
    num_heads : int, default=8
        Number of attention heads.
    qkv_bias : bool, default=False
        If True, add a learnable bias to query, key, value projections.
    attn_drop : float, default=0.0
        Dropout probability for attention weights.
    proj_drop : float, default=0.0
        Dropout probability for output projection.
    is_causal : bool, default=False
        If True, applies a causal mask (temporal causality).
    use_rope : bool, default=False
        If True, applies Rotary Positional Embeddings (RoPE) to queries and keys.
    return_attention : bool, default=False
        If True, returns the attention weights instead of the output tensor.
    """

    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        attn_drop=0.0,
        proj_drop=0.0,
        is_causal=False,
        use_rope=False,
        return_attention=False,
    ):
        super().__init__()
        self.use_rope = use_rope

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.is_causal = is_causal
        self.return_attention = return_attention

    def forward(self, x, freqs=None):
        """
        Forward pass of the attention layer.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, seq_len, embed_dim).
        freqs : torch.Tensor, optional
            Frequencies for Rotary Positional Embeddings (RoPE).
        """
        # qkv: (batch, seq_len, 3 * num_heads * head_dim)
        qkv = self.qkv(x)

        # Reshape to (3, batch, num_heads, seq_len, head_dim)
        qkv = rearrange(
            qkv,
            "batch seq_len (three num_heads head_dim) -> three batch num_heads seq_len head_dim",
            three=3,
            num_heads=self.num_heads,
        )
        q, k, v = qkv[0], qkv[1], qkv[2]  # batch, num_heads, seq_len, head_dim

        # 1. Rotary Positional Embeddings (RoPE)
        # Unlike standard absolute positional encodings, RoPE rotates the
        # query and key vectors to encode relative positions.
        if self.use_rope:
            q = _apply_rotary_emb(freqs, q)
            k = _apply_rotary_emb(freqs, k)

        # 2. Return Attention Weights
        # If return_attention is True, we manually compute attention scores
        # because F.scaled_dot_product_attention doesn't return weights.
        if self.return_attention:
            if self.is_causal:
                attn_mask = torch.ones(
                    q.size(-2), q.size(-2), device=q.device, dtype=torch.bool
                ).tril(diagonal=0)
                attn_zeros = torch.zeros(
                    q.size(-2), q.size(-2), device=q.device, dtype=q.dtype
                )
                attn_mask = attn_zeros.masked_fill(
                    torch.logical_not(attn_mask), -float("inf")
                )
                # Naive attention computation for visualization/debugging
                attn_weight = torch.softmax(
                    (q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))) + attn_mask,
                    dim=-1,
                )
            else:
                attn_weight = torch.softmax(
                    (q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))), dim=-1
                )
            return attn_weight

        # 3. Flash Attention
        # Use PyTorch's optimized scaled_dot_product_attention (SDPA)
        # which internally uses FlashAttention or EfficientAttention kernels.
        y = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.attn_drop.p if self.training else 0,
            is_causal=self.is_causal,
        )

        # Reshape back: (batch, num_heads, seq_len, head_dim) -> (batch, seq_len, embed_dim)
        x = rearrange(
            y,
            "batch num_heads seq_len head_dim -> batch seq_len (num_heads head_dim)",
        )

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class _Block(nn.Module):
    """
    Transformer Block consisting of Self-Attention and MLP.

    This is a standard Transformer encoder block that applies:
    1. Layer Normalization
    2. Multi-Head Self-Attention (with optional RoPE)
    3. Residual Connection with DropPath
    4. Layer Normalization
    5. MLP
    6. Residual Connection with DropPath

    Parameters
    ----------
    dim : int
        Input and output embedding dimension.
    num_heads : int
        Number of attention heads.
    mlp_ratio : float, default=4.0
        Ratio of MLP hidden dimension to embedding dimension.
    qkv_bias : bool, default=False
        If True, add a learnable bias to query, key, value projections.
    drop : float, default=0.0
        Dropout probability for MLP and projection layers.
    attn_drop : float, default=0.0
        Dropout probability for attention weights.
    drop_path : float, default=0.0
        Stochastic depth rate.
    act_layer : nn.Module, default=nn.GELU
        Activation function.
    norm_layer : nn.Module, default=nn.LayerNorm
        Normalization layer.
    is_causal : bool, default=False
        If True, applies a causal mask to attention.
    use_rope : bool, default=False
        If True, applies Rotary Positional Embeddings (RoPE).
    return_attention : bool, default=False
        If True, returns attention weights instead of the output tensor.
    """

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        is_causal=False,
        use_rope=False,
        return_attention=False,
    ):
        super().__init__()
        self.return_attention = return_attention
        self.norm1 = norm_layer(dim)
        self.attn = _Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            is_causal=is_causal,
            use_rope=use_rope,
            return_attention=return_attention,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = _MLP(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x, freqs=None):
        y = self.attn(self.norm1(x), freqs)
        if self.return_attention:
            return y
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class _PatchEmbed(nn.Module):
    """
    Splits the EEG signal into patches and embeds them.

    This layer transforms the input EEG signal (2D: channels x time) into a sequence of
    patch embeddings using a 2D convolution. It treats the channel dimension effectively
    as height=1 for the convolution, sliding over the time dimension.

    Parameters
    ----------
    n_chans : int, default=64
        Number of input channels.
    n_times : int, default=1000
        Number of time samples.
    patch_size : int, default=16
        Size of each patch along the time dimension.
    patch_stride : int, optional
        Stride between patches. If None, defaults to `patch_size` (non-overlapping).
    embed_dim : int, default=768
        Dimension of the output embeddings.
    """

    def __init__(
        self,
        n_chans=64,
        n_times=1000,
        patch_size=16,
        patch_stride=None,
        embed_dim=768,
        apply_norm=False,
    ):
        super().__init__()
        self.n_chans = n_chans
        self.n_times = n_times
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.apply_norm = apply_norm

        self._configure_padding()

        if apply_norm:
            self.unfold = torch.nn.Unfold(
                kernel_size=(1, patch_size),
                stride=(1, patch_stride if patch_stride is not None else patch_size),
            )
            self.proj = nn.Linear(patch_size, embed_dim)
        else:
            self.proj = nn.Conv2d(
                1,
                embed_dim,
                kernel_size=(1, patch_size),
                stride=(1, patch_size if patch_stride is None else patch_stride),
            )

    def forward(self, x):
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input EEG signal of shape (batch, n_chans, n_times).

        Returns
        -------
        torch.Tensor
            Patch embeddings of shape (batch, n_patches, n_chans, embed_dim).
        """
        # x: batch, n_chans, n_times
        x = self.padding_layer(x)
        x = rearrange(x, "batch n_chans n_times -> batch 1 n_chans n_times")

        if self.apply_norm:
            x = self.unfold(x)
            # (batch, patch_size, n_patches * n_chans)

            # Rearrange using Einops to (batch, n_patches, n_chans, patch_size)
            x = rearrange(
                x,
                "batch patch_size (n_chans n_patches) -> batch n_patches n_chans patch_size",
                n_chans=self.n_chans,
            )

            x = torch.layer_norm(x, (self.patch_size,))
            x = self.proj(x)  # (batch, n_patches, n_chans, embed_dim)
            return x
        else:
            # Convolve and rearrange:
            # (batch, embed_dim, n_chans, n_patches) -> (batch, n_patches, n_chans, embed_dim)
            x = self.proj(x)
            x = rearrange(
                x,
                "batch embed_dim n_chans n_patches -> batch n_patches n_chans embed_dim",
            )
            return x

    def _configure_padding(self):
        """
        Validates input size, calculates padding to ensure valid patching,
        and initializes the padding layer.
        """
        # Validation checks
        if self.n_times < self.patch_size:
            raise ValueError(
                f"n_times {self.n_times} must be >= patch_size {self.patch_size}"
            )

        # Padding calculation
        if self.patch_stride is None:
            # Non-overlapping: Just ensure divisibility
            remainder = self.n_times % self.patch_size
            self.padding_size = self.patch_size - remainder if remainder != 0 else 0
        else:
            # Overlapping: Ensure last patch fits perfectly
            remainder = (self.n_times - self.patch_size) % self.patch_stride
            self.padding_size = self.patch_stride - remainder if remainder != 0 else 0

        self.n_times_padded = self.n_times + self.padding_size

        # Layer creation
        if self.padding_size > 0:
            self.padding_layer = nn.ConstantPad1d((0, self.padding_size), 0.0)
        else:
            self.padding_layer = nn.Identity()

        # Num patches calculation
        eff_stride = (
            self.patch_stride if self.patch_stride is not None else self.patch_size
        )
        n_patches = (self.n_times_padded - self.patch_size) // eff_stride + 1
        self.num_patches = (self.n_chans, n_patches)


class _EEGTransformer(nn.Module):
    """
    Backbone of the EEGPT model processing the sequence of patch embeddings.

    This standard Transformer encoder processes the sequence of patch embeddings
    augmented with channel embeddings and a summary token.

    Parameters
    ----------
    n_chans : int
        Number of input channels.
    n_times : int
        Number of time samples.
    patch_size : int
        Size of each patch.
    patch_stride : int
        Stride between patches.
    embed_dim : int
        Embedding dimension.
    embed_num : int
        Number of summary tokens.
    depth : int
        Number of Transformer blocks.
    num_heads : int
        Number of attention heads.
    mlp_ratio : float
        Ratio of MLP hidden dimension to embedding dimension.
    qkv_bias : bool
        If True, add bias to QKV projections.
    drop_rate : float
        Dropout rate.
    attn_drop_rate : float
        Attention dropout rate.
    drop_path_rate : float
        Stochastic depth rate.
    norm_layer : nn.Module
        Normalization layer.
    patch_module : nn.Module
        Module used for patch embedding (e.g., `_PatchEmbed`).
    init_std : float
        Standard deviation for weight initialization.
    """

    def __init__(
        self,
        n_chans=64,
        n_times=1000,
        patch_size=64,
        patch_stride=None,
        embed_dim=768,
        embed_num=1,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        patch_module=_PatchEmbed,
        init_std=0.02,
        return_attention_layer=-1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.embed_num = embed_num
        self.num_heads = num_heads

        self.patch_embed = patch_module(
            n_chans=n_chans,
            n_times=n_times,
            patch_size=patch_size,
            patch_stride=patch_stride,
            embed_dim=embed_dim,
        )
        self.num_patches = self.patch_embed.num_patches

        # Ensure that the embedding layer is large enough to handle the input channels
        # especially when fallback to sequential IDs (0 to n_chans-1) happens.
        num_embeddings = max(len(CHANNEL_DICT), n_chans)
        self.chan_embed = nn.Embedding(num_embeddings, embed_dim)

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule

        self.blocks = nn.ModuleList(
            [
                _Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    is_causal=False,
                    use_rope=False,
                    return_attention=(i + 1) == return_attention_layer,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)
        self.init_std = init_std
        self.summary_token = nn.Parameter(torch.zeros(1, embed_num, embed_dim))

        nn.init.trunc_normal_(self.summary_token, std=self.init_std)
        self.apply(self._init_weights)
        self.fix_init_weight()

    def prepare_chan_ids(self, channels):
        if channels is None:
            # If no channel names are provided, use sequential IDs
            return torch.arange(self.patch_embed.n_chans)

        chan_ids = []
        unknown = []
        for i, ch in enumerate(channels):
            ch_upper = ch.upper().strip(".")
            if ch_upper in CHANNEL_DICT:
                chan_ids.append(CHANNEL_DICT[ch_upper])
            else:
                unknown.append(ch)
                chan_ids.append(i)

        if unknown:
            mne.utils.warn(
                "Unknown channel name(s) in chs_info: "
                f"{unknown}. Falling back to sequential channel IDs for these unknown "
                "channels, while preserving pretrained embeddings for known channels."
                "Map your channel names to EEGPT_CHANNELS to preserve "
                "pretrained channel embeddings.",
            )

        return torch.tensor(chan_ids).unsqueeze_(0).long()

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(torch.sqrt(torch.tensor(2.0 * layer_id)))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Embedding):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, x, chan_ids=None, mask_x=None, mask_t=None):
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input EEG data of shape (batch, n_chans, n_times).
        chan_ids : torch.Tensor, optional
            Channel IDs for channel embedding.
        mask_x : torch.Tensor, optional
            Mask for input patches.
        mask_t : torch.Tensor, optional
            Mask for temporal patches.

        Returns
        -------
        torch.Tensor
            Transformer output.
        """
        # x shape: (batch, n_chans, n_times)

        # -- patchify x
        x = self.patch_embed(x)
        # x shape: (batch, n_patches, n_chans, embed_dim)
        batch, n_patches, n_chans, embed_dim = x.shape

        if n_patches != self.num_patches[1] or n_chans != self.num_patches[0]:
            raise ValueError(
                f"Patch shape mismatch: got ({n_patches}, {n_chans}), "
                f"expected ({self.num_patches[1]}, {self.num_patches[0]})"
            )

        if chan_ids is None:
            chan_ids = torch.arange(0, n_chans)
        chan_ids = chan_ids.to(x.device)

        # -- add channels positional embedding to x
        # chan_embed shape: (1, 1, n_chans, embed_dim)
        x = x + self.chan_embed(chan_ids.long()).unsqueeze(0)

        if mask_x is not None:
            mask_x = mask_x.to(x.device)
            x = _apply_mask(mask_x, x)
            # x shape might change here if masking removes tokens
            batch, n_patches, n_chans, embed_dim = x.shape

        # Flatten batch and patches dimensions for Transformer processing
        # (batch, n_patches, n_chans, embed_dim) -> (batch * n_patches, n_chans, embed_dim)
        x = rearrange(
            x,
            "batch n_patches n_chans embed_dim -> (batch n_patches) n_chans embed_dim",
        )

        # -- concat summary token
        # summary_token shape: (batch * n_patches, embed_num, embed_dim)
        summary_token = self.summary_token.repeat((x.shape[0], 1, 1))
        x = torch.cat([x, summary_token], dim=1)
        # x shape: (batch * n_patches, n_chans + embed_num, embed_dim)

        # -- fwd prop
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if blk.return_attention:
                return x

        # Extract only the summary tokens (used for global representation)
        # x shape: (batch * n_patches, embed_num, embed_dim)
        x = x[:, -summary_token.shape[1] :, :]

        if self.norm is not None:
            x = self.norm(x)

        # Instead of flatten+reshape, let's just rearrange back to separate batch/patches explicitly
        x = rearrange(
            x,
            "(batch n_patches) embed_num embed_dim -> batch n_patches (embed_num embed_dim)",
            batch=batch,
        )

        if mask_t is not None:
            mask_t = mask_t.to(x.device)
            x = _apply_mask_t(mask_t, x)

        # Reshape to final output format: (batch, n_patches, embed_num, embed_dim)
        x = rearrange(
            x,
            "batch n_patches (embed_num embed_dim) -> batch n_patches embed_num embed_dim",
            embed_num=self.embed_num,
        )

        return x
