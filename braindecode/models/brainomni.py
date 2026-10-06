# Authors: Qian Xiao and OpenTSLab BrainOmni contributors
#          Bruno Aristimunha <b.aristimunha@gmail.com> (braindecode adaptation)
#
# License: MIT
#
# Ported from https://github.com/OpenTSLab/BrainOmni (MIT License, 2025 OpenTSLab).
# SEANet/conv/LSTM submodules derive from Meta's EnCodec (MIT License).
# The residual vector quantization lives in
# ``braindecode/modules/quantization.py`` (adapted from
# https://github.com/lucidrains/vector-quantize-pytorch, MIT License).
from __future__ import annotations

import math
import warnings
from collections import OrderedDict

import mne
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.nn import RMSNorm

# Classic weight_norm keeps ``conv.weight_g``/``weight_v`` keys (checkpoint parity).
from torch.nn.utils import weight_norm  # noqa: F401

from braindecode.models.base import EEGModuleMixin
from braindecode.models.util import extract_channel_locations_from_chs_info
from braindecode.modules import FeedForwardBlock
from braindecode.modules.quantization import ResidualVectorQuantizer

_TOKENIZER_CONFIG_RENAMES = {
    "n_dim": "emb_dim",
    "n_head": "tokenizer_num_heads",
    "dropout": "drop_prob",
}


def _translate_opentslab_config(
    config: dict, renames: dict[str, str], ignored: set[str] | None = None
) -> dict:
    """Return a non-mutating translation of an official BrainOmni config."""
    translated = dict(config)
    for key in ignored or ():
        translated.pop(key, None)
    for source, target in renames.items():
        if source not in translated:
            continue
        if target in translated:
            raise ValueError(
                f"OpenTSLab config contains both {source!r} and its Braindecode "
                f"name {target!r}."
            )
        translated[target] = translated.pop(source)
    return translated


class BrainTokenizer(EEGModuleMixin, nn.Module, license="mit"):
    r"""BrainTokenizer from Xiao et al. (2025) [brainomni]_.

    :bdg-danger:`Foundation Model` :bdg-info:`Attention/Transformer`

    ``BrainTokenizer`` is the VQ-VAE tokenizer backbone of BrainOmni
    [brainomni]_. It encodes raw multi-channel EEG/MEG windows into discrete
    neural tokens via a SEANet convolutional encoder, residual vector
    quantization (RVQ), and a cross-attention decoder that reconstructs the
    original waveform. Sensor geometry (position + orientation) is derived from
    ``chs_info`` at initialisation and used by ``_SensorEmbedding`` to build
    geometry-aware channel embeddings, which makes the tokenizer agnostic to the
    channel montage.

    .. rubric:: Architecture Overview

    The tokenizer is a geometry-conditioned VQ-VAE. The end-to-end path is::

        (batch, n_chans, n_times) -> window -> SEANet encode -> collapse channels
        into n_neuro latent sources -> residual VQ -> expand back to channels ->
        SEANet decode -> (batch, n_chans, n_times)

    1. **Sensor-geometry conditioning.** Per-channel position/orientation
       (from ``chs_info``) and sensor type (EEG/MAG/GRAD) are embedded once and
       injected into both cross-attention bridges, decoupling the model from any
       fixed montage.
    2. **Convolutional waveform codec.** A SEANet (EnCodec-style) encoder/decoder
       compresses each window by ``prod(ratios)`` and reconstructs it.
    3. **Discrete bottleneck.** Residual vector quantization turns the continuous
       latent into a stack of ``num_quantizers`` discrete codebook indices.

    .. rubric:: Macro Components

    Windowing (``BrainTokenizer._unfold``)
        **Operations.** Slices ``(batch, n_chans, n_times)`` into windows of
        ``window_length`` samples. A signal shorter than one window is padded;
        with overlap, only the final overlapping window is padded; without
        overlap, an incomplete tail is dropped, matching the released source.
        **Role.** Defines the analysis window that SEANet encodes.

    ``BrainTokenizer.sensor_embed`` (``_SensorEmbedding``)
        **Operations.** Maps each channel's ``(position, orientation)`` 6-vector
        and integer sensor type to an ``emb_dim`` embedding (MLP + type embedding,
        RMSNorm). **Role.** Geometry-aware, montage-agnostic channel identity.

    ``BrainTokenizer.encoder`` (``_TokenizerEncoder``)
        **Operations.** Runs each ``(channel, window)`` waveform through the
        SEANet encoder, then a cross-attention (``_BackwardSolution``) in which
        ``n_neuro`` learnable queries attend to the sensor-conditioned channel
        features, collapsing ``n_chans`` into ``n_neuro`` virtual sources.
        **Role.** Produces ``(batch, n_neuro, n_windows, n_tokens, emb_dim)``
        latent tokens.

    ``BrainTokenizer.quantizer`` (``ResidualVectorQuantizer``)
        **Operations.** Residual vector quantization with EMA codebooks; each
        stage quantizes the residual of the previous one. **Role.** The discrete
        bottleneck, emitting ``num_quantizers`` codebook indices per token.

    ``BrainTokenizer.final_layer`` (``_TokenizerDecoder``)
        **Operations.** Cross-attention (``_ForwardSolution``) lets the sensor
        embeddings query the quantized neural tokens, expanding ``n_neuro`` back
        to ``n_chans``, then the SEANet decoder reconstructs each window.
        **Role.** The output layer; reconstructs ``(batch, n_chans, n_times)``.

    .. rubric:: Temporal, Spatial, and Spectral Encoding

    - **Temporal:** SEANet's strided convolutions and LSTM compress each window
      along time by ``prod(ratios)``; ``n_windows * n_tokens`` is the resulting
      time axis.
    - **Spatial:** ``_SensorEmbedding`` and the two cross-attention bridges map
      physical sensors to and from ``n_neuro`` montage-independent virtual
      sources using sensor geometry.
    - **Spectral:** No explicit transform; frequency selectivity is learned
      implicitly by the SEANet convolutional filterbank.

    .. rubric:: Additional Mechanisms

    - **Residual VQ with the rotation trick.** Codebooks are EMA-updated only in
      ``train`` mode; :meth:`tokenize` runs under :func:`torch.no_grad` in
      ``eval`` mode so the codebooks are never corrupted when extracting tokens.
    - **Weight-normed convolutions.** SEANet uses classic ``weight_norm`` so the
      ``weight_g``/``weight_v`` keys stay aligned with the upstream release.

    .. rubric:: Released Weights

    .. important::

        The authors publish the MIT-licensed ``BrainTokenizer.pt`` artifact in
        ``OpenTSLab/BrainOmni`` on the Hugging Face Hub. It is a plain PyTorch
        state dict rather than a Braindecode Hub repository. The accompanying
        ``model_cfg.json`` can be passed directly to
        :meth:`from_opentslab_config`; then pass the state dict to
        :meth:`load_state_dict` with ``strict=True``. For example::

            config = json.loads(Path(config_path).read_text())
            model = BrainTokenizer.from_opentslab_config(
                config, chs_info=chs_info, n_times=512, sfreq=256.0
            )
            model.load_state_dict(
                torch.load(checkpoint_path, weights_only=True), strict=True
            )

        Input is expected at 256 Hz and must follow the authors' preprocessing.

    .. versionadded:: 1.8

    Parameters
    ----------
    n_outputs : int, optional
        Number of output classes. Retained for the standard model interface;
        reconstruction does not use a classification output.
    n_chans : int, optional
        Number of input channels. Inferred from ``chs_info`` when omitted.
    chs_info : list of dict, optional
        MNE channel information used to derive sensor geometry.
    n_times : int, optional
        Number of input samples.
    input_window_seconds : float, optional
        Input duration in seconds.
    sfreq : float, optional
        Sampling frequency in Hz. Released weights expect 256 Hz.
    emb_dim : int
        Embedding dimensionality throughout the model.
    n_neuro : int
        Number of virtual "neural source" tokens produced by the encoder.
    window_length : int
        Length (samples) of each analysis window fed to SEANet.
    n_filters : int
        Base number of filters in SEANet.
    ratios : tuple of int
        Downsampling ratios in SEANet (encoder) / upsampling ratios
        (decoder).  Total compression equals the product of ratios.
    kernel_size : int
        Kernel size for SEANet convolutions.
    last_kernel_size : int
        Kernel size for the first and last SEANet conv layer.
    tokenizer_num_heads : int
        Number of attention heads in the cross-attention tokenizer blocks.
    codebook_dim : int
        Projected dimension inside each VQ codebook.
    codebook_size : int
        Number of entries per VQ codebook.
    num_quantizers : int
        Number of residual VQ stages.
    rotation_trick : bool
        Whether to use the rotation trick when updating codebook entries.
    quantize_optimize_method : str
        Codebook optimisation method (``"ema"``; the only currently
        supported option).
    drop_prob : float
        Dropout probability applied inside the tokenizer attention blocks.
    activation : type[nn.Module]
        Accepted for braindecode API symmetry; the VQ-VAE uses fixed
        activations baked into the pretrained weights.

    Notes
    -----
    Input is expected at 256 Hz. The released preprocessing applies a
    0.1--96 Hz band-pass, 50/60 Hz notch filtering, bad-channel interpolation,
    per-sensor-type average referencing, then group scaling within EEG, MAG,
    and GRAD. The paper's description of the final scaling is not identical to
    the released implementation; reproduce the source pipeline when using the
    released weights.

    With zero overlap, the released windower drops an incomplete final window.
    :meth:`forward` and :meth:`encode_decode` zero-fill that dropped tail only
    to preserve Braindecode's fixed ``n_times`` reconstruction shape;
    :meth:`tokenize` returns only tokens produced by complete windows.

    References
    ----------
    .. [brainomni] Xiao, Q., Cui, Z., Zhang, C., Chen, S., Wu, W.,
       Thwaites, A., Woolgar, A., Zhou, B., Zhang, C. (2025).
       BrainOmni: A Brain Foundation Model for Unified EEG and MEG Signals.
       NeurIPS 2025.
       Online: https://arxiv.org/abs/2505.18185
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
        # model-specific parameters
        *,
        emb_dim: int = 256,
        n_neuro: int = 16,
        window_length: int = 512,
        n_filters: int = 32,
        ratios: tuple[int, ...] = (8, 4, 2),
        kernel_size: int = 5,
        last_kernel_size: int = 5,
        tokenizer_num_heads: int = 4,
        codebook_dim: int = 256,
        codebook_size: int = 512,
        num_quantizers: int = 4,
        rotation_trick: bool = True,
        quantize_optimize_method: str = "ema",
        drop_prob: float = 0.0,
        activation: type[nn.Module] = nn.SELU,
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

        if not math.isclose(float(self.sfreq), 256.0, rel_tol=0.0, abs_tol=1e-6):
            warnings.warn(
                f"BrainOmni pretrained weights expect sfreq=256 Hz, got "
                f"{self.sfreq}. Use only for training from scratch.",
                UserWarning,
            )

        pos, sensor_type = _geometry_from_chs_info(self.chs_info)
        self.register_buffer("pos", torch.from_numpy(pos))
        self.register_buffer("sensor_type", torch.from_numpy(sensor_type))

        self.emb_dim = emb_dim
        self.n_neuro = n_neuro
        self.window_length = window_length
        self.drop_prob = drop_prob
        self.activation = activation

        if window_length <= 0:
            raise ValueError(f"window_length must be positive, got {window_length}.")
        if n_filters <= 0:
            raise ValueError(f"n_filters must be positive, got {n_filters}.")
        if not ratios or any(ratio <= 0 for ratio in ratios):
            raise ValueError(
                f"ratios must be a non-empty sequence of positive values, got {ratios}."
            )
        if kernel_size <= 0:
            raise ValueError(f"kernel_size must be positive, got {kernel_size}.")
        if last_kernel_size <= 0:
            raise ValueError(
                f"last_kernel_size must be positive, got {last_kernel_size}."
            )
        if tokenizer_num_heads <= 0:
            raise ValueError(
                f"tokenizer_num_heads must be positive, got {tokenizer_num_heads}."
            )
        if emb_dim <= 0 or emb_dim % tokenizer_num_heads:
            raise ValueError(
                f"emb_dim ({emb_dim}) must be positive and divisible by "
                f"tokenizer_num_heads ({tokenizer_num_heads})."
            )
        if n_neuro <= 0:
            raise ValueError(f"n_neuro must be positive, got {n_neuro}.")
        if not 0.0 <= drop_prob <= 1.0:
            raise ValueError(
                f"drop_prob must satisfy 0 <= drop_prob <= 1, got {drop_prob}."
            )

        self.sensor_embed = _SensorEmbedding(emb_dim)
        self.encoder = _TokenizerEncoder(
            n_filters=n_filters,
            ratios=list(ratios),
            kernel_size=kernel_size,
            last_kernel_size=last_kernel_size,
            n_dim=emb_dim,
            n_head=tokenizer_num_heads,
            dropout=drop_prob,
            n_neuro=n_neuro,
        )
        self.quantizer = ResidualVectorQuantizer(
            dim=emb_dim,
            codebook_dim=codebook_dim,
            codebook_size=codebook_size,
            num_quantizers=num_quantizers,
            rotation_trick=rotation_trick,
            quantize_optimize_method=quantize_optimize_method,
        )
        self.final_layer = _TokenizerDecoder(
            n_dim=emb_dim,
            n_head=tokenizer_num_heads,
            n_filters=n_filters,
            ratios=list(ratios),
            kernel_size=kernel_size,
            last_kernel_size=last_kernel_size,
            dropout=drop_prob,
        )
        self.apply(_init_weights)

    @classmethod
    def from_opentslab_config(cls, config: dict, **kwargs) -> BrainTokenizer:
        """Construct from the authors' released ``model_cfg.json``.

        Parameters
        ----------
        config : dict
            Parsed official tokenizer configuration. The input is not mutated.
        **kwargs : dict
            Braindecode metadata such as ``chs_info``, ``n_times``, and
            ``sfreq``, or explicit overrides for translated configuration values.

        Returns
        -------
        BrainTokenizer
            A tokenizer configured for the accompanying official checkpoint.
        """
        translated = _translate_opentslab_config(
            config, renames=_TOKENIZER_CONFIG_RENAMES
        )
        translated.update(kwargs)
        return cls(**translated)

    def _unfold(self, x: torch.Tensor, overlap_ratio: float = 0.0) -> torch.Tensor:
        """Apply the released tokenizer's short/tail windowing policy."""
        step = _window_stride(self.window_length, overlap_ratio)
        if x.shape[-1] < self.window_length:
            x = F.pad(x, (0, self.window_length - x.shape[-1]))
        elif overlap_ratio > 0.0:
            remainder = (x.shape[-1] - self.window_length) % step
            if remainder:
                x = F.pad(x, (0, step - remainder))
        # ``unfold`` drops an incomplete non-overlap tail, as the release does.
        return x.unfold(dimension=-1, size=self.window_length, step=step)

    def load_state_dict(self, state_dict, *args, **kwargs):
        """Load either native or official OpenTSLab BrainTokenizer weights."""
        remapped = OrderedDict()
        metadata = getattr(state_dict, "_metadata", None)
        official = any(
            key.startswith("quantizer.rvq.")
            or key.startswith("decoder.")
            or ".conv.conv." in key
            for key in state_dict
        )
        for key, value in state_dict.items():
            new_key = _rename_official_key(key)
            if new_key is None:
                continue
            if new_key in remapped:
                raise ValueError(
                    f"Checkpoint keys collide after remapping: {new_key!r}."
                )
            remapped[new_key] = value
        if official:
            remapped["pos"] = self.pos
            remapped["sensor_type"] = self.sensor_type
        if metadata is not None:
            remapped._metadata = metadata
        return super().load_state_dict(remapped, *args, **kwargs)

    def _encode_quantize(self, x: torch.Tensor, overlap_ratio: float = 0.0):
        """Unfold -> sensor-embed -> encode -> RVQ -> (feat_q, indices, commit, se)."""
        xu = self._unfold(
            x, overlap_ratio
        )  # (batch, n_chans, n_windows, window_length)
        # Sensor embedding is batch-independent: compute once, broadcast to B.
        se = self.sensor_embed(self.pos, self.sensor_type)  # (n_chans, emb_dim)
        se = se.unsqueeze(0).expand(xu.shape[0], -1, -1)  # (batch, n_chans, emb_dim)
        feat = self.encoder(xu, se)  # (batch, n_neuro, n_windows, n_tokens, emb_dim)
        feat_q, indices, commit = self.quantizer(feat)
        return feat_q, indices, commit, se

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Reconstruct ``x``, zero-filling a dropped non-overlap tail."""
        feat_q, _, _, se = self._encode_quantize(x)
        recon = self.final_layer(
            feat_q, se
        )  # (batch, n_chans, n_windows, window_length)
        recon = recon.reshape(recon.shape[0], recon.shape[1], -1)
        if recon.shape[-1] < x.shape[-1]:
            recon = F.pad(recon, (0, x.shape[-1] - recon.shape[-1]))
        return recon[..., : x.shape[-1]]

    def encode_decode(self, x: torch.Tensor):
        """Return reconstruction, commitment loss, and codebook indices."""
        feat_q, indices, commit, se = self._encode_quantize(x)
        recon = self.final_layer(feat_q, se)
        recon = recon.reshape(recon.shape[0], recon.shape[1], -1)
        if recon.shape[-1] < x.shape[-1]:
            recon = F.pad(recon, (0, x.shape[-1] - recon.shape[-1]))
        recon = recon[..., : x.shape[-1]]
        return recon, commit, indices

    @torch.no_grad()
    def tokenize(self, x: torch.Tensor, overlap_ratio: float = 0.0):
        """Encode ``x`` into quantised features and discrete codebook indices.

        Parameters
        ----------
        x : torch.Tensor
            Input signal, shape ``(batch, n_chans, n_times)``.
        overlap_ratio : float
            Fraction of overlap between consecutive windows (0 = no overlap).

        Returns
        -------
        feat : torch.Tensor
            Quantised features, shape ``(batch, n_neuro, n_windows * n_tokens, emb_dim)``.
        indices : torch.Tensor
            Codebook indices, shape ``(batch, n_neuro, n_windows * n_tokens, num_quantizers)``.

        Notes
        -----
        This method internally switches the tokenizer to eval mode before
        encoding so that VQ codebooks are never EMA-updated (which would
        corrupt them) and dropout is disabled for deterministic output. A fresh
        training-from-scratch codebook still performs the official one-time
        K-means initialization on its first input. The prior training/eval mode
        is restored when the call returns, so calling ``tokenize`` while the
        module is in train mode is safe::

            model.train()
            feat, indices = model.tokenize(x)  # codebooks frozen, mode restored
        """
        was_training = self.training
        self.eval()  # freeze VQ codebooks (no EMA) and disable dropout
        try:
            feat_q, indices, _, _ = self._encode_quantize(x, overlap_ratio)
        finally:
            if was_training:
                self.train()
        # Merge windows (nwin) and per-window tokens (tok) into a single token sequence.
        feat = rearrange(
            feat_q, "batch chans nwin tok dim -> batch chans (nwin tok) dim"
        )
        indices = rearrange(
            indices, "batch chans nwin tok nquant -> batch chans (nwin tok) nquant"
        )
        return feat, indices


def _rename_official_key(key: str) -> str | None:
    """Map an OpenTSLab state-dict key to this port's name.

    Native keys pass through unchanged.
    """
    key = key.replace("quantizer.rvq.", "quantizer.")
    if key.startswith("decoder."):
        key = "final_layer." + key.removeprefix("decoder.")
    key = key.replace(".convtr.convtr.", ".convtr.")
    key = key.replace(".conv.conv.", ".conv.")
    # Released FeedForward is Sequential(Linear, SELU, Linear, Dropout) under
    # ``layer``; FeedForwardBlock numbers its linears 0 and 3.
    key = key.replace("aggregate_mlp.layer.0.", "aggregate_mlp.0.")
    key = key.replace("aggregate_mlp.layer.2.", "aggregate_mlp.3.")
    return key


def _window_stride(window_length: int, overlap_ratio: float) -> int:
    """Validate a BrainOmni overlap and return its integer sample stride."""
    if not 0.0 <= overlap_ratio < 1.0:
        raise ValueError(
            f"overlap_ratio must satisfy 0 <= overlap_ratio < 1, got {overlap_ratio}."
        )
    stride = int(window_length * (1 - overlap_ratio))
    if stride < 1:
        raise ValueError(
            f"overlap_ratio={overlap_ratio} produces a zero-sample stride for "
            f"window_length={window_length}."
        )
    return stride


def _init_weights(module: nn.Module) -> None:
    r"""BrainOmni weight init, faithful to the upstream ``_init_weights``.

    Linear and embedding weights ~ ``trunc_normal_(std=0.02)`` (biases zeroed),
    affine ``RMSNorm`` weights set to 1. Applied via ``self.apply`` at the end of
    ``__init__``; ``weight_norm`` convolutions are deliberately left untouched so
    their ``weight_g``/``weight_v`` parametrization stays intact.
    """
    if isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)
    elif isinstance(module, nn.Embedding):
        nn.init.trunc_normal_(module.weight, std=0.02)
    elif isinstance(module, RMSNorm) and isinstance(module.weight, nn.Parameter):
        nn.init.constant_(module.weight, 1.0)


_SENSOR_CODE = {"eeg": 0, "mag": 1, "grad": 2}


def _normalize_pos(pos: np.ndarray, sensor_type: np.ndarray) -> np.ndarray:
    """Per-modality position normalization (BrainOmni ``normalize_pos``).

    EEG (code 0) and MEG (codes 1, 2) positions are each mean-centered then
    divided by ``sqrt(3 * mean(squared_norm))``.
    """
    pos = pos.copy()
    eeg = sensor_type == 0
    meg = (sensor_type == 1) | (sensor_type == 2)
    for mask in (eeg, meg):
        if not mask.any():
            continue
        xyz = pos[mask, :3]
        xyz = xyz - xyz.mean(axis=0, keepdims=True)
        scale = np.sqrt(3 * np.mean(np.sum(xyz**2, axis=1)))
        scale = scale if scale > 0 else 1.0
        pos[mask, :3] = xyz / scale
    return pos


def _geometry_from_chs_info(chs_info):
    """Derive ``(pos (n_chans, 6) float32, sensor_type (n_chans,) int64)`` from ``chs_info``.

    Positions come from ``extract_channel_locations_from_chs_info`` and the
    EEG/MAG/GRAD type from :func:`mne.channel_type`; the MEG coil orientation and
    the per-modality normalization follow the BrainOmni convention. Raises if any
    channel lacks a finite position.
    """
    # mne.channel_type owns the FIFF kind/unit -> eeg/mag/grad logic (it only needs
    # ``info["chs"][idx]``). Lightweight dicts instead carry a resolved string in
    # ``ch_type`` or ``kind``, so use those directly when present.
    types = []
    for index, ch in enumerate(chs_info):
        resolved_type = ch.get("ch_type")
        if resolved_type is None and isinstance(ch.get("kind"), str):
            resolved_type = ch["kind"]
        if resolved_type is None:
            resolved_type = mne.channel_type({"chs": chs_info}, index)
        types.append(str(resolved_type).lower())

    xyz = extract_channel_locations_from_chs_info(chs_info)
    if xyz is None or len(xyz) != len(chs_info) or not np.isfinite(xyz).all():
        raise ValueError(
            "chs_info lacks finite sensor positions; call raw.set_montage(...)."
        )

    unsupported = set(types) - set(_SENSOR_CODE)
    if unsupported:
        raise ValueError(
            f"Unsupported channel type(s) {sorted(unsupported)}; pass only EEG/MEG."
        )
    sensor_type = np.array([_SENSOR_CODE[t] for t in types], dtype=np.int64)

    grad, mag = sensor_type == _SENSOR_CODE["grad"], sensor_type == _SENSOR_CODE["mag"]
    ori = np.zeros((len(chs_info), 3))  # EEG orientation stays zero
    for index in np.flatnonzero(grad):
        # The released source selects the in-plane x-axis only for VectorView
        # planar coils. Every other MEG coil (including axial gradiometers) uses
        # the coil-normal z-axis.
        coil_type = int(chs_info[index].get("coil_type", 0))
        axis_slice = slice(3, 6) if 3011 <= coil_type <= 3015 else slice(9, 12)
        axis = np.asarray(chs_info[index]["loc"], dtype=np.float64)[axis_slice]
        if axis.shape != (3,) or not np.isfinite(axis).all():
            raise ValueError("chs_info lacks a finite coil orientation for GRAD.")
        ori[index] = axis  # gradiometer: in-plane (ex) axis
    for index in np.flatnonzero(mag):
        axis = np.asarray(chs_info[index]["loc"], dtype=np.float64)[9:12]
        if axis.shape != (3,) or not np.isfinite(axis).all():
            raise ValueError("chs_info lacks a finite coil orientation for MAG.")
        ori[index] = axis  # magnetometer: coil normal (ez) axis

    pos = np.concatenate([xyz, ori], axis=1).astype(np.float32)
    return _normalize_pos(pos, sensor_type), sensor_type


def _safe_pad1d(
    x: torch.Tensor, padding: tuple[int, int], mode: str = "constant"
) -> torch.Tensor:
    """Pad 1-D inputs, extending very short signals before reflection."""
    padding_left, padding_right = padding
    if padding_left < 0 or padding_right < 0:
        raise ValueError(f"padding values must be non-negative, got {padding}.")
    if mode == "zero":
        mode = "constant"
    if mode != "reflect":
        return F.pad(x, padding, mode=mode)
    extra_padding = max(max(padding) - x.shape[-1] + 1, 0)
    if extra_padding:
        x = F.pad(x, (0, extra_padding))
    padded = F.pad(x, padding, mode="reflect")
    if extra_padding:
        padded = padded[..., :-extra_padding]
    return padded


def _unpad1d(x: torch.Tensor, padding: tuple[int, int]) -> torch.Tensor:
    """Remove asymmetric 1-D padding without the ``-0`` slicing trap."""
    padding_left, padding_right = padding
    if padding_left < 0 or padding_right < 0:
        raise ValueError(f"padding values must be non-negative, got {padding}.")
    if padding_left + padding_right > x.shape[-1]:
        raise ValueError(f"Cannot remove padding {padding} from length {x.shape[-1]}.")
    return x[..., padding_left : x.shape[-1] - padding_right]


class _SEANetConv1d(nn.Module):
    """Conv1d with built-in asymmetric / causal padding."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        causal: bool = False,
        norm: str = "none",  # kept for call-site compat; always weight_norm
        norm_kwargs: dict | None = None,  # kept for call-site compat; unused
        pad_mode: str = "reflect",
    ):
        super().__init__()
        if stride > 1 and dilation > 1:
            warnings.warn(
                "_SEANetConv1d has been initialized with stride > 1 and dilation > 1"
                f" (kernel_size={kernel_size} stride={stride}, dilation={dilation})."
            )
        # Classic weight_norm preserves checkpoint key parity with upstream.
        self.conv = weight_norm(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                dilation=dilation,
                groups=groups,
                bias=bias,
            )
        )
        self.causal = causal
        self.pad_mode = pad_mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        kernel_size = self.conv.kernel_size[0]
        stride = self.conv.stride[0]
        dilation = self.conv.dilation[0]
        padding_total = (kernel_size - 1) * dilation - (stride - 1)
        # Right-pad so the last strided window is full: round the effective
        # length up to the next multiple of ``stride`` (EnCodec convention).
        extra_padding = (kernel_size - padding_total - x.shape[-1]) % stride
        if self.causal:
            x = _safe_pad1d(x, (padding_total, extra_padding), mode=self.pad_mode)
        else:
            padding_right = padding_total // 2
            padding_left = padding_total - padding_right
            x = _safe_pad1d(
                x, (padding_left, padding_right + extra_padding), mode=self.pad_mode
            )
        return self.conv(x)


class _SEANetConvTranspose1d(nn.Module):
    """ConvTranspose1d with built-in asymmetric / causal padding trimming."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        causal: bool = False,
        norm: str = "none",  # kept for call-site compat; always weight_norm
        trim_right_ratio: float = 1.0,
        norm_kwargs: dict | None = None,  # kept for call-site compat; unused
    ):
        super().__init__()
        self.causal = causal
        self.trim_right_ratio = trim_right_ratio
        assert self.causal or self.trim_right_ratio == 1.0, (
            "`trim_right_ratio` != 1.0 only makes sense for causal convolutions"
        )
        assert 0.0 <= self.trim_right_ratio <= 1.0
        padding_total = kernel_size - stride
        self.convtr = weight_norm(
            nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride)
        )
        self._padding_total = padding_total

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.convtr(x)
        if self.causal:
            padding_right = math.ceil(self._padding_total * self.trim_right_ratio)
            padding_left = self._padding_total - padding_right
        else:
            padding_right = self._padding_total // 2
            padding_left = self._padding_total - padding_right
        return _unpad1d(y, (padding_left, padding_right))


class _SEANetLSTM(nn.Module):
    """LSTM over convolutional layout (channels-last time axis)."""

    def __init__(
        self,
        dimension: int,
        num_layers: int = 2,
        skip: bool = True,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.bidirectional = bidirectional
        self.skip = skip
        self.lstm = nn.LSTM(
            dimension, dimension, num_layers, bidirectional=bidirectional
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(2, 0, 1)
        y, _ = self.lstm(x)
        if self.bidirectional:
            x = x.repeat(1, 1, 2)
        if self.skip:
            y = y + x
        y = y.permute(1, 2, 0)
        return y


class _SEANetResBlock(nn.Module):
    """SEANet residual block (dilated convs + skip connection)."""

    def __init__(
        self,
        dim: int,
        kernel_sizes: list[int] | None = None,
        dilations: list[int] | None = None,
        activation: str = "ELU",
        activation_params: dict | None = None,
        norm: str = "weight_norm",
        norm_params: dict | None = None,
        causal: bool = False,
        pad_mode: str = "reflect",
        compress: int = 2,
        true_skip: bool = True,
    ):
        super().__init__()
        if kernel_sizes is None:
            kernel_sizes = [3, 1]
        if dilations is None:
            dilations = [1, 1]
        if activation_params is None:
            activation_params = {"alpha": 1.0}
        norm_params = norm_params or {}
        assert len(kernel_sizes) == len(dilations), (
            "Number of kernel sizes must match number of dilations"
        )
        act = getattr(nn, activation)
        hidden = dim // compress
        block: list[nn.Module] = []
        for i, (kernel_size, dilation) in enumerate(zip(kernel_sizes, dilations)):
            in_chs = dim if i == 0 else hidden
            out_chs = dim if i == len(kernel_sizes) - 1 else hidden
            block += [
                act(**activation_params),
                _SEANetConv1d(
                    in_chs,
                    out_chs,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    norm=norm,
                    norm_kwargs=norm_params,
                    causal=causal,
                    pad_mode=pad_mode,
                ),
            ]
        self.block = nn.Sequential(*block)
        self.shortcut: nn.Module
        if true_skip:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = _SEANetConv1d(
                dim,
                dim,
                kernel_size=1,
                norm=norm,
                norm_kwargs=norm_params,
                causal=causal,
                pad_mode=pad_mode,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.shortcut(x) + self.block(x)


class _SEANetEncoder(nn.Module):
    """SEANet encoder: strided conv stack (``prod(ratios)`` downsampling) + LSTM."""

    def __init__(
        self,
        channels: int = 1,
        dimension: int = 128,
        n_filters: int = 32,
        n_residual_layers: int = 1,
        ratios: list[int] | None = None,
        activation: str = "ELU",
        activation_params: dict | None = None,
        norm: str = "weight_norm",
        norm_params: dict | None = None,
        kernel_size: int = 7,
        last_kernel_size: int = 7,
        residual_kernel_size: int = 3,
        dilation_base: int = 2,
        causal: bool = False,
        pad_mode: str = "reflect",
        true_skip: bool = False,
        compress: int = 2,
        lstm: int = 2,
        bidirectional: bool = False,
    ):
        super().__init__()
        if ratios is None:
            ratios = [8, 5, 4, 2]
        if activation_params is None:
            activation_params = {"alpha": 1.0}
        norm_params = norm_params or {}
        self.channels = channels
        self.dimension = dimension
        self.n_filters = n_filters
        self.ratios = list(reversed(ratios))
        self.n_residual_layers = n_residual_layers
        self.hop_length = int(np.prod(self.ratios))

        act = getattr(nn, activation)
        mult = 1
        model: list[nn.Module] = [
            _SEANetConv1d(
                channels,
                mult * n_filters,
                kernel_size,
                norm=norm,
                norm_kwargs=norm_params,
                causal=causal,
                pad_mode=pad_mode,
            )
        ]
        for ratio in self.ratios:
            for j in range(n_residual_layers):
                model += [
                    _SEANetResBlock(
                        mult * n_filters,
                        kernel_sizes=[residual_kernel_size, 1],
                        dilations=[dilation_base**j, 1],
                        norm=norm,
                        norm_params=norm_params,
                        activation=activation,
                        activation_params=activation_params,
                        causal=causal,
                        pad_mode=pad_mode,
                        compress=compress,
                        true_skip=true_skip,
                    )
                ]
            model += [
                act(**activation_params),
                _SEANetConv1d(
                    mult * n_filters,
                    mult * n_filters * 2,
                    kernel_size=ratio * 2,
                    stride=ratio,
                    norm=norm,
                    norm_kwargs=norm_params,
                    causal=causal,
                    pad_mode=pad_mode,
                ),
            ]
            mult *= 2

        if lstm:
            model += [
                _SEANetLSTM(
                    mult * n_filters, num_layers=lstm, bidirectional=bidirectional
                )
            ]

        mult = mult * 2 if bidirectional else mult
        model += [
            act(**activation_params),
            _SEANetConv1d(
                mult * n_filters,
                dimension,
                last_kernel_size,
                norm=norm,
                norm_kwargs=norm_params,
                causal=causal,
                pad_mode=pad_mode,
            ),
        ]
        self.model = nn.Sequential(*model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class _SEANetDecoder(nn.Module):
    """SEANet decoder: mirror of the encoder (transposed-conv upsampling)."""

    def __init__(
        self,
        channels: int = 1,
        dimension: int = 128,
        n_filters: int = 32,
        n_residual_layers: int = 1,
        ratios: list[int] | None = None,
        activation: str = "ELU",
        activation_params: dict | None = None,
        final_activation: str | None = None,
        final_activation_params: dict | None = None,
        norm: str = "weight_norm",
        norm_params: dict | None = None,
        kernel_size: int = 7,
        last_kernel_size: int = 7,
        residual_kernel_size: int = 3,
        dilation_base: int = 2,
        causal: bool = False,
        pad_mode: str = "reflect",
        true_skip: bool = False,
        compress: int = 2,
        lstm: int = 2,
        trim_right_ratio: float = 1.0,
        bidirectional: bool = False,
    ):
        super().__init__()
        if ratios is None:
            ratios = [8, 5, 4, 2]
        if activation_params is None:
            activation_params = {"alpha": 1.0}
        norm_params = norm_params or {}
        self.dimension = dimension
        self.channels = channels
        self.n_filters = n_filters
        self.ratios = ratios
        self.n_residual_layers = n_residual_layers
        self.hop_length = int(np.prod(self.ratios))

        act = getattr(nn, activation)
        mult = int(2 ** len(self.ratios))
        model: list[nn.Module] = [
            _SEANetConv1d(
                dimension,
                mult * n_filters,
                kernel_size,
                norm=norm,
                norm_kwargs=norm_params,
                causal=causal,
                pad_mode=pad_mode,
            )
        ]

        if lstm:
            model += [
                _SEANetLSTM(
                    mult * n_filters, num_layers=lstm, bidirectional=bidirectional
                )
            ]

        for ratio in self.ratios:
            model += [
                act(**activation_params),
                _SEANetConvTranspose1d(
                    mult * n_filters,
                    mult * n_filters // 2,
                    kernel_size=ratio * 2,
                    stride=ratio,
                    norm=norm,
                    norm_kwargs=norm_params,
                    causal=causal,
                    trim_right_ratio=trim_right_ratio,
                ),
            ]
            for j in range(n_residual_layers):
                model += [
                    _SEANetResBlock(
                        mult * n_filters // 2,
                        kernel_sizes=[residual_kernel_size, 1],
                        dilations=[dilation_base**j, 1],
                        activation=activation,
                        activation_params=activation_params,
                        norm=norm,
                        norm_params=norm_params,
                        causal=causal,
                        pad_mode=pad_mode,
                        compress=compress,
                        true_skip=true_skip,
                    )
                ]
            mult //= 2

        model += [
            act(**activation_params),
            _SEANetConv1d(
                n_filters,
                channels,
                last_kernel_size,
                norm=norm,
                norm_kwargs=norm_params,
                causal=causal,
                pad_mode=pad_mode,
            ),
        ]
        if final_activation is not None:
            final_act = getattr(nn, final_activation)
            final_activation_params = final_activation_params or {}
            model += [final_act(**final_activation_params)]
        self.model = nn.Sequential(*model)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.model(z)


class _SensorEmbedding(nn.Module):
    """Embed per-channel position+orientation (n_chans, 6) and type (n_chans,) -> (n_chans, emb_dim)."""

    def __init__(self, n_dim: int) -> None:
        super().__init__()
        self.sensor_embedding_layer = nn.Embedding(3, n_dim)
        self.pos_embedding_layer = nn.Sequential(
            nn.Linear(6, n_dim // 2),
            nn.SELU(),
            nn.Linear(n_dim // 2, n_dim),
        )
        self.aggregate_mlp = FeedForwardBlock(n_dim, 4, 0.0, nn.SELU)
        self.norm = RMSNorm(n_dim, eps=1e-6)

    def forward(self, pos: torch.Tensor, sensor_type: torch.Tensor) -> torch.Tensor:
        x = self.pos_embedding_layer(pos)
        x = x + self.sensor_embedding_layer(sensor_type).type_as(x)
        x = x + self.aggregate_mlp(x)
        return self.norm(x)


class _ForwardSolution(nn.Module):
    """Cross-attention: sensor embeddings query the neural-token key/values."""

    def __init__(self, n_dim: int, n_head: int, dropout: float) -> None:
        super().__init__()
        assert n_dim % n_head == 0
        self.n_dim = n_dim
        self.n_head = n_head
        self.dropout = dropout
        self.kv = nn.Linear(n_dim, 2 * n_dim)
        self.proj = nn.Linear(n_dim, n_dim)

    def forward(
        self,
        sensor_embedding: torch.Tensor,
        neurons: torch.Tensor,
    ) -> torch.Tensor:
        batch, chans, _ = sensor_embedding.shape
        kv = self.kv(neurons)
        k, v = torch.split(kv, split_size_or_sections=self.n_dim, dim=-1)
        q = rearrange(
            sensor_embedding,
            "batch seq (heads head_dim) -> batch heads seq head_dim",
            heads=self.n_head,
        )
        k = rearrange(
            k,
            "batch seq (heads head_dim) -> batch heads seq head_dim",
            heads=self.n_head,
        )
        v = rearrange(
            v,
            "batch seq (heads head_dim) -> batch heads seq head_dim",
            heads=self.n_head,
        )
        output = (
            F.scaled_dot_product_attention(
                query=q,
                key=k,
                value=v,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=False,
            )
            .transpose(1, 2)
            .contiguous()
        )
        output = output.view(batch, chans, -1)
        return self.proj(output)


class _BackwardSolution(nn.Module):
    """Cross-attention: neural queries attend to pre-projected sensor keys/values."""

    def __init__(self, n_dim: int, n_head: int, dropout: float) -> None:
        super().__init__()
        assert n_dim % n_head == 0
        self.n_dim = n_dim
        self.n_head = n_head
        self.dropout = dropout
        self.v = nn.Linear(n_dim, n_dim)
        self.proj = nn.Linear(n_dim, n_dim)

    def forward(
        self,
        neuros: torch.Tensor,
        k: torch.Tensor,
        x: torch.Tensor,
    ) -> torch.Tensor:
        batch, n_queries, _ = neuros.shape
        q = rearrange(
            neuros,
            "batch seq (heads head_dim) -> batch heads seq head_dim",
            heads=self.n_head,
        )
        k = rearrange(
            k,
            "batch seq (heads head_dim) -> batch heads seq head_dim",
            heads=self.n_head,
        )
        v = rearrange(
            self.v(x),
            "batch seq (heads head_dim) -> batch heads seq head_dim",
            heads=self.n_head,
        )
        output = (
            F.scaled_dot_product_attention(
                query=q,
                key=k,
                value=v,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=False,
            )
            .transpose(1, 2)
            .contiguous()
        )
        output = output.view(batch, n_queries, -1)
        return self.proj(output)


class _TokenizerEncoder(nn.Module):
    """SEANet conv-encode windows, then collapse channels to ``n_neuro`` queries.

    ``(batch, n_chans, n_windows, window_length)`` -> ``(batch, n_neuro, n_windows, n_tokens, emb_dim)`` with ``T = L / prod(ratios)``.
    """

    def __init__(
        self,
        n_filters: int,
        ratios: list[int],
        kernel_size: int,
        last_kernel_size: int,
        n_dim: int,
        n_head: int,
        dropout: float,
        n_neuro: int,
    ) -> None:
        super().__init__()
        self.seanet_encoder = _SEANetEncoder(
            channels=1,
            dimension=n_dim,
            n_filters=n_filters,
            ratios=ratios,
            kernel_size=kernel_size,
            last_kernel_size=last_kernel_size,
        )
        self.neuros = nn.Parameter(torch.randn(n_neuro, n_dim))
        self.backwardsolution = _BackwardSolution(
            n_dim=n_dim, n_head=n_head, dropout=dropout
        )
        self.k_proj = nn.Linear(n_dim, n_dim)

    def forward(
        self,
        x: torch.Tensor,
        sensor_embedding: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        assert sensor_embedding is not None, "sensor_embedding is required"
        batch, chans, nwin, wlen = x.shape
        # Each (sample, channel, window) is an independent 1-D waveform for SEANet.
        x = rearrange(x, "batch chans nwin wlen -> (batch chans nwin) 1 wlen")
        x = self.seanet_encoder(x)
        # Re-group encoded windows back so tokens = nwin*tok is the joint time axis.
        x = rearrange(
            x,
            "(batch chans nwin) dim tok -> batch chans (nwin tok) dim",
            batch=batch,
            chans=chans,
            nwin=nwin,
        )
        batch, chans, tokens, _ = x.shape
        # Expand sensor embedding to match every time step, then flatten for cross-attention.
        sensor_embedding = rearrange(
            sensor_embedding.unsqueeze(2).repeat(1, 1, tokens, 1),
            "batch chans tokens dim -> (batch tokens) chans dim",
        )
        x = rearrange(x, "batch chans tokens dim -> (batch tokens) chans dim")
        neuros = self.neuros.type_as(x).unsqueeze(0).repeat(x.shape[0], 1, 1)
        x = self.backwardsolution(neuros, self.k_proj(x + sensor_embedding), x)
        # Collapse the n_neuro-channel axis back into the batch; split window from token dims.
        x = rearrange(
            x,
            "(batch nwin tok) chans dim -> batch chans (nwin tok) dim",
            batch=batch,
            nwin=nwin,
        )
        return rearrange(
            x, "batch chans (nwin tok) dim -> batch chans nwin tok dim", nwin=nwin
        )


class _TokenizerDecoder(nn.Module):
    """Expand ``n_neuro`` tokens back to channels, then SEANet conv-decode.

    ``(batch, n_neuro, n_windows, n_tokens, emb_dim)`` -> ``(batch, n_chans, n_windows, window_length)`` reconstructed waveforms.
    """

    def __init__(
        self,
        n_dim: int,
        n_head: int,
        n_filters: int,
        ratios: list[int],
        kernel_size: int,
        last_kernel_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.forwardsolution = _ForwardSolution(n_dim, n_head, dropout)
        self.seanet_decoder = _SEANetDecoder(
            channels=1,
            dimension=n_dim,
            n_filters=n_filters,
            ratios=ratios,
            kernel_size=kernel_size,
            last_kernel_size=last_kernel_size,
        )

    def forward(
        self,
        x: torch.Tensor,
        sensor_embedding: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert sensor_embedding is not None, "sensor_embedding is required"
        batch, chans, nwin, tok, dim = x.shape
        # Flatten batch × window × token into one axis so cross-attention sees
        # each (sample, window, token) independently across n_neuro latent channels.
        x = rearrange(x, "batch chans nwin tok dim -> (batch nwin tok) chans dim")
        # Match sensor embedding shape to the flattened batch axis.
        sensor_embedding = rearrange(
            sensor_embedding.view(batch, -1, 1, 1, dim).repeat(1, 1, nwin, tok, 1),
            "batch chans nwin tok dim -> (batch nwin tok) chans dim",
        )
        x = self.forwardsolution(sensor_embedding, x)
        # Regroup so each (sample, channel, window) is a separate 1-D sequence for SEANet.
        x = rearrange(
            x,
            "(batch nwin tok) chans dim -> (batch chans nwin) dim tok",
            batch=batch,
            nwin=nwin,
            tok=tok,
        )
        x = self.seanet_decoder(x)
        # Split the fused batch back into sample, channel, and window dimensions.
        return rearrange(
            x,
            "(batch chans nwin) 1 wlen -> batch chans nwin wlen",
            batch=batch,
            nwin=nwin,
        )
