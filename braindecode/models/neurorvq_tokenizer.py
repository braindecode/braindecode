# Authors: Konstantinos Barmpas et al. (original implementation)
#          Braindecode contributors (adaptation)
#
# License: CC BY-NC 4.0
# Adapted from https://github.com/KonstantinosBarmpas/NeuroRVQ (CC BY-NC 4.0).
"""NeuroRVQ residual-vector-quantized biosignal tokenizer."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from braindecode.functional import spectral_input
from braindecode.models.base import EEGModuleMixin
from braindecode.models.neurorvq import (
    _Block,
    _channel_slots,
    _modality,
    _MultiScaleTemporalConv,
)
from braindecode.modules.quantization import _all_reduce_sum


class _EMAEmbedding(nn.Module):
    """Codebook under the released key names, initialized by cosine k-means."""

    def __init__(self, n_codes: int, code_dim: int):
        super().__init__()
        weights = torch.zeros(n_codes, code_dim)
        self.weight = nn.Parameter(weights, requires_grad=False)
        self.cluster_size = nn.Parameter(torch.zeros(n_codes), requires_grad=False)
        self.embed_avg = nn.Parameter(weights.clone(), requires_grad=False)
        self.register_buffer("initted", torch.tensor([0.0]))

    @torch.no_grad()
    def initialize(self, vectors: Tensor) -> None:
        if bool(self.initted.item()):
            return
        n_codes, n_samples = self.weight.shape[0], vectors.shape[0]
        if n_samples >= n_codes:
            indices = torch.randperm(n_samples, device=vectors.device)[:n_codes]
        else:
            indices = torch.randint(n_samples, (n_codes,), device=vectors.device)
        means = vectors[indices]
        for _ in range(10):
            assignments = (vectors @ means.T).argmax(dim=-1)
            counts = torch.bincount(assignments, minlength=n_codes)
            sums = torch.zeros_like(means)
            sums.index_add_(0, assignments, vectors)
            updated = F.normalize(sums / counts.clamp_min(1).unsqueeze(-1), dim=-1)
            means = torch.where(counts[:, None] == 0, means, updated)
        self.weight.copy_(means)
        self.cluster_size.copy_(counts)
        self.initted.fill_(True)


class _EMAVectorQuantizer(nn.Module):
    """EMA vector quantizer on L2-normalized vectors (cosine codebook)."""

    def __init__(self, n_codes: int, code_dim: int, statistic_code_usage=False):
        super().__init__()
        self.decay = 0.99
        self.statistic_code_usage = statistic_code_usage
        self.embedding = _EMAEmbedding(n_codes, code_dim)
        self.register_buffer("cluster_size", torch.zeros(n_codes))

    def _indices(self, vectors: Tensor) -> Tensor:
        self.embedding.initialize(vectors)
        weight = self.embedding.weight
        distances = (
            vectors.square().sum(dim=1, keepdim=True)
            + weight.square().sum(dim=1)
            - 2 * vectors @ weight.T
        )
        return distances.argmin(dim=1)

    def encode(self, z: Tensor) -> Tensor:
        z = F.normalize(z.permute(0, 2, 3, 1), dim=-1)
        return self._indices(z.reshape(-1, z.shape[-1]))

    def forward(self, z: Tensor):
        z = F.normalize(z.permute(0, 2, 3, 1), dim=-1)
        vectors = z.reshape(-1, z.shape[-1])
        indices = self._indices(vectors)
        quantized = F.embedding(indices, self.embedding.weight).view_as(z)

        if self.training or self.statistic_code_usage:
            with torch.no_grad():
                weight = self.embedding.weight
                encodings = F.one_hot(indices, weight.shape[0]).to(z.dtype)
                counts = encodings.sum(0)
                _all_reduce_sum(counts)
                self.cluster_size.mul_(self.decay).add_(counts, alpha=1 - self.decay)
                if self.training:
                    safe_counts = counts.masked_fill(counts == 0, 1.0)
                    embed_sum = vectors.T @ encodings
                    _all_reduce_sum(embed_sum)
                    means = F.normalize(
                        (embed_sum / safe_counts.unsqueeze(0)).T, dim=-1
                    )
                    means = torch.where(counts[:, None] == 0, weight, means)
                    weight.mul_(self.decay).add_(means, alpha=1 - self.decay)
                    weight.copy_(F.normalize(weight, dim=-1))

        quantized = z + (quantized - z).detach()
        return quantized.permute(0, 3, 1, 2).contiguous(), indices


class _ResidualVectorQuantizer(nn.Module):
    def __init__(self, n_quantizers: int, *args):
        super().__init__()
        self.layers = nn.ModuleList(
            [_EMAVectorQuantizer(*args) for _ in range(n_quantizers)]
        )

    def forward(self, x: Tensor):
        quantized_out = torch.zeros_like(x)
        residual = x
        codes = []
        for layer in self.layers:
            quantized, indices = layer(residual)
            residual = residual - quantized
            quantized_out = quantized_out + quantized
            codes.append(indices)
        return quantized_out, torch.stack(codes)

    def encode(self, x: Tensor) -> Tensor:
        """Codes without the straight-through estimator or EMA updates."""
        residual = x
        codes = []
        for layer in self.layers:
            indices = layer.encode(residual)
            quantized = F.embedding(indices, layer.embedding.weight).view(
                x.shape[0], x.shape[2], x.shape[3], x.shape[1]
            )
            residual = residual - quantized.permute(0, 3, 1, 2).contiguous()
            codes.append(indices)
        return torch.stack(codes)


class _PatchProjection(nn.Module):
    def __init__(self, in_chans: int, embed_dim: int):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        x = self.proj(x)
        return x.permute(0, 2, 3, 1).reshape(x.shape[0], -1, x.shape[1])


class _BranchTransformer(nn.Module):
    """Transformer shared by the four scales, with one ``fc_norm_i`` per scale.

    ``patch_embeds`` are registered right after ``cls_token`` so the state-dict
    layout matches the released checkpoint.
    """

    def __init__(
        self,
        patch_embeds: dict[str, nn.Module],
        *,
        n_channels: int,
        max_patches: int,
        embed_dim: int,
        depth: int,
        num_heads: int,
        drop_prob: float,
        attn_drop_rate: float,
        drop_path_rate: float,
        init_values: float,
    ):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        for name, module in patch_embeds.items():
            setattr(self, name, module)
        # One spatial slot per pretrained channel plus slot 0 for the class token.
        self.pos_embed = nn.Parameter(torch.zeros(n_channels + 1, embed_dim))
        self.time_embed = nn.Parameter(torch.zeros(max_patches, embed_dim))
        self.pos_drop = nn.Dropout(drop_prob)
        drop_paths = torch.linspace(0, drop_path_rate, depth).tolist()
        self.blocks = nn.ModuleList(
            [
                _Block(
                    embed_dim,
                    num_heads,
                    4.0,
                    True,
                    nn.LayerNorm,
                    drop_prob,
                    attn_drop_rate,
                    drop_paths[i],
                    init_values,
                )
                for i in range(depth)
            ]
        )
        for i in range(1, 5):
            setattr(self, f"fc_norm_{i}", nn.LayerNorm(embed_dim))

    def embeddings(self, time_indices: Tensor, spatial_indices: Tensor):
        """Spatial (class-token slot 0 prepended) and temporal embeddings."""
        spatial = self.pos_embed[F.pad(spatial_indices, (1, 0), value=0)]
        return spatial, self.time_embed[time_indices]

    def forward(
        self, tokens: Tensor, spatial: Tensor, temporal: Tensor, branch: int
    ) -> Tensor:
        """Encode ``(batch, n_tokens, embed_dim)`` tokens of scale ``branch`` (1-4)."""
        tokens = torch.cat(
            (self.cls_token.expand(tokens.shape[0], -1, -1), tokens), dim=1
        )
        tokens = tokens + spatial
        tokens[:, 1:] = tokens[:, 1:] + temporal
        tokens = self.pos_drop(tokens)
        for block in self.blocks:
            tokens = block(tokens)
        return getattr(self, f"fc_norm_{branch}")(tokens[:, 1:])


class NeuroRVQTokenizer(EEGModuleMixin, nn.Module, license="cc-by-nc-4.0"):
    r"""NeuroRVQ multi-scale residual-vector-quantized biosignal tokenizer from Barmpas et al. [neurorvq]_.

    :bdg-success:`Convolution` :bdg-info:`Attention/Transformer` :bdg-danger:`Foundation Model` :bdg-dark-line:`Channel`

    The tokenizer encodes EEG, ECG, EMG or PPG patches through four temporal
    scales, quantizes each scale with a separate residual vector quantizer,
    and reconstructs the signal from amplitude and phase components. The
    Transformer blocks, temporal convolution, channel lists and ``modality``
    presets are shared with :class:`NeuroRVQ`.

    Inputs must be sampled at the modality's rate, contain complete patches
    and use channel names from the modality's pretrained list (see
    :class:`NeuroRVQ`). The model does not preprocess signals; the released
    EEG example applies notch filters at 50, 60 and 100 Hz, a 0.5-44.5 Hz
    band-pass, clipping at 500 uV and resampling to 200 Hz.

    .. important::
       **Pre-trained Weights Available**

       The released tokenizers (`ntinosbarmpas/NeuroRVQ
       <https://huggingface.co/ntinosbarmpas/NeuroRVQ>`_, revision ``d944b87``,
       CC BY-NC 4.0) are hosted as
       ``braindecode/neurorvq-tokenizer-eeg-pretrained``,
       ``braindecode/neurorvq-tokenizer-ecg-pretrained``,
       ``braindecode/neurorvq-tokenizer-emg-pretrained`` and
       ``braindecode/neurorvq-tokenizer-ppg-pretrained``; the foundation models
       are in ``braindecode/neurorvq-{eeg,ecg,emg}-pretrained``
       (:class:`NeuroRVQ`):

       .. code-block:: python

           from braindecode.models import NeuroRVQTokenizer

           model = NeuroRVQTokenizer.from_pretrained(
               "braindecode/neurorvq-tokenizer-eeg-pretrained", chs_info=raw.info["chs"]
           )

       Raw reconstruction MSE (Table 10 of [neurorvq]_), with the data preparation
       of the EEG-Benchmarking code linked from the NeuroRVQ repository: the port
       gives 0.0858 on Pavlov 2022 (paper: 0.084) and 0.0748 on High Gamma
       (paper: 0.090); the authors' released code gives the same numbers on the
       same data.

    .. versionadded:: 1.9

    `License <https://github.com/KonstantinosBarmpas/NeuroRVQ/blob/main/LICENSE>`_

    Parameters
    ----------
    n_chans : int
        Number of channels.
    n_times : int
        Number of samples; must be divisible by ``patch_size`` and no longer
        than ``patch_size * max_patches``.
    sfreq : float
        Sampling frequency; must be the modality's rate.
    channel_names : sequence of str or None
        Ordered channel names. If omitted, names are inferred from
        ``chs_info`` or default to the first channels in the pretrained order.
    modality : {"eeg", "ecg", "emg", "ppg"}, default="eeg"
        Released configuration (see :class:`NeuroRVQ`); also sets the
        defaults of ``patch_size``, ``max_patches`` and ``num_quantizers``.
    patch_size : int or None, default=None
        Samples per patch; ``None`` uses the modality's value.
    max_patches : int or None, default=None
        Length of the temporal embedding table; ``None`` uses the modality's
        value.
    out_chans : int, default=8
        Number of channels per temporal-convolution branch; must be divisible
        by four. The Transformer width is ``out_chans * patch_size // 8``.
    encoder_depth : int, default=12
        Number of shared encoder Transformer blocks.
    decoder_depth : int, default=3
        Number of shared decoder Transformer blocks.
    num_heads : int, default=10
        Number of attention heads in both Transformer stacks.
    n_code : int, default=8192
        Number of entries per EMA codebook.
    code_dim : int, default=128
        Dimension of each quantized latent vector.
    num_quantizers : int or None, default=None
        Number of residual codebooks per temporal scale (EMG 16, the others
        8); ``None`` uses the modality's value.
    drop_prob : float, default=0.0
        Dropout probability in the Transformer stacks.
    attn_drop_rate : float, default=0.0
        Attention-probability dropout.
    drop_path_rate : float, default=0.0
        Maximum stochastic-depth probability.
    init_values : float, default=0.0
        LayerScale initialization used by the tokenizer checkpoint.
    activation : type[nn.Module], default=nn.GELU
        Activation in the temporal patch embedding.
    statistic_code_usage : bool, default=False
        ``True`` lets eval forwards update the ``cluster_size`` code-usage EMA, as the authors' code.

    Output shape
    ------------
    ``forward`` returns the target and reconstruction patches, each z-scored
    per window, shaped ``(batch, n_chans * n_patches, patch_size)``; their mean
    squared difference is the reconstruction error. ``tokenize`` returns
    integer codes shaped ``(4, num_quantizers, batch, n_chans * n_patches)``.

    Examples
    --------
    >>> import torch
    >>> from braindecode.models import NeuroRVQTokenizer
    >>> model = NeuroRVQTokenizer(
    ...     n_chans=3, n_times=800, sfreq=200, channel_names=["C3", "Cz", "C4"]
    ... ).eval()
    >>> x = torch.randn(2, 3, 800)  # (batch, channels, 4 s at 200 Hz)
    >>> target, reconstruction = model(x)
    >>> target.shape  # (batch, n_chans * n_patches, patch_size)
    torch.Size([2, 12, 200])
    >>> mse = (target - reconstruction).square().mean()
    >>> model.tokenize(x).shape  # (scale, quantizer, batch, n_chans * n_patches)
    torch.Size([4, 8, 2, 12])

    References
    ----------
    .. [neurorvq] Barmpas et al. (2025). NeuroRVQ: Multi-Scale Biosignal
       Tokenization for Generative Foundation Models. https://arxiv.org/abs/2510.13068
    """

    def __init__(
        self,
        n_outputs: int | None = None,
        n_chans: int | None = None,
        chs_info=None,
        n_times: int | None = None,
        input_window_seconds: float | None = None,
        sfreq: float | None = None,
        *,
        channel_names: tuple[str, ...] | list[str] | None = None,
        modality: str = "eeg",
        patch_size: int | None = None,
        max_patches: int | None = None,
        out_chans: int = 8,
        encoder_depth: int = 12,
        decoder_depth: int = 3,
        num_heads: int = 10,
        n_code: int = 8192,
        code_dim: int = 128,
        num_quantizers: int | None = None,
        drop_prob: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        init_values: float = 0.0,
        activation: type[nn.Module] = nn.GELU,
        statistic_code_usage: bool = False,
    ):
        super().__init__(
            n_outputs=n_outputs,
            n_chans=n_chans,
            chs_info=chs_info,
            n_times=n_times,
            input_window_seconds=input_window_seconds,
            sfreq=sfreq,
        )
        del n_outputs, n_chans, chs_info, input_window_seconds
        preset = _modality(self, modality)
        patch_size = patch_size or preset["patch_size"]
        max_patches = max_patches or preset["max_patches"]
        num_quantizers = num_quantizers or preset["num_quantizers"]
        if self.n_times % patch_size:
            raise ValueError(f"n_times must be divisible by patch_size ({patch_size}).")
        self.num_patches = self.n_times // patch_size
        if self.num_patches > max_patches:
            raise ValueError(f"n_times supports at most {max_patches} patches.")
        embed_dim = out_chans * (patch_size // 8)
        # LaBraM's attention would silently floor the head width otherwise.
        if embed_dim % num_heads:
            raise ValueError(
                "The derived embedding width must be divisible by num_heads."
            )

        self.channel_names, slots = _channel_slots(
            self, channel_names, preset["channels"]
        )
        self.patch_size = patch_size
        self.max_patches = max_patches
        self.code_dim = code_dim
        self.num_quantizers = num_quantizers
        self.register_buffer("spatial_embedding_ix", slots, persistent=False)
        transformer: dict = dict(
            n_channels=len(preset["channels"]),
            max_patches=max_patches,
            embed_dim=embed_dim,
            num_heads=num_heads,
            drop_prob=drop_prob,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            init_values=init_values,
        )
        self.encoder = _BranchTransformer(
            {
                "patch_embed": _MultiScaleTemporalConv(
                    out_chans, activation, preset["kernels"]
                )
            },
            depth=encoder_depth,
            **transformer,
        )
        self.decoder = _BranchTransformer(
            {
                f"patch_embed_{i}": _PatchProjection(code_dim, embed_dim)
                for i in range(1, 5)
            },
            depth=decoder_depth,
            **transformer,
        )
        for i in range(1, 5):
            setattr(
                self,
                f"quantize_{i}",
                _ResidualVectorQuantizer(
                    num_quantizers, n_code, code_dim, statistic_code_usage
                ),
            )
            setattr(
                self,
                f"encode_task_layer_{i}",
                nn.Sequential(
                    nn.Linear(embed_dim, embed_dim),
                    nn.Tanh(),
                    nn.Linear(embed_dim, code_dim),
                ),
            )
        self.decode_task_layer_amplitude = nn.Sequential(
            nn.Linear(4 * embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, patch_size),
        )
        self.decode_task_layer_angle_sin = nn.Sequential(
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Tanh(),
            nn.Linear(embed_dim, patch_size),
            nn.Tanh(),
        )
        self.decode_task_layer_angle_cos = nn.Sequential(
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Tanh(),
            nn.Linear(embed_dim, patch_size),
            nn.Tanh(),
        )
        self.final_layer = nn.Identity()
        task_layers = [getattr(self, f"encode_task_layer_{i}") for i in range(1, 5)]
        task_layers += [
            self.decode_task_layer_amplitude,
            self.decode_task_layer_angle_sin,
            self.decode_task_layer_angle_cos,
        ]
        for layer in task_layers:
            for module in layer:
                if isinstance(module, nn.Linear):
                    nn.init.trunc_normal_(module.weight, std=0.02)
                    nn.init.zeros_(module.bias)

    def _embedding_indices(self, device: torch.device):
        time = torch.arange(
            self.max_patches - self.num_patches,
            self.max_patches,
            device=device,
        ).repeat(self.n_chans)
        spatial = self.spatial_embedding_ix.to(device).repeat_interleave(
            self.num_patches
        )
        return time.unsqueeze(0), spatial.unsqueeze(0)

    def _patches(self, x: Tensor) -> Tensor:
        """Validate ``(batch, n_chans, n_times)`` input and split it into patches."""
        if x.ndim != 3 or tuple(x.shape[1:]) != (self.n_chans, self.n_times):
            raise ValueError(
                f"Expected input shape (batch, {self.n_chans}, {self.n_times}), "
                f"got {tuple(x.shape)}."
            )
        return x.reshape(x.shape[0], self.n_chans, self.num_patches, self.patch_size)

    def _branch_latents(self, patches: Tensor, time: Tensor, spatial: Tensor):
        """Project the four encoder scales to ``(batch, code_dim, n_chans, n_patches)``."""
        batch, channels, n_patches, _ = patches.shape
        branches = self.encoder.patch_embed(patches)
        spatial, temporal = self.encoder.embeddings(time, spatial)
        features = [
            self.encoder(branch, spatial, temporal, i)
            for i, branch in enumerate(branches, start=1)
        ]
        latents = []
        for i, feature in enumerate(features, start=1):
            latent = getattr(self, f"encode_task_layer_{i}")(feature)
            latent = latent.reshape(batch, channels, n_patches, self.code_dim)
            latents.append(latent.permute(0, 3, 1, 2).contiguous())
        return latents

    def _encode(self, x: Tensor, time: Tensor, spatial: Tensor):
        quantized, codes = [], []
        for i, latent in enumerate(self._branch_latents(x, time, spatial), start=1):
            q, branch_codes = getattr(self, f"quantize_{i}")(latent)
            quantized.append(q)
            codes.append(branch_codes.reshape(self.num_quantizers, x.shape[0], -1))
        return quantized, torch.stack(codes)

    @torch.no_grad()
    def tokenize(self, x: Tensor) -> Tensor:
        """Return four-scale codes without applying EMA updates.

        A cold codebook is initialized once from the input vectors by cosine
        k-means; load pretrained weights first for pretrained codes.
        """
        patches = self._patches(x)
        time, spatial = self._embedding_indices(x.device)
        scale_codes = []
        for i, latent in enumerate(
            self._branch_latents(patches, time, spatial), start=1
        ):
            codes = getattr(self, f"quantize_{i}").encode(latent)
            scale_codes.append(codes.reshape(self.num_quantizers, x.shape[0], -1))
        return torch.stack(scale_codes)

    @staticmethod
    def _standardize(x: Tensor):
        mean = x.mean(dim=(1, 2, 3), keepdim=True)
        std = torch.sqrt(x.var(dim=(1, 2, 3), keepdim=True).clamp_min(1e-8))
        return (x - mean) / std, mean, std

    def forward(self, x: Tensor):
        """Return the per-window z-scored target and reconstruction."""
        patches = self._patches(x)
        time, spatial = self._embedding_indices(x.device)
        spectrum = torch.fft.fft(spectral_input(patches), dim=-1)
        amplitude = torch.log1p(spectrum.abs()).to(patches)
        amplitude, amp_mean, amp_std = self._standardize(amplitude)
        quantized, _ = self._encode(patches, time, spatial)
        features = []
        for i, q in enumerate(quantized, start=1):
            tokens = getattr(self.decoder, f"patch_embed_{i}")(q)
            embeddings = self.decoder.embeddings(time, spatial)
            features.append(self.decoder(tokens, *embeddings, i))
        decoded = torch.cat(features, dim=-1)
        rec_amp = self.decode_task_layer_amplitude(decoded)
        rec_sin = self.decode_task_layer_angle_sin(decoded).reshape_as(patches)
        rec_cos = self.decode_task_layer_angle_cos(decoded).reshape_as(patches)
        # Undo the log-amplitude standardization, then invert the patch FFT
        # from the amplitude and the predicted phase (cos, sin).
        rec_amp = torch.expm1(rec_amp.reshape_as(patches) * amp_std + amp_mean)
        reconstructed = torch.fft.ifft(
            torch.complex(
                spectral_input(rec_amp * rec_cos), spectral_input(rec_amp * rec_sin)
            ),
            dim=-1,
        ).real.to(patches)
        target_std, _, _ = self._standardize(patches)
        # The authors z-score a contiguous (batch, tokens, 1, patch) copy; the
        # reduction order (float rounding) depends on that shape and layout.
        reconstructed_std, _, _ = self._standardize(
            reconstructed.reshape(x.shape[0], -1, 1, self.patch_size).contiguous()
        )
        return (
            target_std.reshape(x.shape[0], self.n_chans * self.num_patches, -1),
            reconstructed_std.reshape(x.shape[0], self.n_chans * self.num_patches, -1),
        )
