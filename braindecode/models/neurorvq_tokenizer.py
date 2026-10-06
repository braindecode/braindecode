# Authors: Konstantinos Barmpas et al. (original implementation)
#          Braindecode contributors (adaptation)
#
# License: CC BY-NC 4.0
"""NeuroRVQ residual-vector-quantized EEG tokenizer."""

from __future__ import annotations

from functools import partial

import torch
import torch.distributed as distributed
import torch.nn.functional as F
from torch import Tensor, nn

from braindecode.models.base import HAS_HF_HUB, EEGModuleMixin, huggingface_hub
from braindecode.models.neurorvq import (
    _PRETRAINED_REPO_ID,
    _PRETRAINED_REVISION,
    NEURORVQ_CHANNELS,
    _Block,
    _MultiScaleTemporalConv,
    _remap_mlp_state_dict_keys,
)

_PRETRAINED_FILENAME = "pretrained_models/tokenizers/NeuroRVQ_EEG_tokenizer_v1.pt"


def _l2norm(x: Tensor) -> Tensor:
    return F.normalize(x, p=2, dim=-1)


def _kmeans(samples: Tensor, n_clusters: int, n_iters: int = 10):
    """Cosine k-means initialization used by the released codebooks."""
    n_samples = samples.shape[0]
    if n_samples >= n_clusters:
        indices = torch.randperm(n_samples, device=samples.device)[:n_clusters]
    else:
        indices = torch.randint(n_samples, (n_clusters,), device=samples.device)
    means = samples[indices].clone()
    for _ in range(n_iters):
        assignments = (samples @ means.T).argmax(dim=-1)
        counts = torch.bincount(assignments, minlength=n_clusters)
        sums = torch.zeros_like(means)
        sums.index_add_(0, assignments, samples)
        updated = _l2norm(sums / counts.clamp_min(1).unsqueeze(-1))
        means = torch.where(counts[:, None] == 0, means, updated)
    return means, counts


class _EMAEmbedding(nn.Module):
    def __init__(self, n_codes: int, code_dim: int, decay: float, kmeans_init: bool):
        super().__init__()
        weights = (
            torch.zeros(n_codes, code_dim)
            if kmeans_init
            else _l2norm(torch.randn(n_codes, code_dim))
        )
        self.n_codes = n_codes
        self.decay = decay
        self.weight = nn.Parameter(weights, requires_grad=False)
        self.cluster_size = nn.Parameter(torch.zeros(n_codes), requires_grad=False)
        self.embed_avg = nn.Parameter(weights.clone(), requires_grad=False)
        self.register_buffer(
            "initted", torch.tensor([not kmeans_init], dtype=torch.float32)
        )

    @torch.no_grad()
    def initialize(self, vectors: Tensor) -> None:
        if bool(self.initted.item()):
            return
        means, counts = _kmeans(vectors, self.n_codes)
        self.weight.copy_(means)
        self.cluster_size.copy_(counts)
        self.initted.fill_(True)


class _EMAVectorQuantizer(nn.Module):
    def __init__(
        self,
        n_codes: int,
        code_dim: int,
        beta: float = 1.0,
        decay: float = 0.99,
        eps: float = 1e-5,
        kmeans_init: bool = True,
    ):
        super().__init__()
        self.num_tokens = n_codes
        self.codebook_dim = code_dim
        self.beta = beta
        self.decay = decay
        self.eps = eps
        self.embedding = _EMAEmbedding(n_codes, code_dim, decay, kmeans_init)
        self.register_buffer("cluster_size", torch.zeros(n_codes))

    def _indices(self, vectors: Tensor) -> Tensor:
        self.embedding.initialize(vectors)
        distances = (
            vectors.square().sum(dim=1, keepdim=True)
            + self.embedding.weight.square().sum(dim=1)
            - 2 * vectors @ self.embedding.weight.T
        )
        return distances.argmin(dim=1)

    def encode(self, z: Tensor) -> Tensor:
        z = _l2norm(z.permute(0, 2, 3, 1))
        return self._indices(z.reshape(-1, self.codebook_dim))

    def decode(self, indices: Tensor) -> Tensor:
        return F.embedding(indices, self.embedding.weight)

    def forward(self, z: Tensor):
        z = _l2norm(z.permute(0, 2, 3, 1))
        vectors = z.reshape(-1, self.codebook_dim)
        indices = self._indices(vectors)
        quantized = self.decode(indices).view_as(z)
        encodings = F.one_hot(indices, self.num_tokens).to(z.dtype)

        if self.training:
            with torch.no_grad():
                counts = encodings.sum(0)
                if distributed.is_available() and distributed.is_initialized():
                    distributed.all_reduce(counts)
                self.cluster_size.mul_(self.decay).add_(counts, alpha=1 - self.decay)
                safe_counts = counts.masked_fill(counts == 0, 1.0)
                embed_sum = vectors.T @ encodings
                if distributed.is_available() and distributed.is_initialized():
                    distributed.all_reduce(embed_sum)
                means = _l2norm((embed_sum / safe_counts.unsqueeze(0)).T)
                means = torch.where(counts[:, None] == 0, self.embedding.weight, means)
                self.embedding.weight.mul_(self.decay).add_(means, alpha=1 - self.decay)
                self.embedding.weight.copy_(_l2norm(self.embedding.weight))

        loss = self.beta * F.mse_loss(quantized.detach(), z)
        quantized = z + (quantized - z).detach()
        quantized = quantized.permute(0, 3, 1, 2).contiguous()
        return quantized, loss, indices


class _ResidualVectorQuantizer(nn.Module):
    def __init__(self, n_quantizers: int, n_codes: int, code_dim: int):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                _EMAVectorQuantizer(n_codes, code_dim, kmeans_init=True)
                for _ in range(n_quantizers)
            ]
        )

    def forward(self, x: Tensor):
        quantized_out = torch.zeros_like(x)
        residual = x
        codes, losses = [], []
        for layer in self.layers:
            quantized, loss, indices = layer(residual)
            residual = residual - quantized
            quantized_out = quantized_out + quantized
            losses.append(loss + 0.4 * F.mse_loss(quantized, residual.detach()))
            codes.append(indices)
        return quantized_out, torch.stack(codes), torch.stack(losses).mean()

    def encode(self, x: Tensor) -> tuple[Tensor, Tensor]:
        residual = x
        quantized_out = torch.zeros_like(x)
        codes = []
        for layer in self.layers:
            indices = layer.encode(residual)
            quantized = layer.decode(indices).view(
                x.shape[0], x.shape[2], x.shape[3], x.shape[1]
            )
            quantized = quantized.permute(0, 3, 1, 2).contiguous()
            residual = residual - quantized
            quantized_out = quantized_out + quantized
            codes.append(indices)
        return quantized_out, torch.stack(codes)


class _PatchProjection(nn.Module):
    def __init__(self, in_chans: int, embed_dim: int):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        x = self.proj(x)
        return x.permute(0, 2, 3, 1).reshape(x.shape[0], -1, x.shape[1])


def _transformer_blocks(
    *,
    embed_dim: int,
    depth: int,
    num_heads: int,
    drop_prob: float,
    attn_drop_rate: float,
    drop_path_rate: float,
    init_values: float,
) -> nn.ModuleList:
    """Pre-norm Transformer blocks of the NeuroRVQ FM (qk-norm, q/v bias, MLP ratio 4)."""
    drop_paths = torch.linspace(0, drop_path_rate, depth).tolist()
    qk_norm = partial(nn.LayerNorm, eps=1e-6)
    return nn.ModuleList(
        [
            _Block(
                embed_dim,
                num_heads,
                4.0,
                True,
                qk_norm,
                drop_prob,
                attn_drop_rate,
                drop_paths[i],
                init_values,
            )
            for i in range(depth)
        ]
    )


class _BranchTransformer(nn.Module):
    """Embeddings, Transformer blocks and per-branch heads shared by the encoder and decoder.

    Subclasses register ``cls_token`` and their patch embedding first, then call
    :meth:`_init_transformer`, so the parameter order (and the state-dict layout of
    the released checkpoint) is ``cls_token, patch_embed*, pos_embed, time_embed,
    blocks, fc_norm_i/head_i``.
    """

    def _init_transformer(
        self,
        *,
        max_patches: int,
        embed_dim: int,
        depth: int,
        num_heads: int,
        drop_prob: float,
        attn_drop_rate: float,
        drop_path_rate: float,
        init_values: float,
    ) -> None:
        # One spatial slot per montage electrode plus slot 0 for the class token.
        self.pos_embed = nn.Parameter(
            torch.zeros(len(NEURORVQ_CHANNELS) + 1, embed_dim)
        )
        self.time_embed = nn.Parameter(torch.zeros(max_patches, embed_dim))
        self.pos_drop = nn.Dropout(drop_prob)
        self.blocks = _transformer_blocks(
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            drop_prob=drop_prob,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            init_values=init_values,
        )
        self.norm = nn.Identity()
        for i in range(1, 5):
            setattr(self, f"fc_norm_{i}", nn.LayerNorm(embed_dim))
            setattr(self, f"head_{i}", nn.Identity())

    def _embeddings(
        self, time_indices: Tensor, spatial_indices: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Look up the spatial (with class-token slot 0) and temporal embeddings."""
        spatial = self.pos_embed[F.pad(spatial_indices, (1, 0), value=0)]
        return spatial, self.time_embed[time_indices]

    def _run_transformer(
        self, tokens: Tensor, spatial: Tensor, temporal: Tensor, branch: int
    ) -> Tensor:
        """Embed ``(batch, n_tokens, dim)`` patch tokens and apply branch ``branch`` (1-4)."""
        tokens = torch.cat(
            (self.cls_token.expand(tokens.shape[0], -1, -1), tokens), dim=1
        )
        tokens = tokens + spatial
        tokens[:, 1:] = tokens[:, 1:] + temporal
        tokens = self.pos_drop(tokens)
        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.norm(tokens[:, 1:])
        return getattr(self, f"head_{branch}")(
            getattr(self, f"fc_norm_{branch}")(tokens)
        )


class _NeuroRVQEncoder(_BranchTransformer):
    """Multi-scale EEG encoder with state-dict names matching NeuroRVQ v1.

    The four temporal-convolution scales share one Transformer; each scale
    returns its own ``(batch, n_chans * n_patches, embed_dim)`` features.
    """

    def __init__(
        self,
        *,
        max_patches: int,
        embed_dim: int,
        depth: int,
        num_heads: int,
        out_chans: int,
        drop_prob: float,
        attn_drop_rate: float,
        drop_path_rate: float,
        init_values: float,
        activation: type[nn.Module],
    ):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.patch_embed = _MultiScaleTemporalConv(
            out_chans=out_chans, activation=activation
        )
        self._init_transformer(
            max_patches=max_patches,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            drop_prob=drop_prob,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            init_values=init_values,
        )

    def forward(
        self, x: Tensor, time_indices: Tensor, spatial_indices: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        branches = self.patch_embed(x)
        spatial, temporal = self._embeddings(time_indices, spatial_indices)
        return tuple(
            self._run_transformer(branch, spatial, temporal, i)
            for i, branch in enumerate(branches, start=1)
        )


class _NeuroRVQDecoder(_BranchTransformer):
    """Shared Transformer decoder with branch-specific latent projections."""

    def __init__(
        self,
        *,
        max_patches: int,
        code_dim: int,
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
        for i in range(1, 5):
            setattr(self, f"patch_embed_{i}", _PatchProjection(code_dim, embed_dim))
        self._init_transformer(
            max_patches=max_patches,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            drop_prob=drop_prob,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            init_values=init_values,
        )

    def forward_branch(
        self,
        x: Tensor,
        time_indices: Tensor,
        spatial_indices: Tensor,
        branch_index: int,
    ) -> Tensor:
        """Decode the quantized latents of scale ``branch_index`` (0-3)."""
        x = getattr(self, f"patch_embed_{branch_index + 1}")(x)
        spatial, temporal = self._embeddings(time_indices, spatial_indices)
        return self._run_transformer(x, spatial, temporal, branch_index + 1)


class NeuroRVQTokenizer(EEGModuleMixin, nn.Module, license="cc-by-nc-4.0"):
    r"""NeuroRVQ multi-scale residual-vector-quantized EEG tokenizer.

    :bdg-success:`Convolution` :bdg-info:`Attention/Transformer` :bdg-danger:`Foundation Model` :bdg-dark-line:`Channel`

    The tokenizer encodes EEG patches through four temporal scales, quantizes
    each scale with a separate residual vector quantizer, and reconstructs the
    signal from amplitude and phase components. Its design follows
    [neurorvq]_ and its source code is available at [neurorvqcode]_.
    ``tokenize`` returns the discrete code indices in
    ``(scale, quantizer, batch, channel_patch)`` order.

    Inputs must be sampled at 200 Hz, contain complete 200-sample patches, and
    use electrode labels from the released 104-channel montage. The model does
    not apply the source example's band-pass filter, resampling, or clipping.
    The source code and pretrained tokenizer are licensed CC BY-NC 4.0.

    Parameters
    ----------
    n_chans : int
        Number of EEG channels.
    n_times : int
        Number of samples; must be divisible by 200 and no longer than
        ``patch_size * max_patches``.
    sfreq : float
        Sampling frequency. NeuroRVQ-EEG v1 requires 200 Hz.
    channel_names : sequence of str or None
        Ordered electrode names. If omitted, names are inferred from
        ``chs_info`` or default to the first channels in the pretrained order.
    max_patches : int, default=256
        Length of the temporal embedding table.
    patch_size : int, default=200
        Samples per patch. The pretrained version requires 200.
    out_chans : int, default=8
        Number of channels per temporal-convolution branch; must be divisible
        by four. The Transformer width is derived as ``out_chans * 25``.
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
    num_quantizers : int, default=8
        Number of residual codebooks per temporal scale.
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

    Output shape
    ------------
    ``forward`` returns standardized target and reconstruction patches shaped
    ``(batch, n_chans * n_patches, patch_size)``, matching the released
    implementation. ``tokenize`` returns integer codes shaped ``(4,
    num_quantizers, batch, n_chans * n_patches)``.

    Examples
    --------
    Reconstruct 200 Hz EEG windows with the released tokenizer and compute the
    time-domain reconstruction error on the standardized windows:

    >>> import torch
    >>> from braindecode.models import NeuroRVQTokenizer
    >>> model = NeuroRVQTokenizer(
    ...     n_chans=3, n_times=800, sfreq=200, channel_names=["C3", "Cz", "C4"]
    ... )
    >>> model = model.load_pretrained_weights().eval()  # doctest: +SKIP
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
    .. [neurorvqcode] https://github.com/KonstantinosBarmpas/NeuroRVQ
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
        patch_size: int = 200,
        max_patches: int = 256,
        out_chans: int = 8,
        encoder_depth: int = 12,
        decoder_depth: int = 3,
        num_heads: int = 10,
        n_code: int = 8192,
        code_dim: int = 128,
        num_quantizers: int = 8,
        drop_prob: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        init_values: float = 0.0,
        activation: type[nn.Module] = nn.GELU,
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
        if patch_size != 200:
            raise ValueError("NeuroRVQ-EEG v1 requires patch_size=200.")
        if self.n_times % patch_size:
            raise ValueError("n_times must be divisible by patch_size (200 samples).")
        self.num_patches = self.n_times // patch_size
        if self.num_patches > max_patches:
            raise ValueError(f"n_times supports at most {max_patches} patches.")
        model_sfreq = self._sfreq
        if model_sfreq is None and self._input_window_seconds is not None:
            model_sfreq = self.n_times / self._input_window_seconds
        if model_sfreq is not None and model_sfreq != 200:
            raise ValueError("NeuroRVQ-EEG v1 requires a 200 Hz sampling frequency.")
        for name, value in (
            ("max_patches", max_patches),
            ("out_chans", out_chans),
            ("encoder_depth", encoder_depth),
            ("decoder_depth", decoder_depth),
            ("num_heads", num_heads),
            ("n_code", n_code),
            ("code_dim", code_dim),
            ("num_quantizers", num_quantizers),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive.")
        if out_chans % 4:
            raise ValueError("out_chans must be divisible by four for GroupNorm.")
        embed_dim = out_chans * 25
        if embed_dim % num_heads:
            raise ValueError(
                "The derived embedding width must be divisible by num_heads."
            )

        self._has_explicit_channel_mapping = (
            channel_names is not None or self._chs_info is not None
        )
        if channel_names is None and self._chs_info is not None:
            channel_names = [ch["ch_name"] for ch in self._chs_info]
        if channel_names is None:
            channel_names = NEURORVQ_CHANNELS[: self.n_chans]
        normalized = tuple(str(name).strip().lower() for name in channel_names)
        if len(normalized) != self.n_chans or len(set(normalized)) != self.n_chans:
            raise ValueError("channel_names must be unique and match n_chans.")
        channel_to_index = {name: i for i, name in enumerate(NEURORVQ_CHANNELS)}
        unknown = [name for name in normalized if name not in channel_to_index]
        if unknown:
            raise ValueError(f"Unsupported NeuroRVQ channel name(s): {unknown}.")

        self.channel_names = normalized
        self.patch_size = patch_size
        self.max_patches = max_patches
        self.embed_dim = embed_dim
        self.code_dim = code_dim
        self.n_code = n_code
        self.num_quantizers = num_quantizers
        self.register_buffer(
            "spatial_embedding_ix",
            torch.tensor([channel_to_index[name] for name in normalized]),
            persistent=False,
        )
        self.encoder = _NeuroRVQEncoder(
            max_patches=max_patches,
            embed_dim=embed_dim,
            depth=encoder_depth,
            num_heads=num_heads,
            out_chans=out_chans,
            drop_prob=drop_prob,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            init_values=init_values,
            activation=activation,
        )
        self.decoder = _NeuroRVQDecoder(
            max_patches=max_patches,
            code_dim=code_dim,
            embed_dim=embed_dim,
            depth=decoder_depth,
            num_heads=num_heads,
            drop_prob=drop_prob,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            init_values=init_values,
        )
        for i in range(1, 5):
            setattr(
                self,
                f"quantize_{i}",
                _ResidualVectorQuantizer(num_quantizers, n_code, code_dim),
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
        self._init_task_layers()

    def _init_task_layers(self):
        task_layers = [getattr(self, f"encode_task_layer_{i}") for i in range(1, 5)] + [
            self.decode_task_layer_amplitude,
            self.decode_task_layer_angle_sin,
            self.decode_task_layer_angle_cos,
        ]
        for layer in task_layers:
            for module in layer.modules():
                if isinstance(module, nn.Linear):
                    nn.init.trunc_normal_(module.weight, std=0.02)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                elif isinstance(module, nn.LayerNorm):
                    nn.init.ones_(module.weight)
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
        latents = []
        for i, features in enumerate(self.encoder(patches, time, spatial), start=1):
            latent = getattr(self, f"encode_task_layer_{i}")(features)
            latent = latent.reshape(batch, channels, n_patches, self.code_dim)
            latents.append(latent.permute(0, 3, 1, 2).contiguous())
        return latents

    def _encode(self, x: Tensor, time: Tensor, spatial: Tensor):
        quantized, codes = [], []
        for i, latent in enumerate(self._branch_latents(x, time, spatial), start=1):
            q, branch_codes, _ = getattr(self, f"quantize_{i}")(latent)
            quantized.append(q)
            codes.append(branch_codes.reshape(self.num_quantizers, x.shape[0], -1))
        return quantized, torch.stack(codes)

    @torch.no_grad()
    def tokenize(self, x: Tensor) -> Tensor:
        """Return four-scale codes without applying EMA updates.

        If a codebook has not been initialized, it is initialized once from the
        input vectors with the source implementation's cosine k-means routine.
        Load pretrained weights before extracting pretrained representations.
        """
        patches = self._patches(x)
        time, spatial = self._embedding_indices(x.device)
        scale_codes = []
        for i, latent in enumerate(
            self._branch_latents(patches, time, spatial), start=1
        ):
            _, codes = getattr(self, f"quantize_{i}").encode(latent)
            scale_codes.append(codes.reshape(self.num_quantizers, x.shape[0], -1))
        return torch.stack(scale_codes)

    def _decode(self, quantized, time: Tensor, spatial: Tensor):
        features = [
            self.decoder.forward_branch(q, time, spatial, branch_index=i)
            for i, q in enumerate(quantized)
        ]
        decoded = torch.cat(features, dim=-1)
        return (
            self.decode_task_layer_amplitude(decoded),
            self.decode_task_layer_angle_sin(decoded),
            self.decode_task_layer_angle_cos(decoded),
        )

    @staticmethod
    def _standardize(x: Tensor):
        mean = x.mean(dim=(1, 2, 3), keepdim=True)
        std = torch.sqrt(x.var(dim=(1, 2, 3), keepdim=True).clamp_min(1e-8))
        return (x - mean) / std, mean, std

    def forward(self, x: Tensor):
        """Return standardized target and reconstruction windows.

        Both outputs are z-scored per window (over channels, patches and
        samples), as in the released implementation, so their mean squared
        difference is the time-domain reconstruction error of the window.
        """
        patches = self._patches(x)
        time, spatial = self._embedding_indices(x.device)
        spectrum = torch.fft.fft(patches, dim=-1)
        amplitude = torch.log1p(spectrum.abs())
        amplitude, amp_mean, amp_std = self._standardize(amplitude)
        quantized, _ = self._encode(patches, time, spatial)
        rec_amp, rec_sin, rec_cos = self._decode(quantized, time, spatial)
        # Undo the log-amplitude standardization, then invert the patch FFT
        # from the amplitude and the predicted phase (cos, sin).
        rec_amp = torch.expm1(rec_amp.reshape_as(patches) * amp_std + amp_mean)
        rec_sin = rec_sin.reshape_as(patches)
        rec_cos = rec_cos.reshape_as(patches)
        reconstructed = torch.fft.ifft(
            torch.complex(rec_amp * rec_cos, rec_amp * rec_sin), dim=-1
        ).real
        target_std, _, _ = self._standardize(patches)
        reconstructed_std, _, _ = self._standardize(reconstructed)
        return (
            target_std.reshape(x.shape[0], self.n_chans * self.num_patches, -1),
            reconstructed_std.reshape(x.shape[0], self.n_chans * self.num_patches, -1),
        )

    def load_pretrained_weights(self, checkpoint_path: str | None = None):
        """Load the released EEG tokenizer checkpoint from a local path or Hub.

        Released spatial embeddings are electrode-specific, so the model must
        have been constructed with ``channel_names`` or ``chs_info``.
        """
        if not self._has_explicit_channel_mapping:
            raise ValueError(
                "Loading pretrained NeuroRVQ tokenizer weights requires "
                "channel_names or chs_info so input electrodes map to the "
                "released spatial embedding slots. The implicit first-N channel "
                "fallback is only supported for randomly initialized training."
            )
        if checkpoint_path is None:
            if not HAS_HF_HUB:
                raise ImportError(
                    "Loading NeuroRVQ weights from the Hub requires huggingface_hub. "
                    "Install braindecode[hub] or pass checkpoint_path."
                )
            checkpoint_path = huggingface_hub.hf_hub_download(
                repo_id=_PRETRAINED_REPO_ID,
                filename=_PRETRAINED_FILENAME,
                revision=_PRETRAINED_REVISION,
            )
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        self.load_state_dict(_remap_mlp_state_dict_keys(state), strict=True)
        return self
