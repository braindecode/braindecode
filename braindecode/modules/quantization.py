# Authors: Bruno Aristimunha <b.aristimunha@gmail.com>
#
# License: BSD-3

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# Rotation-trick STE helpers (Fifty et al. 2024, §4.2 https://arxiv.org/abs/2410.06424).


def _efficient_rotation_trick_transform(u, q, e):
    # Householder reflection + rank-1 correction: e - 2<e,w>w + 2<e,u>q, w = normalize(u+q).
    w = F.normalize(u + q, p=2, dim=1, eps=1e-6).detach()
    ew = (e * w).sum(dim=1, keepdim=True)
    eu = (e * u.detach()).sum(dim=1, keepdim=True)
    return e - 2 * ew * w + 2 * eu * q.detach()


def _rotate_to(src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
    """Rotation-trick straight-through estimator (Fifty et al. 2024)."""
    orig_shape = src.shape
    src = src.reshape(-1, orig_shape[-1])
    tgt = tgt.reshape(-1, orig_shape[-1])
    norm_src = src.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    norm_tgt = tgt.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    rotated = _efficient_rotation_trick_transform(src / norm_src, tgt / norm_tgt, src)
    rotated = rotated * (norm_tgt / norm_src).detach()
    return rotated.reshape(orig_shape)


class Codebook(nn.Module):
    """EMA-updated Euclidean codebook.

    Buffer keys match the upstream BrainOmni checkpoint for state-dict parity.

    Parameters
    ----------
    dim : int
        Dimensionality of each codebook vector.
    codebook_size : int
        Number of codebook entries.
    decay : float
        Exponential moving-average decay for codebook updates.
    epsilon : float
        Small constant for numerical stability in EMA normalisation.
    threshold_ema_dead_code : int
        Minimum cluster size below which a code is considered dead and
        replaced with a random sample from the current batch.

    References
    ----------
    .. [vqvae] van den Oord, A., Vinyals, O., & Kavukcuoglu, K. (2017).
       Neural Discrete Representation Learning. NeurIPS 2017.
       https://arxiv.org/abs/1711.00937
    .. [rottrick] Fifty, C., et al. (2024). Restructuring Vector Quantization
       with the Rotation Trick. https://arxiv.org/abs/2410.06424
    """

    def __init__(
        self,
        dim: int,
        codebook_size: int,
        decay: float = 0.99,
        epsilon: float = 1e-6,
        threshold_ema_dead_code: int = 2,
    ):
        super().__init__()
        self.decay = decay
        self.epsilon = epsilon
        self.threshold_ema_dead_code = threshold_ema_dead_code
        self.codebook_size = codebook_size

        embed = torch.empty(codebook_size, dim)
        nn.init.kaiming_uniform_(embed)
        # 'inited' buffer kept so state-dict keys match upstream checkpoint.
        self.register_buffer("inited", torch.tensor([True]))
        self.register_buffer("cluster_size", torch.zeros(codebook_size))
        self.register_buffer("embed", embed)
        self.register_buffer("embed_avg", embed.clone())

    def _sample_vectors(self, samples: torch.Tensor, num: int) -> torch.Tensor:
        num_samples, device = samples.shape[0], samples.device
        if num_samples >= num:
            indices = torch.randperm(num_samples, device=device)[:num]
        else:
            indices = torch.randint(0, num_samples, (num,), device=device)
        return samples[indices]

    def replace_(self, samples: torch.Tensor, mask: torch.Tensor) -> None:
        modified = torch.where(
            mask[..., None],
            self._sample_vectors(samples, self.codebook_size),
            self.embed,
        )
        self.embed.data.copy_(modified)

    def expire_codes_(self, batch_samples: torch.Tensor) -> None:
        if self.threshold_ema_dead_code == 0:
            return
        expired = self.cluster_size < self.threshold_ema_dead_code
        if not torch.any(expired):
            return
        batch_samples = rearrange(batch_samples, "... dim -> (...) dim")
        self.replace_(batch_samples, expired)

    @torch.no_grad()
    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        embed = self.embed.t().float()
        dist = (
            x.pow(2).sum(1, keepdim=True)
            - 2 * x @ embed
            + embed.pow(2).sum(0, keepdim=True)
        )
        return dist.argmin(dim=-1)

    def dequantize(self, embed_ind: torch.Tensor) -> torch.Tensor:
        return F.embedding(embed_ind, self.embed)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x = rearrange(x, "... dim -> (...) dim")
        embed_ind = self.quantize(x)
        return embed_ind.view(*shape[:-1])

    def decode(self, embed_ind: torch.Tensor) -> torch.Tensor:
        return self.dequantize(embed_ind)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        shape, dtype = x.shape, x.dtype
        x = rearrange(x, "... dim -> (...) dim")
        embed_ind = self.quantize(x)
        embed_onehot = F.one_hot(embed_ind, self.codebook_size).type(dtype)
        embed_ind = embed_ind.view(*shape[:-1])
        quantize = self.dequantize(embed_ind).type(dtype)

        if self.training:
            self.expire_codes_(x)
            # EMA update of the cluster sizes and code sums (.data detaches grad).
            one_hot_sum = embed_onehot.sum(0)
            self.cluster_size.data.mul_(self.decay).add_(
                one_hot_sum, alpha=1 - self.decay
            )
            embed_sum = (embed_onehot.t() @ x).to(torch.float32)
            self.embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)
            smoothed = (self.cluster_size + self.epsilon) / (
                self.cluster_size.sum() + self.epsilon * self.codebook_size
            )
            cluster_size = smoothed * self.cluster_size.sum()
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(1)
            self.embed.data.copy_(embed_normalized)

        return quantize, embed_ind


class VectorQuantizer(nn.Module):
    """Single-layer vector quantisation with optional rotation-trick STE.

    Parameters
    ----------
    dim : int
        Input/output feature dimensionality.
    codebook_size : int
        Number of codebook entries.
    codebook_dim : int or None
        Internal codebook dimensionality. If different from ``dim``,
        learned projection layers are added. Defaults to ``dim``.
    decay : float
        EMA decay for the underlying :class:`Codebook`.
    epsilon : float
        Numerical stability constant for codebook EMA updates.
    threshold_ema_dead_code : int
        Dead-code replacement threshold (see :class:`Codebook`).
    rotation_trick : bool
        When ``True`` the rotation-trick STE [rottrick]_ is used instead
        of the standard straight-through estimator.

    References
    ----------
    .. [vqvae] van den Oord, A., Vinyals, O., & Kavukcuoglu, K. (2017).
       Neural Discrete Representation Learning. NeurIPS 2017.
       https://arxiv.org/abs/1711.00937
    .. [rottrick] Fifty, C., et al. (2024). Restructuring Vector Quantization
       with the Rotation Trick. https://arxiv.org/abs/2410.06424
    """

    def __init__(
        self,
        dim: int,
        codebook_size: int,
        codebook_dim: int | None = None,
        decay: float = 0.99,
        epsilon: float = 1e-6,
        threshold_ema_dead_code: int = 2,
        rotation_trick: bool = True,
    ):
        super().__init__()
        _codebook_dim: int = codebook_dim if codebook_dim is not None else dim
        requires_projection = _codebook_dim != dim
        self.project_in = (
            nn.Linear(dim, _codebook_dim) if requires_projection else nn.Identity()
        )
        self.project_out = (
            nn.Linear(_codebook_dim, dim) if requires_projection else nn.Identity()
        )
        self._codebook = Codebook(
            dim=_codebook_dim,
            codebook_size=codebook_size,
            decay=decay,
            epsilon=epsilon,
            threshold_ema_dead_code=threshold_ema_dead_code,
        )
        self.rotation_trick = rotation_trick
        self.codebook_size = codebook_size

    @property
    def codebook(self) -> torch.Tensor:
        return self._codebook.embed

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = self.project_in(x)
        return self._codebook.encode(x)

    def decode(self, embed_ind: torch.Tensor) -> torch.Tensor:
        q = self._codebook.decode(embed_ind)
        return self.project_out(q)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        input_dtype = x.dtype
        x = self.project_in(x)
        quantize, embed_ind = self._codebook(x)
        if self.training:
            if self.rotation_trick:
                quantize = _rotate_to(x, quantize).to(input_dtype)
            else:
                quantize = x + (quantize - x).detach()
        loss = F.mse_loss(x.float(), quantize.detach().float()) * 0.25
        if not self.training:
            loss = loss.detach()
        quantize = self.project_out(quantize)
        return quantize, embed_ind, loss


class ResidualVQ(nn.Module):
    """Residual vector quantisation with L2-normalisation.

    Uses the EMA optimisation path with no quantize-dropout. Each residual
    stage is handled by a :class:`VectorQuantizer` layer.

    Parameters
    ----------
    dim : int
        Input feature dimensionality.
    codebook_dim : int
        Internal dimensionality of each :class:`VectorQuantizer` codebook.
    codebook_size : int
        Number of entries in each codebook.
    num_quantizers : int
        Number of residual VQ stages stacked sequentially.
    rotation_trick : bool
        Passed through to each :class:`VectorQuantizer` layer.
    quantize_optimize_method : str
        Codebook update strategy. Only ``"ema"`` is currently supported.

    References
    ----------
    .. [vqvae] van den Oord, A., Vinyals, O., & Kavukcuoglu, K. (2017).
       Neural Discrete Representation Learning. NeurIPS 2017.
       https://arxiv.org/abs/1711.00937
    .. [rottrick] Fifty, C., et al. (2024). Restructuring Vector Quantization
       with the Rotation Trick. https://arxiv.org/abs/2410.06424
    """

    def __init__(
        self,
        dim: int,
        codebook_dim: int,
        codebook_size: int,
        num_quantizers: int,
        rotation_trick: bool = True,
        quantize_optimize_method: str = "ema",
    ):
        super().__init__()
        if quantize_optimize_method != "ema":
            raise ValueError(f"Only 'ema' supported, got {quantize_optimize_method!r}")
        self.num_quantizers = num_quantizers
        self.layers = nn.ModuleList(
            [
                VectorQuantizer(
                    dim=dim,
                    codebook_size=codebook_size,
                    codebook_dim=codebook_dim,
                    rotation_trick=rotation_trick,
                )
                for _ in range(num_quantizers)
            ]
        )

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = F.normalize(x, p=2.0, dim=-1)
        quantized_out: torch.Tensor | float = 0.0
        residual = x
        all_losses: list[torch.Tensor] = []
        all_indices: list[torch.Tensor] = []
        for vq in self.layers:
            quantized, indices, loss = vq(residual)
            residual = residual - quantized.detach()
            quantized_out = quantized_out + quantized
            all_losses.append(loss)
            all_indices.append(indices)
        all_losses_t = torch.stack(all_losses, dim=-1)
        all_indices_t = torch.stack(all_indices, dim=-1)
        return quantized_out, all_indices_t, all_losses_t.mean()  # type: ignore[return-value]

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = F.normalize(x, p=2.0, dim=-1)
        residual = x
        all_indices: list[torch.Tensor] = []
        for vq in self.layers:
            quantized, indices, _ = vq(residual)
            residual = residual - quantized.detach()
            all_indices.append(indices)
        return torch.stack(all_indices, dim=-1)
