# Authors: Jiazhen Hong
#
# License: Apache-2.0
"""SAMBA encoder with a pure PyTorch Mamba-2 backend."""

import math
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from braindecode.models.base import EEGModuleMixin


class PortableGatedNorm(nn.Module):
    """Gate-before-RMS layout and FP32 accumulation used by released weights."""

    def __init__(self, width):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = 1e-05

    def reset_parameters(self):
        nn.init.ones_(self.weight)

    def forward(self, x, z):
        dtype = x.dtype
        # Accumulate the gate and RMS in FP32 for low precision inputs.
        if x.dtype == torch.float16 or x.dtype == torch.bfloat16:
            x, z = x.float(), z.float()
        x = x * F.silu(z)
        y = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + self.eps)
        return (y * self.weight).to(dtype=dtype)


class PortableMamba2(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=2, expand=8, headdim=64):
        super().__init__()
        self.d_inner = d_model * expand
        self.d_state = d_state
        self.headdim = headdim
        if self.d_inner % headdim:
            raise ValueError("Mamba head width mismatch")
        self.nheads = self.d_inner // headdim
        conv_dim = self.d_inner + 2 * d_state
        self.in_proj = nn.Linear(
            d_model, 2 * self.d_inner + 2 * d_state + self.nheads, bias=False
        )
        self.conv1d = nn.Conv1d(
            conv_dim, conv_dim, d_conv, groups=conv_dim, padding=d_conv - 1
        )
        # Match the released Mamba2 constructor, including time-scale priors.
        dt = torch.exp(
            torch.rand(self.nheads) * (math.log(0.1) - math.log(0.001))
            + math.log(0.001)
        ).clamp_min(1e-4)
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        self.A_log = nn.Parameter(torch.empty(self.nheads).uniform_(1, 16).log())
        self.D = nn.Parameter(torch.ones(self.nheads))
        self.dt_bias._no_weight_decay = True
        self.A_log._no_weight_decay = True
        self.D._no_weight_decay = True
        self.norm = PortableGatedNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def reset_parameters(self):
        # Child linear/convolution modules are reset by EEGModuleMixin.
        with torch.no_grad():
            dt = (
                torch.empty_like(self.dt_bias)
                .uniform_(math.log(0.001), math.log(0.1))
                .exp()
                .clamp_min(1e-4)
            )
            self.dt_bias.copy_(dt + torch.log(-torch.expm1(-dt)))
            self.A_log.uniform_(1, 16).log_()
            self.D.fill_(1)

    def components(self, u):
        z, xbc, dt = torch.split(
            self.in_proj(u),
            [self.d_inner, self.d_inner + 2 * self.d_state, self.nheads],
            dim=-1,
        )
        xbc = F.silu(
            self.conv1d(xbc.transpose(1, 2))[..., : u.shape[1]].transpose(1, 2)
        )
        x, b, c = torch.split(xbc, [self.d_inner, self.d_state, self.d_state], dim=-1)
        if x.dtype == torch.float16 or x.dtype == torch.bfloat16:
            x, b, c, dt = x.float(), b.float(), c.float(), dt.float()
        dt = F.softplus(dt + self.dt_bias)
        return (
            z,
            x.reshape(u.shape[0], u.shape[1], self.nheads, self.headdim),
            b,
            c,
            dt,
        )

    def forward(self, u):
        # Chunk-local quadratic scan plus recurrent boundary state. No L x L
        # tensor is formed for the full sequence. Matches Mamba-2 ngroups=1.
        z, x, b, c, dt = self.components(u)
        state = x.new_zeros(u.shape[0], self.nheads, self.headdim, self.d_state)
        ys = []
        a_log = self.A_log
        if a_log.dtype == torch.float16 or a_log.dtype == torch.bfloat16:
            a_log = a_log.float()
        a = -a_log.exp()
        for start in range(0, u.shape[1], 64):
            stop = min(start + 64, u.shape[1])
            xx, bb, cc, dd = (
                x[:, start:stop],
                b[:, start:stop],
                c[:, start:stop],
                dt[:, start:stop],
            )
            cumulative = (dd * a).cumsum(1).transpose(1, 2)
            log_decay = cumulative.unsqueeze(-1) - cumulative.unsqueeze(-2)
            future = torch.ones(
                stop - start, stop - start, dtype=torch.bool, device=u.device
            ).triu(1)
            decay = log_decay.masked_fill(future, float("-inf")).exp()
            cb = torch.matmul(cc, bb.transpose(1, 2)).unsqueeze(1)
            kernel = decay * cb * dd.transpose(1, 2).unsqueeze(-2)
            yy = torch.matmul(kernel, xx.transpose(1, 2)).transpose(1, 2)
            yy = (
                yy
                + torch.einsum("bhpn,btn->bthp", state, cc)
                * cumulative.transpose(1, 2).exp()[..., None]
            )
            end_decay = (cumulative[:, :, -1:] - cumulative).exp().transpose(1, 2)
            state = state * cumulative[:, :, -1].exp()[:, :, None, None] + torch.einsum(
                "bth,bthp,btn->bhpn", end_decay * dd, xx, bb
            )
            ys.append(yy + xx * self.D[None, None, :, None])
        y = torch.cat(ys, dim=1).flatten(2)
        return self.out_proj(self.norm(y, z).to(dtype=u.dtype))

    def recurrence_reference(self, u):
        """Independent sequential calculation for numerical verification only."""
        z, x, b, c, dt = self.components(u)
        state = x.new_zeros(u.shape[0], self.nheads, self.headdim, self.d_state)
        output = []
        a_log = self.A_log
        if a_log.dtype == torch.float16 or a_log.dtype == torch.bfloat16:
            a_log = a_log.float()
        a = -a_log.exp()
        for t in range(u.shape[1]):
            state = (
                state * torch.exp(dt[:, t] * a)[:, :, None, None]
                + dt[:, t, :, None, None] * x[:, t, :, :, None] * b[:, t, None, None, :]
            )
            output.append(
                (state * c[:, t, None, None, :]).sum(-1)
                + x[:, t] * self.D[None, :, None]
            )
        y = torch.stack(output, dim=1).flatten(2)
        return self.out_proj(self.norm(y, z).to(dtype=u.dtype))


def _mixer(config, width, head_dim=None):
    return PortableMamba2(
        width,
        d_state=config.d_state,
        d_conv=config.d_conv,
        expand=config.expand,
        headdim=64 if head_dim is None else head_dim,
    )


class CoordinateDrivenSpatialEmbedding(nn.Module):
    def __init__(self, target_positions: torch.Tensor, hidden_dim: int) -> None:
        super().__init__()
        self.register_buffer(
            "target_positions", target_positions.float(), persistent=True
        )
        self.net = nn.Sequential(
            nn.Linear(3, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )

    def forward(
        self,
        signals: torch.Tensor,
        input_positions: torch.Tensor,
        channel_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if input_positions.ndim == 2:
            input_positions = input_positions.unsqueeze(0).expand(
                signals.shape[0], -1, -1
            )
        if input_positions.shape != (signals.shape[0], signals.shape[1], 3):
            raise ValueError("input_positions must have shape [C, 3] or [B, C, 3]")
        # SAMBA validates fixed coordinates in its constructor. Keeping that
        # tensor-to-bool check out of forward permits torch.export tracing.
        delta = self.target_positions[None, :, None, :] - input_positions[:, None, :, :]
        logits = self.net(delta.to(dtype=self.net[0].weight.dtype)).squeeze(-1)
        if channel_mask is not None:
            if channel_mask.shape != signals.shape[:2]:
                raise ValueError("channel_mask must have shape [B, C]")
            logits = logits.masked_fill(~channel_mask[:, None, :], float("-inf"))
        weights = logits.softmax(dim=-1)
        return (torch.einsum("boc,bct->bot", weights, signals), weights)


class _ResidualMixer(nn.Module):
    """Residual wrapper preserves the released checkpoint parameter layout."""

    def __init__(self, config: SimpleNamespace, width: int) -> None:
        super().__init__()
        self.mixer = _mixer(config, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mixer(x)


class _DifferentialMamba(nn.Module):
    def __init__(self, config: SimpleNamespace) -> None:
        super().__init__()
        width, head = (config.bottleneck_channels, config.head_dim)
        self.head_dim = head
        self.first = nn.ModuleList(
            [_mixer(config, head, head) for _ in range(width // head)]
        )
        self.second = nn.ModuleList(
            [_mixer(config, head, head) for _ in range(width // head)]
        )
        self.lam = nn.Parameter(torch.ones(width))
        self.norms = nn.ModuleList([nn.GroupNorm(1, head) for _ in self.first])
        self.projection = nn.Linear(width, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = []
        for index, (first, second, norm) in enumerate(
            zip(self.first, self.second, self.norms)
        ):
            head = x[..., index * self.head_dim : (index + 1) * self.head_dim]
            lam = self.lam[index * self.head_dim : (index + 1) * self.head_dim]
            difference = first(head.contiguous()) - lam * second(head.contiguous())
            outputs.append(norm(difference.transpose(1, 2)).transpose(1, 2))
        return x + self.projection(torch.cat(outputs, dim=-1))


class _MultiBranchEmbedding(nn.Module):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.branches = nn.ModuleList(
            [
                nn.Conv1d(input_channels, output_channels, kernel, padding=kernel // 2)
                for kernel in (1, 3, 7)
            ]
        )
        self.fuse = nn.Conv1d(output_channels * 3, output_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fuse(torch.cat([branch(x) for branch in self.branches], dim=1))


class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        base, bottleneck = config.base_channels, config.bottleneck_channels
        self.cdse = CoordinateDrivenSpatialEmbedding(
            torch.tensor(config.target_positions), config.cdse_hidden_dim
        )
        self.input_embedding = nn.Sequential(
            _MultiBranchEmbedding(len(config.target_positions), base)
        )
        self.encoder1 = nn.Sequential(
            nn.Linear(base, base), _ResidualMixer(config, base)
        )
        self.encoder2 = nn.Conv1d(base, base, 3, padding=1)
        self.encoder3 = nn.Conv1d(base, bottleneck, 3, padding=1)
        self.bottleneck = nn.Sequential(
            nn.Linear(bottleneck, bottleneck),
            _DifferentialMamba(config),
            nn.Linear(bottleneck, bottleneck),
        )

    def encode(
        self, x: torch.Tensor, positions: torch.Tensor, mask: torch.Tensor | None = None
    ):
        projected, _ = self.cdse(x, positions, mask)
        x = self.encoder1(self.input_embedding(projected).transpose(1, 2)).transpose(
            1, 2
        )
        x = self.encoder2(F.max_pool1d(x, 2))
        x = self.encoder3(F.max_pool1d(x, 2))
        return self.bottleneck(F.max_pool1d(x, 2).transpose(1, 2))


class SAMBA(EEGModuleMixin, nn.Module, license="apache-2.0"):
    """Self-supervised EEG representation learning with Differential Mamba-2 [1]_.

    Coordinates must use the same frame and units as the pretrained target
    montage. Inputs are (batch, channels, time); outputs are unnormalized
    task scores (batch, n_outputs). This downstream encoder omits the
    reconstruction decoder. It pools temporal means and population standard
    deviations before a new linear task head. It does not normalize signals:
    released weights expect per-window, per-channel absolute 95th percentile
    normalization and 128 Hz signals. The encoder is bidirectional within a
    supplied window (convolutions and GroupNorm); it is not a streaming model.

    Parameters
    ----------
    target_positions : list of list of float | None
        Shared montage XYZ coordinates. Defaults to input coordinates for
        training from scratch. Use checkpoint config for pretrained models.
    base_channels : int
        Encoder width. Default 64.
    bottleneck_channels : int
        Differential encoder width. Default 128.
    d_state : int
        Mamba state dimension. Default 16.
    d_conv : int
        Causal Mamba convolution width. Default 2.
    expand : int
        Mamba expansion factor. Default 8.
    head_dim : int
        Differential branch width. Default 16.
    cdse_hidden_dim : int
        Spatial coordinate network width. Default 32.

    Notes
    -----
    Uses a chunked PyTorch recurrence, with no optional CUDA dependencies.
    Chunk length is 64. Numerical results need not be bitwise identical to
    fused CUDA kernels. All encoder parameters support automatic gradients.

    References
    ----------
    .. [1] Hong et al. SAMBA: Toward an EEG Foundation Model via Differential
       Mamba and Coordinate-Driven Spatial Embedding. NeurReps, 2026.
       https://github.com/Jiazhen-Hong/SAMBA
    """

    def __init__(
        self,
        n_outputs=None,
        n_chans=None,
        chs_info=None,
        n_times=None,
        input_window_seconds=None,
        sfreq=None,
        target_positions=None,
        base_channels=64,
        bottleneck_channels=128,
        d_state=16,
        d_conv=2,
        expand=8,
        head_dim=16,
        cdse_hidden_dim=32,
    ):
        super().__init__(
            n_outputs=n_outputs,
            n_chans=n_chans,
            chs_info=chs_info,
            n_times=n_times,
            input_window_seconds=input_window_seconds,
            sfreq=sfreq,
        )
        if chs_info is None:
            raise ValueError("SAMBA requires chs_info with electrode XYZ coordinates")
        positions = torch.tensor(
            [list(ch["loc"][:3]) for ch in chs_info], dtype=torch.float32
        )
        if positions.shape != (self.n_chans, 3) or not torch.isfinite(positions).all():
            raise ValueError(
                "Electrode coordinates must be finite with shape (n_chans, 3)"
            )
        target = positions.tolist() if target_positions is None else target_positions
        target_tensor = torch.as_tensor(target, dtype=torch.float32)
        if (
            target_tensor.ndim != 2
            or target_tensor.shape[1] != 3
            or not len(target_tensor)
            or not torch.isfinite(target_tensor).all()
        ):
            raise ValueError(
                "target_positions must be a finite nonempty (channels, 3) array"
            )
        if (
            min(
                base_channels,
                bottleneck_channels,
                d_state,
                d_conv,
                expand,
                head_dim,
                cdse_hidden_dim,
            )
            < 1
            or bottleneck_channels % head_dim
            or base_channels * expand % 64
        ):
            raise ValueError("Invalid SAMBA dimensions or head divisibility")
        self.target_positions = target
        self.base_channels = base_channels
        self.bottleneck_channels = bottleneck_channels
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.head_dim = head_dim
        self.cdse_hidden_dim = cdse_hidden_dim
        self.register_buffer("input_positions", positions, persistent=False)
        config = SimpleNamespace(
            target_positions=target,
            base_channels=base_channels,
            bottleneck_channels=bottleneck_channels,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            head_dim=head_dim,
            cdse_hidden_dim=cdse_hidden_dim,
        )
        self.encoder = Encoder(config)
        self.final_layer = nn.Linear(2 * bottleneck_channels, self.n_outputs)

    def forward_features(self, X):
        """Return pooled encoder features with gradients enabled."""
        if X.ndim != 3 or X.shape[1] != self.n_chans or X.shape[2] < 16:
            raise ValueError("Expected (batch, n_chans, time), with time >= 16")
        hidden = self.encoder.encode(X, self.input_positions, None)
        return torch.cat((hidden.mean(1), hidden.std(1, correction=0)), dim=-1)

    def forward(self, X):
        return self.final_layer(self.forward_features(X))

    def reset_head(self, n_outputs):
        """Replace only the task head; require a positive number of outputs.

        Use ``forward_features`` for embeddings rather than a zero-sized head.
        """
        if (
            not isinstance(n_outputs, int)
            or isinstance(n_outputs, bool)
            or n_outputs <= 0
        ):
            raise ValueError("n_outputs must be a positive integer")
        self._set_n_outputs(n_outputs)
        self.final_layer = nn.Linear(2 * self.bottleneck_channels, n_outputs)

    def load_pretrained_encoder(self, state_dict):
        """Load a released SambaModel state dict, omitting only decoder keys.

        Download weights explicitly before calling. The task head is initialized
        separately. Missing encoder keys, unknown keys, shapes or a mismatched
        target montage fail instead of silently loading a partial checkpoint.
        """
        expected = self.encoder.state_dict()
        decoder_prefixes = (
            "decoder3.",
            "decoder3_mixer.",
            "decoder2.",
            "decoder2_mixer.",
            "decoder1.",
            "output_embedding.",
            "spatial_decoder.",
        )
        unknown = [
            k
            for k in state_dict
            if k not in expected and not k.startswith(decoder_prefixes)
        ]
        if unknown:
            raise ValueError(f"Unrecognized checkpoint keys: {unknown}")
        if "cdse.target_positions" in state_dict and not torch.equal(
            state_dict["cdse.target_positions"].cpu(),
            expected["cdse.target_positions"].cpu(),
        ):
            raise ValueError(
                "Checkpoint target montage differs from model configuration"
            )
        return self.encoder.load_state_dict(
            {k: v for k, v in state_dict.items() if k in expected}, strict=True
        )
