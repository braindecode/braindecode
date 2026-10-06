# Authors: Pierre Guetschel
#          Bruno Aristimunha <b.aristimunha@gmail.com>
#
# License: BSD (3-clause)
"""Channel strategies: the registry, the shared ``build`` and every strategy but ``source``."""

from __future__ import annotations

import difflib
import math
import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
from scipy.linalg import solve
from scipy.spatial.distance import cdist
from torch import Tensor, nn

from .resolve import (
    ChannelTarget,
    ResolvedMontage,
    TargetSensors,
    match_names,
    nearest_vocabulary,
    resolve_montage,
    standard_position,
)

SUPPORT_SCALE_MM = 30.0
_REGISTRY: dict[str, type["ChannelStrategy"]] = {}


@dataclass
class SpatialMap:
    """What a strategy built for one (montage, target) pair.

    ``weights`` (K, C) is the linear map (``None`` = pass-through);
    ``channel_ids`` (K,) vocabulary ids (``ids`` targets); ``positions`` (K, 3)
    in metres; ``observed`` (K,) marks copies of a measured channel;
    ``support`` (K,) is 1 for name copies, ``exp(-d / 30 mm)`` to the nearest
    used input otherwise, 0 for zero rows; ``extra`` holds per-montage tensors
    of trainable strategies.
    """

    weights: Optional[Tensor]
    channel_ids: Optional[Tensor]
    positions: Optional[Tensor]
    observed: Tensor
    support: Tensor
    extra: dict = field(default_factory=dict)

    def to(self, ref: Tensor) -> "SpatialMap":
        """Copy with float tensors on ``ref``'s device/dtype, the others on its device."""

        def mv(t):
            if t is None:
                return None
            return (
                t.to(ref.device, ref.dtype)
                if t.is_floating_point()
                else t.to(ref.device)
            )

        return SpatialMap(
            mv(self.weights),
            mv(self.channel_ids),
            mv(self.positions),
            mv(self.observed),
            mv(self.support),
            {k: mv(v) for k, v in self.extra.items()},
        )


def register_channel_strategy(name: str):
    """Class decorator adding a :class:`ChannelStrategy` to the registry."""

    def deco(cls):
        if name in _REGISTRY or name == "native":
            raise ValueError(f"Channel strategy {name!r} is already registered.")
        cls.name = name
        _REGISTRY[name] = cls
        return cls

    return deco


def get_channel_strategy(name: str, **kwargs) -> "ChannelStrategy":
    """Instantiate a registered strategy by name."""
    if name not in _REGISTRY:
        close = difflib.get_close_matches(name, list(_REGISTRY), n=3)
        hint = f" Did you mean {close}?" if close else ""
        raise ValueError(
            f"Unknown channel strategy {name!r}.{hint} Available: {['native', *sorted(_REGISTRY)]}."
        )
    return _REGISTRY[name](**kwargs)


def _f32(a) -> Tensor:
    return torch.as_tensor(np.asarray(a), dtype=torch.float32)


def _numpy(t: Tensor) -> np.ndarray:
    """float64 copy of a buffer on any device (MPS has no float64)."""
    return t.detach().cpu().double().numpy()


def _unknown_name(name: str) -> bool:
    return bool(np.isnan(standard_position(name)).any())


def _rows(src: ResolvedMontage, use: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Scatter rows ``(M, k)`` over the used inputs into ``(M, C)``."""
    out = np.zeros((len(w), len(src.names)))
    out[:, use] = w
    return out


def _mlp(d_in: int, d: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(d_in, d), nn.GELU(), nn.Linear(d, d))


def _channel_stats(x: Tensor, used: Tensor) -> Tensor:
    """``(B, C, 2)`` log-variance and log line-length, centred over the used channels."""
    w = used.to(x.dtype)
    stats = torch.stack(
        [
            torch.log(x.var(-1) + 1e-12),
            torch.log(x.diff(dim=-1).abs().mean(-1) + 1e-12),
        ],
        -1,
    )
    return stats - (stats * w[:, None]).sum(1, keepdim=True) / w.sum()


class ChannelStrategy(nn.Module):
    """Map a resolved montage onto a :class:`ChannelTarget`.

    Subclasses implement :meth:`_fill`: rows ``(M, C)`` for the ``M``
    positioned targets the input does not carry, from the inputs ``use``.
    Name copies (exact, then alias; then a coordinate-only input within
    15 mm), non-electrode targets, support and pass-through targets are
    shared here. ``reconstructs=False`` (``exact``) makes a missing target an
    error; ``min_positions`` positioned inputs are required by :meth:`_fill`.
    """

    name: str = ""
    trainable: bool = False
    min_positions: int = 1
    reconstructs: bool = True

    def build(self, src: ResolvedMontage, target: ChannelTarget) -> SpatialMap:
        """Spatial map from ``src`` to what ``target`` consumes."""
        tgt = target.sensors()
        if tgt is None:
            return self._pass_through(src, target)
        if target.interface == "ids" and not self.reconstructs:
            return self._ids_subset(src, tgt)
        W, hit, support = self._copies(src, tgt)
        todo = ~hit & ~tgt.non_electrode & np.isfinite(tgt.positions).all(1)
        if not self.reconstructs:
            missing = [
                n
                for n, h, ne in zip(tgt.names, hit, tgt.non_electrode)
                if not (h or ne)
            ]
            if missing:
                raise ValueError(
                    f"Strategy {self.name!r}: target channels {missing} are not in the input "
                    f"montage {list(src.names)}. Use a reconstructing strategy (e.g. 'spline', "
                    f"'source') or supply them."
                )
        elif todo.any():
            use = self._usable(src)
            W[todo] = self._fill(src, use, tgt.positions[todo])
            filled = todo & (np.abs(W).sum(1) > 0)
            support[filled] = self._fill_support(src, use, tgt.positions[filled])
            gain = np.abs(W[todo]).sum(1).max()
            if gain > 2:
                warnings.warn(
                    f"Strategy {self.name!r}: reconstructed row gain |w|_1 = {gain:.3g} > 2; "
                    f"the map amplifies noise. Supply more input channels or a smoother strategy.",
                    UserWarning,
                    stacklevel=4,
                )
        self._warn_unused(src, W)
        return SpatialMap(
            _f32(W),
            None if tgt.channel_ids is None else torch.as_tensor(tgt.channel_ids),
            _f32(tgt.positions),
            torch.as_tensor(hit),
            _f32(support),
        )

    def project(self, x: Tensor, m: SpatialMap) -> Tensor:
        """Apply a built map to ``x`` of shape ``(B, C, T)``."""
        return x if m.weights is None else m.weights @ x

    def _free_size(self, n_input: int) -> int:
        """Outputs on a ``free`` target for ``n_input`` channels."""
        return n_input

    def _fill(
        self, src: ResolvedMontage, use: np.ndarray, tgt_pos: np.ndarray
    ) -> np.ndarray:
        raise NotImplementedError

    def _copies(self, src: ResolvedMontage, tgt: TargetSensors):
        """Copy rows ``W (K, C)``, the copied targets and their support."""
        copy = match_names(tgt.names, src.names)
        # An input whose name is unknown to 10-05 takes the target within 15 mm.
        free = np.array(
            [
                i not in set(copy) and src.has_position[i] and _unknown_name(n)
                for i, n in enumerate(src.names)
            ],
            dtype=bool,
        )
        open_ = (copy < 0) & ~tgt.non_electrode
        dist = np.zeros(len(copy))
        if free.any() and open_.any():
            cand = np.flatnonzero(free)
            j = nearest_vocabulary(tgt.positions[open_], src.positions[cand])
            rows = np.flatnonzero(open_)[j >= 0]
            copy[rows] = cand[j[j >= 0]]
            dist[rows] = np.linalg.norm(
                tgt.positions[rows] - src.positions[copy[rows]], axis=1
            )
        hit = copy >= 0
        W = np.zeros((len(tgt.names), len(src.names)))
        W[np.flatnonzero(hit), copy[hit]] = 1.0
        return W, hit, np.where(hit, np.exp(-dist / (SUPPORT_SCALE_MM * 1e-3)), 0.0)

    @staticmethod
    def _fill_support(
        src: ResolvedMontage, use: np.ndarray, tgt_pos: np.ndarray
    ) -> np.ndarray:
        d = cdist(tgt_pos, src.positions[use]).min(1, initial=np.inf)
        return np.exp(-d / (SUPPORT_SCALE_MM * 1e-3))

    def _usable(self, src: ResolvedMontage) -> np.ndarray:
        use = src.positioned
        if use.sum() < self.min_positions:
            raise ValueError(
                f"Strategy {self.name!r} needs at least {self.min_positions} channels with a "
                f"position; got {int(use.sum())} of {len(src.names)} ({list(src.names)}). "
                f"Supply 'loc' or standard 10-05 names."
            )
        return use

    def _ids_subset(self, src: ResolvedMontage, tgt: TargetSensors) -> SpatialMap:
        """``ids`` without reconstruction: each input channel -> its vocabulary id."""
        ids = match_names(src.names, tgt.names)
        # Position fallback only for names unknown to 10-05, as in _copies.
        lost = (
            (ids < 0)
            & src.has_position
            & np.array([_unknown_name(n) for n in src.names], dtype=bool)
        )
        if lost.any():
            ids[lost] = nearest_vocabulary(src.positions[lost], tgt.positions)
        bad = [n for n, i in zip(src.names, ids) if i < 0]
        if bad:
            close = difflib.get_close_matches(bad[0], tgt.names, n=3, cutoff=0.0)
            raise ValueError(
                f"Channel {bad[0]!r} is not in the model vocabulary ({len(tgt.names)} names; "
                f"closest: {close}) and has no position within 15 mm of one. Rename it, supply "
                f"its position, or use a reconstructing strategy. Unknown channels: {bad}."
            )
        return SpatialMap(
            None,
            torch.as_tensor(ids),
            _f32(tgt.positions[ids]),
            torch.ones(len(ids), dtype=torch.bool),
            torch.ones(len(ids)),
        )

    def _pass_through(self, src: ResolvedMontage, target: ChannelTarget) -> SpatialMap:
        if target.interface == "positions" and not src.positioned.all():
            missing = [n for n, ok in zip(src.names, src.positioned) if not ok]
            raise ValueError(
                f"Channels {missing} have no position and no standard 10-05 name; this model "
                f"needs coordinates for every channel."
            )
        C = len(src.names)
        return SpatialMap(
            None,
            None,
            _f32(src.positions),
            torch.ones(C, dtype=torch.bool),
            torch.ones(C),
        )

    def _warn_unused(self, src: ResolvedMontage, W: np.ndarray) -> None:
        unused = [n for n, col in zip(src.names, np.abs(W).sum(0)) if col == 0]
        if unused:
            warnings.warn(
                f"{len(unused)} input channel(s) contribute nothing to the output of strategy "
                f"{self.name!r}: {unused[:10]}.",
                UserWarning,
                stacklevel=3,
            )


@register_channel_strategy("exact")
class ExactStrategy(ChannelStrategy):
    """Permutation / subset by resolved name; a missing target is an error."""

    reconstructs = False


@register_channel_strategy("zero")
class ZeroStrategy(ChannelStrategy):
    """Missing targets are zero rows with ``observed=False``."""

    min_positions = 0

    def _fill(self, src, use, tgt_pos):
        return np.zeros((len(tgt_pos), len(src.names)))


@register_channel_strategy("nearest")
class NearestStrategy(ChannelStrategy):
    """Missing target = the nearest positioned input (ties share the weight)."""

    def _fill(self, src, use, tgt_pos):
        d = cdist(tgt_pos, src.positions[use])
        hit = (d == d.min(1, keepdims=True)).astype(float)
        return _rows(src, use, hit / hit.sum(1, keepdims=True))


@register_channel_strategy("idw")
class IDWStrategy(ChannelStrategy):
    """Missing target = inverse-distance weighted mean (``1 / d**p``)."""

    def __init__(self, p: float = 2.0):
        super().__init__()
        self.p = p

    def _fill(self, src, use, tgt_pos):
        w = 1.0 / np.maximum(cdist(tgt_pos, src.positions[use]), 1e-6) ** self.p
        return _rows(src, use, w / w.sum(1, keepdims=True))


@register_channel_strategy("region")
class RegionStrategy(ChannelStrategy):
    """Missing target = mean of the inputs within ``radius_mm`` (none: zero row)."""

    def __init__(self, radius_mm: float = 40.0):
        super().__init__()
        self.radius_mm = radius_mm

    def _fill(self, src, use, tgt_pos):
        near = (cdist(tgt_pos, src.positions[use]) <= self.radius_mm * 1e-3).astype(
            float
        )
        return _rows(src, use, near / np.maximum(near.sum(1, keepdims=True), 1.0))


def _mne_interp_matrix(
    src_pos, tgt_pos, method: str = "spline", reg: float = 0.0
) -> np.ndarray:
    """``(M, k)`` matrix of :meth:`mne.io.Raw.interpolate_to`, from ``eye(k)``."""
    import mne

    src_names = [f"S{i}" for i in range(len(src_pos))]
    try:
        info = mne.create_info(src_names, sfreq=100.0, ch_types="eeg")
        info.set_montage(
            mne.channels.make_dig_montage(
                dict(zip(src_names, src_pos)), coord_frame="head"
            )
        )
        raw = mne.io.RawArray(np.eye(len(src_names)), info, verbose="ERROR")
        montage = mne.channels.make_dig_montage(
            {f"T{i}": p for i, p in enumerate(tgt_pos)}, coord_frame="head"
        )
        with mne.utils.use_log_level("ERROR"):
            return raw.interpolate_to(montage, method=method, reg=reg).get_data()
    except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
        raise ValueError(
            f"MNE could not build the {method!r} interpolation from {len(src_pos)} positions: {exc}"
        ) from exc


@register_channel_strategy("spline")
class SplineStrategy(ChannelStrategy):
    """MNE spherical spline, regularised (``reg=1e-3``; ``0`` = unregularised)."""

    min_positions = 4
    method = "spline"

    def __init__(self, reg: float = 1e-3):
        super().__init__()
        self.reg = reg

    def _fill(self, src, use, tgt_pos):
        return _rows(
            src,
            use,
            _mne_interp_matrix(src.positions[use], tgt_pos, self.method, self.reg),
        )


@register_channel_strategy("field")
class FieldStrategy(SplineStrategy):
    """MNE field mapping (``interpolate_to(method="MNE")``).

    MNE maps average-referenced data, so rows get ``(1 - sum(w)) / k`` added
    on the used inputs: they sum to 1 and keep the input's own reference.
    """

    method = "MNE"

    def __init__(self, reg: float = 0.0):
        super().__init__(reg=reg)

    def _fill(self, src, use, tgt_pos):
        W = _mne_interp_matrix(src.positions[use], tgt_pos, self.method, self.reg)
        return _rows(src, use, W + (1.0 - W.sum(1, keepdims=True)) / W.shape[1])


@register_channel_strategy("wiener")
class WienerStrategy(ChannelStrategy):
    """Linear MMSE from a spatial covariance fitted on dense recordings.

    Call :meth:`fit` (or ``model.channel_tokenizer.fit``) first. Inputs and
    targets are matched to the dense montage by position (within 15 mm). The
    covariance is saved in the ``state_dict``.

    Parameters
    ----------
    noise : float
        Diagonal loading, relative to the mean observed variance.
    """

    cov: Tensor
    dense_positions: Tensor

    def __init__(self, noise: float = 0.01):
        super().__init__()
        self.noise = noise
        self.register_buffer("cov", torch.zeros(0, 0))
        self.register_buffer("dense_positions", torch.zeros(0, 3))

    @property
    def fitted(self) -> bool:
        return self.cov.numel() > 0

    def fit(self, X: np.ndarray, chs_info_dense: list[dict]) -> "WienerStrategy":
        """Fit the covariance on ``X`` of shape ``(n_samples, n_dense_channels)``."""
        dense = resolve_montage(chs_info_dense)
        if not dense.positioned.all() or X.shape[1] != len(dense.names):
            raise ValueError(
                f"fit() needs X of shape (n_samples, n_channels) and a position for every dense "
                f"channel; got X {X.shape} for {len(dense.names)} channels."
            )
        self.cov = _f32(np.cov(np.asarray(X, float).T)).to(self.cov.device)
        self.dense_positions = _f32(dense.positions).to(self.cov.device)
        return self

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        for name in ("cov", "dense_positions"):  # buffers change size when fitted
            if prefix + name in state_dict:
                setattr(self, name, torch.empty_like(state_dict[prefix + name]))
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def build(self, src, target):
        if not self.fitted:
            raise ValueError(
                "The 'wiener' strategy needs a covariance: call fit() first."
            )
        return super().build(src, target)

    def _dense_index(self, positions, names):
        j = nearest_vocabulary(positions, _numpy(self.dense_positions))
        if (j < 0).any():
            raise ValueError(
                f"'wiener': channels {[n for n, i in zip(names, j) if i < 0]} have no electrode "
                f"of the fitted dense montage within 15 mm."
            )
        return j

    def _fill(self, src, use, tgt_pos):
        cov = _numpy(self.cov)
        js = self._dense_index(src.positions[use], np.asarray(src.names)[use])
        jt = self._dense_index(tgt_pos, [f"target at {p.round(3)}" for p in tgt_pos])
        C_obs = cov[np.ix_(js, js)]
        loaded = C_obs + self.noise * np.trace(C_obs) / len(js) * np.eye(len(js))
        return _rows(src, use, solve(loaded, cov[np.ix_(js, jt)], assume_a="sym").T)


@register_channel_strategy("latent")
class LatentStrategy(ChannelStrategy):
    """POYO-style learned cross-attention from any electrode set.

    Keys: MLP of Fourier features of each input position + MLP of its signal
    statistics. Queries: ``n_latents`` learned latents (``free`` targets) or
    an MLP of each missing target position (carried targets are copied).
    """

    trainable = True

    def __init__(self, n_latents: int = 64, d_model: int = 64, n_freqs: int = 8):
        super().__init__()
        self.n_freqs = n_freqs
        self.latents = nn.Parameter(
            torch.randn(n_latents, d_model) / math.sqrt(d_model)
        )
        self.key_position = _mlp(6 * n_freqs, d_model)
        self.key_signal = _mlp(2, d_model)
        self.query_position = _mlp(6 * n_freqs, d_model)

    def _fourier(self, pos: np.ndarray) -> Tensor:
        # Positions in decimetres: head-sized coordinates in [-1, 1].
        arg = (
            np.nan_to_num(pos)[:, :, None] * 10 * np.pi * 2.0 ** np.arange(self.n_freqs)
        )
        return _f32(
            np.concatenate([np.sin(arg), np.cos(arg)], -1).reshape(len(pos), -1)
        )

    def build(self, src, target):
        use = self._usable(src)
        tgt = target.sensors()
        if tgt is None and target.interface != "free":
            return self._pass_through(src, target)
        extra = {"used": torch.as_tensor(use), "src_feat": self._fourier(src.positions)}
        if tgt is None:
            K = self.latents.shape[0]
            on = torch.ones(K, dtype=torch.bool)
            return SpatialMap(
                None, None, None, on, torch.ones(K), {**extra, "recon": on}
            )
        W, hit, support = self._copies(src, tgt)
        recon = ~hit & ~tgt.non_electrode & np.isfinite(tgt.positions).all(1)
        support[recon] = self._fill_support(src, use, tgt.positions[recon])
        self._warn_unused(src, np.vstack([W, recon.any() * use[None]]))
        extra.update(
            recon=torch.as_tensor(recon), tgt_feat=self._fourier(tgt.positions)
        )
        return SpatialMap(
            _f32(W),
            None if tgt.channel_ids is None else torch.as_tensor(tgt.channel_ids),
            _f32(tgt.positions),
            torch.as_tensor(hit),
            _f32(support),
            extra,
        )

    def _free_size(self, n_input: int) -> int:
        return self.latents.shape[0]

    def project(self, x: Tensor, m: SpatialMap) -> Tensor:
        e = m.extra
        if not e:  # pass-through (``positions`` target without chs_info)
            return x
        keys = self.key_position(e["src_feat"]) + self.key_signal(
            _channel_stats(x, e["used"])
        )
        q = self.query_position(e["tgt_feat"]) if "tgt_feat" in e else self.latents
        logits = q @ keys.transpose(1, 2) / math.sqrt(keys.shape[-1])
        attn = logits.masked_fill(~e["used"], float("-inf")).softmax(-1)  # (B, K, C)
        learned = (attn @ x) * e["recon"].to(x.dtype)[:, None]
        return learned if m.weights is None else m.weights @ x + learned
