# Authors: Bruno Aristimunha <b.aristimunha@gmail.com>
#
# License: BSD (3-clause)
"""The channel layer: any montage in, what the backbone consumes out."""

from __future__ import annotations

import warnings
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional, Sequence

import torch
from torch import Tensor, nn

from .resolve import ChannelTarget, ResolvedMontage, _check_kinds, resolve_montage
from .strategies import SpatialMap, get_channel_strategy

CACHE_SIZE = 8


@dataclass
class ChannelEncoding:
    """Output of :class:`ChannelTokenizer`; a backbone reads what it needs.

    ``x`` (B, K, T) is the signal on the target channels; ``channel_ids``,
    ``positions``, ``observed``, ``support`` and ``weights`` are those of
    :class:`~braindecode.modules.channels.SpatialMap` (``None`` /
    all-observed under ``"native"``).
    """

    x: Tensor
    channel_ids: Optional[Tensor]
    positions: Optional[Tensor]
    observed: Tensor
    support: Tensor
    weights: Optional[Tensor]


class ChannelTokenizer(nn.Module):
    """Map any user montage onto what a backbone consumes, inside the model.

    The strategy is chosen at construction and applied in :meth:`forward`; a
    montage given per call is resolved and its map cached (8 montages, keyed
    by names + positions, moved to the input's device and dtype).

    .. warning:: Experimental. Public API may change without a deprecation cycle.

    Parameters
    ----------
    target : ChannelTarget or None
        Channel contract of the backbone (``None`` only with ``"native"``).
    strategy : str
        ``"native"`` (identity) or a registered strategy: ``"exact"``,
        ``"zero"``, ``"nearest"``, ``"idw"``, ``"spline"``, ``"field"``,
        ``"source"``, ``"wiener"``, ``"region"``, ``"latent"``.
    src_chs_info : list of dict, optional
        Montage used when :meth:`forward` gets no ``chs_info``.
    drop_non_eeg : bool
        Drop input channels of another kind (from the signal too) instead of raising.
    kinds : sequence of str
        Accepted input kinds, a subset of ``("eeg", "seeg", "ecog", "dbs")``.
    **strategy_kwargs
        Forwarded to the strategy (e.g. ``reg`` for ``"spline"``).
    """

    def __init__(
        self,
        target: Optional[ChannelTarget],
        strategy: str = "native",
        src_chs_info: Optional[list[dict]] = None,
        *,
        drop_non_eeg: bool = False,
        kinds: Sequence[str] = ("eeg",),
        **strategy_kwargs,
    ) -> None:
        super().__init__()
        self.target = target
        self.strategy_name = strategy
        self.drop_non_eeg = drop_non_eeg
        self.kinds = _check_kinds(kinds)
        self._cache: OrderedDict[str, SpatialMap] = OrderedDict()
        self._src: Optional[ResolvedMontage] = None
        self.strategy = None
        if strategy == "native":
            if strategy_kwargs:
                raise ValueError(
                    f"strategy='native' takes no options; got {sorted(strategy_kwargs)}."
                )
            return
        if target is None:
            raise ValueError(
                f"channel strategy {strategy!r} needs a ChannelTarget: this model has no channel contract."
            )
        self.strategy = get_channel_strategy(strategy, **strategy_kwargs)
        if (
            target.interface == "positions"
            and target.sensors() is None
            and strategy not in ("exact", "zero")
        ):
            warnings.warn(
                f"Channel strategy {strategy!r} has no effect: the target declares no training "
                f"montage, so every input channel passes through with its position.",
                UserWarning,
                stacklevel=2,
            )
        if src_chs_info is not None:
            self._src = self._resolve(src_chs_info)
            if getattr(self.strategy, "fitted", True):
                self._map(self._src)  # surface montage errors at construction

    def _resolve(self, chs_info: list[dict]) -> ResolvedMontage:
        return resolve_montage(
            chs_info, kinds=self.kinds, drop_non_eeg=self.drop_non_eeg
        )

    def _map(self, src: ResolvedMontage, ref: Optional[Tensor] = None) -> SpatialMap:
        m = self._cache.pop(src.key, None)
        if m is None:
            assert self.strategy is not None and self.target is not None
            m = self.strategy.build(src, self.target)
        if ref is not None and (m.support.dtype, m.support.device) != (
            ref.dtype,
            ref.device,
        ):
            m = m.to(ref)  # cache the moved copy
        self._cache[src.key] = m
        while len(self._cache) > CACHE_SIZE:
            self._cache.popitem(last=False)
        return m

    def n_outputs(
        self, chs_info: Optional[list[dict]] = None, *, n_chans: Optional[int] = None
    ) -> int:
        """Number of channels ``K`` :meth:`forward` hands to the backbone.

        The size of the map for a known montage (``chs_info`` or the
        construction one); else the target size (``montage``/``slots``, and
        ``ids`` under a reconstructing strategy), the strategy's own size on
        ``free`` targets (``source`` parcels, ``latent`` latents), or ``n_chans``.
        """
        src = self._resolve(chs_info) if chs_info is not None else self._src
        if src is not None:
            if self.strategy is None:
                return src.n_input
            if getattr(self.strategy, "fitted", True):
                return int(self._map(src).observed.numel())
            n_chans = len(src.picks)
        if self.strategy is not None and self.target is not None:
            sensors = self.target.sensors()
            if sensors is not None and (
                self.target.interface != "ids" or self.strategy.reconstructs
            ):
                return len(sensors.names)
        if n_chans is None:
            raise ValueError(
                "The number of channels the backbone receives depends on the input montage: "
                "pass chs_info (or n_chans) to the model."
            )
        if (
            self.strategy is not None
            and self.target is not None
            and self.target.interface == "free"
        ):
            return self.strategy._free_size(n_chans)
        return n_chans

    def fit(self, *args, **kwargs) -> "ChannelTokenizer":
        """Fit a data-driven strategy (``wiener``) and drop stale maps."""
        if not hasattr(self.strategy, "fit"):
            raise ValueError(
                f"Channel strategy {self.strategy_name!r} has nothing to fit."
            )
        self.strategy.fit(*args, **kwargs)  # type: ignore[union-attr]
        self._cache.clear()
        return self

    def _load_from_state_dict(self, *args, **kwargs):
        self._cache.clear()  # loaded strategy state (a fitted covariance) changes the maps
        super()._load_from_state_dict(*args, **kwargs)

    def forward(
        self, x: Tensor, chs_info: Optional[list[dict]] = None
    ) -> ChannelEncoding:
        """Encode ``x`` (B, C, T) recorded with ``chs_info`` (default: the construction montage)."""
        if self.strategy is None:
            ones = torch.ones(x.shape[1], device=x.device)
            return ChannelEncoding(x, None, None, ones.bool(), ones.to(x.dtype), None)
        src = self._resolve(chs_info) if chs_info is not None else self._src
        if src is None:
            raise ValueError(
                f"No montage: pass chs_info to the model (or to forward) so the channel strategy "
                f"{self.strategy_name!r} knows the input channels."
            )
        if x.shape[1] != src.n_input:
            raise ValueError(
                f"Input has {x.shape[1]} channels but the montage has {src.n_input} channels; they must match."
            )
        if len(src.picks) != src.n_input:
            x = x[:, torch.as_tensor(src.picks, device=x.device)]
        m = self._map(src, x)
        return ChannelEncoding(
            self.strategy.project(x, m),
            m.channel_ids,
            m.positions,
            m.observed,
            m.support,
            m.weights,
        )
