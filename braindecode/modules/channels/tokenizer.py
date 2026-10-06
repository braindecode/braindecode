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

from .resolve import ResolvedMontage, _check_kinds, resolve_montage
from .strategies import SpatialMap, get_channel_strategy
from .target import ChannelTarget

CACHE_SIZE = 8


@dataclass
class ChannelEncoding:
    """Output of :class:`ChannelTokenizer`; a backbone reads what it needs.

    Attributes
    ----------
    x : Tensor (B, K, T)
        Signal on the target channels.
    channel_ids : LongTensor (K,) or None
        Vocabulary ids (``ids`` targets).
    positions : Tensor (K, 3) or None
        Output positions in metres (NaN = unknown).
    observed : BoolTensor (K,)
        Output carries a measured channel (``False`` = reconstructed / zero).
    support : Tensor (K,)
        Confidence in ``[0, 1]`` per output.
    weights : Tensor (K, C) or None
        The linear map applied (``None`` = pass-through / native).
    """

    x: Tensor
    channel_ids: Optional[Tensor]
    positions: Optional[Tensor]
    observed: Tensor
    support: Tensor
    weights: Optional[Tensor]


class ChannelTokenizer(nn.Module):
    """Map any user montage onto what a backbone consumes, inside the model.

    Like a Hugging Face tokenizer: the strategy is chosen at construction and
    applied in :meth:`forward`; a montage given per call is resolved and its
    map cached (at most 8 montages, keyed by names + positions).

    .. warning:: Experimental. Public API may change without a deprecation cycle.

    Parameters
    ----------
    target : ChannelTarget or None
        Channel contract of the backbone (``None`` only with ``"native"``).
    strategy : str
        ``"native"`` (identity: the model's own behaviour) or a registered
        strategy (``"exact"``, ``"zero"``, ``"nearest"``, ``"idw"``,
        ``"spline"``, ``"field"``, ``"source"``, ``"wiener"``, ``"region"``,
        ``"latent"``).
    src_chs_info : list of dict, optional
        Montage used when :meth:`forward` gets no ``chs_info``.
    drop_non_eeg : bool
        Drop input channels of another kind (from the signal too) instead of
        raising.
    kinds : sequence of str
        Accepted input channel kinds (see
        :func:`~braindecode.modules.channels.resolve_montage`). Scalp EEG by
        default; intracranial models pass ``("eeg", "seeg", "ecog", "dbs")``.
        ``"source"`` serves scalp EEG only and raises on other kinds.
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
        if strategy == "native":
            if strategy_kwargs:
                raise ValueError(
                    f"strategy='native' takes no options; got {sorted(strategy_kwargs)}."
                )
            self.strategy = None
            return
        if target is None:
            raise ValueError(
                f"channel strategy {strategy!r} needs a ChannelTarget: this model "
                f"has no channel contract."
            )
        self.strategy = get_channel_strategy(strategy, **strategy_kwargs)
        if (
            target.interface == "positions"
            and target.sensors() is None
            and strategy not in ("exact", "zero")
        ):
            warnings.warn(
                f"Channel strategy {strategy!r} has no effect: the target declares "
                f"no training montage, so every input channel passes through with "
                f"its position. Give the model's ChannelTarget a chs_info to "
                f"reconstruct onto it.",
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
        if ref is not None and (
            m.support.dtype != ref.dtype or m.support.device != ref.device
        ):
            m = m.to(ref)  # cached moved, so the next call is free
        self._cache[src.key] = m
        while len(self._cache) > CACHE_SIZE:
            self._cache.popitem(last=False)
        return m

    def n_outputs(
        self, chs_info: Optional[list[dict]] = None, *, n_chans: Optional[int] = None
    ) -> int:
        """Number of channels ``K`` that :meth:`forward` hands to the backbone.

        Parameters
        ----------
        chs_info : list of dict, optional
            Montage to size for; defaults to the construction montage.
        n_chans : int, optional
            Number of input channels, used when no montage is known (or the
            strategy is not fitted yet) and ``K`` depends on the input.

        Returns
        -------
        int
            With a known montage and a ready strategy: the size of its map.
            Otherwise ``K`` follows from the target and the strategy: the
            target size for ``montage``/``slots`` targets and for ``ids``
            targets under a reconstructing strategy, the strategy's fixed size
            on ``free`` targets (``source`` parcels, ``latent`` latents), else
            the number of input channels.
        """
        src = self._resolve(chs_info) if chs_info is not None else self._src
        if src is not None:
            if self.strategy is None:
                return src.n_input
            if getattr(self.strategy, "fitted", True):
                return int(self._map(src).observed.numel())
            n_chans = len(src.picks)
        if self.strategy is not None:
            assert self.target is not None
            sensors = self.target.sensors()
            if sensors is not None and (
                self.target.interface != "ids" or self.strategy.reconstructs
            ):
                return len(sensors.names)
        if n_chans is None:
            raise ValueError(
                "The number of channels the backbone receives depends on the "
                "input montage: pass chs_info (or n_chans) to the model."
            )
        if self.strategy is not None and self.target.interface == "free":
            return self.strategy._free_size(n_chans)
        return n_chans

    def fit(self, *args, **kwargs) -> "ChannelTokenizer":
        """Fit a data-driven strategy (e.g. ``wiener``) and drop stale maps."""
        if self.strategy is None or not hasattr(self.strategy, "fit"):
            raise ValueError(
                f"Channel strategy {self.strategy_name!r} has nothing to fit."
            )
        self.strategy.fit(*args, **kwargs)
        self._cache.clear()
        return self

    def _load_from_state_dict(self, *args, **kwargs):
        # Loaded strategy state (e.g. a fitted wiener covariance) changes the
        # maps: drop the cached ones.
        self._cache.clear()
        super()._load_from_state_dict(*args, **kwargs)

    def forward(
        self, x: Tensor, chs_info: Optional[list[dict]] = None
    ) -> ChannelEncoding:
        """Encode ``x`` of shape ``(B, C, T)`` recorded with ``chs_info``.

        ``chs_info`` defaults to the construction montage.
        """
        if self.strategy is None:
            C = x.shape[1]
            return ChannelEncoding(
                x,
                None,
                None,
                torch.ones(C, dtype=torch.bool, device=x.device),
                torch.ones(C, dtype=x.dtype, device=x.device),
                None,
            )
        if chs_info is not None:
            src = self._resolve(chs_info)
        elif self._src is not None:
            src = self._src
        else:
            raise ValueError(
                "No montage: pass chs_info to the model (or to forward) so the "
                f"channel strategy {self.strategy_name!r} knows the input channels."
            )
        if x.shape[1] != src.n_input:
            raise ValueError(
                f"Input has {x.shape[1]} channels but the montage has "
                f"{src.n_input} channels; they must match."
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
