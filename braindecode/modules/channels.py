# Authors: Bruno Aristimunha <b.aristimunha@gmail.com>
#
# License: BSD (3-clause)
"""Channel layer: any montage in, the channels a pretrained backbone consumes out.

Every strategy is a matrix ``W`` (K x C) from the C input channels to the K
target channels. Names are copied first (case-insensitive, then legacy names
such as T3 = T7); an input whose name MNE does not know is copied to a target
within 15 mm. A channel named ``A-B`` is ``V(A) - V(B)`` (BIOT), so a target
that is a combination of measured channels is computed exactly
(:func:`braindecode.functional.exact_derivation`): ``F3-C3`` from ``C3-F3``,
``F3-P3`` from ``F3-C3`` and ``C3-P3``, ``F3-C3`` from ``F3`` and ``C3``. The
strategy fills the other targets that have a position from the positioned
inputs; a target without a position (BENDR's ``SCALE``) is zero. ``spline``,
``wiener`` and a kernel passed as the strategy are one Gaussian estimate
(:func:`braindecode.functional.linear_channel_map`), which also gives each
target's posterior variance.
"""

from __future__ import annotations

import difflib
import warnings
from collections import OrderedDict
from typing import cast

import numpy as np
import torch
from scipy.spatial.distance import cdist
from torch import Tensor, nn

from braindecode.functional import check_ch_mask, exact_derivation
from braindecode.functional.channels import (
    _ELECTRODES,
    _channel_electrodes,
    _channel_geometry,
    _derivation_rows,
    _mixing,
    _source,
)
from braindecode.functional.channels import (
    resolve_channel_positions as resolve_channel_positions,
)

_MIN_POSITIONS = {
    "exact": 0,
    "zero": 0,
    "nearest": 1,
    "idw": 1,
    "spline": 4,
    "field": 4,
    "source": 4,
    "wiener": 1,
    "region": 2,
    "latent": 1,
}
STRATEGIES = tuple(_MIN_POSITIONS)


class ChannelLayer(nn.Module):
    """Map any montage onto ``target`` with one matrix per montage.

    .. warning:: Experimental. Public API may change without a deprecation cycle.

    Parameters
    ----------
    target : list of dict
        Channels the backbone consumes (``ch_name``, optional ``loc``).
    strategy : str or callable
        ``"exact"`` (a missing target is an error), ``"zero"``, ``"nearest"``,
        ``"idw"`` (``p``), ``"spline"`` (``reg=1e-3``), ``"field"`` (``reg=0``)
        or ``"source"`` (``n_parcels=64``, ``lam=0.1``, ``grid_mm=15``,
        ``trainable``: a zero-initialised parcel mixing on top of the physics).
        ``"wiener"`` (``noise=0.01``; call :meth:`fit` first), ``"region"``
        (mean within ``radius``, default 1.5 x the median input spacing) or
        ``"latent"`` (``d_model=64``, ``n_freqs=8``: learned cross-attention from
        each missing target's position over the inputs' positions and statistics).
        A callable ``kernel(pos_a, pos_b) -> (n_a, n_b)`` covariance of the
        potentials at two sets of positions (metres, from the centre of the
        sphere fitted to the inputs, as ``"spline"``), e.g.
        :func:`~braindecode.functional.spherical_spline_kernel`, fills them with
        :func:`~braindecode.functional.linear_channel_map` (``reg``: the noise
        variance, default 1e-3; the recording reference is left unknown).
        ``spline`` and ``wiener`` are that estimate with their own covariance;
        ``match_gain=True`` gives each reconstructed target its prior variance.
    chs_info : list of dict, optional
        Montage used when :meth:`forward` gets none.
    drop_non_eeg : bool
        Ignore input channels that are not electrodes instead of raising.
    reference : str, optional
        Reference of the backbone's single-electrode targets: ``None`` (the
        input's), an electrode name (``"Cz"``: target ``A`` is ``V(A) -
        V(Cz)``, with ``Cz`` copied or reconstructed like any target) or
        ``"average"`` (minus their mean). Bipolar targets need none.

    Attributes
    ----------
    variance : torch.Tensor
        Posterior variance of each target for the current montage: 0 when
        measured or exact, from the estimate (before ``match_gain``) for a
        single-electrode target of ``spline``, ``wiener`` or a kernel, ``nan``
        when not computed (other strategies, or a target mixing several
        reconstructed channels, as with ``reference="average"``).

    Notes
    -----
    ``forward`` also takes one montage per sample (``chs_info`` a list of
    montages, ``x`` padded to the largest and ``ch_mask`` the real channels),
    for the strategies without trained state (not ``latent`` or ``trainable``);
    ``observed`` is then ``(batch, n_targets)`` and the buffers (``weight``,
    ``variance``) are those of the last sample.
    A model with a ``channel_strategy`` builds these montages from the ``pos`` and
    ``ch_mask`` of :func:`~braindecode.datasets.pad_channels_collate`. A function
    strategy is not saved by ``save_pretrained``: pass it again when loading.
    """

    def __init__(
        self,
        target,
        strategy,
        chs_info=None,
        *,
        drop_non_eeg=False,
        trainable=False,
        reference=None,
        **kwargs,
    ):
        super().__init__()
        if not callable(strategy) and strategy not in STRATEGIES:
            close = difflib.get_close_matches(strategy, STRATEGIES, n=3)
            raise ValueError(
                f"Unknown channel strategy {strategy!r}. Did you mean {close}? Available: {STRATEGIES}."
            )
        geometry = _channel_geometry(target)
        pos, given = geometry.positions, geometry.has_location
        self.target = [
            c if g or np.isnan(p).any() else {**c, "loc": np.r_[p, np.zeros(9)]}
            for c, p, g in zip(target, pos, given)
        ]
        self.strategy, self.chs_info, self.reference = strategy, chs_info, reference
        self.drop_non_eeg, self.kwargs = drop_non_eeg, kwargs
        # Per-montage maps: LRU of device tensors; the current one is the buffers.
        self._cache, self._key = OrderedDict(), None
        self.register_buffer("weight", torch.zeros(0, 0), persistent=False)
        parts = [_channel_electrodes(str(ch["ch_name"])) for ch in self.target]
        named = {
            ends[0].lower(): ch
            for ends, ch in zip(parts, self.target)
            if len(ends) == 1
        }
        self._referenced = np.array(
            [
                len(ends) == 1 and kind in _ELECTRODES
                for ends, kind in zip(parts, geometry.kinds)
            ]
        )
        unpositioned = {
            row: ends[0]
            for row, ends in enumerate(parts)
            if self._referenced[row] and not np.isfinite(pos[row]).all()
        }
        if reference not in (None, "average"):
            parts = [
                [ends[0], reference] if use_reference else ends
                for ends, use_reference in zip(parts, self._referenced)
            ]
        # A measured A-B takes precedence over reconstructing A and B separately.
        bipolar = {
            row: "-".join(ends)
            for row, ends in enumerate(parts)
            if len(ends) == 2 and all("-" not in end for end in ends)
        }
        derivations, mono = _derivation_rows(parts, bipolar.values())
        columns = {name.lower(): i for i, name in enumerate(mono)}
        self._unpositioned = {
            row: columns[name.lower()] for row, name in unpositioned.items()
        }
        self._whole = [
            columns[bipolar[row].lower()] if row in bipolar else -1
            for row in range(len(parts))
        ]
        geometry = _channel_geometry(
            [named.get(m.lower(), {"ch_name": m}) for m in mono]
        )
        self._mono = mono
        self._tpos = geometry.positions
        self._tstd = geometry.standard_positions
        self._D = derivations.numpy()
        self.trainable = trainable and strategy == "source"
        if self.trainable:
            n = kwargs.get("n_parcels", 64)
            self.parcel_mix = nn.Parameter(torch.zeros(n, n))
        if strategy == "wiener":  # fitted state, saved in the state_dict
            self.register_buffer("cov", torch.zeros(0, 0))
            self.register_buffer("dense_pos", torch.zeros(0, 3))
        if strategy == "latent":
            d, self.n_freqs = kwargs.pop("d_model", 64), kwargs.pop("n_freqs", 8)
            f = 6 * self.n_freqs  # queries: target positions; keys: input positions
            self.query, self.key_pos = (
                nn.Sequential(nn.Linear(f, d), nn.GELU(), nn.Linear(d, d))
                for _ in range(2)
            )
            self.key_stats = nn.Linear(2, d)  # + input statistics
        if chs_info is not None and strategy != "wiener":  # wiener: after fit()
            self._use(chs_info)  # surface montage errors at construction

    def _use(self, chs: list[dict]) -> None:
        """Select the cached map for the current channel metadata, or build one."""
        metadata = tuple(
            (
                str(c["ch_name"]),
                c.get("kind") or c.get("ch_type") or "eeg",
                np.asarray(c.get("loc"), dtype=np.float64).tobytes(),
            )
            for c in chs
        )
        key = (self.drop_non_eeg, metadata)
        if key != self._key:
            maps = self._cache.pop(key, None) or self._build(chs)
            self._cache[key] = maps  # most recent last
            # shortcut: keep 16 maps; revisit if mixed batches repeatedly rebuild evicted maps.
            if len(self._cache) > 16:
                self._cache.popitem(last=False)
            for name, t in maps.items():
                self.register_buffer(name, t, persistent=False)
            self._key = key

    @torch.inference_mode(False)  # maps built in an eval pass serve training
    def _build(self, chs_info: list[dict]) -> dict[str, Tensor]:
        geometry = _channel_geometry(chs_info)
        names, pos = geometry.names, geometry.positions
        std, types = geometry.standard_positions, geometry.kinds
        bad = [n for n, t in zip(names, types) if t not in _ELECTRODES]
        if bad and not self.drop_non_eeg:
            raise ValueError(
                f"Channels {bad} are not EEG electrodes; remove them (e.g. raw.pick('eeg')) or pass drop_non_eeg=True."
            )
        keep = np.array([t in _ELECTRODES for t in types])
        if self.strategy == "source" and {"seeg", "ecog", "dbs"} & set(types):
            raise ValueError(
                "channel_strategy='source' uses a scalp-EEG sphere head; use a sensor strategy for intracranial channels."
            )
        mono, tpos, tstd = self._mono, self._tpos, self._tstd
        # Copy: same name, else same standard position (T3 = T7), else unknown name within 15 mm.
        low = np.array([n.lower() for n in names])
        alias = np.nan_to_num(cdist(tstd, std), nan=np.inf) < 1e-6
        dist = np.nan_to_num(cdist(tpos, pos), nan=np.inf)
        near = np.where(np.isnan(std).any(1), dist, np.inf)
        # Names before positions: B-A shares the midpoint of A-B but needs -1.
        kept = np.flatnonzero(keep)
        derived, exact = exact_derivation(mono, [names[j] for j in kept])
        W = np.zeros((len(mono), len(names)))
        for k, m in enumerate(mono):
            direct = (low == m.lower()) & keep
            if direct.any():
                j = np.flatnonzero(direct)
                W[k, j[np.argmin(near[k][j])]] = 1.0
                continue
            if exact[k]:  # a combination of measured channels
                W[k, kept] = derived[k].numpy()
                continue
            for hit in (alias[k], near[k] <= 0.015):
                if (hit & keep).any():
                    j = np.flatnonzero(hit & keep)
                    W[k, j[np.argmin(near[k][j])]] = 1.0
                    break
        observed = np.abs(W).sum(1) > 0
        D = self._D.copy()
        for row, column in enumerate(self._whole):
            if column >= 0 and observed[column]:
                D[row] = 0
                D[row, column] = 1
        ref = self._referenced.copy()
        for row, electrode in getattr(self, "_unpositioned", {}).items():
            whole = self._whole[row]
            if not observed[electrode] and not (whole >= 0 and observed[whole]):
                # An unmeasured target without a position cannot be re-referenced.
                ref[row] = False
                D[row] = 0
                D[row, electrode] = 1
        measured = (np.abs(D) @ ~observed == 0).any()
        if self.reference == "average" and ref.any():  # minus their mean
            D[ref] -= D[ref].mean(0)
        missing = ~observed & np.abs(D).any(0)
        todo = missing & np.isfinite(tpos).all(1)
        if self.strategy == "exact" and todo.any():
            raise ValueError(
                f"Strategy 'exact': target channels {[m for m, t in zip(mono, todo) if t]} are not in the input montage {names}; use a reconstructing strategy (e.g. 'spline') or supply them."
            )
        seen = np.abs(D) @ missing == 0
        if not measured and (self.strategy == "zero" or not todo.any()):
            raise ValueError(
                f"No target channel of {[c['ch_name'] for c in self.target]} is in the input montage {names}, and strategy {self.strategy!r} reconstructs none: the model would see only zeros. Supply matching names or positions, or a reconstructing strategy (e.g. 'spline')."
            )
        if (missing & ~todo).any():
            warnings.warn(
                f"Target channels {[m for m, t in zip(mono, missing & ~todo) if t]} are not in the input montage and have no position to reconstruct them from; they are zero-filled.",
                UserWarning,
                stacklevel=4,
            )
        use = keep & np.isfinite(pos).all(1)
        need = _MIN_POSITIONS.get(self.strategy, 4)  # a kernel: as spline
        if todo.any() and use.sum() < need:
            raise ValueError(
                f"Strategy {self.strategy!r} needs at least {need} channels with a position; got {int(use.sum())} of {names}. Supply 'loc' or standard 10-05 names."
            )
        fitted = {  # wiener: device-safe float64 copies
            k: getattr(self, k).detach().cpu().double().numpy()
            for k in ("cov", "dense_pos")
            if hasattr(self, k)
        }
        var = np.where(observed, 0.0, np.nan)
        source_factors = None
        if todo.any():
            src_pos, tgt_pos = pos[use], tpos[todo]
            if self.strategy == "source":
                source_factors = _source(src_pos, tgt_pos, **self.kwargs)
                lead, inverse = source_factors
                weights = lead @ inverse + 1.0 / len(src_pos)
            else:
                weights, variance = _mixing(
                    self.strategy, src_pos, tgt_pos, **self.kwargs, **fitted
                )
                if variance is not None:
                    var[todo] = variance
            W[np.ix_(todo, use)] = weights
            gain = np.abs(W[todo]).sum(1).max()
            if gain > 2:
                warnings.warn(
                    f"Strategy {self.strategy!r}: reconstructed row gain |w|_1 = {gain:.3g} > 2; the map amplifies noise. Supply more channels or a smoother strategy.",
                    UserWarning,
                    stacklevel=4,
                )
        dev, dt = self.weight.device, self.weight.dtype  # built where the module is
        # one reconstructed entry in a row: its variance times its weight squared
        hit = (D != 0) & missing
        j, rows = hit.argmax(1), np.arange(len(D))
        own = D[rows, j] ** 2 * var[j]
        variance = np.where(seen, 0.0, np.where(hit.sum(1) == 1, own, np.nan))
        maps = dict(weight=D @ W, variance=variance)
        if self.trainable:  # nothing to reconstruct: zero correction, no inverse
            n = self.parcel_mix.shape[0]
            maps["inverse"] = np.zeros((n, len(names)))
            if source_factors is None:
                maps["lead"] = np.zeros((len(D), n))
            else:
                lead, inverse = source_factors
                maps["lead"] = D[:, todo] @ lead
                maps["inverse"][:, use] = inverse
        if self.strategy == "latent":  # Fourier features of positions (in dm)
            arg = np.nan_to_num(np.vstack([pos, tpos[todo]]))[..., None] * 10 * np.pi
            arg = arg * 2.0 ** np.arange(self.n_freqs)
            feat = np.concatenate([np.sin(arg), np.cos(arg)], -1).reshape(len(arg), -1)
            maps.update(src_feat=feat[: len(pos)], tgt_feat=feat[len(pos) :])
            maps.update(lat_D=D[:, todo], lat_w=use / max(use.sum(), 1))
            maps["lat_mask"] = torch.as_tensor(use, device=dev)
        maps = {
            k: v if torch.is_tensor(v) else torch.tensor(v, device=dev, dtype=dt)
            for k, v in maps.items()
        }
        if self.drop_non_eeg:
            maps["electrode_mask"] = torch.as_tensor(keep, device=dev)
        maps["observed"] = torch.as_tensor(seen, device=dev)
        return maps

    def _apply(self, fn, recurse=True):
        for maps in self._cache.values():  # cached maps follow .to() / .half()
            maps.update({k: fn(t) for k, t in maps.items()})
        super()._apply(fn, recurse)
        self._buffers.update(self._cache.get(self._key, {}))
        return self

    def fit(self, X, chs_info_dense: list[dict]) -> ChannelLayer:
        """Fit the Wiener spatial covariance on a dense recording.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_channels)
            Samples from the densely recorded montage.
        chs_info_dense : list of dict
            Channels in the column order of ``X``. Positions are in metres in
            the head frame. A standard electrode name can supply a missing position.

        Returns
        -------
        self : ChannelLayer
            The fitted layer. The covariance and positions are saved in its state dict.
        """
        pos = _channel_geometry(chs_info_dense).positions
        X = np.asarray(X, float)
        if self.strategy != "wiener" or X.shape[1:] != pos.shape[:1]:
            raise ValueError(
                f"fit() is for 'wiener' and needs X (n_samples, {len(pos)}); got {self.strategy!r}, X {X.shape}."
            )
        dev = self.get_buffer("cov").device
        self.cov = torch.tensor(np.cov(X.T), dtype=torch.float32, device=dev)
        self.dense_pos = torch.tensor(pos, dtype=torch.float32, device=dev)
        self._cache.clear()
        self._key = None
        return self

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        self._cache.clear()  # maps are rebuilt after a load
        self._key = None
        for k in ("cov", "dense_pos"):  # fitted buffers change size
            if hasattr(self, k) and prefix + k in state_dict:
                setattr(self, k, torch.empty_like(state_dict[prefix + k]))
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def forward(
        self,
        x: Tensor,
        chs_info: list[dict] | list[list[dict]] | None = None,
        ch_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Map a batch onto the target montage in target channel order.

        Parameters
        ----------
        x : torch.Tensor of shape (batch, channels, time)
            Input signal, padded to a common channel count for mixed montages.
        chs_info : list of dict or list of list of dict, optional
            Channel metadata in signal order, shared by the batch or given per
            sample. Defaults to the construction montage.
        ch_mask : torch.Tensor of shape (batch, channels), optional
            For per-sample montages, select real channel rows. Without a mask,
            the first ``len(montage)`` rows hold the signal for each sample.

        Returns
        -------
        mapped : torch.Tensor of shape (batch, n_targets, time)
            Signals expressed in the target montage.
        observed : torch.Tensor
            Boolean mask of targets measured or exactly derived. Its shape is
            ``(n_targets,)`` for a shared montage and ``(batch, n_targets)``
            for per-sample montages.
        """
        chs = chs_info if chs_info is not None else self.chs_info
        if chs is None:
            raise ValueError(
                f"No montage: pass chs_info to the model (or to forward) so channel strategy {self.strategy!r} knows the input channels."
            )
        if chs and isinstance(chs[0], (list, tuple)):  # one montage per sample
            big = max(map(len, chs)) > x.shape[1]
            if self.trainable or self.strategy == "latent" or len(chs) != len(x) or big:
                raise ValueError(
                    f"One montage per sample needs a strategy without trained state and one montage of at most {x.shape[1]} channels for each of the {len(x)} samples; got {self.strategy!r} and {len(chs)} montages."
                )
            ch_mask = check_ch_mask(ch_mask, x)
            signals, observed = [], []
            for b, montage in enumerate(cast(list[list[dict]], chs)):
                signal = x[b, : len(montage)] if ch_mask is None else x[b, ch_mask[b]]
                if signal.shape[0] != len(montage):
                    raise ValueError(
                        f"Sample {b}: ch_mask has {signal.shape[0]} channels, "
                        f"its montage {len(montage)}."
                    )
                self._use(montage)
                if self.drop_non_eeg:
                    signal = signal.masked_fill(~self.electrode_mask[:, None], 0)
                signals.append(self.weight @ signal)
                observed.append(self.observed)
            return torch.stack(signals), torch.stack(observed)
        if ch_mask is not None:
            raise ValueError("ch_mask needs one montage per sample in chs_info.")
        if x.shape[1] != len(chs):
            raise ValueError(
                f"Input has {x.shape[1]} channels but the montage has {len(chs)}; they must match."
            )
        self._use(cast(list[dict], chs))
        if self.drop_non_eeg:
            x = x.masked_fill(~self.electrode_mask[:, None], 0)
        out = self.weight @ x
        if self.trainable:
            out = out + self.lead @ (self.parcel_mix @ (self.inverse @ x))
        if self.strategy == "latent" and self.lat_D.shape[1]:
            xf = x.to(torch.promote_types(x.dtype, torch.float32))  # fp16: 1e-12 = 0
            if x.shape[-1] == 1:
                stats = xf.new_zeros((*xf.shape[:2], 2))
            else:
                var = (xf - xf.mean(-1, keepdim=True)).square().sum(-1) / (
                    x.shape[-1] - 1
                )
                stats = torch.stack([var, xf.diff(dim=-1).abs().mean(-1)], -1)
            stats = (stats + 1e-12).log()
            stats = stats - (self.lat_w.to(stats.dtype) @ stats)[:, None]  # (B, C, 2)
            keys = self.key_pos(self.src_feat) + self.key_stats(stats.to(x.dtype))
            q = self.query(self.tgt_feat)
            logits = q @ keys.transpose(1, 2) / keys.shape[-1] ** 0.5
            attn = logits.masked_fill(~self.lat_mask, float("-inf")).softmax(-1)
            out = out + self.lat_D @ (attn @ x)
        return out, self.observed
