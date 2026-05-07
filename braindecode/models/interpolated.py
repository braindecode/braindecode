# Authors: Pierre Guetschel
#
# License: BSD (3-clause)
from __future__ import annotations

from typing import Literal, Optional, Type

import numpy as np
import torch.nn as nn

from braindecode.modules.interpolation import ChannelInterpolationLayer


def InterpolatedModel(
    model_cls: Type,
    target_chs_info: list[dict],
    name: Optional[str] = None,
) -> Type:
    """Return a subclass of ``model_cls`` that interpolates channels to ``target_chs_info``.

    .. warning:: Experimental. Public API may change without a deprecation cycle.

    Parameters
    ----------
    model_cls : type
        A braindecode model class (subclass of
        :class:`~braindecode.models.base.EEGModuleMixin`).
    target_chs_info : list of dict
        The canonical channel set the backbone expects internally. Every
        instance of the returned class projects its input channels to
        this set via :class:`~braindecode.modules.ChannelInterpolationLayer`.
    name : str, optional
        ``__name__`` to assign to the returned class. Defaults to
        ``f"Interpolated{model_cls.__name__}"``.

    Returns
    -------
    type
        A new subclass of ``model_cls`` whose ``__init__`` accepts
        arbitrary user ``chs_info`` and automatically inserts a frozen
        (by default) channel-interpolation layer before the backbone.
    """
    _is_sequential = issubclass(model_cls, nn.Sequential)

    class _Interpolated(model_cls):
        _TARGET_CHS_INFO = target_chs_info

        def __init__(
            self,
            chs_info,
            n_outputs=None,
            n_times=None,
            input_window_seconds=None,
            sfreq=None,
            n_chans=None,
            interpolation_method: str = "spline",
            interpolation_mode: Literal["always", "name_match"] = "name_match",
            trainable: bool = False,
            **kwargs,
        ):
            # Signal-related params are declared EXPLICITLY here so that
            # skorch's ``EEGClassifier._set_signal_args`` (which inspects
            # the ``__init__`` signature via ``inspect.signature``) can
            # auto-forward them from the training dataset. They would not
            # be discoverable if they were collapsed into ``**kwargs``.
            # ``n_chans`` is declared for the same discoverability reason
            # but intentionally ignored: the backbone must see
            # ``len(target_chs_info)`` derived from ``chs_info``.
            del n_chans
            # Backbone init uses the target channels. During this call,
            # some backbones run a dummy forward (e.g. to size the head);
            # ``self.interpolation_layer`` does not exist yet — the
            # ``forward`` override below falls back to pass-through when
            # the attribute is absent. Assigning an ``nn.Identity()``
            # before ``super()`` is impossible: ``nn.Module.__setattr__``
            # requires ``nn.Module.__init__`` to have run, and the chain
            # would wipe ``self._modules`` when it reaches it again.
            super().__init__(
                chs_info=target_chs_info,
                n_outputs=n_outputs,
                n_times=n_times,
                input_window_seconds=input_window_seconds,
                sfreq=sfreq,
                **kwargs,
            )

            layer = ChannelInterpolationLayer(
                src_chs_info=chs_info,
                tgt_chs_info=target_chs_info,
                mode=interpolation_mode,
                method=interpolation_method,
                trainable=trainable,
            )
            if _is_sequential:
                # For nn.Sequential subclasses, prepend the interpolation
                # layer so that nn.Sequential.forward runs it first.
                # Registering via attribute assignment appends to _modules;
                # instead we rebuild _modules with the layer first.
                old_modules = list(self._modules.items())  # type: ignore[has-type]
                self._modules.clear()  # type: ignore[has-type]
                self._modules["interpolation_layer"] = layer  # type: ignore[index]
                for k, v in old_modules:
                    self._modules[k] = v  # type: ignore[index]
            else:
                self.interpolation_layer = layer

            # Rebind private attrs so the user-facing view (.chs_info,
            # .n_chans, .input_shape, build_model_config) reflects the
            # user's channels. Properties are NOT overridden — we mutate
            # the private attrs the base-class properties read from.
            self._chs_info = chs_info
            self._n_chans = len(chs_info)

        if not _is_sequential:

            def forward(self, x, *args, **kwargs):
                # During super().__init__() the interpolation_layer attr
                # does not exist yet; any dummy forward call (e.g. from
                # get_output_shape) must pass through unchanged so the
                # backbone sees its expected target-shape input.
                # Forward *args / **kwargs so backbone-specific flags
                # like ``return_features`` keep working through the
                # wrapper.
                interp = getattr(self, "interpolation_layer", None)
                if interp is not None:
                    x = interp(x)
                return super().forward(x, *args, **kwargs)

    _Interpolated.__name__ = name or f"Interpolated{model_cls.__name__}"
    _Interpolated.__qualname__ = _Interpolated.__name__
    # Propagate the backbone docstring so Sphinx and the categorization tests
    # can read the class badges.  Prepend a short header so the class shows
    # up clearly in documentation as distinct from the backbone.
    backbone_doc = model_cls.__doc__ or ""
    _Interpolated.__doc__ = (
        f"Channel-interpolating wrapper around :class:`{model_cls.__name__}`.\n\n"
        ":bdg-dark-line:`Channel`\n\n"
        f"Accepts arbitrary user ``chs_info`` and projects them to the\n"
        f"backbone's canonical channel set via\n"
        f":class:`~braindecode.modules.ChannelInterpolationLayer`.\n\n"
        f"For all other parameters and behavior see the backbone\n"
        f"documentation reproduced below.\n\n" + backbone_doc
    )
    return _Interpolated


def _build_chs_info_from_montage(names: list[str], montage: str) -> list[dict]:
    """Build a ``list[dict]`` ``chs_info`` from channel names + an MNE montage.

    Each returned dict has ``ch_name``, ``kind="eeg"``, and ``loc`` (shape
    ``(3,)``). Used by braindecode's shipped ``Interpolated*`` variants to
    turn a bare list of canonical channel names into the dict form
    ``ChannelInterpolationLayer`` expects.

    Parameters
    ----------
    names : list of str
        Channel names in the desired order.
    montage : str
        Name of an MNE standard montage (e.g. ``"standard_1005"``).

    Returns
    -------
    list of dict

    Raises
    ------
    ValueError
        If a name is not found in the montage.
    """
    import mne

    mtg = mne.channels.make_standard_montage(montage)
    ch_pos = mtg.get_positions()["ch_pos"]
    out = []
    for n in names:
        if n not in ch_pos:
            raise ValueError(f"Channel {n!r} not found in montage {montage!r}.")
        out.append(
            {"ch_name": n, "kind": "eeg", "loc": np.asarray(ch_pos[n], dtype=float)}
        )
    return out
