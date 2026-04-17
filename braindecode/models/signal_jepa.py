# Authors: Pierre Guetschel <pierre.guetschel@gmail.com>
#
# License: BSD (3-clause)
from __future__ import annotations

import math
from copy import deepcopy
from pathlib import Path
from typing import Any, Optional, Sequence

import torch
from einops.layers.torch import Rearrange
from torch import nn

from braindecode.models.base import EEGModuleMixin

_DEFAULT_CONV_LAYER_SPEC = (  # downsampling: 128Hz -> 1Hz, receptive field 1.1875s, stride 1s
    (8, 32, 8),
    (16, 2, 2),
    (32, 2, 2),
    (64, 2, 2),
    (64, 2, 2),
)

# The 62 channels used to pre-train Signal-JEPA. Derived from
# `moabb.datasets.Lee2019_SSVEP` subject 1 session '1' run '1train'
# (the first 62 ch_names; the last 5 of the dataset's 67 channels are
# 4 EMG + 1 stim). Positions are from MNE's `standard_1005` montage.
# Names and ORDER MUST match the _ChannelEmbedding row order of the
# published HuggingFace checkpoint `braindecode/signal-jepa`.
_PRETRAIN_CHS_INFO: list[dict] = [
    {"ch_name": "Fp1", "loc": [-0.0294367, 0.0839171, -0.00699]},
    {"ch_name": "Fp2", "loc": [0.0298723, 0.0848959, -0.00708]},
    {"ch_name": "F7", "loc": [-0.0702629, 0.0424743, -0.01142]},
    {"ch_name": "F3", "loc": [-0.0502438, 0.0531112, 0.042192]},
    {"ch_name": "Fz", "loc": [0.0003122, 0.058512, 0.066462]},
    {"ch_name": "F4", "loc": [0.0518362, 0.0543048, 0.040814]},
    {"ch_name": "F8", "loc": [0.0730431, 0.0444217, -0.012]},
    {"ch_name": "FC5", "loc": [-0.0772149, 0.0186433, 0.02446]},
    {"ch_name": "FC1", "loc": [-0.0340619, 0.0260111, 0.079987]},
    {"ch_name": "FC2", "loc": [0.0347841, 0.0264379, 0.078808]},
    {"ch_name": "FC6", "loc": [0.0795341, 0.0199357, 0.024438]},
    {"ch_name": "T7", "loc": [-0.0841611, -0.0160187, -0.009346]},
    {"ch_name": "C3", "loc": [-0.0653581, -0.0116317, 0.064358]},
    {"ch_name": "Cz", "loc": [0.0004009, -0.009167, 0.100244]},
    {"ch_name": "C4", "loc": [0.0671179, -0.0109003, 0.06358]},
    {"ch_name": "T8", "loc": [0.0850799, -0.0150203, -0.00949]},
    {"ch_name": "TP9", "loc": [-0.0856192, -0.0465147, -0.045707]},
    {"ch_name": "CP5", "loc": [-0.0795922, -0.0465507, 0.030949]},
    {"ch_name": "CP1", "loc": [-0.0355131, -0.0472919, 0.091315]},
    {"ch_name": "CP2", "loc": [0.0383838, -0.0470731, 0.090695]},
    {"ch_name": "CP6", "loc": [0.0833218, -0.0461013, 0.031206]},
    {"ch_name": "TP10", "loc": [0.0861618, -0.0470353, -0.045869]},
    {"ch_name": "P7", "loc": [-0.0724343, -0.0734527, -0.002487]},
    {"ch_name": "P3", "loc": [-0.0530073, -0.0787878, 0.05594]},
    {"ch_name": "Pz", "loc": [0.0003247, -0.081115, 0.082615]},
    {"ch_name": "P4", "loc": [0.0556667, -0.0785602, 0.056561]},
    {"ch_name": "P8", "loc": [0.0730557, -0.0730683, -0.00254]},
    {"ch_name": "PO9", "loc": [-0.0549104, -0.0980448, -0.035465]},
    {"ch_name": "O1", "loc": [-0.0294134, -0.112449, 0.008839]},
    {"ch_name": "Oz", "loc": [0.0001076, -0.114892, 0.014657]},
    {"ch_name": "O2", "loc": [0.0298426, -0.112156, 0.0088]},
    {"ch_name": "PO10", "loc": [0.0549876, -0.0980911, -0.035541]},
    {"ch_name": "FC3", "loc": [-0.0601819, 0.0227162, 0.055544]},
    {"ch_name": "FC4", "loc": [0.0622931, 0.0237228, 0.05563]},
    {"ch_name": "C5", "loc": [-0.0802801, -0.0137597, 0.02916]},
    {"ch_name": "C1", "loc": [-0.036158, -0.0099839, 0.089752]},
    {"ch_name": "C2", "loc": [0.037672, -0.0096241, 0.088412]},
    {"ch_name": "C6", "loc": [0.0834559, -0.0127763, 0.029208]},
    {"ch_name": "CP3", "loc": [-0.0635562, -0.0470088, 0.065624]},
    {"ch_name": "CPz", "loc": [0.0003858, -0.047318, 0.099432]},
    {"ch_name": "CP4", "loc": [0.0666118, -0.0466372, 0.06558]},
    {"ch_name": "P1", "loc": [-0.0286203, -0.0805249, 0.075436]},
    {"ch_name": "P2", "loc": [0.0319197, -0.0804871, 0.076716]},
    {"ch_name": "POz", "loc": [0.0002156, -0.102178, 0.050608]},
    {"ch_name": "FT9", "loc": [-0.0840759, 0.0145673, -0.050429]},
    {"ch_name": "FTT9h", "loc": [-0.084125, -0.0018467, -0.029794]},
    {"ch_name": "TTP7h", "loc": [-0.0855651, -0.0306287, 0.011153]},
    {"ch_name": "TP7", "loc": [-0.0848302, -0.0460217, -0.007056]},
    {"ch_name": "TPP9h", "loc": [-0.0781602, -0.0607567, -0.023824]},
    {"ch_name": "FT10", "loc": [0.0841131, 0.0143647, -0.050538]},
    {"ch_name": "FTT10h", "loc": [0.084123, -0.0018083, -0.029638]},
    {"ch_name": "TPP8h", "loc": [0.0785198, -0.0604323, 0.012902]},
    {"ch_name": "TP8", "loc": [0.0855488, -0.0455453, -0.00713]},
    {"ch_name": "TPP10h", "loc": [0.0789027, -0.0609553, -0.023805]},
    {"ch_name": "F9", "loc": [-0.0701019, 0.0416523, -0.049952]},
    {"ch_name": "F10", "loc": [0.0721141, 0.0420667, -0.050452]},
    {"ch_name": "AF7", "loc": [-0.0548397, 0.0685722, -0.01059]},
    {"ch_name": "AF3", "loc": [-0.0337007, 0.0768371, 0.021227]},
    {"ch_name": "AF4", "loc": [0.0357123, 0.0777259, 0.021956]},
    {"ch_name": "AF8", "loc": [0.0557433, 0.0696568, -0.010755]},
    {"ch_name": "PO3", "loc": [-0.0365114, -0.1008529, 0.037167]},
    {"ch_name": "PO4", "loc": [0.0367816, -0.1008491, 0.036397]},
]


def _resolve_channel_embedding_config(
    channel_embedding: str,
    chs_info: list[dict] | None,
) -> tuple[list[dict], list[list[float] | None], torch.LongTensor]:
    """Resolve the construction parameters for ``_ChannelEmbedding``.

    Parameters
    ----------
    channel_embedding : {"scratch", "pretrain_aligned"}
        How to initialize ``_ChannelEmbedding``. See
        ``SignalJEPA`` docstring.
    chs_info : list of dict | None
        User-supplied channel information. Each element must be a
        mapping with at least the keys ``"ch_name"`` and ``"loc"``.

    Returns
    -------
    effective_chs_info : list of dict
        The ``chs_info`` that ``EEGModuleMixin`` should see. Equal to
        ``chs_info`` when provided, else ``_PRETRAIN_CHS_INFO``.
    channel_locations : list of (list of float or None)
        Locations used to initialize ``_ChannelEmbedding``. Length
        equals ``num_embeddings`` (N in ``'scratch'`` mode, 62 in
        ``'pretrain_aligned'`` mode).
    ch_idxs : torch.LongTensor, shape ``(len(effective_chs_info),)``
        Mapping from the user channel order to embedding indices.

    Raises
    ------
    ValueError
        If ``channel_embedding`` is not recognized, if
        ``channel_embedding='scratch'`` is combined with
        ``chs_info=None``, or if ``channel_embedding='pretrain_aligned'``
        is combined with a ``chs_info`` that contains a channel name
        absent from ``_PRETRAIN_CHS_INFO`` (case-insensitive match).
    """
    if channel_embedding not in ("scratch", "pretrain_aligned"):
        raise ValueError(
            "channel_embedding must be 'scratch' or 'pretrain_aligned', "
            f"got {channel_embedding!r}"
        )

    if channel_embedding == "scratch":
        if chs_info is None:
            raise ValueError("chs_info is required when channel_embedding='scratch'")
        effective_chs_info = list(chs_info)
        channel_locations = [ch["loc"] for ch in effective_chs_info]
        ch_idxs = torch.arange(len(effective_chs_info), dtype=torch.long)
        return effective_chs_info, channel_locations, ch_idxs

    # channel_embedding == "pretrain_aligned"
    pretrain_name_to_idx = {
        ch["ch_name"].lower(): i for i, ch in enumerate(_PRETRAIN_CHS_INFO)
    }
    channel_locations = [ch["loc"] for ch in _PRETRAIN_CHS_INFO]

    if chs_info is None:
        effective_chs_info = [dict(ch) for ch in _PRETRAIN_CHS_INFO]
        ch_idxs = torch.arange(len(_PRETRAIN_CHS_INFO), dtype=torch.long)
        return effective_chs_info, channel_locations, ch_idxs

    missing = [
        ch["ch_name"]
        for ch in chs_info
        if ch["ch_name"].lower() not in pretrain_name_to_idx
    ]
    if missing:
        raise ValueError(
            f"Channel(s) {missing} not in the Signal-JEPA pre-training set. "
            "To load pretrained weights while keeping these channels, use:\n"
            "  channel_embedding='scratch'\n"
            "  hub_id='braindecode/signal-jepa_without-chans'\n"
            "  strict=False  # on load_state_dict / from_pretrained\n"
            "The pretrained channel embedding weights will not be loaded."
        )

    effective_chs_info = list(chs_info)
    ch_idxs = torch.tensor(
        [pretrain_name_to_idx[ch["ch_name"].lower()] for ch in chs_info],
        dtype=torch.long,
    )
    return effective_chs_info, channel_locations, ch_idxs


class _BaseSignalJEPA(EEGModuleMixin, nn.Module):
    r"""Base class for the SignalJEPA models

    Parameters
    ----------
    channel_embedding : {"scratch", "pretrain_aligned"}, default ``"scratch"``
        How to initialize the :class:`_ChannelEmbedding` table.

        * ``"scratch"``: table has ``len(chs_info)`` rows, initialized from
          user locations. ``chs_info`` is required.
        * ``"pretrain_aligned"``: table has 62 rows, initialized from the
          pre-training locations. If ``chs_info`` is provided, every channel
          name must match one in the pre-training set (case-insensitive);
          ``forward`` will then expect input with as many channels as
          ``chs_info`` has. If ``chs_info=None``, the model runs on the
          full 62 channels in the pre-training order.

    feature_encoder__conv_layers_spec: list of tuple
        tuples have shape ``(dim, k, stride)`` where:

        * ``dim`` : number of output channels of the layer (unrelated to EEG channels);
        * ``k`` : temporal length of the layer's kernel;
        * ``stride`` : temporal stride of the layer's kernel.

    drop_prob: float
    feature_encoder__mode: str
        Normalisation mode. Either ``default`` or ``layer_norm``.
    feature_encoder__conv_bias: bool
    activation: nn.Module
        Activation layer for the feature encoder.
    pos_encoder__spat_dim: int
        Number of dimensions to use to encode the spatial position of the patch,
        i.e. the EEG channel.
    pos_encoder__time_dim: int
        Number of dimensions to use to encode the temporal position of the patch.
    pos_encoder__sfreq_features: float
        The "downsampled" sampling frequency returned by the feature encoder.
    pos_encoder__spat_kwargs: dict
        Additional keyword arguments to pass to the :class:`nn.Embedding` layer used to
        embed the channel names.
    transformer__d_model: int
        The number of expected features in the encoder/decoder inputs.
    transformer__num_encoder_layers: int
        The number of encoder layers in the transformer.
    transformer__num_decoder_layers: int
        The number of decoder layers in the transformer.
    transformer__nhead: int
        The number of heads in the multiheadattention models.
    _init_feature_encoder : bool
        Do not change the default value (used for internal purposes).
    _init_transformer : bool
        Do not change the default value (used for internal purposes).
    """

    feature_encoder: _ConvFeatureEncoder | None
    pos_encoder: _PosEncoder | None
    transformer: nn.Transformer | None

    _feature_encoder_channels: str = "n_chans"

    def __init__(
        self,
        n_outputs=None,
        n_chans=None,
        chs_info=None,
        n_times=None,
        input_window_seconds=None,
        sfreq=None,
        *,
        # feature_encoder
        feature_encoder__conv_layers_spec: Sequence[
            tuple[int, int, int]
        ] = _DEFAULT_CONV_LAYER_SPEC,
        drop_prob: float = 0.0,
        feature_encoder__mode: str = "default",
        feature_encoder__conv_bias: bool = False,
        activation: type[nn.Module] = nn.GELU,
        # pos_encoder
        pos_encoder__spat_dim: int = 30,
        pos_encoder__time_dim: int = 34,
        pos_encoder__sfreq_features: float = 1.0,
        pos_encoder__spat_kwargs: dict | None = None,
        # transformer
        transformer__d_model: int = 64,
        transformer__num_encoder_layers: int = 8,
        transformer__num_decoder_layers: int = 4,
        transformer__nhead: int = 8,
        # other
        channel_embedding: str = "scratch",
        _init_feature_encoder: bool,
        _init_transformer: bool,
    ):
        # Resolve channel embedding config before calling super().__init__
        if _init_transformer:
            effective_chs_info, channel_locations, ch_idxs = (
                _resolve_channel_embedding_config(channel_embedding, chs_info)
            )
        else:
            effective_chs_info = chs_info
            channel_locations = None
            ch_idxs = None

        super().__init__(
            n_outputs=n_outputs,
            n_chans=n_chans,
            chs_info=effective_chs_info,
            n_times=n_times,
            input_window_seconds=input_window_seconds,
            sfreq=sfreq,
        )
        del n_outputs, n_chans, chs_info, n_times, input_window_seconds, sfreq

        # Store channel_embedding after super().__init__ so it's tracked by build_model_config
        self._channel_embedding = channel_embedding

        self.feature_encoder = None
        self.pos_encoder = None
        self.transformer = None
        if _init_feature_encoder:
            self.feature_encoder = _ConvFeatureEncoder(
                conv_layers_spec=feature_encoder__conv_layers_spec,
                channels=getattr(self, self._feature_encoder_channels),
                drop_prob=drop_prob,
                mode=feature_encoder__mode,
                conv_bias=feature_encoder__conv_bias,
                activation=activation,
            )

        if _init_transformer:
            assert channel_locations is not None  # narrowing for type checker
            assert ch_idxs is not None
            self.pos_encoder = _PosEncoder(
                spat_dim=pos_encoder__spat_dim,
                time_dim=pos_encoder__time_dim,
                channel_locations=channel_locations,
                ch_idxs=ch_idxs,
                sfreq_features=pos_encoder__sfreq_features,
                spat_kwargs=pos_encoder__spat_kwargs,
            )
            self.transformer = nn.Transformer(
                d_model=transformer__d_model,
                nhead=transformer__nhead,
                num_encoder_layers=transformer__num_encoder_layers,
                num_decoder_layers=transformer__num_decoder_layers,
                batch_first=True,
            )


class SignalJEPA(_BaseSignalJEPA):
    r"""Architecture introduced in signal-JEPA for self-supervised pre-training, Guetschel, P et al (2024) [1]_

    :bdg-success:`Convolution` :bdg-dark-line:`Channel` :bdg-danger:`Foundation Model`

    This model is not meant for classification but for SSL pre-training.
    Its output shape depends on the input shape.
    For classification purposes, three variants of this model are available:

    * :class:`SignalJEPA_Contextual`
    * :class:`SignalJEPA_PostLocal`
    * :class:`SignalJEPA_PreLocal`

    The classification architectures can either be instantiated from scratch
    (random parameters) or from a pre-trained :class:`SignalJEPA` model.

    .. versionadded:: 0.9

    References
    ----------
    .. [1] Guetschel, P., Moreau, T., & Tangermann, M. (2024).
        S-JEPA: towards seamless cross-dataset transfer through dynamic spatial attention.
        In 9th Graz Brain-Computer Interface Conference, https://www.doi.org/10.3217/978-3-99161-014-4-003
    """

    def __init__(
        self,
        n_outputs=None,
        n_chans=None,
        chs_info=None,
        n_times=None,
        input_window_seconds=None,
        sfreq=None,
        *,
        # feature_encoder
        feature_encoder__conv_layers_spec: Sequence[
            tuple[int, int, int]
        ] = _DEFAULT_CONV_LAYER_SPEC,
        drop_prob: float = 0.0,
        feature_encoder__mode: str = "default",
        feature_encoder__conv_bias: bool = False,
        activation: type[nn.Module] = nn.GELU,
        # pos_encoder
        pos_encoder__spat_dim: int = 30,
        pos_encoder__time_dim: int = 34,
        pos_encoder__sfreq_features: float = 1.0,
        pos_encoder__spat_kwargs: dict | None = None,
        # transformer
        transformer__d_model: int = 64,
        transformer__num_encoder_layers: int = 8,
        transformer__num_decoder_layers: int = 4,
        transformer__nhead: int = 8,
        # other
        channel_embedding: str = "scratch",
    ):
        super().__init__(
            n_outputs=n_outputs,
            n_chans=n_chans,
            chs_info=chs_info,
            n_times=n_times,
            input_window_seconds=input_window_seconds,
            sfreq=sfreq,
            feature_encoder__conv_layers_spec=feature_encoder__conv_layers_spec,
            drop_prob=drop_prob,
            feature_encoder__mode=feature_encoder__mode,
            feature_encoder__conv_bias=feature_encoder__conv_bias,
            activation=activation,
            pos_encoder__spat_dim=pos_encoder__spat_dim,
            pos_encoder__time_dim=pos_encoder__time_dim,
            pos_encoder__sfreq_features=pos_encoder__sfreq_features,
            pos_encoder__spat_kwargs=pos_encoder__spat_kwargs,
            transformer__d_model=transformer__d_model,
            transformer__num_encoder_layers=transformer__num_encoder_layers,
            transformer__num_decoder_layers=transformer__num_decoder_layers,
            transformer__nhead=transformer__nhead,
            channel_embedding=channel_embedding,
            _init_feature_encoder=True,
            _init_transformer=True,
        )
        del n_outputs, n_chans, chs_info, n_times, input_window_seconds, sfreq
        self.final_layer = nn.Identity()

    def forward(self, X, return_features=False):  # type: ignore
        local_features = self.feature_encoder(X)  # type: ignore
        pos_encoding = self.pos_encoder(local_features)  # type: ignore
        local_features += pos_encoding  # type: ignore
        contextual_features = self.transformer.encoder(local_features)  # type: ignore
        if return_features:
            return {"features": contextual_features, "cls_token": None}
        y = self.final_layer(contextual_features)  # type: ignore
        return y  # type: ignore


class SignalJEPA_Contextual(_BaseSignalJEPA):
    r"""Contextual downstream architecture introduced in signal-JEPA Guetschel, P et al (2024) [1]_.

    :bdg-success:`Convolution` :bdg-dark-line:`Channel` :bdg-danger:`Foundation Model`

    This architecture is one of the variants of :class:`SignalJEPA`
    that can be used for classification purposes.

    .. figure:: https://braindecode.org/dev/_static/model/sjepa_contextual.jpg
        :align: center
        :alt: sJEPA Contextual.

    .. versionadded:: 0.9

    Parameters
    ----------
    n_spat_filters : int
        Number of spatial filters.

    References
    ----------
    .. [1] Guetschel, P., Moreau, T., & Tangermann, M. (2024).
        S-JEPA: towards seamless cross-dataset transfer through dynamic spatial attention.
        In 9th Graz Brain-Computer Interface Conference, https://www.doi.org/10.3217/978-3-99161-014-4-003
    """

    def __init__(
        self,
        n_outputs=None,
        n_chans=None,
        chs_info=None,
        n_times=None,
        input_window_seconds=None,
        sfreq=None,
        *,
        n_spat_filters: int = 4,
        # feature_encoder
        feature_encoder__conv_layers_spec: Sequence[
            tuple[int, int, int]
        ] = _DEFAULT_CONV_LAYER_SPEC,
        drop_prob: float = 0.0,
        feature_encoder__mode: str = "default",
        feature_encoder__conv_bias: bool = False,
        activation: type[nn.Module] = nn.GELU,
        # pos_encoder
        pos_encoder__spat_dim: int = 30,
        pos_encoder__time_dim: int = 34,
        pos_encoder__sfreq_features: float = 1.0,
        pos_encoder__spat_kwargs: dict | None = None,
        # transformer
        transformer__d_model: int = 64,
        transformer__num_encoder_layers: int = 8,
        transformer__num_decoder_layers: int = 4,
        transformer__nhead: int = 8,
        # other
        channel_embedding: str = "scratch",
        _init_feature_encoder: bool = True,
        _init_transformer: bool = True,
    ):
        super().__init__(
            n_outputs=n_outputs,
            n_chans=n_chans,
            chs_info=chs_info,
            n_times=n_times,
            input_window_seconds=input_window_seconds,
            sfreq=sfreq,
            feature_encoder__conv_layers_spec=feature_encoder__conv_layers_spec,
            drop_prob=drop_prob,
            feature_encoder__mode=feature_encoder__mode,
            feature_encoder__conv_bias=feature_encoder__conv_bias,
            activation=activation,
            pos_encoder__spat_dim=pos_encoder__spat_dim,
            pos_encoder__time_dim=pos_encoder__time_dim,
            pos_encoder__sfreq_features=pos_encoder__sfreq_features,
            pos_encoder__spat_kwargs=pos_encoder__spat_kwargs,
            transformer__d_model=transformer__d_model,
            transformer__num_encoder_layers=transformer__num_encoder_layers,
            transformer__num_decoder_layers=transformer__num_decoder_layers,
            transformer__nhead=transformer__nhead,
            channel_embedding=channel_embedding,
            _init_feature_encoder=_init_feature_encoder,
            _init_transformer=_init_transformer,
        )
        del n_outputs, n_chans, chs_info, n_times, input_window_seconds, sfreq
        self._clf_conv_layers_spec = feature_encoder__conv_layers_spec
        self._clf_n_spat_filters = n_spat_filters
        self.final_layer = _get_separable_clf_layer(
            conv_layers_spec=feature_encoder__conv_layers_spec,
            n_chans=self.n_chans,
            n_times=self.n_times,
            n_classes=self.n_outputs,
            n_spat_filters=n_spat_filters,
        )

    def reset_head(self, n_outputs):
        self._n_outputs = n_outputs
        self.final_layer = _get_separable_clf_layer(
            conv_layers_spec=self._clf_conv_layers_spec,
            n_chans=self.n_chans,
            n_times=self.n_times,
            n_classes=n_outputs,
            n_spat_filters=self._clf_n_spat_filters,
        )

    def forward(self, X, return_features=False):  # type: ignore
        local_features = self.feature_encoder(X)  # type: ignore
        pos_encoding = self.pos_encoder(local_features)  # type: ignore
        local_features += pos_encoding  # type: ignore
        contextual_features = self.transformer.encoder(local_features)  # type: ignore
        if return_features:
            return {"features": contextual_features, "cls_token": None}
        y = self.final_layer(contextual_features)  # type: ignore
        return y  # type: ignore

    @classmethod
    def from_pretrained(
        cls,
        model: Optional[SignalJEPA | str | Path] = None,  # type: ignore
        n_outputs: Optional[int] = None,  # type: ignore
        n_spat_filters: int = 4,
        chs_info: Optional[list[dict[str, Any]]] = None,  # type: ignore
        **kwargs,
    ):
        """Instantiate a new model from a pre-trained :class:`SignalJEPA` model or from Hub.

        Parameters
        ----------
        model: SignalJEPA, str, Path, or None
            Either a pre-trained :class:`SignalJEPA` model, a string/Path to a local directory
            (for Hub-style loading), or None (for Hub loading via kwargs).
        n_outputs: int or None
            Number of classes for the new model. Required when loading from a SignalJEPA model,
            optional when loading from Hub (will be read from config).
        n_spat_filters: int
            Number of spatial filters.
        chs_info: list of dict | None
            Information about each individual EEG channel. This should be filled with
            ``info["chs"]``. Refer to :class:`mne.Info` for more details.
        **kwargs
            Additional keyword arguments passed to the parent class for Hub loading.
        """
        # Check if this is a Hub-style load (from a directory path)
        if isinstance(model, (str, Path)) or (model is None and kwargs):
            # Forward named params that would otherwise be dropped
            if n_outputs is not None:
                kwargs["n_outputs"] = n_outputs
            if chs_info is not None:
                kwargs["chs_info"] = chs_info
            # This is a Hub load, delegate to parent class
            if isinstance(model, (str, Path)):
                # model is actually the repo_id or directory path
                return super().from_pretrained(model, **kwargs)
            else:
                # model is None, treat as hub-style load
                return super().from_pretrained(**kwargs)

        # This is the original SignalJEPA transfer learning case
        if not isinstance(model, SignalJEPA):
            raise TypeError(
                f"model must be a SignalJEPA instance, a path string, or Path object, got {type(model)}"
            )
        if n_outputs is None:
            raise ValueError(
                "n_outputs must be provided when loading from a SignalJEPA model"
            )

        feature_encoder = model.feature_encoder
        pos_encoder = model.pos_encoder
        transformer = model.transformer
        assert feature_encoder is not None
        assert pos_encoder is not None
        assert transformer is not None

        new_model = cls(
            n_outputs=n_outputs,
            n_chans=model.n_chans,
            n_times=model.n_times,
            chs_info=chs_info,
            n_spat_filters=n_spat_filters,
            feature_encoder__conv_layers_spec=feature_encoder.conv_layers_spec,
            _init_feature_encoder=False,
            _init_transformer=False,
        )
        new_model.feature_encoder = deepcopy(feature_encoder)
        new_model.pos_encoder = deepcopy(pos_encoder)
        new_model.transformer = deepcopy(transformer)

        return new_model


class SignalJEPA_PostLocal(_BaseSignalJEPA):
    r"""Post-local downstream architecture introduced in signal-JEPA Guetschel, P et al (2024) [1]_.

    :bdg-success:`Convolution` :bdg-dark-line:`Channel` :bdg-danger:`Foundation Model`

    This architecture is one of the variants of :class:`SignalJEPA`
    that can be used for classification purposes.

    .. figure:: https://braindecode.org/dev/_static/model/sjepa_post-local.jpg
        :align: center
        :alt: sJEPA Pre-Local.

    .. versionadded:: 0.9

    Parameters
    ----------
    n_spat_filters : int
        Number of spatial filters.

    References
    ----------
    .. [1] Guetschel, P., Moreau, T., & Tangermann, M. (2024).
        S-JEPA: towards seamless cross-dataset transfer through dynamic spatial attention.
        In 9th Graz Brain-Computer Interface Conference, https://www.doi.org/10.3217/978-3-99161-014-4-003
    """

    def __init__(
        self,
        n_outputs=None,
        n_chans=None,
        chs_info=None,
        n_times=None,
        input_window_seconds=None,
        sfreq=None,
        *,
        n_spat_filters: int = 4,
        # feature_encoder
        feature_encoder__conv_layers_spec: Sequence[
            tuple[int, int, int]
        ] = _DEFAULT_CONV_LAYER_SPEC,
        drop_prob: float = 0.0,
        feature_encoder__mode: str = "default",
        feature_encoder__conv_bias: bool = False,
        activation: type[nn.Module] = nn.GELU,
        # pos_encoder
        pos_encoder__spat_dim: int = 30,
        pos_encoder__time_dim: int = 34,
        pos_encoder__sfreq_features: float = 1.0,
        pos_encoder__spat_kwargs: dict | None = None,
        # transformer
        transformer__d_model: int = 64,
        transformer__num_encoder_layers: int = 8,
        transformer__num_decoder_layers: int = 4,
        transformer__nhead: int = 8,
        # other
        _init_feature_encoder: bool = True,
    ):
        super().__init__(
            n_outputs=n_outputs,
            n_chans=n_chans,
            chs_info=chs_info,
            n_times=n_times,
            input_window_seconds=input_window_seconds,
            sfreq=sfreq,
            feature_encoder__conv_layers_spec=feature_encoder__conv_layers_spec,
            drop_prob=drop_prob,
            feature_encoder__mode=feature_encoder__mode,
            feature_encoder__conv_bias=feature_encoder__conv_bias,
            activation=activation,
            pos_encoder__spat_dim=pos_encoder__spat_dim,
            pos_encoder__time_dim=pos_encoder__time_dim,
            pos_encoder__sfreq_features=pos_encoder__sfreq_features,
            pos_encoder__spat_kwargs=pos_encoder__spat_kwargs,
            transformer__d_model=transformer__d_model,
            transformer__num_encoder_layers=transformer__num_encoder_layers,
            transformer__num_decoder_layers=transformer__num_decoder_layers,
            transformer__nhead=transformer__nhead,
            _init_feature_encoder=_init_feature_encoder,
            _init_transformer=False,
        )
        del n_outputs, n_chans, chs_info, n_times, input_window_seconds, sfreq
        self._clf_conv_layers_spec = feature_encoder__conv_layers_spec
        self._clf_n_spat_filters = n_spat_filters
        self.final_layer = _get_separable_clf_layer(
            conv_layers_spec=feature_encoder__conv_layers_spec,
            n_chans=self.n_chans,
            n_times=self.n_times,
            n_classes=self.n_outputs,
            n_spat_filters=n_spat_filters,
        )

    def reset_head(self, n_outputs):
        self._n_outputs = n_outputs
        self.final_layer = _get_separable_clf_layer(
            conv_layers_spec=self._clf_conv_layers_spec,
            n_chans=self.n_chans,
            n_times=self.n_times,
            n_classes=n_outputs,
            n_spat_filters=self._clf_n_spat_filters,
        )

    @classmethod
    def from_pretrained(
        cls,
        model: SignalJEPA | str | Path = None,  # type: ignore
        n_outputs: int = None,  # type: ignore
        n_spat_filters: int = 4,
        **kwargs,
    ):
        """Instantiate a new model from a pre-trained :class:`SignalJEPA` model or from Hub.

        Parameters
        ----------
        model: SignalJEPA, str, Path, or None
            Either a pre-trained :class:`SignalJEPA` model, a string/Path to a local directory
            (for Hub-style loading), or None (for Hub loading via kwargs).
        n_outputs: int or None
            Number of classes for the new model. Required when loading from a SignalJEPA model,
            optional when loading from Hub (will be read from config).
        n_spat_filters: int
            Number of spatial filters.
        **kwargs
            Additional keyword arguments passed to the parent class for Hub loading.
        """
        # Check if this is a Hub-style load (from a directory path)
        if isinstance(model, (str, Path)) or (model is None and kwargs):
            # Forward named params that would otherwise be dropped
            if n_outputs is not None:
                kwargs["n_outputs"] = n_outputs
            # This is a Hub load, delegate to parent class
            if isinstance(model, (str, Path)):
                # model is actually the repo_id or directory path
                return super().from_pretrained(model, **kwargs)
            else:
                # model is None, treat as hub-style load
                return super().from_pretrained(**kwargs)

        # This is the original SignalJEPA transfer learning case
        if not isinstance(model, SignalJEPA):
            raise TypeError(
                f"model must be a SignalJEPA instance, a path string, or Path object, got {type(model)}"
            )
        if n_outputs is None:
            raise ValueError(
                "n_outputs must be provided when loading from a SignalJEPA model"
            )

        feature_encoder = model.feature_encoder
        assert feature_encoder is not None
        new_model = cls(
            n_outputs=n_outputs,
            n_chans=model.n_chans,
            n_times=model.n_times,
            n_spat_filters=n_spat_filters,
            feature_encoder__conv_layers_spec=feature_encoder.conv_layers_spec,
            _init_feature_encoder=False,
        )
        new_model.feature_encoder = deepcopy(feature_encoder)
        return new_model

    def forward(self, X, return_features=False):
        local_features = self.feature_encoder(X)
        if return_features:
            return {"features": local_features, "cls_token": None}
        y = self.final_layer(local_features)
        return y


class SignalJEPA_PreLocal(_BaseSignalJEPA):
    r"""Pre-local downstream architecture introduced in signal-JEPA Guetschel, P et al (2024) [1]_.

    :bdg-success:`Convolution` :bdg-dark-line:`Channel` :bdg-danger:`Foundation Model`

    This architecture is one of the variants of :class:`SignalJEPA`
    that can be used for classification purposes.

    .. figure:: https://braindecode.org/dev/_static/model/sjepa_pre-local.jpg
        :align: center
        :alt: sJEPA Pre-Local.

    .. versionadded:: 0.9

    .. important::
       **Pre-trained Weights Available**

       This model has pre-trained weights available on the Hugging Face Hub.
       You can load them using:

       .. code-block:: python

           from braindecode.models import SignalJEPA_PreLocal

           # Load pre-trained model from Hugging Face Hub
           model = SignalJEPA_PreLocal.from_pretrained(
               "braindecode/SignalJEPA-PreLocal-pretrained"
           )

       To push your own trained model to the Hub:

       .. code-block:: python

           # After training your model
           model.push_to_hub(
               repo_id="username/my-sjepa-model",
               commit_message="Upload trained SignalJEPA model",
           )

       Requires installing ``braindecode[hug]`` for Hub integration.

    Parameters
    ----------
    n_spat_filters : int
        Number of spatial filters.

    References
    ----------
    .. [1] Guetschel, P., Moreau, T., & Tangermann, M. (2024).
        S-JEPA: towards seamless cross-dataset transfer through dynamic spatial attention.
        In 9th Graz Brain-Computer Interface Conference, https://www.doi.org/10.3217/978-3-99161-014-4-003
    """

    _feature_encoder_channels: str = "n_spat_filters"

    def __init__(
        self,
        n_outputs=None,
        n_chans=None,
        chs_info=None,
        n_times=None,
        input_window_seconds=None,
        sfreq=None,
        *,
        n_spat_filters: int = 4,
        # feature_encoder
        feature_encoder__conv_layers_spec: Sequence[
            tuple[int, int, int]
        ] = _DEFAULT_CONV_LAYER_SPEC,
        drop_prob: float = 0.0,
        feature_encoder__mode: str = "default",
        feature_encoder__conv_bias: bool = False,
        activation: type[nn.Module] = nn.GELU,
        # pos_encoder
        pos_encoder__spat_dim: int = 30,
        pos_encoder__time_dim: int = 34,
        pos_encoder__sfreq_features: float = 1.0,
        pos_encoder__spat_kwargs: dict | None = None,
        # transformer
        transformer__d_model: int = 64,
        transformer__num_encoder_layers: int = 8,
        transformer__num_decoder_layers: int = 4,
        transformer__nhead: int = 8,
        # other
        _init_feature_encoder: bool = True,
    ):
        self.n_spat_filters = n_spat_filters
        super().__init__(
            n_outputs=n_outputs,
            n_chans=n_chans,
            chs_info=chs_info,
            n_times=n_times,
            input_window_seconds=input_window_seconds,
            sfreq=sfreq,
            feature_encoder__conv_layers_spec=feature_encoder__conv_layers_spec,
            drop_prob=drop_prob,
            feature_encoder__mode=feature_encoder__mode,
            feature_encoder__conv_bias=feature_encoder__conv_bias,
            activation=activation,
            pos_encoder__spat_dim=pos_encoder__spat_dim,
            pos_encoder__time_dim=pos_encoder__time_dim,
            pos_encoder__sfreq_features=pos_encoder__sfreq_features,
            pos_encoder__spat_kwargs=pos_encoder__spat_kwargs,
            transformer__d_model=transformer__d_model,
            transformer__num_encoder_layers=transformer__num_encoder_layers,
            transformer__num_decoder_layers=transformer__num_decoder_layers,
            transformer__nhead=transformer__nhead,
            _init_feature_encoder=_init_feature_encoder,
            _init_transformer=False,
        )
        del n_outputs, n_chans, chs_info, n_times, input_window_seconds, sfreq
        self.spatial_conv = nn.Sequential(
            Rearrange("b channels time -> b 1 channels time"),
            nn.Conv2d(1, n_spat_filters, (self.n_chans, 1)),
            Rearrange("b spat_filters 1 time -> b spat_filters time"),
        )
        self._out_emb_dim = _get_out_emb_dim(
            conv_layers_spec=feature_encoder__conv_layers_spec,
            n_times=self.n_times,
            n_spat_filters=n_spat_filters,
        )
        self.final_layer = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(self._out_emb_dim, self.n_outputs),
        )

    def reset_head(self, n_outputs):
        self._n_outputs = n_outputs
        self.final_layer = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(self._out_emb_dim, n_outputs),
        )

    @classmethod
    def from_pretrained(
        cls,
        model: SignalJEPA | str | Path = None,  # type: ignore
        n_outputs: int = None,  # type: ignore
        n_spat_filters: int = 4,
        **kwargs,
    ):
        """Instantiate a new model from a pre-trained :class:`SignalJEPA` model or from Hub.

        Parameters
        ----------
        model: SignalJEPA, str, Path, or None
            Either a pre-trained :class:`SignalJEPA` model, a string/Path to a local directory
            (for Hub-style loading), or None (for Hub loading via kwargs).
        n_outputs: int or None
            Number of classes for the new model. Required when loading from a SignalJEPA model,
            optional when loading from Hub (will be read from config).
        n_spat_filters: int
            Number of spatial filters.
        **kwargs
            Additional keyword arguments passed to the parent class for Hub loading.
        """
        # Check if this is a Hub-style load (from a directory path)
        if isinstance(model, (str, Path)) or (model is None and kwargs):
            # Forward named params that would otherwise be dropped
            if n_outputs is not None:
                kwargs["n_outputs"] = n_outputs
            # This is a Hub load, delegate to parent class
            if isinstance(model, (str, Path)):
                # model is actually the repo_id or directory path
                return super().from_pretrained(model, **kwargs)
            else:
                # model is None, treat as hub-style load
                return super().from_pretrained(**kwargs)

        # This is the original SignalJEPA transfer learning case
        if not isinstance(model, SignalJEPA):
            raise TypeError(
                f"model must be a SignalJEPA instance, a path string, or Path object, got {type(model)}"
            )
        if n_outputs is None:
            raise ValueError(
                "n_outputs must be provided when loading from a SignalJEPA model"
            )

        feature_encoder = model.feature_encoder
        assert feature_encoder is not None
        new_model = cls(
            n_outputs=n_outputs,
            n_chans=model.n_chans,
            n_times=model.n_times,
            n_spat_filters=n_spat_filters,
            feature_encoder__conv_layers_spec=feature_encoder.conv_layers_spec,
            _init_feature_encoder=False,
        )
        new_model.feature_encoder = deepcopy(feature_encoder)
        return new_model

    def forward(self, X, return_features=False):
        X = self.spatial_conv(X)
        local_features = self.feature_encoder(X)
        if return_features:
            return {"features": local_features, "cls_token": None}
        y = self.final_layer(local_features)
        return y


class _ConvFeatureEncoder(nn.Sequential):
    r"""Convolutional feature encoder for EEG data.

    Computes successive 1D convolutions (with activations) over the time
    dimension of the input EEG signal.

    Inspiration from https://github.com/facebookresearch/fairseq/blob/main/fairseq/models/wav2vec/wav2vec2.py
    and https://github.com/SPOClab-ca/BENDR/blob/main/dn3_ext.py

    Parameters
    ----------
    conv_layers_spec: list of tuple
        tuples have shape ``(dim, k, stride)`` where:

        * ``dim`` : number of output channels of the layer (unrelated to EEG channels);
        * ``k`` : temporal length of the layer's kernel;
        * ``stride`` : temporal stride of the layer's kernel.

    channels: int
    drop_prob: float
    mode: str
        Normalisation mode. Either ``default`` or ``layer_norm``.
    conv_bias: bool
    activation: nn.Module
    """

    def __init__(
        self,
        conv_layers_spec: Sequence[tuple[int, int, int]],
        channels: int,
        drop_prob: float = 0.0,
        mode: str = "default",
        conv_bias: bool = False,
        activation: type[nn.Module] = nn.GELU,
    ):
        assert mode in {"default", "layer_norm"}

        input_channels = 1
        conv_layers = []
        for i, layer_spec in enumerate(conv_layers_spec):
            # Each layer_spec should be a tuple: (output_channels, kernel_size, stride)
            assert len(layer_spec) == 3, "Invalid conv definition: " + str(layer_spec)
            output_channels, kernel_size, stride = layer_spec
            conv_layers.append(
                self._get_block(
                    input_channels,
                    output_channels,
                    kernel_size,
                    stride,
                    drop_prob,
                    activation,
                    is_layer_norm=(mode == "layer_norm"),
                    is_group_norm=(mode == "default" and i == 0),
                    conv_bias=conv_bias,
                )
            )
            input_channels = output_channels
        all_layers = [
            Rearrange("b channels time -> (b channels) 1 time", channels=channels),
            *conv_layers,
            Rearrange(
                "(b channels) emb_dim time_out -> b (channels time_out) emb_dim",
                channels=channels,
            ),
        ]
        super().__init__(*all_layers)
        self.emb_dim = (
            output_channels  # last output dimension becomes the embedding dimension
        )
        self.conv_layers_spec = conv_layers_spec

    @staticmethod
    def _get_block(
        input_channels,
        output_channels,
        kernel_size,
        stride,
        drop_prob,
        activation,
        is_layer_norm=False,
        is_group_norm=False,
        conv_bias=False,
    ):
        assert not (is_layer_norm and is_group_norm), (
            "layer norm and group norm are exclusive"
        )

        conv = nn.Conv1d(
            input_channels,
            output_channels,
            kernel_size,
            stride=stride,
            bias=conv_bias,
        )
        nn.init.kaiming_normal_(conv.weight)
        if is_layer_norm:
            return nn.Sequential(
                conv,
                nn.Dropout(p=drop_prob),
                nn.Sequential(
                    Rearrange("... channels time -> ... time channels"),
                    nn.LayerNorm(output_channels, elementwise_affine=True),
                    Rearrange("... time channels -> ... channels time"),
                ),
                activation(),
            )
        elif is_group_norm:
            return nn.Sequential(
                conv,
                nn.Dropout(p=drop_prob),
                nn.GroupNorm(output_channels, output_channels, affine=True),
                activation(),
            )
        else:
            return nn.Sequential(conv, nn.Dropout(p=drop_prob), activation())

    def n_times_out(self, n_times):
        return _n_times_out(self.conv_layers_spec, n_times)


class _ChannelEmbedding(nn.Embedding):
    r"""Embedding layer for EEG channels.

    The difference with a regular :class:`nn.Embedding` is that the embedding
    vectors are initialized with a positional encodding of the channel locations.

    Parameters
    ----------
    channel_locations: list of (list of float or None)
        List of the n-dimensions locations of the EEG channels.
    embedding_dim: int
        Dimensionality of the embedding vectors. Must be a multiple of the number
        of dimensions of the channel locations.
    """

    def __init__(
        self, channel_locations: list[list[float] | None], embedding_dim: int, **kwargs
    ):
        self.coordinate_ranges = [
            (min(coords), max(coords))
            for coords in zip(
                *[
                    loc[3:6] if len(loc) == 12 else loc
                    for loc in channel_locations
                    if loc is not None
                ]
            )
        ]
        channel_mins, channel_maxs = zip(*self.coordinate_ranges)
        global_min = min(channel_mins)
        global_max = max(channel_maxs)
        self.max_abs_coordinate = max(abs(global_min), abs(global_max))
        self.embedding_dim_per_coordinate = embedding_dim // len(self.coordinate_ranges)
        self.channel_locations = list(channel_locations)

        assert embedding_dim % len(self.coordinate_ranges) == 0

        super().__init__(len(channel_locations), embedding_dim, **kwargs)

    def reset_parameters(self):
        for i, loc in enumerate(self.channel_locations):
            if loc is None:
                nn.init.zeros_(self.weight[i])
            else:
                for j, (x, (x0, x1)) in enumerate(zip(loc, self.coordinate_ranges)):
                    with torch.no_grad():
                        self.weight[
                            i,
                            j * self.embedding_dim_per_coordinate : (j + 1)
                            * self.embedding_dim_per_coordinate,
                        ].copy_(
                            _pos_encode_contineous(
                                x,
                                0,
                                10 * self.max_abs_coordinate,
                                self.embedding_dim_per_coordinate,
                                device=self.weight.device,
                            ),
                        )


class _PosEncoder(nn.Module):
    r"""Positional encoder for EEG data.

    Parameters
    ----------
    spat_dim: int
        Number of dimensions to use to encode the spatial position of the patch,
        i.e. the EEG channel.
    time_dim: int
        Number of dimensions to use to encode the temporal position of the patch.
    channel_locations: list of (list of float or None)
        List of the n-dimensional locations of each entry in the channel
        embedding table. Length equals the number of embedding rows
        (N for ``'scratch'`` mode, 62 for ``'pretrain_aligned'`` mode).
    ch_idxs: torch.LongTensor, shape ``(n_chans,)``
        Default mapping from the model's channel order to rows of the
        channel embedding table. Stored as a non-persistent buffer
        (device-aware but excluded from ``state_dict``).
    sfreq_features: float
        The "downsampled" sampling frequency returned by the feature encoder.
    spat_kwargs: dict
        Additional keyword arguments to pass to the :class:`nn.Embedding` layer used to
        embed the channel names.
    max_seconds: float
        Maximum number of seconds to consider for the temporal encoding.
    """

    def __init__(
        self,
        spat_dim: int,
        time_dim: int,
        channel_locations: list[list[float] | None],
        ch_idxs: torch.LongTensor,
        sfreq_features: float,
        spat_kwargs: dict | None = None,
        max_seconds: float = 600.0,  # 10 minutes
    ):
        super().__init__()
        spat_kwargs = spat_kwargs or {}
        self.spat_dim = spat_dim
        self.time_dim = time_dim
        self.max_n_times = int(max_seconds * sfreq_features)

        # Positional encoder for the spatial dimension:
        self.pos_encoder_spat = _ChannelEmbedding(
            channel_locations, spat_dim, **spat_kwargs
        )  # (batch_size, n_channels, spat_dim)

        # Default channel index mapping. Registered as a non-persistent
        # buffer so it follows .to(device) but is NOT saved to state_dict.
        # The mapping is fully determined by chs_info + channel_embedding
        # at __init__, so persisting it would duplicate config.json.
        self.register_buffer(
            "default_ch_idxs", ch_idxs.to(torch.long), persistent=False
        )

        # Pre-computed tensor for positional encoding on the time dimension:
        self.encoding_time = torch.zeros(0, dtype=torch.float32, requires_grad=False)

    def _check_encoding_time(self, n_times: int):
        if self.encoding_time.size(0) < n_times:
            self.encoding_time = self.encoding_time.new_empty((n_times, self.time_dim))
            self.encoding_time[:] = _pos_encode_time(
                n_times=n_times,
                n_dim=self.time_dim,
                max_n_times=self.max_n_times,
                device=self.encoding_time.device,
            )

    def forward(self, local_features, ch_idxs: torch.Tensor | None = None):
        """
        Parameters
        ----------
        local_features : (batch_size, n_chans * n_times_out, emb_dim)
        ch_idxs : (batch_size, n_chans) | None
            Indices of the channels into the embedding table. When ``None``,
            ``self.default_ch_idxs`` (shape ``(n_chans,)``) is broadcast
            across the batch.

        Returns
        -------
        pos_encoding : (batch_size, n_chans * n_times_out, emb_dim)
            The first ``spat_dim`` dimensions encode the channel positional
            encoding; the following ``time_dim`` dimensions encode the
            temporal positional encoding.
        """
        batch_size, n_chans_times, emb_dim = local_features.shape
        if ch_idxs is None:
            ch_idxs = self.default_ch_idxs[None, :].expand(batch_size, -1)

        batch_size_chs, n_chans = ch_idxs.shape
        assert emb_dim >= self.spat_dim + self.time_dim
        assert n_chans_times % n_chans == 0
        n_times = n_chans_times // n_chans

        pos_encoding = local_features.new_empty(
            (batch_size_chs, n_chans, n_times, emb_dim)
        )
        # Channel pos. encoding
        pos_encoding[:, :, :, : self.spat_dim] = self.pos_encoder_spat(ch_idxs)[
            :, :, None, :
        ]
        # Temporal pos. encoding
        self._check_encoding_time(n_times)
        _ = pos_encoding[:, :, :, self.spat_dim : self.spat_dim + self.time_dim].copy_(
            self.encoding_time[None, None, :n_times, :],
        )

        return pos_encoding.view(batch_size, n_chans_times, emb_dim)


def _n_times_out(conv_layers_spec, n_times):
    # it would be equal to n_times//ds_factor without edge effects:
    n_times_out_ = n_times
    for _, width, stride in conv_layers_spec:
        n_times_out_ = int((n_times_out_ - width) / stride) + 1
    return n_times_out_


def _get_out_emb_dim(conv_layers_spec, n_times, n_spat_filters=4):
    n_time_out = _n_times_out(conv_layers_spec, n_times)
    emb_dim = conv_layers_spec[-1][0]
    return n_spat_filters * n_time_out * emb_dim


def _get_separable_clf_layer(
    conv_layers_spec, n_chans, n_times, n_classes, n_spat_filters=4
):
    out_emb_dim = _get_out_emb_dim(
        conv_layers_spec=conv_layers_spec,
        n_times=n_times,
        n_spat_filters=n_spat_filters,
    )
    clf_layer = nn.Sequential()
    clf_layer.add_module(
        "unflatten_tokens",
        Rearrange("b (n_chans tokens) d -> b 1 n_chans tokens d", n_chans=n_chans),
    )
    clf_layer.add_module("spat_conv", nn.Conv3d(1, n_spat_filters, (n_chans, 1, 1)))
    clf_layer.add_module("flatten", nn.Flatten(start_dim=1))
    clf_layer.add_module("linear", nn.Linear(out_emb_dim, n_classes))
    return clf_layer


def _pos_encode_time(
    n_times: int,
    n_dim: int,
    max_n_times: int,
    device: torch.device = torch.device("cpu"),
):
    """1-dimensional positional encoding.

    Parameters
    ----------
        n_times: int
            Number of time samples to encode.
        n_dim: int
            Number of dimensions of the positional encoding. Must be even.
        max_n_times: int
            The largest possible number of time samples to encode.
            Used to scale the positional encoding.
        device: torch.device
            Device to put the output on.
    Returns
    -------
        pos_encoding: (n_times, n_dim)
    """
    assert n_dim % 2 == 0
    position = torch.arange(n_times, device=device).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, n_dim, 2, device=device) * (-math.log(max_n_times) / n_dim)
    )
    pos_encoding = torch.empty((n_times, n_dim), dtype=torch.float32, device=device)
    pos_encoding[:, 0::2] = torch.sin(position * div_term)
    pos_encoding[:, 1::2] = torch.cos(position * div_term)
    return pos_encoding


def _pos_encode_contineous(
    x, x_min, x_max, n_dim, device: torch.device = torch.device("cpu")
):
    """1-dimensional positional encoding.

    Parameters
    ----------
        x: float
            The position to encode.
        x_min: float
            The minimum possible value of x.
        x_max: float
            The maximum possible value of x.
        n_dim: int
            Number of dimensions of the positional encoding. Must be even.
         device: torch.device
            Device to put the output on.
    Returns
    -------
        pos_encoding: (n_dim,)
    """
    assert n_dim % 2 == 0
    div_term = torch.exp(
        (1 - torch.arange(0, n_dim, 2, device=device) / n_dim) * 2 * math.pi
    )
    pos_encoding = torch.empty((n_dim,), dtype=torch.float32, device=device)
    xx = (x - x_min) / (x_max - x_min)
    pos_encoding[0::2] = torch.sin(xx * div_term)
    pos_encoding[1::2] = torch.cos(xx * div_term)
    return pos_encoding
