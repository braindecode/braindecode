# Authors: Pierre Guetschel <pierre.guetschel@gmail.com>
#
# License: BSD (3-clause)
from __future__ import annotations

import math
from copy import deepcopy
from typing import Any, Sequence

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


class _BaseSignalJEPA(EEGModuleMixin, nn.Module):
    """Base class for the SignalJEPA models

    Parameters
    ----------
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
        _init_feature_encoder: bool,
        _init_transformer: bool,
    ):
        super().__init__(
            n_outputs=n_outputs,
            n_chans=n_chans,
            chs_info=chs_info,
            n_times=n_times,
            input_window_seconds=input_window_seconds,
            sfreq=sfreq,
        )
        del n_outputs, n_chans, chs_info, n_times, input_window_seconds, sfreq

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
            ch_locs = [ch["loc"] for ch in self.chs_info]  # type: ignore
            self.pos_encoder = _PosEncoder(
                spat_dim=pos_encoder__spat_dim,
                time_dim=pos_encoder__time_dim,
                ch_locs=ch_locs,
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
    """Architecture introduced in signal-JEPA for self-supervised pre-training, Guetschel, P et al (2024) [1]_

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
            _init_feature_encoder=True,
            _init_transformer=True,
        )
        del n_outputs, n_chans, chs_info, n_times, input_window_seconds, sfreq
        self.final_layer = nn.Identity()

    def forward(self, X, ch_idxs: torch.Tensor | None = None):  # type: ignore
        local_features = self.feature_encoder(X)  # type: ignore
        pos_encoding = self.pos_encoder(local_features, ch_idxs=ch_idxs)  # type: ignore
        local_features += pos_encoding  # type: ignore
        contextual_features = self.transformer.encoder(local_features)  # type: ignore
        y = self.final_layer(contextual_features)  # type: ignore
        return y  # type: ignore


class SignalJEPA_Contextual(_BaseSignalJEPA):
    """Contextual downstream architecture introduced in signal-JEPA Guetschel, P et al (2024) [1]_.

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
            _init_feature_encoder=_init_feature_encoder,
            _init_transformer=_init_transformer,
        )
        del n_outputs, n_chans, chs_info, n_times, input_window_seconds, sfreq
        self.final_layer = _get_separable_clf_layer(
            conv_layers_spec=feature_encoder__conv_layers_spec,
            n_chans=self.n_chans,
            n_times=self.n_times,
            n_classes=self.n_outputs,
            n_spat_filters=n_spat_filters,
        )

    @classmethod
    def from_pretrained(
        cls,
        model: SignalJEPA,
        n_outputs: int,
        n_spat_filters: int = 4,
        chs_info: list[dict[str, Any]] | None = None,
    ):
        """Instantiate a new model from a pre-trained :class:`SignalJEPA` model.

        Parameters
        ----------
        model: SignalJEPA
            Pre-trained model.
        n_outputs: int
            Number of classes for the new model.
        n_spat_filters: int
            Number of spatial filters.
        chs_info: list of dict | None
            Information about each individual EEG channel. This should be filled with
            ``info["chs"]``. Refer to :class:`mne.Info` for more details.
        """
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

        if chs_info is not None:
            ch_names = [ch["ch_name"] for ch in chs_info]
            new_model.pos_encoder.set_fixed_ch_names(ch_names)

        return new_model

    def forward(self, X, ch_idxs: torch.Tensor | None = None):  # type: ignore
        local_features = self.feature_encoder(X)  # type: ignore
        pos_encoding = self.pos_encoder(local_features, ch_idxs=ch_idxs)  # type: ignore
        local_features += pos_encoding  # type: ignore
        contextual_features = self.transformer.encoder(local_features)  # type: ignore
        y = self.final_layer(contextual_features)  # type: ignore
        return y  # type: ignore


class SignalJEPA_PostLocal(_BaseSignalJEPA):
    """Post-local downstream architecture introduced in signal-JEPA Guetschel, P et al (2024) [1]_.

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
        self.final_layer = _get_separable_clf_layer(
            conv_layers_spec=feature_encoder__conv_layers_spec,
            n_chans=self.n_chans,
            n_times=self.n_times,
            n_classes=self.n_outputs,
            n_spat_filters=n_spat_filters,
        )

    @classmethod
    def from_pretrained(
        cls, model: SignalJEPA, n_outputs: int, n_spat_filters: int = 4
    ):
        """Instantiate a new model from a pre-trained :class:`SignalJEPA` model.

        Parameters
        ----------
        model: SignalJEPA
            Pre-trained model.
        n_outputs: int
            Number of classes for the new model.
        n_spat_filters: int
            Number of spatial filters.
        chs_info: list of dict | None
            Information about each individual EEG channel. This should be filled with
            ``info["chs"]``. Refer to :class:`mne.Info` for more details.
        """
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

    def forward(self, X):
        local_features = self.feature_encoder(X)
        y = self.final_layer(local_features)
        return y


class SignalJEPA_PreLocal(_BaseSignalJEPA):
    """Pre-local downstream architecture introduced in signal-JEPA Guetschel, P et al (2024) [1]_.

    This architecture is one of the variants of :class:`SignalJEPA`
    that can be used for classification purposes.

    .. figure:: https://braindecode.org/dev/_static/model/sjepa_pre-local.jpg
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
        out_emb_dim = _get_out_emb_dim(
            conv_layers_spec=feature_encoder__conv_layers_spec,
            n_times=self.n_times,
            n_spat_filters=n_spat_filters,
        )
        self.final_layer = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(out_emb_dim, self.n_outputs),
        )

    @classmethod
    def from_pretrained(
        cls, model: SignalJEPA, n_outputs: int, n_spat_filters: int = 4
    ):
        """Instantiate a new model from a pre-trained :class:`SignalJEPA` model.

        Parameters
        ----------
        model: SignalJEPA
            Pre-trained model.
        n_outputs: int
            Number of classes for the new model.
        n_spat_filters: int
            Number of spatial filters.
        chs_info: list of dict | None
            Information about each individual EEG channel. This should be filled with
            ``info["chs"]``. Refer to :class:`mne.Info` for more details.
        """
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

    def forward(self, X):
        X = self.spatial_conv(X)
        local_features = self.feature_encoder(X)
        y = self.final_layer(local_features)
        return y


class _ConvFeatureEncoder(nn.Sequential):
    """Convolutional feature encoder for EEG data.

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
    """Embedding layer for EEG channels.

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
    """Positional encoder for EEG data.

    Parameters
    ----------
    spat_dim: int
        Number of dimensions to use to encode the spatial position of the patch,
        i.e. the EEG channel.
    time_dim: int
        Number of dimensions to use to encode the temporal position of the patch.
    ch_locs: list of list of float or 2d array
        List of the n-dimensions locations of the EEG channels.
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
        ch_locs,
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
            ch_locs, spat_dim, **spat_kwargs
        )  # (batch_size, n_channels, spat_dim)

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
        * local_features: (batch_size, n_chans * n_times_out, emb_dim)
        * ch_idxs: (batch_size, n_chans) | None
            Indices of the channels to use in the ``ch_names`` list passed
            as argument plus one. Index 0 is reserved for an unknown channel.

        Returns
        -------
        pos_encoding: (batch_size, n_chans * n_times_out, emb_dim)
            The first ``spat_dim`` dimensions encode the channels positional encoding
            and the following ``time_dim`` dimensions encode the temporal positional encoding.
        """
        batch_size, n_chans_times, emb_dim = local_features.shape
        if ch_idxs is None:
            ch_idxs = torch.arange(
                0,
                self.pos_encoder_spat.num_embeddings,
                device=local_features.device,
            ).repeat(batch_size, 1)

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
