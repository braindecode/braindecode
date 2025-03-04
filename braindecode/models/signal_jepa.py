# Authors: Pierre Guetschel <pierre.guetschel@gmail.com>
#
# License: BSD (3-clause)
from __future__ import annotations
from typing import Any, Sequence
import math
from copy import deepcopy

import torch
from torch import nn


from braindecode.models.base import EEGModuleMixin
from braindecode.models.modules import TransposeLast


def pos_encode_time(n_times, n_dim, max_n_times, device="cpu"):
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
        device: str
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


def pos_encode_contineous(x, x_min, x_max, n_dim, device="cpu"):
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
         device: str
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


class UnflattenTokens(nn.Module):
    def __init__(self, n_chans: int):
        super().__init__()
        self.n_chans = n_chans

    def forward(self, x: torch.Tensor):
        batch_size, _, emb_dim = x.shape
        return x.view(batch_size, 1, self.n_chans, -1, emb_dim)


class ReshapeConv2d(nn.Module):
    """Conv 2d layer for EEG spatial filtering. Also packs the batch."""

    def __init__(self, n_chs, n_spat_filters, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(1, n_spat_filters, (n_chs, 1), **kwargs)
        self.n_spat_filters = n_spat_filters

    def forward(self, X):
        assert X.ndim == 3
        batch_size, n_chs, n_times = X.shape
        X = X.view(batch_size, 1, n_chs, n_times)
        X = self.conv(X)
        X = X.view(batch_size, self.n_spat_filters, n_times)
        return X


class ConvFeatureEncoder(nn.Module):
    """Convolutional feature encoder for EEG data.

    Computes successive 1D convolutions (with activations) over the time
    dimension of the input EEG signal.

    Inspiration from https://github.com/facebookresearch/fairseq/blob/main/fairseq/models/wav2vec/wav2vec2.py
    and https://github.com/SPOClab-ca/BENDR/blob/main/dn3_ext.py

    Parameters
    ----------
        conv_layers_spec: list of tuples (dim, k, stride) where:

            * dim: number of output channels of the layer (unrelated to EEG channels);
            * k: temporal length of the layer's kernel;
            * stride: temporal stride of the layer's kernel.

        drop_prob: float
        mode: str
            Normalisation mode. Either``default`` or ``layer_norm``.
        conv_bias: bool
        activation: nn.Module
    """

    def __init__(
        self,
        conv_layers_spec: Sequence[tuple[int, int, int]],
        drop_prob: float = 0.0,
        mode: str = "default",
        conv_bias: bool = False,
        activation: type[nn.Module] = nn.GELU,
    ):
        assert mode in {"default", "layer_norm"}
        super().__init__()

        def block(
            n_in,
            n_out,
            k,
            stride,
            is_layer_norm=False,
            is_group_norm=False,
            conv_bias=False,
        ):

            def make_conv():
                conv = nn.Conv1d(n_in, n_out, k, stride=stride, bias=conv_bias)
                nn.init.kaiming_normal_(conv.weight)
                return conv

            assert not (
                is_layer_norm and is_group_norm
            ), "layer norm and group norm are exclusive"

            if is_layer_norm:
                return nn.Sequential(
                    make_conv(),
                    nn.Dropout(p=drop_prob),
                    nn.Sequential(
                        TransposeLast(),
                        nn.LayerNorm(dim, elementwise_affine=True),
                        TransposeLast(),
                    ),
                    activation(),
                )
            elif is_group_norm:
                return nn.Sequential(
                    make_conv(),
                    nn.Dropout(p=drop_prob),
                    nn.GroupNorm(dim, dim, affine=True),
                    activation(),
                )
            else:
                return nn.Sequential(make_conv(), nn.Dropout(p=drop_prob), activation())

        in_d = 1
        conv_layers = []
        for i, cl in enumerate(conv_layers_spec):
            assert len(cl) == 3, "invalid conv definition: " + str(cl)
            (dim, k, stride) = cl
            conv_layers.append(
                block(
                    in_d,
                    dim,
                    k,
                    stride,
                    is_layer_norm=mode == "layer_norm",
                    is_group_norm=mode == "default" and i == 0,
                    conv_bias=conv_bias,
                )
            )
            in_d = dim
        self.emb_dim = dim  # last dim
        self.conv_layers_spec = conv_layers_spec
        self.cnn = nn.Sequential(*conv_layers)

    def forward(self, batch):
        """
        Parameters
        ----------
            batch: dict with keys:

                * X: (batch_size, n_chans, n_times)
                    Batched EEG signal.

        Returns
        -------
            local_features: (batch_size, n_chans * n_times_out, emb_dim)
                Local features extracted from the EEG signal.
                ``emb_dim`` corresponds to the ``dim`` of the last element of
                ``conv_layers_spec``.
        """
        x = batch["X"]
        batch_size, n_chans, n_times = x.shape
        x = x.view(batch_size * n_chans, 1, n_times)
        x: torch.Tensor = self.cnn(x)  # (batch_size * n_chans, emb_dim, n_times_out)
        x = x.transpose(1, 2)
        x = x.reshape(batch_size, -1, self.emb_dim)
        return x  # (batch_size, n_chans * n_times_out, emb_dim)

    @property
    def receptive_fields(self):
        rf = 1
        receptive_fields = [rf]
        for _, width, stride in reversed(self.conv_layers_spec):
            rf = (rf - 1) * stride + width  # assumes no padding and no dilation
            receptive_fields.append(rf)
        return list(reversed(receptive_fields))

    def n_times_out(self, n_times):
        return n_times_out(self.conv_layers_spec, n_times)

    def description(self, sfreq=None, n_times=None):
        dims, _, strides = zip(*self.conv_layers_spec)
        receptive_fields = self.receptive_fields
        rf = receptive_fields[0]
        desc = f"Receptive field: {rf} samples"
        if sfreq is not None:
            desc += f", {rf / sfreq:.2f} seconds"

        ds_factor = math.prod(strides)
        desc += f" | Downsampled by {ds_factor}"
        if sfreq is not None:
            desc += f", new sfreq: {sfreq / ds_factor:.2f} Hz"
        desc += f" | Overlap of {rf - ds_factor} samples"
        if n_times is not None:
            n_times_out = self.n_times_out(n_times)
            desc += f" | {n_times_out} encoded samples/trial"

        n_features = [
            f"{dim}*{rf}" for dim, rf in zip([1] + list(dims), receptive_fields)
        ]
        desc += f" | #features/sample at each layer (n_channels*n_times): [{', '.join(n_features)}] = {[eval(x) for x in n_features]}"
        return desc


class ChannelEmbedding(nn.Embedding):
    """Embedding layer for EEG channels.

    The difference with a regular :class:`nn.Embedding` is that the embedding
    vectors are initialized with a positional encodding of the channel locations.

    Parameters
    ----------
        locs: list of (list of float or None)
            List of the n-dimensions locations of the EEG channels.
        embedding_dim: int
            Dimensionality of the embedding vectors. Must be a multiple of the number
            of dimensions of the channel locations.
    """

    def __init__(self, locs: list[list[float] | None], embedding_dim: int, **kwargs):
        self.ranges = [
            (min(col), max(col))
            for col in zip(
                *[
                    row[3:6] if len(row) == 12 else row
                    for row in locs
                    if row is not None
                ]
            )
        ]
        x_min_list, x_max_list = zip(*self.ranges)
        x_min = min(x_min_list)
        x_max = max(x_max_list)
        self.x_max = max(abs(x_min), abs(x_max))
        self.coord_dim = embedding_dim // len(self.ranges)
        self.locs = list(locs)
        print(f"{self.ranges=}")
        assert embedding_dim % len(self.ranges) == 0
        super().__init__(len(locs), embedding_dim, **kwargs)

    def reset_parameters(self):
        for i, loc in enumerate(self.locs):
            if loc is None:
                nn.init.zeros_(self.weight[i])
            else:
                for j, (x, (x0, x1)) in enumerate(zip(loc, self.ranges)):
                    with torch.no_grad():
                        self.weight[
                            i, j * self.coord_dim : (j + 1) * self.coord_dim
                        ].copy_(
                            pos_encode_contineous(
                                x,
                                0,
                                10 * self.x_max,
                                self.coord_dim,
                                device=self.weight.device,
                            ),
                        )


class PosEncoder(nn.Module):
    """Positional encoder for EEG data.

    Parameters
    ----------
        spat_dim: int
            Number of dimensions to use to encode the spatial position of the patch,
            i.e. the EEG channel.
        time_dim: int
            Number of dimensions to use to encode the temporal position of the patch.
        ch_names: list[str]
            List of all the EEG channel names that could be encountered in the data.
        ch_locs: list of list of float or 2d array
            List of the n-dimensions locations of the EEG channels.
        sfreq_features: float
            The "downsampled" sampling frequency returned by the feature encoder.
        spat_kwargs: dict
            Additional keyword arguments to pass to the :class:`nn.Embedding` layer used to
            embed the channel names.
    """

    max_seconds: float = 600.0  # 10 minutes
    fixed_ch_names: list[str] | None = None

    def __init__(
        self,
        spat_dim: int,
        time_dim: int,
        ch_names: list[str],
        ch_locs: list[list[float]],
        sfreq_features: float,
        spat_kwargs: dict | None = None,
    ):
        assert len(ch_names) == len(ch_locs)
        super().__init__()
        spat_kwargs = spat_kwargs or {}
        ch_locs_plus_ukn = [None] + list(ch_locs)
        self.ch_names = ch_names
        self.pos_encoder_spat = ChannelEmbedding(
            ch_locs_plus_ukn, spat_dim, **spat_kwargs
        )  # (batch_size, n_channels, spat_dim)
        self.spat_dim = spat_dim
        self.time_dim = time_dim
        self.max_n_times = int(self.max_seconds * sfreq_features)
        self.encoding_time = torch.zeros(0, dtype=torch.float32, requires_grad=False)

    def _check_encoding_time(self, n_times):
        if self.encoding_time.size(0) < n_times:
            self.encoding_time = self.encoding_time.new_empty((n_times, self.time_dim))
            self.encoding_time[:] = pos_encode_time(
                n_times=n_times,
                n_dim=self.time_dim,
                max_n_times=self.max_n_times,
                device=self.encoding_time.device,
            )

    def forward(self, batch):
        """
        Parameters
        ----------
            batch: dict with keys:

                * local_features: (batch_size, n_chans * n_times_out, emb_dim)
                * ch_idxs: (batch_size, n_chans)
                    Indices of the channels to use in the ``ch_names`` list passed
                    as argument plus one. Index 0 is reserved for an unknown channel.
                    Only needed if ``set_fixed_ch_names`` has not been called.

        Returns
        -------
            pos_encoding: (batch_size, n_chans * n_times_out, emb_dim)
                The first ``spat_dim`` dimensions encode the channels positional encoding
                and the following ``time_dim`` dimensions encode the temporal positional encoding.
        """
        local_features = batch["local_features"]
        ch_idxs = self.get_ch_idxs(batch).to(local_features.device)
        batch_size, n_chans_times, emb_dim = local_features.shape
        batch_size_chs, n_chans = ch_idxs.shape  # ==(1,_) if fixed_ch_names
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
        if batch_size_chs == 1:  # case with fixed_ch_names
            pos_encoding = pos_encoding.tile(batch_size, 1, 1, 1)
        return pos_encoding.view(batch_size, n_chans_times, emb_dim)

    def set_fixed_ch_names(self, ch_names: list[str]):
        """Sets a fixed list of channels so that the batch
        does not need to contain ``ch_names``.
        """
        self.fixed_ch_names = list(ch_names)

    def unset_fixed_ch_names(self):
        """
        Unsets the fixed list of channels.
        """
        self.fixed_ch_names = None

    def get_ch_idxs(self, batch):
        if self.fixed_ch_names is not None:
            return (
                torch.tensor(
                    [self.ch_names.index(c) for c in self.fixed_ch_names]
                ).unsqueeze(0)
                + 1  # 0 is reserved for unknown channel
            )
        return batch["ch_idxs"]


def n_times_out(conv_layers_spec, n_times):
    # it would be equal to n_times//ds_factor without edge effects:
    n_times_out_ = n_times
    for _, width, stride in conv_layers_spec:
        n_times_out_ = int((n_times_out_ - width) / stride) + 1
    return n_times_out_


def get_out_emb_dim(conv_layers_spec, n_times, n_spat_filters=4):
    n_time_out = n_times_out(conv_layers_spec, n_times)
    emb_dim = conv_layers_spec[-1][0]
    return n_spat_filters * n_time_out * emb_dim


def get_separable_clf_layer(
    conv_layers_spec, n_chans, n_times, n_classes, n_spat_filters=4
):
    out_emb_dim = get_out_emb_dim(
        conv_layers_spec=conv_layers_spec,
        n_times=n_times,
        n_spat_filters=n_spat_filters,
    )
    clf_layer = nn.Sequential()
    clf_layer.add_module("unflatter_tokens", UnflattenTokens(n_chans))
    clf_layer.add_module("spat_conv", nn.Conv3d(1, n_spat_filters, (n_chans, 1, 1)))
    clf_layer.add_module("flatten", nn.Flatten(start_dim=1))
    clf_layer.add_module("linear", nn.Linear(out_emb_dim, n_classes))
    return clf_layer


_DEFAULT_CONV_LAYER_SPEC = (  # downsampling: 128Hz -> 1Hz, receptive field 1.1875s, stride 1s
    (8, 32, 8),
    (16, 2, 2),
    (32, 2, 2),
    (64, 2, 2),
    (64, 2, 2),
)


class _BaseSignalJEPA(EEGModuleMixin, nn.Module):
    """Base class for the SignalJEPA models.

    Parameters
    ----------
    feature_encoder__conv_layers_spec: list of tuples (dim, k, stride) where:

        * dim: number of output channels of the layer (unrelated to EEG channels);
        * k: temporal length of the layer's kernel;
        * stride: temporal stride of the layer's kernel.

    drop_prob: float
    feature_encoder__mode: str
        Normalisation mode. Either``default`` or ``layer_norm``.
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

    feature_encoder: ConvFeatureEncoder | None
    pos_encoder: PosEncoder | None
    transformer: nn.Transformer | None

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
            self.feature_encoder = ConvFeatureEncoder(
                conv_layers_spec=feature_encoder__conv_layers_spec,
                drop_prob=drop_prob,
                mode=feature_encoder__mode,
                conv_bias=feature_encoder__conv_bias,
                activation=activation,
            )

        if _init_transformer:
            ch_names = [ch["ch_name"] for ch in self.chs_info]
            ch_locs = [ch["loc"] for ch in self.chs_info]
            self.pos_encoder = PosEncoder(
                spat_dim=pos_encoder__spat_dim,
                time_dim=pos_encoder__time_dim,
                ch_names=ch_names,
                ch_locs=ch_locs,
                sfreq_features=pos_encoder__sfreq_features,
                spat_kwargs=pos_encoder__spat_kwargs,
            )
            self.pos_encoder.set_fixed_ch_names(ch_names)
            self.transformer = nn.Transformer(
                d_model=transformer__d_model,
                nhead=transformer__nhead,
                num_encoder_layers=transformer__num_encoder_layers,
                num_decoder_layers=transformer__num_decoder_layers,
                batch_first=True,
            )


class SignalJEPA(_BaseSignalJEPA):
    """Architecture introduced in signal-JEPA [sJEPA]_ and used for SSL pre-training.

    This model is not meant for classification but for SSL pre-training.
    Its output shape depends on the input shape.

    References
    ----------
    .. [sJEPA] Guetschel, P., Moreau, T., & Tangermann, M. (2024).
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

    def forward(self, X):
        batch = {"X": X}
        local_features = self.feature_encoder(batch)
        pos_encoding = self.pos_encoder(dict(local_features=local_features, **batch))
        local_features += pos_encoding
        contextual_features = self.transformer.encoder(local_features)
        y = self.final_layer(contextual_features)
        return y


class SignalJEPA_Contextual(_BaseSignalJEPA):
    """Contextual downstream architecture introduced in signal-JEPA [sJEPA]_.

    Parameters
    ----------
    n_spat_filters : int
        Number of spatial filters.

    References
    ----------
    .. [sJEPA] Guetschel, P., Moreau, T., & Tangermann, M. (2024).
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
        self.final_layer = get_separable_clf_layer(
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

    def forward(self, X):
        batch = {"X": X}
        local_features = self.feature_encoder(batch)
        pos_encoding = self.pos_encoder(dict(local_features=local_features, **batch))
        local_features += pos_encoding
        contextual_features = self.transformer.encoder(local_features)
        y = self.final_layer(contextual_features)
        return y


class SignalJEPA_PostLocal(_BaseSignalJEPA):
    """Post-local downstream architecture introduced in signal-JEPA [sJEPA]_.

    Parameters
    ----------
    n_spat_filters : int
        Number of spatial filters.

    References
    ----------
    .. [sJEPA] Guetschel, P., Moreau, T., & Tangermann, M. (2024).
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
        feature_encoder__conv_layers_spec: list[
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
        self.final_layer = get_separable_clf_layer(
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
        local_features = self.feature_encoder({"X": X})
        y = self.final_layer(local_features)
        return y


class SignalJEPA_PreLocal(_BaseSignalJEPA):
    """Pre-local downstream architecture introduced in signal-JEPA [sJEPA]_.

    Parameters
    ----------
    n_spat_filters : int
        Number of spatial filters.

    References
    ----------
    .. [sJEPA] Guetschel, P., Moreau, T., & Tangermann, M. (2024).
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
        feature_encoder__conv_layers_spec: list[
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
        self.spatial_conv = ReshapeConv2d(self.n_chans, n_spat_filters)
        out_emb_dim = get_out_emb_dim(
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
        local_features = self.feature_encoder({"X": X})
        y = self.final_layer(local_features)
        return y
