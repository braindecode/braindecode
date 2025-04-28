# This source code is licensed under Attribution-NonCommercial 4.0
# International.
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# Braindecode adaptation made by
# Bruno Aristimunha <b.aristimunha@gmail.com>

import logging
import math
import random
import typing as tp
from functools import partial

import mne
import torch
import torchaudio as ta
from torch import nn
from torch.nn import functional as F

logger = logging.getLogger(__name__)


class DeepRecurrentEncoder(nn.Module):
    """Deep Recurrent Encoder (DRE) for MEG data based on [chehab2022]_.

    .. figure:: _static/model/deeprecurrent.jpg
       :align: center
       :alt: Deep Recurrent Encoder Architecture Diagram [chehab2022]_

    This model aims to learn a generative model of brain activity (:math:`h_t`)
    and its mapping to observable MEG signals (:math:`x_t`). It generalizes
    traditional linear models like Temporal Receptive Fields (TRF) and
    Recurrent TRF (RTRF) by employing non-linear components, specifically
    LSTMs for dynamics and convolutional layers for input/output mapping.

    The underlying assumption, following [chehab2022]_, is that brain
    dynamics :math:`h_t` evolve based on past dynamics and external stimuli :math:`u_t`,
    and MEG sensors measure a transformation of these dynamics. The DRE
    parameterizes this relationship using deep learning components.

    The general structure implemented corresponds to Equation (6) in [chehab2022]_:

    .. math::

        \\begin{cases}
        \\mathbf{x}_{t:t-\\tau_x+1} & = d_{\\theta_2}(h_t) \\\\
        h_{t} & = f_{\\theta_1} (\\mathbf{h}_{t-1:t-\\tau_h},
                                e_{\\theta_3}(\\mathbf{\\tilde{u}}_{t:t-\\tau_u}))
        \\end{cases}

    Where:
        - :math:`x_t \\in \\mathbb{R}^{d_x}` is the MEG recording at time :math:`t`.
        - :math:`h_t \\in \\mathbb{R}^{d_h}` represents the latent brain state (or model estimate).
        - :math:`\\tilde{u}_t` is an potentially augmented control input, including
          external stimuli :math:`u_t`, subject embeddings :math:`s`, and optionally
          masked past MEG signals :math:`p_t \\odot x_t`.
        - :math:`e_{\\theta_3}` is an encoder function (typically convolutional layers,
          potentially processing STFT features) mapping inputs to a sequence
          representation suitable for the recurrent core.
        - :math:`f_{\\theta_1}` governs the latent state dynamics (implemented here via
          a recurrent component like LSTM/`DualPathRNN`).
        - :math:`d_{\\theta_2}` is a decoder function (typically transposed convolutions)
          mapping the latent state :math:`h_t` back to predicted MEG signals :math:`\\hat{x}_t`.
        - :math:`\\mathbf{v}_{t:t-\\tau+1}` denotes the flattened concatenation
          :math:`[v_t, ..., v_{t-\\tau+1}]`. DRE aims to minimize receptive fields
          (:math:`\\tau_h=1, \\tau_u=0` ideally for the LSTM core).

    Key features adapted from [chehab2022]_ include:

    - **Non-linear Dynamics:** Uses recurrent layers (e.g., LSTM inspired)
      to model :math:`f_{\\theta_1}`, allowing complex temporal dependencies.
    - **Convolutional Encoder/Decoder:** Employs 1D convolutions for :math:`e_{\\theta_3}`
      and transposed convolutions for :math:`d_{\\theta_2}`, potentially reducing
      temporal resolution via striding for efficiency.
    - **Subject Embeddings:** Learns a vector :math:`s \\in \\mathbb{R}^{d_s}` for each
      subject (`subject_dim`) to capture subject-specific variations within a
      single model, augmenting the input :math:`\\tilde{u}_t`.
    - **Optional STFT:** Can process time-frequency representations of MEG
      data (`n_fft`) as input to the encoder :math:`e_{\\theta_3}`.
    - **End-to-End Training:** Learns the encoder, dynamics, and decoder
      jointly by minimizing a prediction loss, typically
      :math:`\\sum_t \\| x_t - \\hat{x}_t \\|_2^2`.

    Parameters
    ----------
    in_channels : dict[str, int]
        Dictionary mapping input modality names (e.g., 'meg', 'audio') to their
        respective number of input channels. Keys must match `hidden`.
        The 'meg' modality undergoes special processing (STFT, subject adapt.).
    out_channels : int
        Number of output channels for the final layer (dimension :math:`d_x` of :math:`\\hat{x}_t`).
    hidden : dict[str, int]
        Dictionary mapping input modality names to the number of hidden channels
        for the first convolutional layer of their respective branch in the
        encoder :math:`e_{\\theta_3}`. Keys must match `in_channels`.
    depth : int, default=4
        Number of convolutional layers in each encoder branch (:math:`e_{\\theta_3}`)
        and implicitly in the decoder :math:`d_{\\theta_2}` if `linear_out=False`.
    concatenate : bool, default=False
        If True, concatenate the outputs of different modality branches from
        :math:`e_{\\theta_3}` before feeding into the dynamics model :math:`f_{\\theta_1}`.
    linear_out : bool, default=False
        If True, use a simple linear ConvTranspose1d layer for the decoder :math:`d_{\\theta_2}`.
        Mutually exclusive with `complex_out`.
    complex_out : bool, default=False
        If True, use a Conv1d -> Activation -> ConvTranspose1d sequence for the
        decoder :math:`d_{\\theta_2}`.
    kernel_size : int, default=5
        Kernel size for the 1D convolutional layers in :math:`e_{\\theta_3}` and :math:`d_{\\theta_2}`.
        Must be odd. Corresponds to :math:`K` in the paper's figure.
    growth : float, default=1.0
        Factor by which the number of channels changes in successive convolutional
        layers within :math:`e_{\\theta_3}`. ``channels = hidden * growth ** layer_index``.
    dilation_growth : int, default=2
        Factor by which dilation increases in successive convolutional layers.
    dilation_period : int, optional
        Number of layers after which dilation resets to 1. Default is None.
    skip : bool, default=False
        If True, use skip connections within the convolutional layers (:math:`e_{\\theta_3}, d_{\\theta_2}`).
    post_skip : bool, default=False
        If True, apply activation *after* skip connection addition.
    scale : float, optional
        Rescaling factor for skip connections.
    rewrite : bool, default=False
        If True, use weight normalization and potentially learnable scaling.
    groups : int, default=1
        Number of groups for grouped convolutions.
    glu : int, default=0
        If > 0, use Gated Linear Units (GLU) in convolutional layers.
    glu_context : int, default=0
        Context size for the GLU gate projection.
    glu_glu : bool, default=True
        If True and `glu > 0`, use GLU for both main path and gate.
    activation : type[torch.nn.Module], default=nn.Identity
        Activation function class (e.g., nn.ReLU, nn.GELU) used within :math:`e_{\\theta_3}` and :math:`d_{\\theta_2}`.
    dual_path : int, default=0
        If > 0, enables a `DualPathRNN` (LSTM-based) component to model :math:`f_{\\theta_1}`
        with the specified hidden size (:math:`d_h`). If 0, dynamics are implicitly handled
        by the convolutional structure only.
    conv_dropout : float, default=0.0
        Dropout rate applied within convolutional layers.
    dropout_input : float, default=0.0
        Dropout rate applied to the input of each convolutional layer.
    batch_norm : bool, default=False
        If True, use Batch Normalization in convolutional layers.
    relu_leakiness : float, default=0.0
        Leakiness factor if `activation` is LeakyReLU.
    n_subjects : int, default=200
        Total number of unique subjects for subject embeddings (:math:`s`).
    subject_dim : int, default=64
        Dimension (:math:`d_s`) of the subject-specific embedding vectors :math:`s`. If 0, no embedding.
    subject_layers : bool, default=False
        If True, add subject-specific linear adaptation layers to the 'meg' branch.
    subject_layers_dim : str, default="input"
        Target dimension ('input' or 'hidden') for subject layer adaptation.
    subject_layers_id : bool, default=False
        If True, concatenate subject embedding :math:`s` to the input of subject layers.
    embedding_scale : float, default=1.0
        Scaling factor applied to subject embeddings :math:`s`.
    n_fft : int, optional
        If set, applies STFT preprocessing to 'meg' inputs within :math:`e_{\\theta_3}`.
    fft_complex : bool, default=True
        If `n_fft` is set, determines if STFT output is complex (real/imag parts)
        or magnitude.
    merger : bool, default=False
        If True, enables `ChannelMerger` module for 'meg' channels early in :math:`e_{\\theta_3}`.
    merger_pos_dim : int, default=256
        Dimensionality of positional embeddings for `ChannelMerger`.
    merger_channels : int, default=270
        Number of output channels after `ChannelMerger`.
    merger_dropout : float, default=0.2
        Dropout rate within `ChannelMerger`.
    merger_penalty : float, default=0.0
        Usage penalty for `ChannelMerger` attention.
    merger_per_subject : bool, default=False
        If True, `ChannelMerger` parameters are subject-specific.
    dropout : float, default=0.0
        Dropout rate for initial `ChannelDropout` applied to 'meg' inputs.
    dropout_rescale : bool, default=True
        If True, rescale channels after `ChannelDropout`.
    initial_linear : int, default=0
        If > 0, adds initial 1x1 Conv layer(s) to 'meg' inputs in :math:`e_{\\theta_3}`.
    initial_depth : int, default=1
        Number of layers in the initial linear stack.
    initial_nonlin : bool, default=False
        If True, apply `activation` after the initial linear stack.

    Attributes
    ----------
    subject_embedding_ : ScaledEmbedding or None
        The learned subject embedding layer (producing :math:`s`).
    subject_layers_ : SubjectLayers or None
        The subject-specific adaptation layers module.
    stft_ : torchaudio.transforms.Spectrogram or None
        The STFT transformation layer.
    merger_ : ChannelMerger or None
        The channel merging module.
    dropout_ : ChannelDropout or None
        The initial channel dropout module.
    initial_linear_ : torch.nn.Sequential or None
        The initial linear layer(s) module.
    encoders_ : torch.nn.ModuleDict
        Dictionary containing the convolutional encoder (:math:`e_{\\theta_3}`) modules
        for each input modality.
    dual_path_ : DualPathRNN or None
        The recurrent module (:math:`f_{\\theta_1}`) if `dual_path > 0`.
    final_ : torch.nn.Module or None
        The final output layer(s) corresponding to the decoder (:math:`d_{\\theta_2}`).
    activation_ : torch.nn.Module
        Instance of the activation function used.

    Notes
    -----
    - The implementation closely follows the architecture described in [chehab2022]_,
      mapping the components to the functions :math:`e_{\\theta_3}` (encoder),
      :math:`f_{\\theta_1}` (dynamics), and :math:`d_{\\theta_2}` (decoder) from Equation (6).
    - The optional `DualPathRNN` (`dual_path` > 0) explicitly models the
      recurrent dynamics :math:`f_{\\theta_1}`. If `dual_path` is 0, the model relies
      solely on the convolutional structure for temporal processing.
    - Subject adaptation via embeddings (`subject_dim`) or layers (`subject_layers`)
      allows a single model to handle multi-subject data effectively.
    - The use of strided convolutions in the encoder :math:`e_{\\theta_3}` can reduce
      the sequence length processed by the recurrent component :math:`f_{\\theta_1}`,
      improving computational efficiency, as mentioned in [chehab2022]_.

    References
    ----------
    .. [chehab2022] Chehab, O., Défossez, A., Loiseau, J. C., Gramfort, A.,
       & King, J. R. (2022). Deep Recurrent Encoder: an end-to-end network to model
       magnetoencephalography at scale. Neurons, Behavior, Data Analysis, and Theory.
    .. [brainmagik] Défossez, A., Caucheteux, C., Rapin, J., Kabeli, O., & King, J. R.
       (2023). Decoding speech perception from non-invasive brain recordings. Nature
       Machine Intelligence, 5(10), 1097-1107.

    .. versionadded:: 1.0
    """

    def __init__(
        self,
        # Channels
        in_channels: tp.Dict[str, int],
        out_channels: int,
        hidden: tp.Dict[str, int],
        # Overall structure
        depth: int = 4,
        concatenate: bool = False,  # concatenate the inputs
        linear_out: bool = False,
        complex_out: bool = False,
        # Conv layer
        kernel_size: int = 5,
        growth: float = 1.0,
        dilation_growth: int = 2,
        dilation_period: tp.Optional[int] = None,
        skip: bool = False,
        post_skip: bool = False,
        scale: tp.Optional[float] = None,
        rewrite: bool = False,
        groups: int = 1,
        glu: int = 0,
        glu_context: int = 0,
        glu_glu: bool = True,
        activation: type[nn.Module] = nn.Identity,
        # Dual path RNN
        dual_path: int = 0,
        # Dropouts, BN, activations
        conv_dropout: float = 0.0,
        dropout_input: float = 0.0,
        batch_norm: bool = False,
        relu_leakiness: float = 0.0,
        # Subject specific settings
        n_subjects: int = 200,
        subject_dim: int = 64,
        subject_layers: bool = False,
        subject_layers_dim: str = "input",  # or hidden
        subject_layers_id: bool = False,
        embedding_scale: float = 1.0,
        # stft transform
        n_fft: tp.Optional[int] = None,
        fft_complex: bool = True,
        # Attention multi-dataset support
        merger: bool = False,
        merger_pos_dim: int = 256,
        merger_channels: int = 270,
        merger_dropout: float = 0.2,
        merger_penalty: float = 0.0,
        merger_per_subject: bool = False,
        dropout: float = 0.0,
        dropout_rescale: bool = True,
        initial_linear: int = 0,
        initial_depth: int = 1,
        initial_nonlin: bool = False,
    ):
        super().__init__()

        # check inputs
        if set(in_channels.keys()) != set(hidden.keys()):
            raise ValueError(
                "Channels and hidden keys must match "
                f"({set(in_channels.keys())} and {set(hidden.keys())})"
            )
        assert kernel_size % 2 == 1, "For padding to work, this must be verified"

        self._concatenate = concatenate
        self.out_channels = out_channels

        self.merger = None
        self.dropout = None
        self.initial_linear = None
        self.activation = activation()

        if dropout > 0.0:
            self.dropout = ChannelDropout(dropout, dropout_rescale)
        if merger:
            self.merger = ChannelMerger(
                merger_channels,
                pos_dim=merger_pos_dim,
                dropout=merger_dropout,
                usage_penalty=merger_penalty,
                n_subjects=n_subjects,
                per_subject=merger_per_subject,
            )
            in_channels["meg"] = merger_channels

        if initial_linear:
            init = [nn.Conv1d(in_channels["meg"], initial_linear, 1)]
            for _ in range(initial_depth - 1):
                init += [activation(), nn.Conv1d(initial_linear, initial_linear, 1)]
            if initial_nonlin:
                init += [activation()]
            self.initial_linear = nn.Sequential(*init)
            in_channels["meg"] = initial_linear

        self.subject_layers = None
        if subject_layers:
            assert "meg" in in_channels
            meg_dim = in_channels["meg"]
            dim = {"hidden": hidden["meg"], "input": meg_dim}[subject_layers_dim]
            self.subject_layers = SubjectLayers(
                meg_dim, dim, n_subjects, subject_layers_id
            )
            in_channels["meg"] = dim

        self.stft = None
        if n_fft is not None:
            assert "meg" in in_channels
            self.fft_complex = fft_complex
            self.n_fft = n_fft
            self.stft = ta.transforms.Spectrogram(
                n_fft=n_fft,
                hop_length=n_fft // 2,
                normalized=True,
                power=None if fft_complex else 1,
                return_complex=True,
            )
            in_channels["meg"] *= n_fft // 2 + 1
            if fft_complex:
                in_channels["meg"] *= 2

        self.subject_embedding = None
        if subject_dim:
            self.subject_embedding = ScaledEmbedding(
                n_subjects, subject_dim, embedding_scale
            )
            in_channels["meg"] += subject_dim

        # concatenate inputs if need be
        if concatenate:
            in_channels = {"concat": sum(in_channels.values())}
            hidden = {"concat": sum(hidden.values())}

        # compute the sequences of channel sizes
        sizes = {}
        for name in in_channels:
            sizes[name] = [in_channels[name]]
            sizes[name] += [int(round(hidden[name] * growth**k)) for k in range(depth)]

        params: tp.Dict[str, tp.Any]
        params = dict(
            kernel=kernel_size,
            stride=1,
            leakiness=relu_leakiness,
            dropout=conv_dropout,
            dropout_input=dropout_input,
            batch_norm=batch_norm,
            dilation_growth=dilation_growth,
            groups=groups,
            dilation_period=dilation_period,
            skip=skip,
            post_skip=post_skip,
            scale=scale,
            rewrite=rewrite,
            glu=glu,
            glu_context=glu_context,
            glu_glu=glu_glu,
            activation=activation,
        )

        final_channels = sum([x[-1] for x in sizes.values()])
        self.dual_path = None
        if dual_path:
            self.dual_path = DualPathRNN(final_channels, dual_path)
        self.final = None
        pad = 0
        kernel = 1
        stride = 1
        if n_fft is not None:
            pad = n_fft // 4
            kernel = n_fft
            stride = n_fft // 2

        if linear_out:
            assert not complex_out
            self.final = nn.ConvTranspose1d(
                final_channels, out_channels, kernel, stride, pad
            )
        elif complex_out:
            self.final = nn.Sequential(
                nn.Conv1d(final_channels, 2 * final_channels, 1),
                activation(),
                nn.ConvTranspose1d(
                    2 * final_channels, out_channels, kernel, stride, pad
                ),
            )
        else:
            assert len(sizes) == 1, "if no linear_out, there must be a single branch."
            params["activation_on_last"] = False
            list(sizes.values())[0][-1] = out_channels

        self.encoders = nn.ModuleDict(
            {name: ConvSequence(channels, **params) for name, channels in sizes.items()}
        )

    def forward(self, inputs, batch):
        subjects = batch.subject_index
        length = next(iter(inputs.values())).shape[-1]  # length of any of the inputs

        if self.dropout is not None:
            inputs["meg"] = self.dropout(inputs["meg"], batch)

        if self.merger is not None:
            inputs["meg"] = self.merger(inputs["meg"], batch)

        if self.initial_linear is not None:
            inputs["meg"] = self.initial_linear(inputs["meg"])

        if self.subject_layers is not None:
            inputs["meg"] = self.subject_layers(inputs["meg"], subjects)

        if self.stft is not None:
            x = inputs["meg"]
            pad = self.n_fft // 4
            x = F.pad(pad_multiple(x, self.n_fft // 2), (pad, pad), mode="reflect")
            z = self.stft(inputs["meg"])
            B, C, Fr, T = z.shape
            if self.fft_complex:
                z = torch.view_as_real(z).permute(0, 1, 2, 4, 3)
            z = z.reshape(B, -1, T)
            inputs["meg"] = z

        if self.subject_embedding is not None:
            emb = self.subject_embedding(subjects)[:, :, None]
            inputs["meg"] = torch.cat(
                [inputs["meg"], emb.expand(-1, -1, length)], dim=1
            )

        if self._concatenate:
            input_list = [x[1] for x in sorted(inputs.items())]
            inputs = {"concat": torch.cat(input_list, dim=1)}

        encoded = {}
        for name, x in inputs.items():
            encoded[name] = self.encoders[name](x)

        inputs = [x[1] for x in sorted(encoded.items())]
        x = torch.cat(inputs, dim=1)
        if self.dual_path is not None:
            x = self.dual_path(x)
        if self.final is not None:
            x = self.final(x)
        assert x.shape[-1] >= length
        return x[:, :, :length]


class Recording:
    recording_index: int
    study_name: str
    recording_uid: str
    mne_info: mne.io.BaseRaw

    def __init__(
        self,
        recording_index: int,
        study_name: str,
        recording_uid: str,
        mne_info: mne.io.BaseRaw,
    ):
        self.recording_index = recording_index
        self.study_name = study_name
        self.recording_uid = recording_uid
        self.mne_info = mne_info


def pad_multiple(x: torch.Tensor, base: int):
    length = x.shape[-1]
    target = math.ceil(length / base) * base
    return torch.nn.functional.pad(x, (0, target - length))


class ScaledEmbedding(nn.Module):
    """Scale up learning rate for the embedding, otherwise, it can move too slowly."""

    def __init__(self, num_embeddings: int, embedding_dim: int, scale: float = 10.0):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data /= scale
        self.scale = scale

    @property
    def weight(self):
        return self.embedding.weight * self.scale

    def forward(self, x):
        return self.embedding(x) * self.scale


class SubjectLayers(nn.Module):
    """Per subject linear layer."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_subjects: int,
        init_id: bool = False,
    ):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(n_subjects, in_channels, out_channels))
        if init_id:
            assert in_channels == out_channels
            self.weights.data[:] = torch.eye(in_channels)[None]
        self.weights.data *= 1 / in_channels**0.5

    def forward(self, x, subjects):
        _, C, D = self.weights.shape
        weights = self.weights.gather(0, subjects.view(-1, 1, 1).expand(-1, C, D))
        return torch.einsum("bct,bcd->bdt", x, weights)

    def __repr__(self):
        S, C, D = self.weights.shape
        return f"SubjectLayers({C}, {D}, {S})"


class LayerScale(nn.Module):
    """Layer scale from [Touvron et al 2021] (https://arxiv.org/pdf/2103.17239.pdf).
    This rescales diagonally residual outputs close to 0 initially, then learnt.
    """

    def __init__(self, channels: int, init: float = 0.1, boost: float = 5.0):
        super().__init__()
        self.scale = nn.Parameter(torch.zeros(channels, requires_grad=True))
        self.scale.data[:] = init / boost
        self.boost = boost

    def forward(self, x):
        return (self.boost * self.scale[:, None]) * x


class ConvSequence(nn.Module):
    def __init__(
        self,
        channels: tp.Sequence[int],
        kernel: int = 4,
        dilation_growth: int = 1,
        dilation_period: tp.Optional[int] = None,
        stride: int = 2,
        dropout: float = 0.0,
        leakiness: float = 0.0,
        groups: int = 1,
        decode: bool = False,
        batch_norm: bool = False,
        dropout_input: float = 0,
        skip: bool = False,
        scale: tp.Optional[float] = None,
        rewrite: bool = False,
        activation_on_last: bool = True,
        post_skip: bool = False,
        glu: int = 0,
        glu_context: int = 0,
        glu_glu: bool = True,
        activation: tp.Any = None,
    ) -> None:
        super().__init__()
        dilation = 1
        channels = tuple(channels)
        self.skip = skip
        self.sequence = nn.ModuleList()
        self.glus = nn.ModuleList()
        if activation is None:
            activation = partial(nn.LeakyReLU, leakiness)
        Conv = nn.Conv1d if not decode else nn.ConvTranspose1d
        # build layers
        for k, (chin, chout) in enumerate(zip(channels[:-1], channels[1:])):
            layers: tp.List[nn.Module] = []
            is_last = k == len(channels) - 2

            # Set dropout for the input of the conv sequence if defined
            if k == 0 and dropout_input:
                assert 0 < dropout_input < 1
                layers.append(nn.Dropout(dropout_input))

            # conv layer
            if dilation_growth > 1:
                assert kernel % 2 != 0, "Supports only odd kernel with dilation for now"
            if dilation_period and (k % dilation_period) == 0:
                dilation = 1
            pad = kernel // 2 * dilation
            layers.append(
                Conv(
                    chin,
                    chout,
                    kernel,
                    stride,
                    pad,
                    dilation=dilation,
                    groups=groups if k > 0 else 1,
                )
            )
            dilation *= dilation_growth
            # non-linearity
            if activation_on_last or not is_last:
                if batch_norm:
                    layers.append(nn.BatchNorm1d(num_features=chout))
                layers.append(activation())
                if dropout:
                    layers.append(nn.Dropout(dropout))
                if rewrite:
                    layers += [nn.Conv1d(chout, chout, 1), nn.LeakyReLU(leakiness)]
                    # layers += [nn.Conv1d(chout, 2 * chout, 1), nn.GLU(dim=1)]
            if chin == chout and skip:
                if scale is not None:
                    layers.append(LayerScale(chout, scale))
                if post_skip:
                    layers.append(Conv(chout, chout, 1, groups=chout, bias=False))

            self.sequence.append(nn.Sequential(*layers))
            if glu and (k + 1) % glu == 0:
                ch = 2 * chout if glu_glu else chout
                act = nn.GLU(dim=1) if glu_glu else activation()
                self.glus.append(
                    nn.Sequential(
                        nn.Conv1d(chout, ch, 1 + 2 * glu_context, padding=glu_context),
                        act,
                    )
                )
            else:
                self.glus.append(None)

    def forward(self, x: tp.Any) -> tp.Any:
        for module_idx, module in enumerate(self.sequence):
            old_x = x
            x = module(x)
            if self.skip and x.shape == old_x.shape:
                x = x + old_x
            glu = self.glus[module_idx]
            if glu is not None:
                x = glu(x)
        return x


class DualPathRNN(nn.Module):
    def __init__(self, channels: int, depth: int, inner_length: int = 10):
        super().__init__()
        self.lstms = nn.ModuleList(
            [nn.LSTM(channels, channels, 1) for _ in range(depth * 4)]
        )
        self.inner_length = inner_length

    def forward(self, x: torch.Tensor):
        B, C, L = x.shape
        IL = self.inner_length
        x = pad_multiple(x, self.inner_length)
        x = x.permute(2, 0, 1).contiguous()
        for idx, lstm in enumerate(self.lstms):
            y = x.reshape(-1, IL, B, C)
            if idx % 2 == 0:
                y = y.transpose(0, 1).reshape(IL, -1, C)
            else:
                y = y.reshape(-1, IL * B, C)
            y, _ = lstm(x)
            if idx % 2 == 0:
                y = y.reshape(IL, -1, B, C).transpose(0, 1).reshape(-1, B, C)
            else:
                y = y.reshape(-1, B, C)
            x = x + y

            if idx % 2 == 1:
                x = x.flip(dims=(0,))
        return x[:L].permute(1, 2, 0).contiguous()


class PositionGetter:
    INVALID = -0.1

    def __init__(self) -> None:
        self._cache: tp.Dict[int, torch.Tensor] = {}
        self._invalid_names: tp.Set[str] = set()

    def get_recording_layout(self, recording: Recording) -> torch.Tensor:
        index = recording.recording_index
        if index in self._cache:
            return self._cache[index]
        else:
            info = recording.mne_info
            layout = mne.find_layout(info)
            indexes: tp.List[int] = []
            valid_indexes: tp.List[int] = []
            for meg_index, name in enumerate(info.ch_names):
                name = name.rsplit("-", 1)[0]
                try:
                    indexes.append(layout.names.index(name))
                except ValueError:
                    if name not in self._invalid_names:
                        logger.warning(
                            "Channels %s not in layout for recording %s of %s.",
                            name,
                            recording.study_name,
                            recording.recording_uid,
                        )
                        self._invalid_names.add(name)
                else:
                    valid_indexes.append(meg_index)

            positions = torch.full((len(info.ch_names), 2), self.INVALID)
            x, y = layout.pos[indexes, :2].T
            x = (x - x.min()) / (x.max() - x.min())
            y = (y - y.min()) / (y.max() - y.min())
            x = torch.from_numpy(x).float()
            y = torch.from_numpy(y).float()
            positions[valid_indexes, 0] = x
            positions[valid_indexes, 1] = y
            self._cache[index] = positions
            return positions

    def get_positions(self, batch):
        meg = batch.meg
        B, C, T = meg.shape
        positions = torch.full((B, C, 2), self.INVALID, device=meg.device)
        for idx in range(len(batch)):
            recording = batch._recordings[idx]
            rec_pos = self.get_recording_layout(recording)
            positions[idx, : len(rec_pos)] = rec_pos.to(meg.device)
        return positions

    def is_invalid(self, positions):
        return (positions == self.INVALID).all(dim=-1)


class FourierEmb(nn.Module):
    """
    Fourier positional embedding.
    Unlike trad. embedding this is not using exponential periods
    for cosines and sinuses, but typical `2 pi k` which can represent
    any function over [0, 1]. As this function would be necessarily periodic,
    we take a bit of margin and do over [-0.2, 1.2].
    """

    def __init__(self, dimension: int = 256, margin: float = 0.2):
        super().__init__()
        n_freqs = (dimension // 2) ** 0.5
        assert int(n_freqs**2 * 2) == dimension
        self.dimension = dimension
        self.margin = margin

    def forward(self, positions):
        *O, D = positions.shape
        assert D == 2
        *O, D = positions.shape
        n_freqs = (self.dimension // 2) ** 0.5
        freqs_y = torch.arange(n_freqs).to(positions)
        freqs_x = freqs_y[:, None]
        width = 1 + 2 * self.margin
        positions = positions + self.margin
        p_x = 2 * math.pi * freqs_x / width
        p_y = 2 * math.pi * freqs_y / width
        positions = positions[..., None, None, :]
        loc = (positions[..., 0] * p_x + positions[..., 1] * p_y).view(*O, -1)
        emb = torch.cat(
            [
                torch.cos(loc),
                torch.sin(loc),
            ],
            dim=-1,
        )
        return emb


class ChannelDropout(nn.Module):
    def __init__(self, dropout: float = 0.1, rescale: bool = True):
        """
        Args:
            dropout: dropout radius in normalized [0, 1] coordinates.
            rescale: at valid, rescale all channels.
        """
        super().__init__()
        self.dropout = dropout
        self.rescale = rescale
        self.position_getter = PositionGetter()

    def forward(self, meg, batch):
        if not self.dropout:
            return meg

        B, C, T = meg.shape
        meg = meg.clone()
        positions = self.position_getter.get_positions(batch)
        valid = (~self.position_getter.is_invalid(positions)).float()
        meg = meg * valid[:, :, None]

        if self.training:
            center_to_ban = torch.rand(2, device=meg.device)
            kept = (positions - center_to_ban).norm(dim=-1) > self.dropout
            meg = meg * kept.float()[:, :, None]
            if self.rescale:
                proba_kept = torch.zeros(B, C, device=meg.device)
                n_tests = 100
                for _ in range(n_tests):
                    center_to_ban = torch.rand(2, device=meg.device)
                    kept = (positions - center_to_ban).norm(dim=-1) > self.dropout
                    proba_kept += kept.float() / n_tests
                meg = meg / (1e-8 + proba_kept[:, :, None])

        return meg


class ChannelMerger(nn.Module):
    def __init__(
        self,
        chout: int,
        pos_dim: int = 256,
        dropout: float = 0,
        usage_penalty: float = 0.0,
        n_subjects: int = 200,
        per_subject: bool = False,
    ):
        super().__init__()
        assert pos_dim % 4 == 0
        self.position_getter = PositionGetter()
        self.per_subject = per_subject
        if self.per_subject:
            self.heads = nn.Parameter(
                torch.randn(n_subjects, chout, pos_dim, requires_grad=True)
            )
        else:
            self.heads = nn.Parameter(torch.randn(chout, pos_dim, requires_grad=True))
        self.heads.data /= pos_dim**0.5
        self.dropout = dropout
        self.embedding = FourierEmb(pos_dim)
        self.usage_penalty = usage_penalty
        self._penalty = torch.tensor(0.0)

    @property
    def training_penalty(self):
        return self._penalty.to(next(self.parameters()).device)

    def forward(self, meg, batch):
        B, C, T = meg.shape
        meg = meg.clone()
        positions = self.position_getter.get_positions(batch)
        embedding = self.embedding(positions)
        score_offset = torch.zeros(B, C, device=meg.device)
        score_offset[self.position_getter.is_invalid(positions)] = float("-inf")

        if self.training and self.dropout:
            center_to_ban = torch.rand(2, device=meg.device)
            radius_to_ban = self.dropout
            banned = (positions - center_to_ban).norm(dim=-1) <= radius_to_ban
            score_offset[banned] = float("-inf")

        if self.per_subject:
            _, cout, pos_dim = self.heads.shape
            subject = batch.subject_index
            heads = self.heads.gather(
                0, subject.view(-1, 1, 1).expand(-1, cout, pos_dim)
            )
        else:
            heads = self.heads[None].expand(B, -1, -1)

        scores = torch.einsum("bcd,bod->boc", embedding, heads)
        scores += score_offset[:, None]
        weights = torch.softmax(scores, dim=2)
        out = torch.einsum("bct,boc->bot", meg, weights)
        if self.training and self.usage_penalty > 0.0:
            usage = weights.mean(dim=(0, 1)).sum()
            self._penalty = self.usage_penalty * usage
        return out


if __name__ == "__main__":
    # Work in Progress
    model = DeepRecurrentEncoder(
        in_channels={"meg": 10}, out_channels=2, hidden={"meg": 20}
    )
    x = torch.randn(2, 10, 100)
    batch = type("Batch", (object,), {"subject_index": torch.tensor([0, 1])})
    y = model(x, batch)
