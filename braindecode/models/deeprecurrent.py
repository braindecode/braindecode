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
    def __init__(
        self,
        ########################
        # Braindecode parameters
        #########################
        # Channels
        n_chans: int,  # in_channels: int,
        n_outputs: int,  # out_channels: int,
        ####################
        # Model parameters
        hidden_channels: int,
        # Overall structure
        depth: int = 4,
        linear_out: bool = True,
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
        initial_linear=20,
        initial_depth: int = 1,
        initial_nonlin: bool = False,
        # Final layer
        decode: bool = True,  # True for braindecode
        # Braindecode parameters to replace the other parameters
        n_times=None,
        chs_info=None,
        input_window_seconds=None,
        sfreq=None,
    ):
        super().__init__()

        # check inputs
        assert kernel_size % 2 == 1, "For padding to work, this must be verified"
        # number of classes
        self.n_outputs = n_outputs

        self.merger = None
        self.dropout = None
        self.initial_linear = None

        # initialize dummy layers that will be replaced by the real ones
        self.activation_ = activation()
        self.initial_linear_ = None
        self.subject_layers_ = None
        self.subject_embedding_ = None
        self.stft_ = None
        self.dual_path_ = None
        self.final_ = None
        self.final_ = None

        current_in_channels = n_chans

        # Parameters for the encoder
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
            decode=decode,
        )
        # parameters for final
        pad = 0
        kernel = 1
        stride = 1
        #############################################################
        # Creating the layer...
        if dropout > 0.0:
            self.dropout_ = ChannelDropout(dropout, dropout_rescale)
        if merger:
            self.merger_ = ChannelMerger(
                merger_channels,
                pos_dim=merger_pos_dim,
                dropout=merger_dropout,
                usage_penalty=merger_penalty,
                n_subjects=n_subjects,
                per_subject=merger_per_subject,
            )
            current_in_channels = merger_channels

        # So confused this part... We have one conv layer for each channel (?)
        # very similar to eeg-simple conv, but as optional (?)
        if initial_linear:
            init = [nn.Conv1d(current_in_channels, initial_linear, 1)]
            for _ in range(initial_depth - 1):
                init += [activation(), nn.Conv1d(initial_linear, initial_linear, 1)]
            if initial_nonlin:
                init += [activation()]
            self.initial_linear_ = nn.Sequential(*init)
            # overwrite the input channels
            current_in_channels = initial_linear

        if subject_layers:
            # dim is equal to n_times in braindecode
            # SubjectLayers adapts the channel dimension
            input_dim_for_subj = current_in_channels
            subj_layer_out_dim = {
                "hidden": hidden_channels,
                "input": input_dim_for_subj,
            }[subject_layers_dim]

            self.subject_layers_ = SubjectLayers(
                input_dim_for_subj, subj_layer_out_dim, n_subjects, subject_layers_id
            )
            current_in_channels = subj_layer_out_dim

        if n_fft is not None:
            self.fft_complex = fft_complex
            self.n_fft = n_fft
            self.stft_ = ta.transforms.Spectrogram(
                n_fft=n_fft,
                hop_length=n_fft // 2,
                normalized=True,
                power=None if fft_complex else 1,
                return_complex=True,
            )
            stft_freq_bins = n_fft // 2 + 1
            stft_out_channels = current_in_channels * stft_freq_bins
            if fft_complex:
                # real and imag
                stft_out_channels *= 2
            current_in_channels = stft_out_channels

        if subject_layers:
            self.subject_embedding_ = ScaledEmbedding(
                n_subjects, subject_dim, embedding_scale
            )
            current_in_channels += subject_dim

        # compute the sequences of channel sizes
        conv_seq_channels = [current_in_channels]
        conv_seq_channels += [
            int(round(hidden_channels * growth**k)) for k in range(depth)
        ]

        if not linear_out and not complex_out:
            params["activation_on_last"] = False
            conv_seq_channels[-1] = n_outputs  # Output channels defined by last layer

        self.encoder_ = ConvSequence(conv_seq_channels, **params)

        final_channels = conv_seq_channels[-1]

        # --- Recurrent Layer ---
        if dual_path:
            # Input to DualPathRNN is the output of the encoder
            self.dual_path_ = DualPathRNN(final_channels, dual_path)
            # DualPathRNN output channels == input channels

        # --- Final Layer ---

        if n_fft is not None:
            # Adjust ConvTranspose1d for STFT hop length
            pad = n_fft // 4
            kernel = n_fft
            stride = n_fft // 2

        # Check if encoder already produced the output channels
        if linear_out:
            assert not complex_out
            self.final_ = nn.ConvTranspose1d(
                final_channels, self.n_outputs, kernel, stride, pad
            )
        elif complex_out:
            self.final_ = nn.Sequential(
                nn.Conv1d(final_channels, 2 * final_channels, 1),
                self.activation_(),
                nn.ConvTranspose1d(
                    2 * final_channels, self.n_outputs, kernel, stride, pad
                ),
            )

    def forward(
        self,
        x,
        batch=None,
    ):
        # subjects = batch.subject_index
        # Estimate original length before potential STFT downsampling
        original_length = x.shape[-1]

        # Apply ""preprocessing"" layers sequentially
        # if self.dropout_ is not None:
        #     x = self.dropout_(x, batch)

        # if self.merger_ is not None:
        #     x = self.merger_(x, batch)

        if self.initial_linear_ is not None:
            x = self.initial_linear_(x)

        # if self.subject_layers_ is not None:
        #     x = self.subject_layers_(x, subjects)

        if self.stft_ is not None:
            pad_amount = self.n_fft // 4  # Use attribute directly
            # Pad for STFT analysis window overlap
            x_padded = F.pad(
                pad_multiple(x, self.n_fft // 2),
                (pad_amount, pad_amount),
                mode="reflect",
            )
            z = self.stft_(x_padded)  # Apply STFT
            B, C, Fr, T_stft = z.shape

            if self.fft_complex:
                # Convert complex tensor to real representation (B, C, Freq, 2, Time) -> (B, C * Freq * 2, Time)
                z = torch.view_as_real(z).permute(
                    0, 1, 2, 4, 3
                )  # B, C, Fr, T, 2 -> TO-DO: replace for einops
                z = z.reshape(
                    B, C * Fr * 2, T_stft
                )  # Combine C, Fr, Real/Imag into one dim
            else:
                # If power=1 (magnitude), reshape directly
                z = z.reshape(B, C * Fr, T_stft)  # Combine C, Fr
            x = z  # Update x to be the STFT representation

        # if self.subject_embedding_ is not None:
        #     current_length = x.shape[-1]
        #     emb = self.subject_embedding_(subjects)[:, :, None]
        #     x = torch.cat(
        #         [x, emb.expand(-1, -1, current_length)], dim=1
        #     )

        # Main Encoder
        x = self.encoder_(x)

        # Optional Recurrent Layer
        if self.dual_path_ is not None:
            x = self.dual_path_(x)

        # Final Decoder/Output Layer
        if self.final_ is not None:
            x = self.final_(x)

        current_length = x.shape[-1]
        if current_length < original_length:
            # This might happen if strides reduce length too much
            logger.warning(
                f"Output length ({current_length}) is less than input length ({original_length}). Padding..."
            )
            x = F.pad(x, (0, original_length - current_length))
        elif current_length > original_length:
            # Crop the output to match the original input length
            x = x[:, :, :original_length]

        return x


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
        kernel: int = 4,  # should not be 5 here?
        dilation_growth: int = 1,
        dilation_period: tp.Optional[int] = None,
        stride: int = 2,
        dropout: float = 0.0,
        leakiness: float = 0.0,
        groups: int = 1,
        decode: bool = False,  # This param should be True or False for braindecode use?
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
        self.skip = skip  # what is skip?
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
    def __init__(self, ch_info, dropout: float = 0.1, rescale: bool = True):
        """
        Args:
            dropout: dropout radius in normalized [0, 1] coordinates.
            rescale: at valid, rescale all channels.
        """
        super().__init__()
        self.dropout = dropout
        self.rescale = rescale
        self.ch_info = ch_info

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
        self.position_getter = PositionGetter()  # type: ignore

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
    n_batch = 4
    n_channels_in = 20
    n_channels_out = 2  # classes (?)
    n_times = 1000
    hidden = 32
    depth = 3
    sfreq = 100.0

    print("\n--- Example with STFT ---")
    n_fft_val = 32
    model = DeepRecurrentEncoder(
        n_chans=n_channels_in,
        n_outputs=n_channels_out,
        hidden_channels=hidden,
        depth=depth,
        kernel_size=3,  # Smaller kernel often used with STFT
        subject_layers=False,
        dropout=0.1,
        merger=False,
        n_fft=n_fft_val,
        fft_complex=True,
        dual_path=1,
        linear_out=False,
    )
    subject_index = torch.randint(0, 200, (n_batch,))
    x_tensor = torch.randn(n_batch, n_channels_in, n_times)

    y = model(x_tensor)
    print(y.shape)
