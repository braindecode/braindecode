# Authors: Meta Platforms, Inc. and affiliates (original)
#          Bruno Aristimunha <b.aristimunha@gmail.com> (Braindecode adaptation)
#          Hubert Banville <hubertjb@meta.com> (Braindecode adaptation and Review)
#
# License: Attribution-NonCommercial 4.0 International

"""BrainModule: Dilated Convolutional Encoder for EEG classification."""

from __future__ import annotations

import math
import typing as tp

import torch
import torchaudio as ta
from torch import nn
from torch.nn import functional as F

from braindecode.models.base import EEGModuleMixin
from braindecode.modules.layers import SubjectLayers

__all__ = ["BrainModule"]


class BrainModule(EEGModuleMixin, nn.Module):
    r"""BrainModule from Brain Module [brainmagik]_, also known as SimpleConv.

    A dilated convolutional encoder for EEG classification, using residual
    connections and optional GLU gating for improved expressivity.

    :bdg-success:`Convolution`

    .. figure:: ../_static/model/simpleconv.png
        :align: center
        :alt: BrainModule Architecture
        :width: 1000px

    Parameters
    ----------
    hidden_dim : int, default=320
        Hidden dimension for convolutional layers. Input is projected to this
        dimension before the convolutional blocks.
    depth : int, default=4
        Number of convolutional blocks. Each block contains a dilated convolution
        with batch normalization and activation, followed by a residual connection.
    kernel_size : int, default=3
        Convolutional kernel size. Must be odd for proper padding with dilation.
    growth : float, default=1.0
        Channel size multiplier: hidden_dim * (growth ** layer_index).
        Values > 1.0 grow channels deeper; < 1.0 shrink them.
        Note: growth != 1.0 disables residual connections between layers
        with different channel sizes.
    dilation_growth : int, default=2
        Dilation multiplier per layer (e.g., 2 means dilation doubles each layer).
        Improves receptive field exponentially. Requires odd kernel_size.
    dilation_period : int, default=5
        Reset dilation to 1 every N layers. Prevents dilation from growing
        too large and maintains local connectivity.
    conv_drop_prob : float, default=0.0
        Dropout probability for convolutional layers.
    dropout_input : float, default=0.0
        Dropout probability applied to model input only.
    batch_norm : bool, default=True
        If True, apply batch normalization after each convolution.
    activation : type[nn.Module], default=nn.GELU
        Activation function class to use (e.g., nn.GELU, nn.ReLU, nn.ELU).
    n_subjects : int, default=200
        Number of unique subjects (for subject-specific pathways).
        Only used if subject_dim > 0.
    subject_dim : int, default=0
        Dimension of subject embeddings. If 0, no subject-specific features.
        If > 0, adds subject embeddings to the input before encoding.
    subject_layers : bool, default=False
        If True, apply subject-specific linear transformations to input channels.
        Each subject has its own weight matrix. Requires subject_dim > 0.
    subject_layers_dim : str, default="input"
        Where to apply subject layers: "input" or "hidden".
    subject_layers_id : bool, default=False
        If True, initialize subject layers as identity matrices.
    embedding_scale : float, default=1.0
        Scaling factor for subject embeddings learning rate.
    n_fft : int, optional
        FFT size for STFT processing. If None, no STFT is applied.
        If specified, applies spectrogram transform before encoding.
    fft_complex : bool, default=True
        If True, keep complex spectrogram. If False, use power spectrogram.
        Only used when n_fft is not None.
    channel_dropout_prob : float, default=0.0
        Probability of dropping each channel during training (0.0 to 1.0).
        If 0.0, no channel dropout is applied.
    channel_dropout_type : str, optional
        If specified with chs_info, only drop channels of this type
        (e.g., 'eeg', 'ref', 'eog'). If None with dropout_prob > 0, drops any channel.
    glu : int, default=2
        If > 0, applies Gated Linear Units (GLU) every N convolutional layers.
        GLUs gate intermediate representations for more expressivity.
        If 0, no GLU is applied.
    glu_context : int, default=1
        Context window size for GLU gates. If > 0, uses contextual information
        from neighboring time steps for gating. Requires glu > 0.

    References
    ----------
    .. [brainmagik] Défossez, A., Caucheteux, C., Rapin, J., Kabeli, O., & King, J. R.
       (2023). Decoding speech perception from non-invasive brain recordings. Nature
       Machine Intelligence, 5(10), 1097-1107.

    Notes
    -----
    - Input shape: (batch, n_chans, n_times)
    - Output shape: (batch, n_outputs)
    - The model uses dilated convolutions with stride=1 to maintain temporal
      resolution while achieving large receptive fields.
    - Residual connections are applied at every layer where input and output
      channels match.
    - Subject-specific features (subject_dim > 0, subject_layers) require passing
      subject indices in the forward pass as an optional parameter or via batch.
    - STFT processing (n_fft > 0) automatically transforms input to spectrogram domain.

    .. versionadded:: 1.2

    """

    def __init__(
        self,
        # braindecode EEGModuleMixin parameters
        n_chans: int | None = None,
        n_outputs: int | None = None,
        n_times: int | None = None,
        sfreq: float | None = None,
        chs_info: list[dict] | None = None,
        input_window_seconds: float | None = None,
        ########
        # Model related parameters
        # Architecture
        hidden_dim: int = 320,
        depth: int = 4,
        kernel_size: int = 3,
        growth: float = 1.0,
        dilation_growth: int = 2,
        dilation_period: int = 5,
        # Regularization
        conv_drop_prob: float = 0.0,
        dropout_input: float = 0.0,
        batch_norm: bool = True,
        activation: type[nn.Module] = nn.GELU,
        # Subject-specific features (optional)
        n_subjects: int = 200,
        subject_dim: int = 0,
        subject_layers: bool = False,
        subject_layers_dim: str = "input",
        subject_layers_id: bool = False,
        embedding_scale: float = 1.0,
        # STFT/Spectrogram (optional)
        n_fft: int | None = None,
        fft_complex: bool = True,
        # Channel dropout (optional)
        channel_dropout_prob: float = 0.0,
        channel_dropout_type: str | None = None,
        # GLU gates (optional)
        glu: int = 2,
        glu_context: int = 1,
    ):
        # Initialize EEGModuleMixin
        super().__init__(
            n_outputs=n_outputs,
            n_chans=n_chans,
            chs_info=chs_info,
            n_times=n_times,
            input_window_seconds=input_window_seconds,
            sfreq=sfreq,
        )

        # Store parameters for later use
        self.subject_dim = subject_dim
        self.n_subjects = n_subjects
        self.n_fft = n_fft
        self.fft_complex = fft_complex
        self.hidden_dim = hidden_dim

        # Validate inputs
        _validate_brainmodule_params(
            subject_layers=subject_layers,
            subject_dim=subject_dim,
            depth=depth,
            kernel_size=kernel_size,
            growth=growth,
            dilation_growth=dilation_growth,
            channel_dropout_prob=channel_dropout_prob,
            channel_dropout_type=channel_dropout_type,
            glu=glu,
            glu_context=glu_context,
        )

        # Initialize channel dropout (optional)
        self.channel_dropout = None
        if channel_dropout_prob > 0:
            self.channel_dropout = _ChannelDropout(
                dropout_prob=channel_dropout_prob,
                ch_info=chs_info,
                channel_type=channel_dropout_type,
            )

        # Initialize subject-specific modules (optional)
        self.subject_embedding = None
        self.subject_layers_module = None
        input_channels = self.n_chans

        if subject_dim > 0:
            self.subject_embedding = _ScaledEmbedding(
                n_subjects, subject_dim, embedding_scale
            )
            input_channels += subject_dim

        if subject_layers:
            assert subject_dim > 0, "subject_layers requires subject_dim > 0"
            meg_dim = input_channels
            dim = hidden_dim if subject_layers_dim == "hidden" else meg_dim
            self.subject_layers_module = SubjectLayers(
                meg_dim, dim, n_subjects, subject_layers_id
            )
            input_channels = dim

        # Initialize STFT module (optional)
        self.stft = None
        if n_fft is not None:
            self.stft = ta.transforms.Spectrogram(
                n_fft=n_fft,
                hop_length=n_fft // 2,
                normalized=True,
                power=None if fft_complex else 1,
                return_complex=True,
            )
            # Update input channels for spectrogram
            freq_bins = n_fft // 2 + 1
            if fft_complex:
                input_channels *= 2 * freq_bins
            else:
                input_channels *= freq_bins

        # Initial projection layer: project input channels to hidden_dim
        # This is crucial for residual connections to work properly
        self.input_projection = nn.Conv1d(input_channels, hidden_dim, 1)

        # Build channel dimensions for encoder (all same size for residuals)
        # With growth=1.0, all layers have hidden_dim channels (residuals work)
        # With growth!=1.0, channels vary (residuals only where dims match)
        encoder_dims = [hidden_dim] + [
            int(round(hidden_dim * growth**k)) for k in range(depth)
        ]

        # Build encoder (stride=1, no downsampling)
        self.encoder = _ConvSequence(
            channels=encoder_dims,
            kernel_size=kernel_size,
            dilation_growth=dilation_growth,
            dilation_period=dilation_period,
            dropout=conv_drop_prob,
            dropout_input=dropout_input,
            batch_norm=batch_norm,
            glu=glu,
            glu_context=glu_context,
            activation=activation,
        )

        # Final layer: temporal aggregation + output projection
        # Use the last encoder dimension (may differ from hidden_dim if growth != 1)
        final_hidden_dim = encoder_dims[-1]
        self.final_layer = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(start_dim=1),
            nn.Linear(final_hidden_dim, self.n_outputs),
        )

    def forward(
        self, x: torch.Tensor, subject_index: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input EEG data of shape (batch, n_chans, n_times).
        subject_index : torch.Tensor, optional
            Subject indices of shape (batch,). Required if subject_dim > 0.

        Returns
        -------
        torch.Tensor
            Output logits/predictions of shape (batch, n_outputs).
        """
        # Validate input shape
        if x.dim() != 3:
            raise ValueError(
                f"Expected 3D input (batch, channels, time), got shape {x.shape}"
            )
        if x.shape[1] != self.n_chans:
            raise ValueError(f"Expected {self.n_chans} channels, got {x.shape[1]}")

        # Apply STFT if enabled
        if self.stft is not None:
            # Pad for STFT window
            assert self.n_fft is not None, "n_fft must be set if stft is not None"
            pad_size = self.n_fft // 4
            x = F.pad(
                _pad_multiple(x, self.n_fft // 2), (pad_size, pad_size), mode="reflect"
            )

            # Apply STFT
            spec = self.stft(x)  # (batch, channels, freq, time)
            B, C, Fr, T = spec.shape

            if self.fft_complex:
                # Convert complex to real/imag channels
                spec = torch.view_as_real(spec).permute(0, 1, 2, 4, 3)
                x = spec.reshape(B, C * 2 * Fr, T)
            else:
                x = spec.reshape(B, C * Fr, T)

        # Apply channel dropout if enabled
        if self.channel_dropout is not None:
            x = self.channel_dropout(x)

        # Apply subject layers if enabled
        if self.subject_layers_module is not None:
            if subject_index is None:
                raise ValueError(
                    "subject_index is required when subject_layers is enabled"
                )
            x = self.subject_layers_module(x, subject_index)

        # Apply subject embedding if enabled
        if self.subject_embedding is not None:
            if subject_index is None:
                raise ValueError("subject_index is required when subject_dim > 0")
            emb = self.subject_embedding(subject_index)  # (batch, subject_dim)
            emb = emb[:, :, None].expand(
                -1, -1, x.shape[-1]
            )  # (batch, subject_dim, time)
            x = torch.cat([x, emb], dim=1)  # Concatenate along channel dimension

        # Project input to hidden dimension
        x = self.input_projection(x)

        # Encode with residual dilated convolutions
        x = self.encoder(x)

        # Apply final layer (pool + linear)
        x = self.final_layer(x)

        return x


class _ConvSequence(nn.Module):
    """Sequence of residual dilated convolutional layers with GLU activation.

    This is a simplified encoder-only architecture that maintains temporal
    resolution (stride=1) and applies residual connections at every layer
    where input and output channels match.

    Parameters
    ----------
    channels : Sequence[int]
        Channel dimensions for each layer. E.g., [320, 320, 320, 320] for
        a 3-layer network with 320 hidden dims.
    kernel_size : int, default=3
        Convolutional kernel size. Must be odd for proper padding with dilation.
    dilation_growth : int, default=2
        Dilation multiplier per layer. Improves receptive field exponentially.
    dilation_period : int, default=5
        Reset dilation to 1 every N layers.
    dropout : float, default=0.0
        Dropout probability after activation.
    dropout_input : float, default=0.0
        Dropout probability applied to input only.
    batch_norm : bool, default=True
        Whether to apply batch normalization.
    glu : int, default=2
        Apply GLU gating every N layers. If 0, no GLU.
    glu_context : int, default=1
        Context window for GLU convolution.
    activation : type, default=nn.GELU
        Activation function class.
    """

    def __init__(
        self,
        channels: tp.Sequence[int],
        kernel_size: int = 3,
        dilation_growth: int = 2,
        dilation_period: int = 5,
        dropout: float = 0.0,
        dropout_input: float = 0.0,
        batch_norm: bool = True,
        glu: int = 2,
        glu_context: int = 1,
        activation: tp.Any = None,
    ) -> None:
        super().__init__()

        if dilation_growth > 1:
            assert kernel_size % 2 != 0, (
                "Supports only odd kernel with dilation for now"
            )

        if activation is None:
            activation = nn.GELU

        self.sequence = nn.ModuleList()
        self.glus = nn.ModuleList()
        self.skip_projections = nn.ModuleList()  # For when chin != chout

        dilation = 1
        channels = tuple(channels)

        for k, (chin, chout) in enumerate(zip(channels[:-1], channels[1:])):
            layers: tp.List[nn.Module] = []

            # Input dropout (only on first layer)
            if k == 0 and dropout_input > 0:
                layers.append(nn.Dropout(dropout_input))

            # Reset dilation periodically
            if dilation_period and (k % dilation_period) == 0:
                dilation = 1

            # Dilated convolution with proper padding to maintain temporal size
            pad = kernel_size // 2 * dilation
            layers.extend(
                [
                    nn.Conv1d(
                        chin,
                        chout,
                        kernel_size=kernel_size,
                        stride=1,  # Always stride=1 for residual connections
                        padding=pad,
                        dilation=dilation,
                    ),
                ]
            )

            # Batch norm + activation + dropout
            if batch_norm:
                layers.append(nn.BatchNorm1d(num_features=chout))
            layers.append(activation())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

            dilation *= dilation_growth

            self.sequence.append(nn.Sequential(*layers))

            # Add skip projection if channels don't match (for growth != 1.0)
            if chin != chout:
                self.skip_projections.append(nn.Conv1d(chin, chout, 1))
            else:
                self.skip_projections.append(None)

            # GLU gating every N layers
            if glu > 0 and (k + 1) % glu == 0:
                self.glus.append(
                    nn.Sequential(
                        nn.Conv1d(
                            chout, 2 * chout, 1 + 2 * glu_context, padding=glu_context
                        ),
                        nn.GLU(dim=1),
                    )
                )
            else:
                self.glus.append(None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for module, glu, skip_proj in zip(
            self.sequence, self.glus, self.skip_projections
        ):
            # Apply residual connection
            # If channels match, add directly; otherwise use projection
            if skip_proj is not None:
                x = skip_proj(x) + module(x)
            else:
                x = x + module(x)
            if glu is not None:
                x = glu(x)
        return x


def _pad_multiple(x: torch.Tensor, base: int) -> torch.Tensor:
    """Pad tensor to be a multiple of base."""
    length = x.shape[-1]
    target = math.ceil(length / base) * base
    return F.pad(x, (0, target - length))


class _ScaledEmbedding(nn.Module):
    """Scaled embedding layer for subjects.

    Scales up the learning rate for the embedding to prevent slow convergence.
    Used for subject-specific representations.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, scale: float = 10.0):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data /= scale
        self.scale = scale

    @property
    def weight(self) -> torch.Tensor:
        """Get scaled embedding weights."""
        return self.embedding.weight * self.scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Subject indices of shape (batch,).

        Returns
        -------
        torch.Tensor
            Scaled embeddings of shape (batch, embedding_dim).
        """
        return self.embedding(x) * self.scale


class _ChannelDropout(nn.Module):
    """Channel dropout with rescaling and optional ch_info support.

    Randomly drops channels during training and rescales output to maintain
    expected value. Optionally supports selective channel dropout based on
    channel type (EEG, reference, EOG, etc.) using ch_info metadata.

    Parameters
    ----------
    dropout_prob : float, default=0.0
        Probability of dropping each channel (0.0 to 1.0).
        If 0.0, no dropout is applied.
    ch_info : list of dict, optional
        Channel information from MNE (e.g., from raw.info['chs']).
        Each dict should have 'ch_name' and 'ch_type' keys.
        If provided, enables selective channel dropout by type.
    channel_type : str, optional
        If specified with ch_info, only drop channels of this type
        (e.g., 'eeg', 'ref', 'eog'). If None, drop from all available channels.
    rescale : bool, default=True
        If True, rescale output to maintain expected value.
        scale_factor = n_channels / (n_channels - n_dropped)

    Examples
    --------
    >>> # Random channel dropout
    >>> dropout = _ChannelDropout(dropout_prob=0.1)
    >>> x = torch.randn(4, 32, 1000)
    >>> x_dropped = dropout(x)

    >>> # Selective EEG dropout using ch_info
    >>> ch_info = [{'ch_name': 'Fp1', 'ch_type': 'eeg'}, ...]
    >>> dropout_eeg = _ChannelDropout(
    ...     dropout_prob=0.1,
    ...     ch_info=ch_info,
    ...     channel_type='eeg'  # Only drop EEG channels
    ... )
    >>> x_dropped = dropout_eeg(x)  # Reference channels never dropped
    """

    def __init__(
        self,
        dropout_prob: float = 0.0,
        ch_info: list[dict] | None = None,
        channel_type: str | None = None,
        rescale: bool = True,
    ):
        super().__init__()

        if not 0.0 <= dropout_prob <= 1.0:
            raise ValueError(f"dropout_prob must be in [0.0, 1.0], got {dropout_prob}")
        if channel_type is not None and ch_info is None:
            raise ValueError("channel_type requires ch_info to be provided")

        self.dropout_prob = dropout_prob
        self.rescale = rescale
        self.ch_info = ch_info
        self.channel_type = channel_type

        # Compute droppable channel indices
        self.droppable_indices: list[int] | None = None
        if ch_info is not None:
            if channel_type is not None:
                # Drop only specific type
                self.droppable_indices = [
                    i
                    for i, ch in enumerate(ch_info)
                    if ch.get("ch_type", ch.get("kind")) == channel_type
                ]
            else:
                # Drop any channel
                self.droppable_indices = list(range(len(ch_info)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with channel dropout.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape (batch, channels, time).

        Returns
        -------
        torch.Tensor
            Output of same shape as input, with selected channels randomly zeroed.
        """
        if not self.training or self.dropout_prob == 0:
            return x

        _, channels, _ = x.shape

        # Determine which channels to drop
        if self.droppable_indices is not None:
            # Only drop from specified indices
            n_droppable = len(self.droppable_indices)
            n_to_drop = max(1, int(n_droppable * self.dropout_prob))
            if n_to_drop > 0:
                drop_indices = torch.tensor(
                    self.droppable_indices, device=x.device, dtype=torch.long
                )
                # Randomly select which droppable indices to actually drop
                selected = torch.randperm(n_droppable, device=x.device)[:n_to_drop]
                drop_indices = drop_indices[selected]
            else:
                drop_indices = torch.tensor([], device=x.device, dtype=torch.long)
        else:
            # Drop from any channel
            n_to_drop = max(1, int(channels * self.dropout_prob))
            drop_indices = torch.randperm(channels, device=x.device)[:n_to_drop]

        # Clone and apply dropout
        if len(drop_indices) > 0:
            x_out = x.clone()
            x_out[:, drop_indices, :] = 0

            # Rescale to maintain expected value
            if self.rescale:
                scale_factor = channels / (channels - len(drop_indices))
                x_out = x_out * scale_factor
        else:
            x_out = x

        return x_out

    def __repr__(self) -> str:
        return (
            f"ChannelDropout(dropout_prob={self.dropout_prob}, "
            f"rescale={self.rescale}, "
            f"channel_type={self.channel_type})"
        )


def _validate_brainmodule_params(
    subject_layers: bool,
    subject_dim: int,
    depth: int,
    kernel_size: int,
    growth: float,
    dilation_growth: int,
    channel_dropout_prob: float,
    channel_dropout_type: str | None,
    glu: int,
    glu_context: int,
) -> None:
    """Validate BrainModule parameters.

    Parameters
    ----------
    subject_layers : bool
        Whether to use subject-specific layer transformations.
    subject_dim : int
        Dimension of subject embeddings.
    depth : int
        Number of convolutional blocks.
    kernel_size : int
        Convolutional kernel size.
    growth : float
        Channel size multiplier per layer.
    dilation_growth : int
        Dilation multiplier per layer.
    channel_dropout_prob : float
        Channel dropout probability.
    channel_dropout_type : str or None
        Channel type to selectively drop.
    glu : int
        GLU gating interval.
    glu_context : int
        GLU context window size.

    Raises
    ------
    ValueError
        If any parameter combination is invalid.
    """
    validations = [
        (
            subject_layers and subject_dim == 0,
            "subject_layers=True requires subject_dim > 0",
        ),
        (depth < 1, "depth must be >= 1"),
        (kernel_size <= 0, "kernel_size must be > 0"),
        (kernel_size % 2 == 0, "kernel_size must be odd for proper padding"),
        (growth <= 0, "growth must be > 0"),
        (dilation_growth < 1, "dilation_growth must be >= 1"),
        (
            not 0.0 <= channel_dropout_prob <= 1.0,
            "channel_dropout_prob must be in [0.0, 1.0]",
        ),
        (
            channel_dropout_type is not None and channel_dropout_prob == 0.0,
            "channel_dropout_type requires channel_dropout_prob > 0",
        ),
        (glu < 0, "glu must be >= 0"),
        (glu_context < 0, "glu_context must be >= 0"),
        (glu_context > 0 and glu == 0, "glu_context > 0 requires glu > 0"),
        (glu_context >= kernel_size, "glu_context must be < kernel_size"),
    ]

    for condition, message in validations:
        if condition:
            raise ValueError(message)
