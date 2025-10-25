# Authors: Meta Platforms, Inc. and affiliates (original)
#          Bruno Aristimunha <b.aristimunha@gmail.com> (Braindecode adaptation)
#
# License: Attribution-NonCommercial 4.0 International

"""Dilated Convolutional Decoder model adapted for Braindecode."""

from __future__ import annotations

import math
import typing as tp
from functools import partial

import torch
import torchaudio as ta
from torch import nn
from torch.nn import functional as F

from braindecode.models.base import EEGModuleMixin
from braindecode.modules.attention import LocalSelfAttention
from braindecode.modules.layers import SubjectLayers

__all__ = ["DilatedConvDecoder"]


class DilatedConvDecoder(EEGModuleMixin, nn.Module):
    r"""Dilated Convolutional Decoder (SimpleConv) as the Brain Module from [brainmagik]_.

    :bdg-secondary:`Recurrent` :bdg-info:`Small Attention` :bdg-success:`Convolution`

    .. figure:: ../_static/model/simpleconv.png
        :align: center
        :alt: SimpleConv Architecture
        :width: 1000px

        For each layer, we note first the number of output channels, while the number of time steps is constant throughout the layers.
        The model is composed of a spatial attention layer, then a 1x1 convolution without activation.
        A 'Subject Layer' is selected based on the subject index s, which consists in a 1x1 convolution learnt only for that subject with no activation.
        Then, we apply five convolutional blocks made of three convolutions.
        The first two use residual skip connection and increasing dilation, followed by a BatchNorm layer and a GELU activation.
        The third convolution is not residual, and uses a GLU activation (which halves the number of channels) and no normalization.
        Finally, we apply two 1x1 convolutions with a GELU in between.

    The **Dilated Convolutional Decoder**, sometimes referred to as SimpleConv [brainmagik]_,
    functions as the **Brain Module** (or deep convolutional network) described in the
    original paper by Défossez et al. (2023) for decoding speech perception from
    non-invasive MEG/EEG recordings.

    The model is a flexible encoder-decoder architecture adapted for Braindecode's
    EEG workflows, originally designed for MEG decoding [brainmagik]_. Its primary role is to
    map raw brain activity :math:`X \in \mathbb{R}^{C \times T}` into a latent
    representation :math:`Z \in \mathbb{R}^{F \times T}` that is maximally aligned
    with deep speech representations (e.g., wav2vec 2.0 features) using a contrastive
    learning objective (CLIP loss).

    The architecture leverages **dilated convolutions** to achieve large receptive
    fields while maintaining temporal resolution [brainmagik]_.

    Architectural Overview
    ----------------------

    The Brain Module processes input through a series of stages: input conditioning,
    encoding via dilated convolution blocks, optional temporal processing (LSTM and
    attention), and a symmetrical decoder.

    1. **Input Conditioning:** Handles Spectrogram transformation (optional STFT/n_fft),
       Channel Dropout, Subject-Specific Embeddings, and initial 1x1 convolutions.
    2. **Encoder:** A series of dilated convolutional blocks that downsample the signal.
    3. **Temporal Modeling:** Optional LSTM and local self-attention layers for sequence processing.
    4. **Decoder:** Convolutional blocks that upsample the representation back to the original time dimension.
    5. **Final Layer:** Optional output convolution and pooling for dimensionality matching or classification.

    Macro Components
    ----------------

    - `DilatedConvDecoder.initial_layer` **(Initial Linear/1x1 Conditioning)**
        - *Original Paper Role.* Used for feature conditioning and domain adaptation, consisting of one or more 1x1 convolutions applied before the main encoder.
        - *Presence.* Enabled if `initial_linear > 0`.
    - `DilatedConvDecoder.subject_layers_module` **(Participant Layer)**
        - *Original Paper Role.* Learns a unique linear transformation matrix :math:`M_s \in \mathbb{R}^{D_1, D_1}` for each participant :math:`s`, applied along the channel dimension after the conceptual Spatial Attention layer.
          This is designed to leverage inter-individual variability.
        - *Presence.* Enabled if `subject_layers=True` and `subject_dim > 0`.
    - `DilatedConvDecoder.encoder` **(Dilated Convolutional Encoding)**
        - *Original Paper Role.* Maps the conditioned input to a compressed latent space. The paper describes this stack as **five blocks of three convolutional layers**.
        - *Presence.* Implemented via the internal `_ConvSequence` module.
    - `DilatedConvDecoder.lstm` **(Optional Temporal Modeling)**
        - *Original Paper Role.* Provides temporal sequence modeling capability]. The implementation supports bidirectional processing and sequence flipping.
        - *Presence.* Enabled if `lstm_layers > 0`.
    - `DilatedConvDecoder.attentions` **(Optional Local Self-Attention)**
        - *Original Paper Role.* Attention layers added after the LSTM for further temporal refinement.
        - *Presence.* Enabled if `attention_layers > 0`, instantiated as :class:`braindecode.modules.attention.LocalSelfAttention` with windowed self-attention that preserves sequence length via residual connections.
    - `DilatedConvDecoder.decoder` **(Dilated Convolutional Decoding)**
        - *Original Paper Role.* Symmetrical reconstruction layers that map the temporal representations back to the original time dimension.
        - *Presence.* Implemented via the internal `_ConvSequence` with `decode=True`.
    - `DilatedConvDecoder.final_layer` **(Output Convolution/Pooling)**
        - *Original Paper Role.* Applies a final 1x1 convolution (often followed by activation and another 1x1 convolution if `complex_out=True`) to achieve the final output dimensionality :math:`F` (matching the speech representations).
          It also includes global average pooling if required for classification tasks.
        - *Presence.* Always present, though its components vary based on `linear_out` and `complex_out`.

    Key Architectural Mechanisms
    ----------------------------

    * **Dilated Convolutions and Receptive Field.**
        The encoder blocks utilize dilated convolutions (in the first two layers of each block) where the dilation rate increases exponentially per layer (e.g., :math:`2^{2k \mod 5}` and :math:`2^{2k+1 \mod 5}` for block :math:`k`).
        This mechanism allows the model to capture context across a **large receptive field** without drastically reducing the time dimension.
    * **Residual Skip Connections.**
        Skip (residual) connections are applied across the convolutional layers in the encoder and decoder blocks to stabilize training and improve gradient flow, particularly in deep networks. These can optionally use :class:`~braindecode.models.dilated_conv_decoder._LayerScale`.
    * **Gated Activation.** The convolutional layers rely primarily on GELU activation.
        Specifically, the third layer in each block uses a **Gated Linear Unit (GLU)** activation (implemented with :class:`torch.nn.GLU`), which gates intermediate representations for enhanced expressivity.
    * **Input Transformation (STFT).**
        The module optionally includes a Spectrogram transformation (if `n_fft` is set) before encoding, allowing the model to operate in the **spectrogram domain** rather than the raw time domain.
    * **Local Self-Attention Refinement.**
        Each attention stage wraps the encoded sequence in :class:`braindecode.modules.attention.LocalSelfAttention`, applying sliding-window self-attention with residual connections so temporal saliency can be adjusted without destabilising the convolutional backbone.
    * **Regularization & Robustness Tooling.**
        Dropout knobs (input, per-conv, LSTM), :class:`~braindecode.models.dilated_conv_decoder._ChannelDropout`, optional LayerScale, GLU gating, and subject-aware adapters (embeddings + :class:`braindecode.modules.layers.SubjectLayers`) provide avenues to combat overfitting and deal with inter-participant variability.
    * **Temporal Alignment.**
        To compensate for the expected delay between acoustic stimulus and neural response, the original paper's architecture shifts the input brain signal by **150 ms into the future** to facilitate alignment between brain representation :math:`Z` and speech representation :math:`Y`.

    How the information is encoded temporally, spatially, and spectrally
    --------------------------------------------------------------------

    * **Temporal.**
        The temporal axis is the primary processing dimension. The encoder :class:`~braindecode.models.dilated_conv_decoder._ConvSequence`
        applies stacks of dilated 1D convolutions whose dilation grows geometrically
        (`dilation_growth`) so each successive block aggregates progressively longer
        context without collapsing the sample rate. Residual shortcuts (optionally scaled
        via LayerScale) stabilise these wide receptive fields. When enabled, the :class:`~braindecode.models.dilated_conv_decoder._LSTM`
        layer (optionally flipped or bidirectional) and the :class:`braindecode.modules.attention.LocalSelfAttention` blocks
        refine sequential structure on the `(time, feature)` axis before the decoder
        mirrors the stride pattern with transposed convolutions. Padding/cropping through
        :meth:`~braindecode.models.dilated_conv_decoder.DilatedConvDecoder.pad_to_valid_length` and the final adaptive pooling ensure that temporal
        resolution matches the requested output while still leveraging the enlarged
        receptive field.

    * **Spatial.**
        Spatial structure (sensors/electrodes) is shaped ahead of the temporal stack.
        :class:`~braindecode.models.dilated_conv_decoder._ChannelDropout` can drop entire electrodes to promote robustness to missing
        channels. When `subject_layers` is active, :class:`braindecode.modules.layers.SubjectLayers` injects per-subject
        1x1 projections across the channel axis and :class:`~braindecode.models.dilated_conv_decoder._ScaledEmbedding` concatenates learned
        subject embeddings as extra "virtual" channels. The optional initial 1x1 block
        (and the ungrouped first convolution in :class:`~braindecode.models.dilated_conv_decoder._ConvSequence`) mixes all channels,
        letting every temporal filter access the full montage, while later grouped
        convolutions or residual skips preserve locality if desired.

    * **Spectral.**
        Spectral cues can be made explicit by enabling the STFT branch (`n_fft`), which
        converts the waveform into torchaudio spectrogram bins (real/imag pairs when
        `fft_complex=True`) that are flattened into the channel dimension. Otherwise,
        spectral structure is learned implicitly: long temporal kernels act as trainable
        band-pass filters, the dilated stack captures multi-scale rhythms, and GLU-gated
        layers modulate harmonic content. The decoder recombines these multi-scale
        features before the final readout, allowing the network to represent both narrow-
        band oscillations and broadband transients relevant for downstream decoding.

    .. rubric:: Additional Mechanisms

    - **Attention.**
        When `attention_layers > 0`, each stage adds a
        :class:`braindecode.modules.attention.LocalSelfAttention` block after the
        encoder (or :class:`~braindecode.models.dilated_conv_decoder._LSTM` output if
        present). These modules form sliding-window self-attention over the time axis,
        allowing the network to reweight features using local contextual cues while
        keeping computational cost linear in sequence length. The attention output is
        merged residually with the incoming representation so it acts as an adaptive,
        low-cost refinement on top of the dilated convolutions.

    - **Regularization.**
        Several mechanisms curb overfitting and improve robustness: per-layer dropout
        (`conv_drop_prob`) and input dropout (`dropout_input`) use
        :class:`torch.nn.Dropout`, recurrent dropout (`lstm_drop_prob`) regularises the
        LSTM stack, and :class:`~braindecode.models.dilated_conv_decoder._ChannelDropout`
        randomly drops full sensor channels. Optional :class:`~braindecode.models.dilated_conv_decoder._LayerScale`
        gently rescales residual branches, while GLU gating (`glu`, `glu_context`, :class:`torch.nn.GLU`) and subject-specific parameter
        sharing (via :class:`~braindecode.modules.layers.SubjectLayers` and
        :class:`~braindecode.models.dilated_conv_decoder._ScaledEmbedding`) encourage
        sparse, participant-aware representations that generalise across recordings.

    Notes
    -----
    * **Input/Output Shape:** Input is typically (batch, n_chans, n_times); output depends on configuration but is often (batch, n_outputs, n_times) or (batch, n_outputs) if pooling is applied.
    * **Subject-Specific Features:** Subject embeddings (`subject_dim > 0`) or subject layers (`subject_layers=True`) require passing subject indices in the forward pass.
    * **Temporal Padding:** Padding logic is utilized (`pad_to_valid_length`) to ensure temporal dimensions are maintained correctly through the encoder-decoder roundtrip.
    * **Original Context:** The brain module was trained using a **contrastive CLIP loss**  to maximize discrimination between segments, showing improved performance over regression losses targeting low-level features like Mel spectrograms.

    Parameters
    ----------
    hidden_dim : int, default=64
        Base hidden dimension for convolutional layers. Actual layer sizes
        grow or shrink according to the `growth` parameter.
    depth : int, default=2
        Number of encoder/decoder layer pairs.
    kernel_size : int, default=5
        Convolutional kernel size. Must be odd (e.g., 3, 5, 7) for dilation to work properly.
        Default changed from 4 to 5 to support dilation_growth > 1.
    stride : int, default=2
        Stride for downsampling in encoder, upsampling in decoder.
    growth : float, default=1.0
        Channel size multiplier: hidden_dim * (growth ** layer_index).
        Values > 1.0 grow channels deeper; < 1.0 shrink them.
    dilation_growth : int, default=1
        Dilation multiplier per layer (e.g., 2 means dilation doubles each layer).
        Improves receptive field. Note: requires odd kernel if > 1.
    dilation_period : int, optional
        Reset dilation to 1 every N layers. Useful to prevent dilation explosion.
    lstm_layers : int, default=0
        Number of LSTM layers. If 0, no LSTM is used.
    lstm_dropout : float, default=0.0
        Dropout probability for LSTM layers.
    flip_lstm : bool, default=False
        If True, LSTM processes the sequence backward then flips output.
    bidirectional_lstm : bool, default=False
        If True, LSTM is bidirectional. Reduces hidden size to keep output dim constant.
    attention_layers : int, default=0
        Number of attention layers to add after LSTM. If 0, no attention.
    attention_heads : int, default=4
        Number of attention heads (must divide hidden_dim evenly).
    conv_dropout : float, default=0.0
        Dropout probability for convolutional layers.
    dropout_input : float, default=0.0
        Dropout probability applied to model input only.
    batch_norm : bool, default=False
        If True, apply batch normalization after each convolution.
    relu_leakiness : float, default=0.0
        Leakiness of LeakyReLU (0.0 = standard ReLU, >0 = leaky).
    gelu : bool, default=False
        If True, use GELU activation instead of ReLU/LeakyReLU.
        Only applies when relu_leakiness is 0.0.
    linear_out : bool, default=False
        If True, apply a final 1x1 convolution to produce outputs.
        Otherwise, the decoder directly outputs n_outputs channels.
    complex_out : bool, default=False
        If True and linear_out=True, apply a non-linearity between
        intermediate and output convolutions.
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
    initial_linear : int, default=0
        Number of output channels for initial linear (1x1 conv) preprocessing layer.
        If 0, no initial layer is applied. If > 0, applies one or more 1x1 convolutions
        before the main encoder for feature conditioning and domain adaptation.
    initial_depth : int, default=1
        Number of 1x1 convolutional layers to apply before the encoder.
        Only used if initial_linear > 0.
    initial_nonlin : bool, default=False
        If True, apply a non-linearity after the final initial layer.
        Only used if initial_linear > 0.
    layer_scale : float, optional
        If not None, applies LayerScale to residual skip connections in encoder/decoder.
        Helps stabilize training in deep networks with skip connections.
        Typical value: 0.1. Higher values mean more aggressive scaling.
    skip : bool, default=False
        If True, adds skip (residual) connections in encoder and decoder blocks.
        Skip connections help with gradient flow and training stability.
    post_skip : bool, default=False
        If True, applies skip connections after activation instead of before.
        Only used if skip=True. Changes residual connection pattern.
    channel_dropout_prob : float, default=0.0
        Probability of dropping each channel during training (0.0 to 1.0).
        If 0.0, no channel dropout is applied.
    channel_dropout_type : str, optional
        If specified with chs_info, only drop channels of this type
        (e.g., 'eeg', 'ref', 'eog'). If None with dropout_prob > 0, drops any channel.
    glu : int, default=0
        If > 0, applies Gated Linear Units (GLU) every N encoder/decoder layers
        where N = glu. GLUs gate intermediate representations for more expressivity.
        If 0, no GLU is applied.
    glu_context : int, default=0
        Context window size for GLU gates. If > 0, uses contextual information
        from neighboring time steps for gating. Requires glu > 0.
    glu_glu : bool, default=True
        If True with glu > 0, uses nn.GLU for gating (doubles channels internally).
        If False, uses standard activation for gating (no channel doubling).

    References
    ----------
    .. [brainmagik] Défossez, A., Caucheteux, C., Rapin, J., Kabeli, O., & King, J. R.
       (2023). Decoding speech perception from non-invasive brain recordings. Nature
       Machine Intelligence, 5(10), 1097-1107.

    Notes
    -----
    - Input shape: (batch, n_chans, n_times)
    - Output shape: (batch, n_outputs, n_times) [if decoder is used] or
                    (batch, n_outputs) [if linear_out or final aggregation is applied]
    - The model uses dilated convolutions to achieve large receptive fields
      without losing temporal resolution excessively.
    - Padding is computed to maintain temporal dimensions through layers when possible.
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
        # Architecture
        hidden_dim: int = 64,
        depth: int = 2,
        kernel_size: int = 5,
        stride: int = 2,
        growth: float = 1.0,
        dilation_growth: int = 1,
        dilation_period: int | None = None,
        # LSTM (optional)
        lstm_layers: int = 0,
        lstm_drop_prob: float = 0.0,
        flip_lstm: bool = False,
        bidirectional_lstm: bool = False,
        # Attention
        attention_layers: int = 0,
        attention_heads: int = 4,
        # Regularization
        conv_drop_prob: float = 0.0,
        dropout_input: float = 0.0,
        batch_norm: bool = False,
        relu_leakiness: float = 0.0,
        gelu: bool = False,
        # final layer
        linear_out: bool = False,
        complex_out: bool = False,
        final_activation: type[nn.Module] = nn.ReLU,
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
        # Initial linear layers (optional)
        initial_linear: int = 0,
        initial_depth: int = 1,
        initial_nonlin: bool = False,
        # Skip connection features (optional)
        layer_scale: float | None = None,
        skip: bool = False,
        post_skip: bool = False,
        # Channel dropout (optional)
        channel_dropout_prob: float = 0.0,
        channel_dropout_type: str | None = None,
        # GLU gates (optional)
        glu: int = 0,
        glu_context: int = 0,
        glu_glu: bool = True,
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

        # Validate inputs
        if subject_layers and subject_dim == 0:
            raise ValueError("subject_layers=True requires subject_dim > 0")
        if complex_out and not linear_out:
            raise ValueError(
                "complex_out=True requires linear_out=True; "
                "otherwise the decoder directly produces outputs."
            )
        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth}")
        if kernel_size <= 0:
            raise ValueError(f"kernel_size must be > 0, got {kernel_size}")
        if stride < 1:
            raise ValueError(f"stride must be >= 1, got {stride}")
        if growth <= 0:
            raise ValueError(f"growth must be > 0, got {growth}")
        if dilation_growth < 1:
            raise ValueError(f"dilation_growth must be >= 1, got {dilation_growth}")
        if dilation_growth > 1 and kernel_size % 2 == 0:
            raise ValueError(
                "dilation_growth > 1 requires odd kernel_size "
                f"(got kernel_size={kernel_size})"
            )
        if attention_heads < 1:
            raise ValueError(f"attention_heads must be >= 1, got {attention_heads}")
        if lstm_layers < 0:
            raise ValueError(f"lstm_layers must be >= 0, got {lstm_layers}")
        if attention_layers < 0:
            raise ValueError(f"attention_layers must be >= 0, got {attention_layers}")
        if initial_linear < 0:
            raise ValueError(f"initial_linear must be >= 0, got {initial_linear}")
        if initial_depth < 1:
            raise ValueError(f"initial_depth must be >= 1, got {initial_depth}")
        if initial_depth > 1 and initial_linear == 0:
            raise ValueError("initial_depth > 1 requires initial_linear > 0")
        if layer_scale is not None and layer_scale <= 0:
            raise ValueError(f"layer_scale must be > 0, got {layer_scale}")
        if post_skip and not skip:
            raise ValueError("post_skip=True requires skip=True")
        if not 0.0 <= channel_dropout_prob <= 1.0:
            raise ValueError(
                f"channel_dropout_prob must be in [0.0, 1.0], "
                f"got {channel_dropout_prob}"
            )
        if channel_dropout_type is not None and channel_dropout_prob == 0.0:
            raise ValueError("channel_dropout_type requires channel_dropout_prob > 0")
        if glu < 0:
            raise ValueError(f"glu must be >= 0, got {glu}")
        if glu_context < 0:
            raise ValueError(f"glu_context must be >= 0, got {glu_context}")
        if glu_context > 0 and glu == 0:
            raise ValueError("glu_context > 0 requires glu > 0")
        if glu_context >= kernel_size:
            raise ValueError(
                f"glu_context must be < kernel_size "
                f"(got {glu_context} >= {kernel_size})"
            )

        self.depth = depth
        self.kernel_size = kernel_size
        self.stride = stride
        self.hidden_dim = hidden_dim
        self.growth = growth
        self.dilation_growth = dilation_growth
        self.dilation_period = dilation_period
        self.lstm_layers = lstm_layers
        self.attention_layers = attention_layers
        self.skip = skip
        self.post_skip = post_skip
        self.layer_scale = layer_scale

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

        # Initialize initial linear layers (optional)
        self.initial_layer = None
        if initial_linear > 0:
            initial_modules: list[nn.Module] = []
            for k in range(initial_depth):
                in_chans = input_channels if k == 0 else initial_linear
                initial_modules.append(
                    nn.Conv1d(in_chans, initial_linear, 1, bias=True)
                )
                if k < initial_depth - 1 or initial_nonlin:
                    initial_modules.append(nn.LeakyReLU(relu_leakiness))
                    if batch_norm:
                        initial_modules.append(nn.BatchNorm1d(initial_linear))
                    if conv_drop_prob > 0:
                        initial_modules.append(nn.Dropout(conv_drop_prob))
            self.initial_layer = nn.Sequential(*initial_modules)
            input_channels = initial_linear

        # Build channel dimension sequences for encoder and decoder
        # Use self.n_chans property to infer from chs_info if needed
        encoder_dims = [input_channels] + [
            int(round(hidden_dim * growth**k)) for k in range(depth)
        ]
        decoder_input_dim = encoder_dims[-1]

        # LSTM may expand dimension
        lstm_hidden = decoder_input_dim
        if lstm_layers > 0:
            lstm_hidden = decoder_input_dim

        # Configure activation function based on gelu parameter
        if gelu:
            activation_fn = nn.GELU
        elif relu_leakiness:
            activation_fn = partial(nn.LeakyReLU, relu_leakiness)
        else:
            activation_fn = nn.ReLU

        # Build encoder
        encoder_params: dict[str, tp.Any] = dict(
            kernel=kernel_size,
            stride=stride,
            leakiness=relu_leakiness,
            dropout=conv_drop_prob,
            dropout_input=dropout_input,
            batch_norm=batch_norm,
            dilation_growth=dilation_growth,
            dilation_period=dilation_period,
            skip=skip,
            scale=layer_scale,
            post_skip=post_skip,
            glu=glu,
            glu_context=glu_context,
            glu_glu=glu_glu,
            activation=activation_fn,
        )

        self.encoder = _ConvSequence(encoder_dims, **encoder_params)

        # Build LSTM (optional)
        self.lstm = None
        if lstm_layers > 0:
            self.lstm = _LSTM(
                input_size=decoder_input_dim,
                hidden_size=lstm_hidden,
                dropout=lstm_drop_prob,
                num_layers=lstm_layers,
                bidirectional=bidirectional_lstm,
            )
            self._flip_lstm = flip_lstm

        # Build attention layers (optional)
        self.attentions = nn.ModuleList()
        for _ in range(attention_layers):
            self.attentions.append(
                LocalSelfAttention(lstm_hidden, heads=attention_heads)
            )

        # Build decoder
        decoder_dims = [int(round(lstm_hidden / growth**k)) for k in range(depth + 1)]

        # Decoder outputs directly to n_outputs (or to intermediate dims if linear_out)
        if not linear_out:
            encoder_params["activation_on_last"] = False
            decoder_dims[-1] = self.n_outputs

        self.decoder = _ConvSequence(decoder_dims, decode=True, **encoder_params)

        # Final layer: combine output conv, pooling, and squeezing for classification
        # This is a named module for compatibility with test_model_integration_full_last_layer
        final_modules = nn.ModuleDict()

        if linear_out:
            if complex_out:
                final_modules["final_conv"] = nn.Sequential(
                    nn.Conv1d(decoder_dims[-1], 2 * decoder_dims[-1], 1),
                    final_activation(),
                    nn.Conv1d(2 * decoder_dims[-1], self.n_outputs, 1),
                )
            else:
                final_modules["final_conv"] = nn.Conv1d(
                    decoder_dims[-1], self.n_outputs, 1
                )

        # Global pooling layer for classification tasks
        final_modules["pool"] = nn.AdaptiveAvgPool1d(1)
        self.final_layer = nn.Sequential(*final_modules.values())

    def compute_valid_length(self, length: int) -> int:
        """
        Compute valid output length after encoder-decoder roundtrip.

        Returns the nearest valid length such that convolution operations
        do not leave partial time steps. If input length is already valid,
        output will have the same length.

        Parameters
        ----------
        length : int
            Input time dimension.

        Returns
        -------
        int
            Valid length after encoder-decoder.
        """
        # Encoder: downsample by stride per layer
        for _ in range(self.depth):
            length = math.ceil(length / self.stride) + 1
            length = max(length, 1)
        # Decoder: upsample by stride per layer
        for _ in range(self.depth):
            length = (length - 1) * self.stride
        return int(length)

    def pad_to_valid_length(self, x: torch.Tensor) -> tuple[torch.Tensor, int]:
        """
        Pad input to valid length.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, channels, time).

        Returns
        -------
        padded_x : torch.Tensor
            Padded tensor.
        original_length : int
            Original time dimension (for later cropping).
        """
        original_length = x.shape[-1]
        valid_length = self.compute_valid_length(original_length)
        delta = valid_length - original_length
        if delta > 0:
            x = F.pad(x, (0, delta))
        return x, original_length

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
            Output logits/predictions. Shape depends on model configuration:
            - If decoder without final layer: (batch, n_outputs, n_times)
            - If linear_out=True: (batch, n_outputs, n_times)
            Output is cropped to original input time dimension.
        """
        # Validate input shape
        if x.dim() != 3:
            raise ValueError(
                f"Expected 3D input (batch, channels, time), got shape {x.shape}"
            )
        if x.shape[1] != self.n_chans:
            raise ValueError(f"Expected {self.n_chans} channels, got {x.shape[1]}")

        original_length = x.shape[-1]

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

        # Apply initial linear layers if enabled
        if self.initial_layer is not None:
            x = self.initial_layer(x)

        # Pad to valid length for convolutions
        x, _ = self.pad_to_valid_length(x)

        # Encode
        encoded = self.encoder(x)

        # Optional LSTM
        if self.lstm is not None:
            # LSTM expects (time, batch, features)
            encoded = encoded.permute(2, 0, 1)
            if self._flip_lstm:
                encoded = encoded.flip([0])
            encoded, _ = self.lstm(encoded)
            if self._flip_lstm:
                encoded = encoded.flip([0])
            encoded = encoded.permute(1, 2, 0)

        # Optional attention
        for attention in self.attentions:
            encoded = encoded + attention(encoded)

        # Decode
        decoded = self.decoder(encoded)

        # Apply final layer (output conv + pooling)
        decoded = self.final_layer(decoded).squeeze(-1)

        # Crop to original length
        if decoded.shape[-1] > original_length:
            decoded = decoded[..., :original_length]

        return decoded


class _ConvSequence(nn.Module):
    """Sequence of convolutional layers with optional dilation, skip connections, GLU."""

    def __init__(
        self,
        channels: tp.Sequence[int],
        kernel: int = 4,
        dilation_growth: int = 1,
        dilation_period: int | None = None,
        stride: int = 2,
        dropout: float = 0.0,
        leakiness: float = 0.0,
        groups: int = 1,
        decode: bool = False,
        batch_norm: bool = False,
        dropout_input: float = 0.0,
        skip: bool = False,
        scale: float | None = None,
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
            if chin == chout and skip:
                if scale is not None:
                    layers.append(_LayerScale(chout, scale))
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


class _LayerScale(nn.Module):
    """Layer scale: rescales residual outputs close to 0, learnable.

    From [Touvron et al 2021] (https://arxiv.org/pdf/2103.17239.pdf).
    Helps with training stability in deep networks with residual paths.
    """

    def __init__(self, channels: int, init: float = 0.1, boost: float = 5.0):
        super().__init__()
        self.scale = nn.Parameter(torch.zeros(channels, requires_grad=True))
        self.scale.data[:] = init / boost
        self.boost = boost

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (self.boost * self.scale[:, None]) * x


class _LSTM(nn.Module):
    """Wrapper for LSTM that normalizes output dimension for bidirectional mode.

    If bidirectional, LSTM output is linearly projected to hidden_size so
    downstream layers see consistent dimensions.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
        )
        self.linear = None
        if bidirectional:
            self.linear = nn.Linear(2 * hidden_size, hidden_size)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape (time, batch, input_size).

        Returns
        -------
        output : torch.Tensor
            Output of shape (time, batch, hidden_size).
        (h, c) : tuple
            LSTM hidden state and cell state.
        """
        x, (h, c) = self.lstm(x)
        if self.linear is not None:
            x = self.linear(x)
        return x, (h, c)


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
