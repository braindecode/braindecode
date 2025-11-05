import copy

import torch
from einops.layers.torch import Rearrange
from torch import nn

from braindecode.models.base import EEGModuleMixin


class BENDR(EEGModuleMixin, nn.Module):
    """BENDR (BErt-inspired Neural Data Representations) from Kostas et al. (2021) [bendr]_.

    :bdg-success:`Convolution` :bdg-danger:`Large Brain Model`

    .. figure:: https://www.frontiersin.org/files/Articles/653659/fnhum-15-653659-HTML/image_m/fnhum-15-653659-g001.jpg
        :align: center
        :alt: BENDR Architecture
        :width: 1000px


    The **BENDR** architecture adapts techniques used for language modeling (LM) toward the
    development of encephalography modeling (EM) [bendr]_. It utilizes a self-supervised
    training objective to learn compressed representations of raw EEG signals [bendr]_. The
    model is capable of modeling completely novel raw EEG sequences recorded with differing
    hardware and subjects, aiming for transferable performance across a variety of downstream
    BCI and EEG classification tasks [bendr]_.

    .. rubric:: Architectural Overview

    BENDR is adapted from wav2vec 2.0 [wav2vec2]_ and is composed of two main stages: a
    feature extractor (Convolutional stage) that produces BErt-inspired Neural Data
    Representations (BENDR), followed by a transformer encoder (Contextualizer) [bendr]_.

    .. rubric:: Macro Components

    - `BENDR.encoder` **(Convolutional Stage/Feature Extractor)**
        - *Operations.* A stack of six short-receptive field 1D convolutions [bendr]_. Each
          block consists of 1D convolution, GroupNorm, and GELU activation.
        - *Role.* Takes raw data :math:`X_{raw}` and dramatically downsamples it to a new
          sequence of vectors (BENDR) [bendr]_. Each resulting vector has a length of 512.
    - `BENDR.contextualizer` **(Transformer Encoder)**
        - *Operations.* A transformer encoder that uses layered, multi-head self-attention
          [bendr]_. It employs T-Fixup weight initialization [tfixup]_ and uses 8 layers
          and 8 heads.
        - *Role.* Maps the sequence of BENDR vectors to a contextualized sequence. The output
          of a fixed start token is typically used as the aggregate representation for
          downstream classification [bendr]_.
    - `Contextualizer.position_encoder` **(Positional Encoding)**
        - *Operations.* An additive (grouped) convolution layer with a receptive field of 25
          and 16 groups [bendr]_.
        - *Role.* Encodes position information before the input enters the transformer.

    .. rubric:: How the information is encoded temporally, spatially, and spectrally

    * **Temporal.**
      The convolutional encoder uses a stack of blocks where the stride matches the receptive
      field (e.g., 3 for the first block, 2 for subsequent blocks) [bendr]_. This process
      downsamples the raw data by a factor of 96, resulting in an effective sampling frequency
      of approximately 2.67 Hz.
    * **Spatial.**
      To maintain simplicity and reduce complexity, the convolutional stage uses **1D
      convolutions** and elects not to mix EEG channels across the first stage [bendr]_. The
      input includes 20 channels (19 EEG channels and one relative amplitude channel).
    * **Spectral.**
      The convolution operations implicitly extract features from the raw EEG signal [bendr]_.
      The representations (BENDR) are derived from the raw waveform using convolutional
      operations followed by sequence modeling [wav2vec2]_.

    .. rubric:: Additional Mechanisms

    - **Self-Supervision (Pre-training).** Uses a masked sequence learning approach (adapted
      from wav2vec 2.0 [wav2vec2]_) where contiguous spans of BENDR sequences are masked, and
      the model attempts to reconstruct the original underlying encoded vector based on the
      transformer output and a set of negative distractors [bendr]_.
    - **Regularization.** LayerDrop [layerdrop]_ and Dropout (at probabilities 0.01 and 0.15,
      respectively) are used during pre-training [bendr]_. The implementation also uses T-Fixup
      scaling for parameter initialization [tfixup]_.
    - **Input Conditioning.** A fixed token (a vector filled with the value **-5**) is
      prepended to the BENDR sequence before input to the transformer, serving as the aggregate
      representation token [bendr]_.

    Notes
    -----
    * The full BENDR architecture contains a large number of parameters; configuration (1)
      involved training over **one billion parameters** [bendr]_.
    * Randomly initialized full BENDR architecture was generally ineffective at solving
      downstream tasks without prior self-supervised training [bendr]_.
    * The pre-training task (contrastive predictive coding via masking) is generalizable,
      exhibiting strong uniformity of performance across novel subjects, hardware, and
      tasks [bendr]_.

    .. warning::

        **Important:** To utilize the full potential of BENDR, the model requires
        **self-supervised pre-training** on large, unlabeled EEG datasets (like TUEG) followed
        by subsequent fine-tuning on the specific downstream classification task [bendr]_.

    Parameters
    ----------
    encoder_h : int, default=512
        Hidden size (number of output channels) of the convolutional encoder. This determines
        the dimensionality of the BENDR feature vectors produced by the encoder.
    contextualizer_hidden : int, default=3076
        Hidden size of the feedforward layer within each transformer block. The paper uses
        approximately 2x the transformer dimension (3076 ~ 2 x 1536).
    projection_head : bool, default=False
        If True, adds a projection layer at the end of the encoder to project back to the
        input feature size. This is used during self-supervised pre-training but typically
        disabled during fine-tuning.
    drop_prob : float, default=0.1
        Dropout probability applied throughout the model. The paper recommends 0.15 for
        pre-training and 0.0 for fine-tuning. Default is 0.1 as a compromise.
    layer_drop : float, default=0.0
        Probability of dropping entire transformer layers during training (LayerDrop
        regularization [layerdrop]_). The paper uses 0.01 for pre-training and 0.0 for
        fine-tuning.
    activation : :class:`torch.nn.Module`, default=:class:`torch.nn.GELU`
        Activation function used in the encoder convolutional blocks. The paper uses GELU
        activation throughout.
    transformer_layers : int, default=8
        Number of transformer encoder layers in the contextualizer. The paper uses 8 layers.
    transformer_heads : int, default=8
        Number of attention heads in each transformer layer. The paper uses 8 heads with
        head dimension of 192 (1536 / 8).
    position_encoder_length : int, default=25
        Kernel size for the convolutional positional encoding layer. The paper uses a
        receptive field of 25 with 16 groups.
    enc_width : tuple of int, default=(3, 2, 2, 2, 2, 2)
        Kernel sizes for each of the 6 convolutional blocks in the encoder. Each value
        corresponds to one block.
    enc_downsample : tuple of int, default=(3, 2, 2, 2, 2, 2)
        Stride values for each of the 6 convolutional blocks in the encoder. The total
        downsampling factor is the product of all strides (3 x 2 x 2 x 2 x 2 x 2 = 96).
    start_token : int or float, default=-5
        Value used to fill the start token embedding that is prepended to the BENDR sequence
        before input to the transformer. This token's output is used as the aggregate
        representation for classification.
    final_layer : bool, default=True
        If True, includes a final linear classification layer that maps from encoder_h to
        n_outputs. If False, the model outputs the contextualized features directly.

    References
    ----------
    .. [bendr] Kostas, D., Aroca-Ouellette, S., & Rudzicz, F. (2021).
       BENDR: Using transformers and a contrastive self-supervised learning task to learn from
       massive amounts of EEG data.
       Frontiers in Human Neuroscience, 15, 653659.
       https://doi.org/10.3389/fnhum.2021.653659
    .. [wav2vec2] Baevski, A., Zhou, Y., Mohamed, A., & Auli, M. (2020).
       wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations.
       In H. Larochelle, M. Ranzato, R. Hadsell, M. F. Balcan, & H. Lin (Eds),
       Advances in Neural Information Processing Systems (Vol. 33, pp. 12449-12460).
       https://dl.acm.org/doi/10.5555/3495724.3496768
    .. [tfixup] Huang, T. K., Liang, S., Jha, A., & Salakhutdinov, R. (2020).
       Improving Transformer Optimization Through Better Initialization.
       In International Conference on Machine Learning (pp. 4475-4483). PMLR.
       https://dl.acm.org/doi/10.5555/3524938.3525354
    .. [layerdrop] Fan, A., Grave, E., & Joulin, A. (2020).
       Reducing Transformer Depth on Demand with Structured Dropout.
       International Conference on Learning Representations.
       Retrieved from https://openreview.net/forum?id=SylO2yStDr
    """

    def __init__(
        self,
        # Signal related parameters
        n_chans=None,
        n_outputs=None,
        n_times=None,
        chs_info=None,
        input_window_seconds=None,
        sfreq=None,
        # Model parameters
        encoder_h=512,  # Hidden size of the encoder convolutional layers
        contextualizer_hidden=3076,  # Feedforward hidden size in transformer
        projection_head=False,  # Whether encoder should project back to input feature size (unused in original fine-tuning)
        drop_prob=0.1,  # General dropout probability (paper: 0.15 for pretraining, 0.0 for fine-tuning)
        layer_drop=0.0,  # Probability of dropping transformer layers during training (paper: 0.01 for pretraining)
        activation=nn.GELU,  # Activation function
        # Transformer specific parameters
        transformer_layers=8,
        transformer_heads=8,
        position_encoder_length=25,  # Kernel size for positional encoding conv
        enc_width=(3, 2, 2, 2, 2, 2),
        enc_downsample=(3, 2, 2, 2, 2, 2),
        start_token=-5,  # Value for start token embedding
        final_layer=True,  # Whether to include the final linear layer
    ):
        super().__init__(
            n_outputs=n_outputs,
            n_chans=n_chans,
            chs_info=chs_info,
            n_times=n_times,
            input_window_seconds=input_window_seconds,
            sfreq=sfreq,
        )

        # Keep these parameters if needed later, otherwise they are captured by the mixin
        del n_outputs, n_chans, chs_info, n_times, input_window_seconds, sfreq

        self.encoder_h = encoder_h
        self.contextualizer_hidden = contextualizer_hidden
        self.include_final_layer = final_layer

        # Encoder: Use parameters from __init__
        self.encoder = _ConvEncoderBENDR(
            in_features=self.n_chans,
            encoder_h=encoder_h,
            dropout=drop_prob,
            projection_head=projection_head,
            enc_width=enc_width,
            enc_downsample=enc_downsample,
            activation=activation,
        )

        self.contextualizer = _BENDRContextualizer(
            in_features=self.encoder.encoder_h,  # Use the output feature size of the encoder
            hidden_feedforward=contextualizer_hidden,
            heads=transformer_heads,
            layers=transformer_layers,
            dropout=drop_prob,  # Use general dropout probability
            activation=activation,
            position_encoder=position_encoder_length,  # Pass position encoder kernel size
            layer_drop=layer_drop,
            start_token=start_token,  # Keep fixed start token value
        )
        in_features = self.encoder.encoder_h  # Set in_features for final layer

        self.final_layer = None
        if self.include_final_layer:
            # Input to Linear will be [batch_size, encoder_h] after taking last timestep
            linear = nn.Linear(in_features=in_features, out_features=self.n_outputs)
            self.final_layer = nn.utils.parametrizations.weight_norm(
                linear, name="weight", dim=1
            )

    def forward(self, x):
        # Input x: [batch_size, n_chans, n_times]
        encoded = self.encoder(x)
        # encoded: [batch_size, encoder_h, n_encoded_times]

        context = self.contextualizer(encoded)
        # context: [batch_size, encoder_h, n_encoded_times + 1] (due to start token)

        # Extract features - use the output corresponding to the start token (index 0)
        # The output has shape [batch_size, features, seq_len+1]
        # The start token was prepended at position 0 in the contextualizer before the
        # transformer layers. After permutation back to [batch, features, seq_len+1],
        # the start token output is at index 0.
        # According to the paper (Kostas et al. 2021): "The transformer output of this
        # initial position was not modified during pre-training, and only used for
        # downstream tasks." This follows BERT's [CLS] token convention where the
        # first token aggregates sequence information via self-attention.
        feature = context[:, :, 0]
        # feature: [batch_size, encoder_h]

        if self.final_layer is not None:
            feature = self.final_layer(feature)
            # feature: [batch_size, n_outputs]

        return feature


class _ConvEncoderBENDR(nn.Module):
    def __init__(
        self,
        in_features,
        encoder_h=512,
        enc_width=(3, 2, 2, 2, 2, 2),
        dropout=0.0,
        projection_head=False,
        enc_downsample=(3, 2, 2, 2, 2, 2),
        activation=nn.GELU,
    ):
        super().__init__()
        self.encoder_h = encoder_h
        self.in_features = in_features

        if not isinstance(enc_width, (list, tuple)):
            enc_width = [enc_width]
        if not isinstance(enc_downsample, (list, tuple)):
            enc_downsample = [enc_downsample]
        if len(enc_downsample) != len(enc_width):
            raise ValueError(
                "Encoder width and downsampling factors must have the same length."
            )

        # Centerable convolutions make life simpler
        enc_width = [
            e if e % 2 != 0 else e + 1 for e in enc_width
        ]  # Ensure odd kernel size
        self._downsampling = enc_downsample
        self._width = enc_width

        current_in_features = in_features
        self.encoder = nn.Sequential()
        for i, (width, downsample) in enumerate(zip(enc_width, enc_downsample)):
            self.encoder.add_module(
                "Encoder_{}".format(i),
                nn.Sequential(
                    nn.Conv1d(
                        current_in_features,
                        encoder_h,
                        width,
                        stride=downsample,
                        padding=width
                        // 2,  # Correct padding for 'same' output length before stride
                    ),
                    nn.Dropout2d(dropout),  # 2D dropout (matches paper specification)
                    nn.GroupNorm(
                        encoder_h // 2, encoder_h
                    ),  # Consider making num_groups configurable or ensure encoder_h is divisible by 2
                    activation(),
                ),
            )
            current_in_features = encoder_h
        if projection_head:
            self.encoder.add_module(
                "projection_head",
                nn.Conv1d(encoder_h, encoder_h, 1),
            )

    def forward(self, x):
        return self.encoder(x)


class _BENDRContextualizer(nn.Module):
    """Transformer-based contextualizer for BENDR."""

    def __init__(
        self,
        in_features,
        hidden_feedforward=3076,
        heads=8,
        layers=8,
        dropout=0.1,  # Default dropout
        activation=nn.GELU,  # Activation for transformer FF layer
        position_encoder=25,  # Kernel size for conv positional encoding
        layer_drop=0.0,  # Probability of dropping a whole layer
        start_token=-5,  # Value for start token embedding
    ):
        super().__init__()
        self.dropout = dropout
        self.layer_drop = layer_drop
        self.start_token = start_token  # Store start token value

        # Paper specification: transformer_dim = 3 * encoder_h (1536 for encoder_h=512)
        self.transformer_dim = 3 * in_features
        self.in_features = in_features
        # --- Positional Encoding --- (Applied before projection)
        self.position_encoder = None
        if position_encoder > 0:
            conv = nn.Conv1d(
                in_features,
                in_features,
                kernel_size=position_encoder,
                padding=position_encoder // 2,
                groups=16,  # Number of groups for depthwise separation
            )
            # T-Fixup positional encoding initialization (paper specification)
            nn.init.normal_(conv.weight, mean=0, std=2 / self.transformer_dim)
            nn.init.constant_(conv.bias, 0)

            conv = nn.utils.parametrizations.weight_norm(conv, name="weight", dim=2)
            self.relative_position = nn.Sequential(conv, activation())

        # --- Input Conditioning --- (Includes projection up to transformer_dim)
        # Rearrange, Norm, Dropout, Project, Rearrange
        self.input_conditioning = nn.Sequential(
            Rearrange(
                "batch channel time -> batch time channel"
            ),  # Batch, Time, Channels
            nn.LayerNorm(in_features),
            nn.Dropout(dropout),
            nn.Linear(in_features, self.transformer_dim),  # Project up using Linear
            # nn.Conv1d(in_features, self.transformer_dim, 1), # Alternative: Project up using Conv1d
            Rearrange(
                "batch time channel -> time batch channel"
            ),  # Time, Batch, Channels (Transformer expected format)
        )

        # --- Transformer Encoder Layers ---
        # Paper uses T-Fixup: remove internal LayerNorm layers

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.transformer_dim,  # Use projected dimension
            nhead=heads,
            dim_feedforward=hidden_feedforward,
            dropout=dropout,  # Dropout within transformer layer
            activation=activation(),
            batch_first=False,  # Expects (T, B, C)
            norm_first=False,  # Standard post-norm architecture
        )

        # T-Fixup: Replace internal LayerNorms with Identity
        encoder_layer.norm1 = nn.Identity()
        encoder_layer.norm2 = nn.Identity()

        self.transformer_layers = nn.ModuleList(
            [copy.deepcopy(encoder_layer) for _ in range(layers)]
        )

        # Add final LayerNorm after all transformer layers (paper specification)
        self.norm = nn.LayerNorm(self.transformer_dim)

        # --- Output Layer --- (Project back down to in_features)
        # Paper uses Conv1d with kernel=1 (equivalent to Linear but expects (B,C,T) input)
        self.output_layer = nn.Conv1d(self.transformer_dim, in_features, 1)

        # Initialize parameters with T-Fixup (paper specification)
        self.apply(self._init_bert_params)

    def _init_bert_params(self, module):
        """Initialize linear layers and apply T-Fixup scaling."""
        if isinstance(module, nn.Linear):
            # Standard init
            nn.init.xavier_uniform_(module.weight.data)
            if module.bias is not None:
                module.bias.data.zero_()
            # Tfixup Scaling
            module.weight.data = (
                0.67 * len(self.transformer_layers) ** (-0.25) * module.weight.data
            )
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input x: [batch_size, in_features, seq_len]

        # Apply relative positional encoding
        if hasattr(self, "relative_position"):
            pos_enc = self.relative_position(x)
            x = x + pos_enc

        # Apply input conditioning (includes projection up and rearrange)
        x = self.input_conditioning(x)
        # x: [seq_len, batch_size, transformer_dim]

        # Prepend start token
        if self.start_token is not None:
            token_emb = torch.full(
                (1, x.shape[1], x.shape[2]),
                float(self.start_token),
                device=x.device,
                requires_grad=False,
            )
            x = torch.cat([token_emb, x], dim=0)
        # x: [seq_len + 1, batch_size, transformer_dim]

        # Apply transformer layers with layer drop
        for layer in self.transformer_layers:
            if not self.training or torch.rand(1) > self.layer_drop:
                x = layer(x)
        # x: [seq_len + 1, batch_size, transformer_dim]

        # Apply final LayerNorm (T-Fixup: norm only at the end)
        x = self.norm(x)
        # x: [seq_len + 1, batch_size, transformer_dim]

        # Permute to (B, C, T) format for Conv1d output layer
        x = Rearrange("time batch channel -> batch channel time")(x)
        # x: [batch_size, transformer_dim, seq_len + 1]

        # Apply output projection (Conv1d expects B, C, T)
        x = self.output_layer(x)
        # x: [batch_size, in_features, seq_len + 1]

        return x
