import copy

import torch
from einops.layers.torch import Rearrange
from torch import nn

from braindecode.models.base import EEGModuleMixin


class BENDR(EEGModuleMixin, nn.Module):
    """BENDR from Can Han et al (2025) [Han2025]_.

    :bdg-success:`Convolution` :bdg-danger:`Large Brain Model`

    .. figure:: https://www.frontiersin.org/files/Articles/653659/fnhum-15-653659-HTML/image_m/fnhum-15-653659-g001.jpg
        :align: center
        :alt: BENDR Architecture
        :width: 1000px

    The **Spatial-Spectral** and **Temporal - Dual Prototype Network** (SST-DPN)
    is an end-to-end 1D convolutional architecture designed for motor imagery (MI) EEG decoding,
    aiming to address challenges related to discriminative feature extraction and
    small-sample sizes [Han2025]_.

    The framework systematically addresses three key challenges: multi-channel spatial-spectral
    features and long-term temporal features [Han2025]_.

    .. rubric:: Architectural Overview

    SST-DPN consists of a feature extractor (_SSTEncoder, comprising Adaptive Spatial-Spectral
    Fusion and Multi-scale Variance Pooling) followed by Dual Prototype Learning classification [Han2025]_.

    1. **Adaptive Spatial-Spectral Fusion (ASSF)**: Uses :class:`_DepthwiseTemporalConv1d` to generate a
        multi-channel spatial-spectral representation, followed by :class:`_SpatSpectralAttn`
        (Spatial-Spectral Attention) to model relationships and highlight key spatial-spectral
        channels [Han2025]_.

    2. **Multi-scale Variance Pooling (MVP)**: Applies :class:`_MultiScaleVarPooler` with variance pooling
        at multiple temporal scales to capture long-range temporal dependencies, serving as an
        efficient alternative to transformers [Han2025]_.

    3. **Dual Prototype Learning (DPL)**: A training strategy that employs two sets of
        prototypes—Inter-class Separation Prototypes (proto_sep) and Intra-class Compact
        Prototypes (proto_cpt)—to optimize the feature space, enhancing generalization ability and
        preventing overfitting on small datasets [Han2025]_. During inference (forward pass),
        classification decisions are based on the distance (dot product) between the
        feature vector and proto_sep for each class [Han2025]_.

    .. rubric:: Macro Components

    - `SSTDPN.encoder` **(Feature Extractor)**

        - *Operations.* Combines Adaptive Spatial-Spectral Fusion and Multi-scale Variance Pooling
          via an internal :class:`_SSTEncoder`.
        - *Role.* Maps the raw MI-EEG trial :math:`X_i \in \mathbb{R}^{C \times T}` to the
          feature space :math:`z_i \in \mathbb{R}^d`.

    - `_SSTEncoder.temporal_conv` **(Depthwise Temporal Convolution for Spectral Extraction)**

        - *Operations.* Internal :class:`_DepthwiseTemporalConv1d` applying separate temporal
          convolution filters to each channel with kernel size `temporal_conv_kernel_size` and
          depth multiplier `n_spectral_filters_temporal` (equivalent to :math:`F_1` in the paper).
        - *Role.* Extracts multiple distinct spectral bands from each EEG channel independently.

    - `_SSTEncoder.spt_attn` **(Spatial-Spectral Attention for Channel Gating)**

        - *Operations.* Internal :class:`_SpatSpectralAttn` module using Global Context Embedding
          via variance-based pooling, followed by adaptive channel normalization and gating.
        - *Role.* Reweights channels in the spatial-spectral dimension to extract efficient and
          discriminative features by emphasizing task-relevant regions and frequency bands.

    - `_SSTEncoder.chan_conv` **(Pointwise Fusion across Channels)**

        - *Operations.* A 1D pointwise convolution with `n_fused_filters` output channels
          (equivalent to :math:`F_2` in the paper), followed by BatchNorm and the specified
          `activation` function (default: ELU).
        - *Role.* Fuses the weighted spatial-spectral features across all electrodes to produce
          a fused representation :math:`X_{fused} \in \mathbb{R}^{F_2 \times T}`.

    - `_SSTEncoder.mvp` **(Multi-scale Variance Pooling for Temporal Extraction)**

        - *Operations.* Internal :class:`_MultiScaleVarPooler` using :class:`_VariancePool1D`
          layers at multiple scales (`mvp_kernel_sizes`), followed by concatenation.
        - *Role.* Captures long-range temporal features at multiple time scales. The variance
          operation leverages the prior that variance represents EEG spectral power.

    - `SSTDPN.proto_sep` / `SSTDPN.proto_cpt` **(Dual Prototypes)**

        - *Operations.* Learnable vectors optimized during training using prototype learning losses.
          The `proto_sep` (Inter-class Separation Prototype) is constrained via L2 weight-normalization
          (:math:`\lVert s_i \rVert_2 \leq` `proto_sep_maxnorm`) during inference.
        - *Role.* `proto_sep` achieves inter-class separation; `proto_cpt` enhances intra-class compactness.

    .. rubric:: How the information is encoded temporally, spatially, and spectrally

    * **Temporal.**
       The initial :class:`_DepthwiseTemporalConv1d` uses a large kernel (e.g., 75). The MVP module employs pooling
       kernels that are much larger (e.g., 50, 100, 200 samples) to capture long-term temporal
       features effectively. Large kernel pooling layers are shown to be superior to transformer
       modules for this task in EEG decoding according to [Han2025]_.

    * **Spatial.**
       The initial convolution at the classes :class:`_DepthwiseTemporalConv1d` groups parameter :math:`h=1`,
       meaning :math:`F_1` temporal filters are shared across channels. The Spatial-Spectral Attention
       mechanism explicitly models the relationships among these channels in the spatial-spectral
       dimension, allowing for finer-grained spatial feature modeling compared to conventional
       GCNs according to the authors [Han2025]_.
       In other words, all electrode channels share :math:`F_1` temporal filters
       independently to produce the spatial-spectral representation.

    * **Spectral.**
       Spectral information is implicitly extracted via the :math:`F_1` filters in :class:`_DepthwiseTemporalConv1d`.
       Furthermore, the use of Variance Pooling (in MVP) explicitly leverages the neurophysiological
       prior that the **variance of EEG signals represents their spectral power**, which is an
       important feature for distinguishing different MI classes [Han2025]_.

    .. rubric:: Additional Mechanisms

    - **Attention.** A lightweight Spatial-Spectral Attention mechanism models spatial-spectral relationships
        at the channel level, distinct from applying attention to deep feature dimensions,
        which is common in comparison methods like :class:`ATCNet`.
    - **Regularization.** Dual Prototype Learning acts as a regularization technique
        by optimizing the feature space to be compact within classes and separated between
        classes. This enhances model generalization and classification performance, particularly
        useful for limited data typical of MI-EEG tasks, without requiring external transfer
        learning data, according to [Han2025]_.

    Notes
    ----------
    * The implementation of the DPL loss functions (:math:`\mathcal{L}_S`, :math:`\mathcal{L}_C`, :math:`\mathcal{L}_{EF}`)
      and the optimization of ICPs are typically handled outside the primary ``forward`` method, within the training strategy
      (see Ref. 52 in [Han2025]_).
    * The default parameters are configured based on the BCI Competition IV 2a dataset.
    * The use of Prototype Learning (PL) methods is novel in the field of EEG-MI decoding.
    * **Lowest FLOPs:** Achieves the lowest Floating Point Operations (FLOPs) (9.65 M) among competitive
      SOTA methods, including braindecode models like :class:`ATCNet` (29.81 M) and
      :class:`EEGConformer` (63.86 M), demonstrating computational efficiency [Han2025]_.
    * **Transformer Alternative:** Multi-scale Variance Pooling (MVP) provides a accuracy
      improvement over temporal attention transformer modules in ablation studies, offering a more
      efficient alternative to transformer-based approaches like :class:`EEGConformer` [Han2025]_.

    .. warning::

        **Important:** To utilize the full potential of SSTDPN with Dual Prototype Learning (DPL),
        users must implement the DPL optimization strategy outside the model's forward method.
        For implementation details and training strategies, please consult the official code at
        [Han2025Code]_:
        https://github.com/hancan16/SST-DPN/blob/main/train.py


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
        # extra model parameters
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
        self.include_final_layer = final_layer  # Renamed to avoid conflict

        # Encoder: Use parameters from __init__
        self.encoder = _ConvEncoderBENDR(
            in_features=self.n_chans,  # Use n_chans from mixin/init
            encoder_h=encoder_h,
            dropout=drop_prob,
            projection_head=projection_head,  # Pass projection_head parameter
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
            # activation="gelu", # Pass activation name string
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

        # Extract features - take the state corresponding to the *last input timestep*
        # The output has shape [batch_size, features, seq_len+1]
        # The last element context[:,:,-1] corresponds to the *last input time step*
        # (assuming start token is at index 0 after permute in contextualizer)
        # However, often the output corresponding to the *start token* (index 0) is used
        # as the aggregate representation. Let's assume you want the last input timestep's state.
        # Check the transformer's output dimensions carefully based on start_token handling.
        # If output is [batch_size, features, seq_len+1], then last *input* timestep is -1
        feature = context[:, :, -1]
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
    def __init__(
        self,
        in_features,
        hidden_feedforward=3076,
        heads=8,
        layers=8,
        dropout=0.1,  # Default dropout
        activation="gelu",  # Activation for transformer FF layer
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
            self.relative_position = nn.Sequential(conv, nn.GELU())
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
            activation=activation,
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
        """Initialize linear layers and apply TFixup scaling."""
        if isinstance(module, nn.Linear):
            # Standard init
            # module.weight.data.normal_(mean=0.0, std=0.02)
            nn.init.xavier_uniform_(module.weight.data)
            if module.bias is not None:
                module.bias.data.zero_()
            # Tfixup Scaling
            module.weight.data = (
                0.67 * len(self.transformer_layers) ** (-0.25) * module.weight.data
            )
        # You might want to initialize LayerNorm layers as well
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _init_simplified_params(self, module):
        """Initialize linear layers with Xavier and LayerNorms with defaults."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight.data)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, x):
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
        x = x.permute(1, 2, 0)
        # x: [batch_size, transformer_dim, seq_len + 1]

        # Apply output projection (Conv1d expects B, C, T)
        x = self.output_layer(x)
        # x: [batch_size, in_features, seq_len + 1]

        return x
