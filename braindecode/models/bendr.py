import copy

import torch
from einops.layers.torch import Rearrange
from torch import nn

from braindecode.models.base import EEGModuleMixin


class BENDR(EEGModuleMixin, nn.Module):
    """ """

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
        drop_prob=0.1,  # General dropout probability
        layer_drop=0.0,  # Probability of dropping transformer layers during training
        activation=nn.GELU,  # Activation function
        # Transformer specific parameters (add defaults matching original if needed)
        transformer_layers=8,
        transformer_heads=8,
        position_encoder_length=25,  # Kernel size for positional encoding conv
        enc_width=(3, 2, 2, 2, 2, 2),
        enc_downsample=(3, 2, 2, 2, 2, 2),
        # extra model parameters
        start_token=-5,  # Value for start token embedding
        final_layer=True,  # Whether to include the final linear layer
    ):
        # Initialize EEGModuleMixin first if it provides n_chans, n_outputs etc.
        # Ensure required parameters like n_chans, n_outputs are set before use
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
            # Input to LazyLinear will be [batch_size, encoder_h] after taking last timestep
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
                    nn.Dropout1d(dropout),  # 1D dropout for 1D conv
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

        # Let's follow the original's projection:
        self.transformer_dim = in_features
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
            # Initialize weights first
            nn.init.normal_(conv.weight, mean=0, std=0.02)  # Basic init
            nn.init.constant_(conv.bias, 0)

            conv = nn.utils.parametrizations.weight_norm(conv, name="weight", dim=2)
            self.relative_position = nn.Sequential(conv, nn.GELU())
        # --- Input Conditioning --- (Includes projection up to transformer_dim)
        # Rearrange, Norm, Dropout, Project, Rearrange
        self.input_conditioning = nn.Sequential(
            Rearrange("b c t -> b t c"),  # Batch, Time, Channels
            nn.LayerNorm(in_features),
            nn.Dropout(dropout),
            nn.Linear(in_features, self.transformer_dim),  # Project up using Linear
            # nn.Conv1d(in_features, self.transformer_dim, 1), # Alternative: Project up using Conv1d
            Rearrange(
                "b t c -> t b c"
            ),  # Time, Batch, Channels (Transformer expected format)
        )

        # --- Transformer Encoder Layers ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.transformer_dim,  # Use projected dimension
            nhead=heads,
            dim_feedforward=hidden_feedforward,
            dropout=dropout,  # Dropout within transformer layer
            activation=activation,
            batch_first=False,  # Expects (T, B, C)
            norm_first=False,  # Standard post-norm architecture
        )
        self.transformer_layers = nn.ModuleList(
            [copy.deepcopy(encoder_layer) for _ in range(layers)]
        )

        # --- Output Layer --- (Project back down to in_features)
        # Input is (T, B, C_transformer), output should be (B, C_original, T)
        self.output_layer = nn.Linear(self.transformer_dim, in_features)

        # Initialize parameters like BERT / TFixup
        self.apply(self._init_simplified_params)

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

        # Apply final projection back to original feature dimension
        x = self.output_layer(x)
        # x: [seq_len + 1, batch_size, in_features]

        # Rearrange back to [batch_size, in_features, seq_len + 1]
        x = x.permute(1, 2, 0)

        return x
