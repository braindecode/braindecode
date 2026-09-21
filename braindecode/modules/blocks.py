from collections.abc import Sequence
from warnings import warn

import torch
from einops.layers.torch import Rearrange
from torch import nn


class PatchTokenizer(nn.Module):
    r"""Tokenize an EEG signal into non-overlapping temporal patches.

    Transforms ``(batch, n_chans, n_times)`` into
    ``(batch, n_chans, n_patches, patch_dim)`` by splitting the time axis into
    non-overlapping patches of ``patch_size`` samples. This is the shared
    patch / "tokenization" step used by transformer EEG foundation models
    (e.g. LaBraM, CBraMod, EEG-DINO).

    By default, as in the filter-bank models
    (:class:`~braindecode.models.FBCNet`, :class:`~braindecode.models.FBMSNet`),
    when ``n_times`` is not a multiple of ``patch_size`` the input is right
    zero-padded (a warning is emitted at construction). Set
    ``on_non_divisible="crop"`` to hard-crop the trailing samples instead, or
    ``"error"`` to reject non-divisible inputs. Padding/cropping is applied at
    ``forward`` time to the actual input length (not the construction-time
    ``n_times``), so the module also accepts inputs of a different length.

    Two modes:

    - **non-learnable** (``learnable=False``, default): a pure reshape, so
      ``patch_dim == patch_size`` and the raw samples of each patch are kept
      (the patch embedding, if any, lives in the model).
    - **learnable** (``learnable=True``): maps each patch to ``emb_dim``
      features, so ``patch_dim == emb_dim``. ``projection="conv"`` keeps the
      historical strided ``Conv1d`` behavior; ``projection="linear"`` first
      forms raw patches and then applies one ``Linear(patch_size, emb_dim)`` to
      each patch.

    Parameters
    ----------
    patch_size : int
        Number of time samples per patch.
    n_times : int
        Number of time samples of the input, used to set up the right-padding
        when ``n_times`` is not a multiple of ``patch_size``.
    emb_dim : int, optional
        Output features per patch in learnable mode. Defaults to ``patch_size``.
        Ignored when ``learnable=False``.
    learnable : bool, default=False
        Whether the tokenizer is a learned convolution or a fixed reshape.
    on_non_divisible : {"pad", "crop", "error"}, default="pad"
        How to handle a time dimension that is not divisible by ``patch_size``.
        ``"pad"`` right-pads with zeros, ``"crop"`` drops the trailing samples,
        and ``"error"`` raises a :class:`ValueError`.
    projection : {"conv", "linear"}, default="conv"
        Learnable projection type. Only used when ``learnable=True``.
    output_order : {"channel_patch", "patch_channel"}, default="channel_patch"
        Axis order after the batch dimension. ``"channel_patch"`` returns
        ``(batch, n_chans, n_patches, patch_dim)`` and preserves the historical
        output. ``"patch_channel"`` returns
        ``(batch, n_patches, n_chans, patch_dim)``.

    Examples
    --------
    >>> import torch
    >>> from braindecode.modules import PatchTokenizer
    >>> tokenizer = PatchTokenizer(patch_size=200, n_times=1000)
    >>> tokenizer(torch.randn(2, 19, 1000)).shape
    torch.Size([2, 19, 5, 200])
    """

    def __init__(
        self,
        patch_size,
        n_times,
        emb_dim=None,
        learnable=False,
        on_non_divisible="pad",
        projection="conv",
        output_order="channel_patch",
    ):
        super().__init__()
        if on_non_divisible not in ("pad", "crop", "error"):
            raise ValueError(
                "on_non_divisible must be 'pad', 'crop', or 'error', "
                f"got {on_non_divisible!r}."
            )
        if projection not in ("conv", "linear"):
            raise ValueError(
                f"projection must be 'conv' or 'linear', got {projection!r}."
            )
        if output_order not in ("channel_patch", "patch_channel"):
            raise ValueError(
                "output_order must be 'channel_patch' or 'patch_channel', "
                f"got {output_order!r}."
            )
        self.patch_size = patch_size
        self.learnable = learnable
        self.on_non_divisible = on_non_divisible
        self.projection = projection
        self.output_order = output_order
        self.emb_dim = (emb_dim or patch_size) if learnable else patch_size
        if n_times % patch_size:
            if on_non_divisible == "pad":
                warn(
                    f"Time dimension ({n_times}) is not divisible by patch_size "
                    f"({patch_size}). Input will be padded.",
                    UserWarning,
                )
            elif on_non_divisible == "error":
                raise ValueError(
                    f"Time dimension ({n_times}) is not divisible by patch_size "
                    f"({patch_size})."
                )
        # Padding/cropping for a non-divisible time axis is applied at runtime in
        # _prepare_input (which works for any input length, not just the
        # construction-time n_times), so no padding submodule is stored here.
        # Defined unconditionally (Identity when not learnable) so the attribute
        # always exists for torch.jit.script, which type-checks both branches.
        self.patcher = nn.Identity()
        self.proj = nn.Identity()
        if learnable and projection == "conv":
            self.patcher = nn.Conv1d(1, self.emb_dim, patch_size, stride=patch_size)
        elif learnable and projection == "linear":
            self.proj = nn.Linear(patch_size, self.emb_dim)

    def _prepare_input(self, x):
        n_times = x.shape[-1]
        remainder = n_times % self.patch_size
        if remainder == 0:
            return x
        if self.on_non_divisible == "pad":
            return nn.functional.pad(x, (0, self.patch_size - remainder))
        if self.on_non_divisible == "crop":
            cropped_n_times = n_times - remainder
            if cropped_n_times < self.patch_size:
                raise ValueError(
                    f"Need at least one full patch of {self.patch_size} samples, "
                    f"got {n_times}."
                )
            return x[..., :cropped_n_times]
        raise ValueError(
            f"Time dimension ({n_times}) is not divisible by patch_size "
            f"({self.patch_size})."
        )

    def forward(self, x):
        x = self._prepare_input(x)
        batch_size, n_chans, _ = x.shape
        if self.learnable and self.projection == "conv":
            x = x.flatten(0, 1).unsqueeze(1)  # (batch * chans, 1, time)
            x = self.patcher(x)  # (batch * chans, emb, patches)
            x = x.reshape(batch_size, n_chans, x.shape[-2], x.shape[-1]).permute(
                0, 1, 3, 2
            )
        else:
            x = x.reshape(batch_size, n_chans, -1, self.patch_size)
            if self.learnable:
                x = self.proj(x)
        if self.output_order == "patch_channel":
            return x.transpose(1, 2)
        return x


class InceptionBlock(nn.Module):
    """
    Inception block module.

    This module applies multiple convolutional branches to the input and concatenates
    their outputs along the channel dimension. Each branch can have a different
    configuration, allowing the model to capture multi-scale features.

    Parameters
    ----------
    branches : list of nn.Module
        List of convolutional branches to apply to the input.

    Examples
    --------
    >>> import torch
    >>> from torch import nn
    >>> from braindecode.modules import InceptionBlock
    >>> block = InceptionBlock(
    ...     [
    ...         nn.Conv1d(3, 4, kernel_size=1),
    ...         nn.Conv1d(3, 4, kernel_size=3, padding=1),
    ...     ]
    ... )
    >>> inputs = torch.randn(2, 3, 100)
    >>> outputs = block(inputs)
    >>> outputs.shape
    torch.Size([2, 8, 100])
    """

    def __init__(self, branches):
        super().__init__()
        self.branches = nn.ModuleList(branches)

    def forward(self, x):
        return torch.cat([branch(x) for branch in self.branches], 1)


class MLP(nn.Sequential):
    r"""Multilayer Perceptron (MLP) with GELU activation and optional dropout.

    Also known as fully connected feedforward network, an MLP is a sequence of
    non-linear parametric functions

    .. math:: h_{i + 1} = a_{i + 1}(h_i W_{i + 1}^T + b_{i + 1}),

    over feature vectors :math:`h_i`, with the input and output feature vectors
    :math:`x = h_0` and :math:`y = h_L`, respectively. The non-linear functions
    :math:`a_i` are called activation functions. The trainable parameters of an
    MLP are its weights and biases :math:`\\phi = \{W_i, b_i | i = 1, \dots, L\}`.

    Parameters
    ----------
    in_features: int
        Number of input features.
    hidden_features: Sequential[int] (default=None)
        Number of hidden features, if None, set to in_features.
        You can increase the size of MLP just passing more int in the
        hidden features vector. The model size increase follow the
        rule 2n (hidden layers)+2 (in and out layers)
    out_features: int (default=None)
        Number of output features, if None, set to in_features.
    act_layer: nn.GELU (default)
        The activation function constructor. If ``None``, use
        :class:`torch.nn.GELU` instead.
    drop: float (default=0.0)
        Dropout rate.
    normalize: bool (default=False)
        Whether to apply layer normalization.

    Examples
    --------
    >>> import torch
    >>> from braindecode.modules import MLP
    >>> module = MLP(in_features=32, hidden_features=(64,), out_features=16)
    >>> inputs = torch.randn(2, 10, 32)
    >>> outputs = module(inputs)
    >>> outputs.shape
    torch.Size([2, 10, 16])
    """

    def __init__(
        self,
        in_features: int,
        hidden_features=None,
        out_features=None,
        activation=nn.GELU,
        drop=0.0,
        normalize=False,
    ):
        self.normalization = nn.LayerNorm if normalize else lambda: None
        self.in_features = in_features
        self.out_features = out_features or self.in_features
        if hidden_features:
            self.hidden_features = hidden_features
        else:
            self.hidden_features = (self.in_features, self.in_features)
        self.activation = activation

        layers = []

        for before, after in zip(
            (self.in_features, *self.hidden_features),
            (*self.hidden_features, self.out_features),
        ):
            layers.extend(
                [
                    nn.Linear(in_features=before, out_features=after),
                    self.activation(),
                    self.normalization(),
                ]
            )

        layers = layers[:-2]
        layers.append(nn.Dropout(p=drop))

        # Cleaning if we are not using the normalization layer
        layers = list(filter(lambda layer: layer is not None, layers))

        super().__init__(*layers)


class FeedForwardBlock(nn.Sequential):
    """Feedforward network block.

    Parameters
    ----------
    emb_size : int
        Embedding dimension.
    expansion : int
        Expansion factor for the hidden layer size.
    drop_p : float
        Dropout probability.
    activation : type[nn.Module], default=nn.GELU
        Activation function constructor.

    Examples
    --------
    >>> import torch
    >>> from braindecode.modules import FeedForwardBlock
    >>> module = FeedForwardBlock(emb_size=32, expansion=2, drop_p=0.1)
    >>> inputs = torch.randn(2, 10, 32)
    >>> outputs = module(inputs)
    >>> outputs.shape
    torch.Size([2, 10, 32])
    """

    def __init__(
        self, emb_size, expansion, drop_p, activation: type[nn.Module] = nn.GELU
    ):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            activation(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )


class _TDSConv2dBlock(nn.Module):
    r"""Time-depth-separable convolutional block.

    The feature dimension is factorized as ``channels * width`` and processed
    by a valid ``(1, kernel_width)`` convolution. The result is combined with
    the aligned tail of the input before layer normalization.

    Parameters
    ----------
    channels : int
        Number of convolution channels in the feature factorization.
    width : int
        Width of each convolution channel. The input feature dimension must be
        ``channels * width``.
    kernel_width : int
        Size of the valid temporal convolution kernel.
    activation : type[nn.Module], default=nn.ReLU
        Activation constructor applied after the convolution.
    time_first : bool, default=True
        If ``True``, inputs and outputs use ``(time, batch, features)``. If
        ``False``, they use ``(batch, features, time)``.
    """

    def __init__(
        self,
        channels: int,
        width: int,
        kernel_width: int,
        activation: type[nn.Module] = nn.ReLU,
        time_first: bool = True,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.width = width
        self.time_first = time_first
        self.conv2d = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=(1, kernel_width),
        )
        self.activation = activation()
        self.layer_norm = nn.LayerNorm(channels * width)
        if time_first:
            self.fold_features = Rearrange(
                "time batch (channels width) -> batch channels width time",
                channels=channels,
                width=width,
            )
            self.unfold_features = Rearrange(
                "batch channels width time -> time batch (channels width)"
            )
        else:
            self.fold_features = Rearrange(
                "batch (channels width) time -> batch channels width time",
                channels=channels,
                width=width,
            )
            self.unfold_features = Rearrange(
                "batch channels width time -> batch (channels width) time"
            )
        self.features_to_sequence = Rearrange(
            "batch features time -> batch time features"
        )
        self.sequence_to_features = Rearrange(
            "batch time features -> batch features time"
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        folded = self.fold_features(inputs)
        convolved = self.conv2d(folded)
        activated = self.activation(convolved)
        conv_out = self.unfold_features(activated)
        n_out_frames = activated.shape[-1]

        if self.time_first:
            residual_added = conv_out + inputs[-n_out_frames:]
            output = self.layer_norm(residual_added)
            return output

        residual_added = conv_out + inputs[..., -n_out_frames:]
        sequence = self.features_to_sequence(residual_added)
        normalized = self.layer_norm(sequence)
        output = self.sequence_to_features(normalized)
        return output


class _TDSFullyConnectedBlock(nn.Module):
    r"""Residual position-wise feed-forward TDS block.

    Parameters
    ----------
    num_features : int
        Size of the feature dimension.
    activation : type[nn.Module], default=nn.ReLU
        Activation constructor used between the two linear layers.
    drop_prob : float, default=0.0
        Dropout probability after the activation and second linear layer.
    time_first : bool, default=True
        If ``True``, inputs and outputs use ``(time, batch, features)``. If
        ``False``, they use ``(batch, features, time)``.
    """

    def __init__(
        self,
        num_features: int,
        activation: type[nn.Module] = nn.ReLU,
        drop_prob: float = 0.0,
        time_first: bool = True,
    ) -> None:
        super().__init__()
        self.time_first = time_first
        layers: list[nn.Module] = [
            nn.Linear(num_features, num_features),
            activation(),
        ]
        if drop_prob > 0:
            layers.append(nn.Dropout(drop_prob))
        layers.append(nn.Linear(num_features, num_features))
        if drop_prob > 0:
            layers.append(nn.Dropout(drop_prob))
        self.fc_block = nn.Sequential(*layers)
        self.layer_norm = nn.LayerNorm(num_features)
        self.features_to_sequence = Rearrange(
            "batch features time -> batch time features"
        )
        self.sequence_to_features = Rearrange(
            "batch time features -> batch features time"
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.time_first:
            transformed = self.fc_block(inputs)
            residual_added = transformed + inputs
            output = self.layer_norm(residual_added)
            return output

        sequence = self.features_to_sequence(inputs)
        transformed = self.fc_block(sequence)
        residual_added = transformed + sequence
        normalized = self.layer_norm(residual_added)
        output = self.sequence_to_features(normalized)
        return output


class TDSConvEncoder(nn.Module):
    r"""Stack time-depth-separable convolution and feed-forward blocks.

    With ``K = len(block_channels)`` blocks and valid convolutions, the encoder
    removes ``K * (kernel_width - 1)`` frames. Every entry in
    ``block_channels`` must evenly divide ``num_features``.

    Parameters
    ----------
    num_features : int
        Size of the input and output feature dimension.
    block_channels : sequence of int, default=(24, 24, 24, 24)
        Convolution channel count for each TDS block.
    kernel_width : int, default=32
        Size of each valid temporal convolution kernel.
    activation : type[nn.Module], default=nn.ReLU
        Activation constructor used in every block.
    drop_prob : float, default=0.0
        Dropout probability in the feed-forward blocks.
    time_first : bool, default=True
        If ``True``, inputs and outputs use ``(time, batch, features)``. If
        ``False``, they use ``(batch, features, time)``.
    """

    def __init__(
        self,
        num_features: int,
        block_channels: Sequence[int] = (24, 24, 24, 24),
        kernel_width: int = 32,
        activation: type[nn.Module] = nn.ReLU,
        drop_prob: float = 0.0,
        time_first: bool = True,
    ) -> None:
        super().__init__()
        if not block_channels:
            raise ValueError("block_channels must be non-empty.")
        blocks: list[nn.Module] = []
        for n_block_channels in block_channels:
            if num_features % n_block_channels != 0:
                raise ValueError(
                    f"block_channels entry {n_block_channels} does not evenly "
                    f"divide num_features={num_features}."
                )
            blocks.extend(
                [
                    _TDSConv2dBlock(
                        channels=n_block_channels,
                        width=num_features // n_block_channels,
                        kernel_width=kernel_width,
                        activation=activation,
                        time_first=time_first,
                    ),
                    _TDSFullyConnectedBlock(
                        num_features=num_features,
                        activation=activation,
                        drop_prob=drop_prob,
                        time_first=time_first,
                    ),
                ]
            )
        self.tds_conv_blocks = nn.Sequential(*blocks)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = self.tds_conv_blocks(inputs)
        return output
