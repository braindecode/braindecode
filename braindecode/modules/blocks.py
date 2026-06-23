from typing import Optional
from warnings import warn

import torch
from torch import nn


class PatchTokenizer(nn.Module):
    r"""Tokenize an EEG signal into temporal patches (optionally overlapping).

    Transforms ``(batch, n_chans, n_times)`` into
    ``(batch, n_chans, n_patches, patch_dim)`` by sliding a window of
    ``patch_size`` samples along the time axis with a step of ``stride``
    samples. The default ``stride = patch_size`` gives non-overlapping patches;
    a smaller ``stride`` gives overlapping ones. This is the shared patch /
    "tokenization" step used by transformer EEG foundation models (e.g. LaBraM,
    CBraMod, EEG-DINO, BrainOmni).

    As in the filter-bank models (:class:`~braindecode.models.FBCNet`,
    :class:`~braindecode.models.FBMSNet`), the time axis is always right
    zero-padded so the windows tile it fully (no samples are dropped) and at
    least one window is produced; it is never an error. Padding is recomputed
    from the actual input length at every forward, so inputs whose length
    differs from ``n_times`` are handled too. A warning is emitted at
    construction when the configured ``n_times`` does not tile cleanly.

    Two modes:

    - **non-learnable** (``learnable=False``, default): a pure windowing view,
      so ``patch_dim == patch_size`` and the raw samples of each patch are kept
      (the patch embedding, if any, lives in the model). The ``stride`` may be
      overridden per call (see :meth:`forward`), so one tokenizer can be reused
      at several overlaps.
    - **learnable** (``learnable=True``): a strided ``Conv1d`` (kernel
      ``patch_size``, stride ``stride``, applied per channel) maps each patch to
      ``emb_dim`` features, so ``patch_dim == emb_dim``. The stride is fixed at
      construction.

    Parameters
    ----------
    patch_size : int
        Number of time samples per patch (the window length).
    n_times : int
        Number of time samples of the input, used to emit the construction
        warning when ``n_times`` does not tile cleanly into patches.
    emb_dim : int, optional
        Output features per patch in learnable mode. Defaults to ``patch_size``.
        Ignored when ``learnable=False``.
    learnable : bool, default=False
        Whether the tokenizer is a learned convolution or a fixed windowing.
    stride : int, optional
        Step between consecutive patches, in samples. Defaults to ``patch_size``
        (non-overlapping). Use ``stride < patch_size`` for overlapping patches.

    Examples
    --------
    >>> import torch
    >>> from braindecode.modules import PatchTokenizer
    >>> tokenizer = PatchTokenizer(patch_size=200, n_times=1000)
    >>> tokenizer(torch.randn(2, 19, 1000)).shape
    torch.Size([2, 19, 5, 200])
    >>> tokenizer(torch.randn(2, 19, 1000), stride=100).shape  # 50% overlap
    torch.Size([2, 19, 9, 200])
    """

    def __init__(self, patch_size, n_times, emb_dim=None, learnable=False, stride=None):
        super().__init__()
        self.patch_size = patch_size
        self.stride = patch_size if stride is None else stride
        self.learnable = learnable
        self.emb_dim = (emb_dim or patch_size) if learnable else patch_size
        if self._pad_to_tile(n_times, self.stride):
            warn(
                f"Time dimension ({n_times}) does not tile into patches of size "
                f"{patch_size} at stride {self.stride}. Input will be padded.",
                UserWarning,
            )
        # Defined unconditionally (Identity when not learnable) so the attribute
        # always exists for torch.jit.script, which type-checks both branches.
        if learnable:
            self.patcher = nn.Conv1d(1, self.emb_dim, patch_size, stride=self.stride)
        else:
            self.patcher = nn.Identity()

    def _pad_to_tile(self, n_times: int, stride: int) -> int:
        """Right-pad needed so ``patch_size`` windows tile ``n_times`` at ``stride`` (>= 1 window)."""
        n_eff = max(n_times, self.patch_size)
        remainder = (n_eff - self.patch_size) % stride
        return (n_eff - n_times) + (stride - remainder) % stride

    def forward(self, x: torch.Tensor, stride: Optional[int] = None) -> torch.Tensor:
        """Patchify ``(batch, n_chans, n_times)`` -> ``(batch, n_chans, n_patches, patch_dim)``.

        ``stride`` overrides the patch step for this call (non-learnable mode
        only), so a single tokenizer can be reused at several overlaps; it
        defaults to the ``stride`` set at construction.
        """
        step = self.stride if stride is None else stride
        pad = self._pad_to_tile(x.shape[-1], step)
        if pad:
            x = nn.functional.pad(x, (0, pad))
        batch_size, n_chans, _ = x.shape
        if self.learnable:
            if step != self.stride:
                raise ValueError(
                    "stride override is only supported for non-learnable "
                    f"PatchTokenizer (got stride={step}, built with "
                    f"stride={self.stride})."
                )
            x = x.flatten(0, 1).unsqueeze(1)  # (batch * chans, 1, time)
            x = self.patcher(x)  # (batch * chans, emb, patches)
            return x.reshape(batch_size, n_chans, x.shape[-2], x.shape[-1]).permute(
                0, 1, 3, 2
            )
        # Non-overlapping unfold is contiguous and equals the former reshape;
        # overlapping unfold returns a strided view.
        return x.unfold(dimension=-1, size=self.patch_size, step=step)


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
