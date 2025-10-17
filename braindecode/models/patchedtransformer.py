# Authors: Jose Mauricio <josemaurici3991@gmail.com>
#          Bruno Aristimunha <b.aristimunha@gmail.com>
# License: BSD (3-clause)
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from braindecode.models.base import EEGModuleMixin


class PBT(EEGModuleMixin, nn.Module):
    """Patched Brain Transformer (PBT) model from Klein et al. (2025) [pbt]_.

    :bdg-info:`Small Attention`

    This implementation was based in https://github.com/timonkl/PatchedBrainTransformer/

    .. figure:: https://raw.githubusercontent.com/timonkl/PatchedBrainTransformer/refs/heads/main/PBT_sketch.png
       :align: center
       :alt:  Patched Brain Transformer Architecture
       :width: 680px

    PBT tokenizes EEG trials into per-channel patches, linearly projects each
    patch to a model embedding dimension, prepends a classification token and
    adds channel-aware positional embeddings. The token sequence is processed
    by a Transformer encoder stack and classification is performed from the
    classification token.

    .. rubric:: Architectural Overview

    - Tokenization: The pre-processed EEG signals `(batch, n_chans, n_times)` is divided into non-overlapping
      patches of size `d_input` along the time axis. Since the original implementation does
      this process inside a custom Dataloader, we've adapted to apply this inside the own model.
      First the number of total patches is calculated using `n_chans`, `n_times`, `d_input` and `num_tokens_per_channel`,
      We have segment `X` input into these windows to fit the together with a positional encoder built internally
      (since only one dataset can be used at time) `Xp`

    - Positional indexing: a :class:`_ChannelEncoding` provides per-sample positional
      indices which are mapped to embeddings via :class:`nn.Embedding`.

    - Projection: linear projection `d_input -> d_model` maps tokens into the
      Transformer embedding space to be input into the Transformer encoder.

    - Transformer encoder: a stack of `n_blocks` Transformer encoder layers with `num_heads` attention heads.

    - Classification head: a linear layer applied to the CLS token.


    References
    ----------
    .. [pbt] Klein, T., Minakowski, P., & Sager, S. (2025).
        Flexible Patched Brain Transformer model for EEG decoding.
        Scientific Reports, 15(1), 1-12.
        https://www.nature.com/articles/s41598-025-86294-3
    .. [visualtransformer]  Dosovitskiy, A., Beyer, L., Kolesnikov, A.,
        Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M.,
        Minderer, M., Heigold, G., Gelly, S., Uszkoreit, J. & Houlsby,
        N. (2021). An Image is Worth 16x16 Words: Transformers for Image
        Recognition at Scale. International Conference on Learning
        Representations (ICLR).
    .. [efficient-batchpacking] Krell, M. M., Kosec, M., Perez, S. P., &
        Fitzgibbon, A. (2021). Efficient sequence packing without
        cross-contamination: Accelerating large language models without
        impacting performance. arXiv preprint arXiv:2107.02027.

    Parameters
    ----------
    d_input : int, optional
        Size (in samples) of each patch (token) extracted along the time axis.
    num_tokens_per_channel : int, optional
        Number of token indices reserved per channel (positional embedding indexing).
    d_model : int, optional
        Transformer embedding dimensionality.
    n_blocks : int, optional
        Number of Transformer encoder layers.
    num_heads : int, optional
        Number of attention heads.
    drop_prob : float, optional
        Dropout probability used in Transformer components.
    learnable_cls : bool, optional
        Whether the classification token is learnable.
    bias_transformer : bool, optional
        Whether to use bias in Transformer linear layers.
    activation : nn.Module, optional
        Activation function class to use in Transformer feed-forward layers.

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
        d_input: int = 64,
        num_tokens_per_channel: int = 8,
        d_model: int = 128,
        n_blocks: int = 4,
        num_heads: int = 4,
        drop_prob: float = 0.1,
        learnable_cls=True,
        bias_transformer=False,
        activation: nn.Module = nn.GELU,
    ) -> None:
        super().__init__(
            n_outputs=n_outputs,
            n_chans=n_chans,
            chs_info=chs_info,
            n_times=n_times,
            input_window_seconds=input_window_seconds,
            sfreq=sfreq,
        )
        del n_outputs, n_chans, chs_info, n_times, input_window_seconds, sfreq
        # Store hyperparameters
        self.d_input = d_input
        self.num_tokens_per_channel = num_tokens_per_channel
        self.d_model = d_model
        self.n_blocks = n_blocks
        self.num_heads = num_heads
        self.drop_prob = drop_prob

        # number of distinct positional indices (per-channel tokens + cls)
        self.num_embeddings = self.num_tokens_per_channel * self.n_chans + 1

        # number of windows (how many disjoint chunks of size d_input fit in the trial)
        self.windows = (self.n_chans * self.n_times) // (
            (self.num_embeddings - 1) * self.d_input
        )

        if self.windows == 0:
            raise ValueError(
                f"Unable to form windows with the current parameters.\n"
                f"num_embeddings = {self.num_embeddings}, d_input = {self.d_input}\n"
                f"Consider reducing num_embeddings or d_input."
            )

        # Linear projection from token raw-size -> d_model
        self.linear_projection = nn.Linear(
            in_features=self.d_input, out_features=self.d_model, bias=False
        )

        # Classification token (learnable or fixed zero)
        if learnable_cls:
            self.cls_token = nn.Parameter(torch.randn(1, 1, self.d_model) * 0.002)
        else:
            # non-learnable zeroed tensor
            self.cls_token = torch.full(
                size=(1, 1, self.d_model),
                fill_value=0,
                requires_grad=False,
                dtype=torch.float32,
            )

        # Channel-aware positional index generator (registers buffer internally)
        self.positional_embedding = _ChannelEncoding(
            n_chans=self.n_chans,
            n_times=self.n_times,
            num_tokens_per_channel=self.num_tokens_per_channel,
        )

        # actual embedding table mapping indices -> d_model
        self.pos_embedding = nn.Embedding(
            num_embeddings=self.num_embeddings, embedding_dim=self.d_model
        )

        # Transformer encoder stack
        self.transformer_encoder = _TransformerEncoder(
            n_blocks=n_blocks,
            d_model=self.d_model,
            n_head=num_heads,
            drop_prob=drop_prob,
            bias=bias_transformer,
            activation=activation,
        )

        # classification head on classify token - CLS token
        self.final_layer = nn.Linear(
            in_features=d_model, out_features=self.n_outputs, bias=True
        )

        # initialize weights
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        """Weight initialization following the original implementation."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.002)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """The implementation follows the original code logic

        - split input into windows of size `(num_embeddings - 1) * d_input`
        - for each window: reshape into tokens, map positional indices to embeddings,
          add cls token, run Transformer encoder and collect CLS outputs
        - aggregate CLS outputs across windows (if >1) and pass through `final_layer`

        Parameters
        ----------
        X : torch.Tensor
            Input tensor with shape (batch_size, n_chans, n_times)

        Returns
        -------
        torch.Tensor
            Output logits with shape (batch_size, n_outputs).
        """
        # positional indices per-sample (batch_size, n_chans, n_times) ->
        # values in [0, num_embeddings-1]
        Xpositional = self.positional_embedding(X)
        batch_size = X.shape[0]

        concat = []
        for i in range(self.windows):
            start_idx = i * ((self.num_embeddings - 1) * self.d_input)
            end_idx = (i + 1) * ((self.num_embeddings - 1) * self.d_input)

            # X_patched: (B, num_embeddings - 1, d_input)
            X_patched = X.view(batch_size, -1)[:, start_idx:end_idx].view(
                batch_size, (self.num_embeddings - 1), self.d_input
            )

            # Xpos_patched: (B, num_embeddings - 1, d_input) ->
            # reduce to single index per token
            Xpos_patched = Xpositional.view(batch_size, -1)[:, start_idx:end_idx].view(
                batch_size, (self.num_embeddings - 1), self.d_input
            )

            # reduce positional block to a single index per token (take first element)
            Xpos_patched = Xpos_patched[:, :, 0].long()
            # shape (batch_size, num_embeddings-1)

            # project patches -> (batch_size, num_embeddings-1, d_model)
            tokens = self.linear_projection(X_patched)

            # expand cls token -> (batch_size, 1, d_model)
            cls_token = self.cls_token.expand(batch_size, -1, -1)

            # add cls token to tokens -> (batch_size, num_embeddings, d_model)
            tokens = torch.cat([cls_token, tokens], dim=1)

            # build positional indices including CLS (0 reserved for CLS)
            cls_idx = torch.zeros((batch_size, 1), dtype=torch.long, device=X.device)
            int_pos = torch.cat(
                [cls_idx, Xpos_patched], dim=1
            )  # (batch_size, num_embeddings)

            # lookup positional embeddings -> (batch_size, num_embeddings, d_model)
            pos_emb = self.pos_embedding(int_pos)

            # transformer forward -> (batch_size, num_embeddings, d_model)
            transformer_out = self.transformer_encoder(tokens + pos_emb)

            concat.append(transformer_out[:, 0])  # CLS vector (batch_size, d_model)

        # aggregate across windows (original code creates mean over windows but
        # returns last transformer_out CLS)
        concat_agg = torch.stack(concat, dim=0)
        concat_agg = torch.mean(concat_agg, dim=0)  # (batch_size, d_model)

        return self.final_layer(concat_agg)


class _LayerNorm(nn.Module):
    """Layer normalization with optional bias.

    Simple wrapper around :func:`torch.nn.functional.layer_norm` exposing a
    learnable scale and optional bias.

    Parameters
    ----------
    ndim : int
        Number of features (normalized shape).
    bias : bool
        Whether to include a learnable bias term.
    """

    def __init__(self, ndim: int, bias: bool) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply layer normalization.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor where the last dimension has size `ndim`.

        Returns
        -------
        torch.Tensor
            Normalized tensor with the same shape as input.
        """
        return F.layer_norm(
            x,
            normalized_shape=self.weight.shape,
            weight=self.weight,
            bias=self.bias,
            eps=1e-5,
        )


class _MHSA(nn.Module):
    """Multi-head self-attention (MHSA) block.

    Implements a standard multi-head attention mechanism with optional
    use of PyTorch's scaled_dot_product_attention (FlashAttention) when
    available and requested.

    Parameters
    ----------
    d_model : int
        Dimensionality of the model / embeddings.
    n_head : int
        Number of attention heads.
    bias : bool
        Whether linear layers use bias.
    drop_prob : float, optional
        drop_prob probability applied to attention weights and residual projection.
    """

    def __init__(
        self,
        d_model: int,
        n_head: int,
        bias: bool,
        drop_prob: float = 0.0,
    ) -> None:
        super().__init__()

        assert d_model % n_head == 0, "d_model must be divisible by n_head"

        # qkv and output projection
        self.attn = nn.Linear(d_model, 3 * d_model, bias=bias)
        self.proj = nn.Linear(d_model, d_model, bias=bias)

        # dropout modules
        self.attn_drop_prob = nn.Dropout(drop_prob)
        self.resid_drop_prob = nn.Dropout(drop_prob)

        self.n_head = n_head
        self.d_model = d_model
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for MHSA.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, T, C) where C == d_model.

        Returns
        -------
        torch.Tensor
            Output tensor of shape (B, T, C).
        """

        B, T, C = x.size()

        # project to q, k, v and reshape for multi-head attention
        q, k, v = self.attn(x).split(self.d_model, dim=2)

        # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        y = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.drop_prob if self.training else 0.0,
            is_causal=False,
        )

        # re-assemble heads -> (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # final linear projection + residual drop_prob
        y = self.resid_drop_prob(self.proj(y))
        return y


class _FeedForward(nn.Module):
    """Position-wise feed-forward network from Transformer.

    Implements the two-layer MLP with GELU activation and dropout used in
    Transformer architectures.

    Parameters
    ----------
    d_model : int
        Input and output dimensionality.
    dim_feedforward : int, optional
        Hidden dimensionality of the feed-forward layer. If None, must be provided by caller.
    drop_prob : float, optional
        Dropout probability.
    bias : bool, optional
        Whether linear layers use bias.
    """

    def __init__(
        self,
        d_model: int,
        dim_feedforward: Optional[int] = None,
        drop_prob: float = 0.0,
        bias: bool = False,
        activation: nn.Module = nn.GELU,
    ) -> None:
        super().__init__()

        if dim_feedforward is None:
            raise ValueError("dim_feedforward must be provided")

        self.proj_in = nn.Linear(d_model, dim_feedforward, bias=bias)
        self.activation = activation()
        self.proj = nn.Linear(dim_feedforward, d_model, bias=bias)
        self.drop_prob = nn.Dropout(drop_prob)
        self.drop_prob1 = nn.Dropout(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the feed-forward block."""
        x = self.proj_in(x)
        x = self.activation(x)
        x = self.drop_prob1(x)
        x = self.proj(x)
        x = self.drop_prob(x)
        return x


class _TransformerEncoderLayer(nn.Module):
    """Single Transformer encoder layer (pre-norm) combining MHSA and feed-forward.

    The block follows the pattern:
    x <- x + MHSA(_LayerNorm(x))
    x <- x + FF(_LayerNorm(x))
    """

    def __init__(
        self,
        d_model: int,
        n_head: int,
        drop_prob: float = 0.0,
        dim_feedforward: Optional[int] = None,
        bias: bool = False,
        activation: nn.Module = nn.GELU,
    ) -> None:
        super().__init__()

        if dim_feedforward is None:
            dim_feedforward = 4 * d_model
            # note: preserve the original behaviour (print) from the provided code
            print(
                "dim_feedforward is set to 4*d_model, the default in Vaswani et al. (Attention is all you need)"
            )

        self.layer_norm_att = _LayerNorm(d_model, bias=bias)
        self.mhsa = _MHSA(d_model, n_head, bias, drop_prob=drop_prob)
        self.layer_norm_ff = _LayerNorm(d_model, bias=bias)
        self.feed_forward = _FeedForward(
            d_model=d_model,
            dim_feedforward=dim_feedforward,
            drop_prob=drop_prob,
            bias=bias,
            activation=activation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Execute one encoder layer.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape (B, T, d_model).

        Returns
        -------
        torch.Tensor
            Output of the same shape as input.
        """
        x = x + self.mhsa(self.layer_norm_att(x))
        x = x + self.feed_forward(self.layer_norm_ff(x))
        return x


class _TransformerEncoder(nn.Module):
    """Stack of Transformer encoder layers.

    Parameters
    ----------
    n_blocks : int
        Number of encoder layers to stack.
    d_model : int
        Dimensionality of embeddings.
    n_head : int
        Number of attention heads per layer.
    drop_prob : float
        Dropout probability.
    bias : bool
        Whether linear layers use bias.
    """

    def __init__(
        self,
        n_blocks: int,
        d_model: int,
        n_head: int,
        drop_prob: float,
        bias: bool,
        activation: nn.Module = nn.GELU,
    ) -> None:
        super().__init__()

        self.encoder_block = nn.ModuleList(
            [
                _TransformerEncoderLayer(
                    d_model=d_model,
                    n_head=n_head,
                    drop_prob=drop_prob,
                    dim_feedforward=None,
                    bias=bias,
                    activation=activation,
                )
                for _ in range(n_blocks)
            ]
        )

        # GPT2-like initialization for linear layers
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * n_blocks))

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward through all encoder blocks sequentially."""
        for block in self.encoder_block:
            x = block(x)
        return x


class _ChannelEncoding(nn.Module):
    """Channel-aware positional encoding helper.

    This module builds a per-channel positional index buffer that maps each
    time sample to a token index for that channel. It is registered as a
    buffer (`x_pos_single`) and expanded to batch-size at runtime.

    Parameters
    ----------
    n_chans : int
        Number of EEG channels.
    n_times : int
        Number of time samples per trial.
    num_tokens_per_channel : int, optional
        Number of distinct token indices to use per channel.
    """

    def __init__(self, n_chans=2, n_times=1000, num_tokens_per_channel=8) -> None:
        super().__init__()

        x_pos_single = torch.zeros((n_chans, n_times), dtype=torch.long)

        for c in range(n_chans):
            start = c * num_tokens_per_channel + 1
            end = (c + 1) * num_tokens_per_channel + 1
            seq = torch.arange(start, end, dtype=torch.long)
            seq_rep = seq.repeat((n_times // num_tokens_per_channel) + 1)[:n_times]
            x_pos_single[c, :] = seq_rep

        self.register_buffer("x_pos_single", x_pos_single)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Return per-sample positional indices expanded to batch dimension.

        Parameters
        ----------
        X : torch.Tensor
            Input tensor only used for batch size inference (B, C, T).

        Returns
        -------
        torch.LongTensor
            Tensor of shape (B, C, T) containing positional token indices.
        """
        b, c, t = X.shape
        return self.x_pos_single.unsqueeze(0).expand(b, -1, -1)
