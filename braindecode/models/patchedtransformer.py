# Authors: Your Name <you@domain.com>
# License: BSD (3-clause)
from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from braindecode.models.base import EEGModuleMixin

class PBT(EEGModuleMixin, nn.Sequential):
    """Patched Brain Transformer (PBT) model from T Klein et al. (2025).
    This implementation was based in https://github.com/timonkl/PatchedBrainTransformer/

    PBT tokenizes EEG trials into per-channel patches, linearly projects each
    patch to a model embedding dimension, prepends a classification token and
    adds channel-aware positional embeddings. The token sequence is processed
    by a Transformer encoder stack and classification is performed from the
    classification token.

    .. rubric:: Architectural Overview

    - Tokenization: The pre-processed EEG signals `(Batch, Channel, Timestep)` is divided into non-overlapping
      patches of size `d_input` along the time axis. Since the original implementation does 
      this process inside a custom Dataloader, we've adapted to apply this inside the own model.
      First the number of total patches is calculated using C, T, `d_input` and `num_tokens_per_channel`,
      We've segment X input into these windows to fit the together with a positional encoder built internally 
      (since only one dataset can be used at time) Xp
    - Positional indexing: a `_ChannelEncoding` provides per-sample positional
      indices which are mapped to embeddings via :class:`nn.Embedding`.
    - Projection: linear projection `d_input -> d_model` maps tokens into the
      Transformer embedding space to be input into the Transformer encoder.
    - Transformer encoder: a stack of `n_blocks` Transformer encoder layers with `num_heads` attention heads.
    - Classification head: a linear layer applied to the CLS token.

    Parameters
    ----------
    n_chans : int, optional
        Number of EEG channels.
    n_outputs : int, optional
        Number of output classes.
    n_times : int, optional
        Number of time samples per trial.
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
    dropout : float, optional
        Dropout probability used in Transformer components.
    device : str or torch.device, optional
        Device for initialization of optional buffers/params.
    learnable_cls : bool, optional
        Whether the classification token is learnable.
    bias_transformer : bool, optional
        Whether to use bias in Transformer linear layers.
    input_window_seconds, sfreq, multiple_datasets, chs_info : optional
        Passed to :class:`EEGModuleMixin` for metadata compatibility.
    """

    def __init__(
        self,
        n_chans: Optional[int] = None,
        n_outputs: Optional[int] = None,
        n_times: Optional[int] = None,
        d_input: int = 64,
        num_tokens_per_channel: int = 8,
        d_model: int = 128,
        n_blocks: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
        device: str = "cpu",
        learnable_cls: bool = True,
        bias_transformer: bool = False,
        input_window_seconds=None,
        sfreq=None,
        multiple_datasets=None,
        chs_info: Optional[list[Dict]] = None,
    ) -> None:
        super().__init__(
            n_outputs=n_outputs,
            n_chans=n_chans,
            chs_info=chs_info,
            n_times=n_times,
            input_window_seconds=input_window_seconds,
            sfreq=sfreq,
        )

        # Store hyperparameters
        self.d_input = d_input
        self.num_tokens_per_channel = num_tokens_per_channel
        self.d_model = d_model
        self.n_blocks = n_blocks
        self.num_heads = num_heads
        self.dropout = dropout
        self.device = device

        # number of distinct positional indices (per-channel tokens + cls)
        self.num_embeddings = self.num_tokens_per_channel * self.n_chans + 1

        # number of windows (how many disjoint chunks of size d_input fit in the trial)
        self.windows = (self.n_chans * self.n_times) // ((self.num_embeddings - 1) * self.d_input)

        if self.windows == 0:
            raise ValueError(
                f"Unable to form windows with the current parameters.\n"
                f"num_embeddings = {self.num_embeddings}, d_input = {self.d_input}\n"
                f"Consider reducing num_embeddings or d_input."
            )

        # Linear projection from token raw-size -> d_model
        self.linear_projection = nn.Linear(in_features=self.d_input, out_features=self.d_model, bias=False)

        # Classification token (learnable or fixed zero)
        if learnable_cls:
            self.cls_token = nn.Parameter(torch.randn(1, 1, self.d_model) * 0.002)
        else:
            # non-learnable zeroed tensor on specified device
            self.cls_token = torch.full(
                size=(1, 1, self.d_model),
                fill_value=0,
                requires_grad=False,
                dtype=torch.float32,
                device=device,
            )

        # Channel-aware positional index generator (registers buffer internally)
        self.positional_embedding = _ChannelEncoding(
            n_chans=n_chans, n_times=n_times, num_tokens_per_channel=num_tokens_per_channel, device=device
        )

        # actual embedding table mapping indices -> d_model
        self.pos_embedding = nn.Embedding(num_embeddings=self.num_embeddings, embedding_dim=self.d_model)

        # Transformer encoder stack
        self.transformer_encoder = _TransformerEncoder(
            n_blocks=n_blocks, d_model=self.d_model, n_head=num_heads, dropout=dropout, bias=bias_transformer
        )

        # classification head on CLS token
        self.cls_head = nn.Linear(in_features=d_model, out_features=n_outputs, bias=True)

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

    def forward(self, X: torch.Tensor, split_sections=None) -> torch.Tensor:
        """Forward pass.

        The implementation follows the original code logic:
        - split input into windows of size `(num_embeddings - 1) * d_input`
        - for each window: reshape into tokens, map positional indices to embeddings,
          add cls token, run Transformer encoder and collect CLS outputs
        - aggregate CLS outputs across windows (if >1) and pass through `cls_head`

        Parameters
        ----------
        X : torch.Tensor
            Input tensor with shape (B, C, T).
        split_sections : optional
            Passed to transformer layers for memory-optimized attention.

        Returns
        -------
        torch.Tensor
            Output logits with shape (B, n_outputs).
        """
        # positional indices per-sample (B, C, T) -> values in [0, num_embeddings-1]
        Xp = self.positional_embedding(X)
        B = X.shape[0]

        concat = []
        for i in range(self.windows):
            start_idx = i * ((self.num_embeddings - 1) * self.d_input)
            end_idx = (i + 1) * ((self.num_embeddings - 1) * self.d_input)

            # Xa: (B, num_embeddings - 1, d_input)
            X_ = X.view(B, -1)[:, start_idx:end_idx].view(B, (self.num_embeddings - 1), self.d_input)

            # Xp_: (B, num_embeddings - 1, d_input) -> reduce to single index per token
            Xp_ = Xp.view(B, -1)[:, start_idx:end_idx].view(B, (self.num_embeddings - 1), self.d_input)

            # reduce positional block to a single index per token (take first element)
            Xp_ = Xp_[:, :, 0].long()  # shape (B, num_embeddings-1)

            # project patches -> (B, num_embeddings-1, d_model)
            tokens = self.linear_projection(X_)

            # expand cls token -> (B, 1, d_model)
            cls_token = self.cls_token.expand(B, -1, -1)

            # add cls token to tokens -> (B, num_embeddings, d_model)
            tokens = torch.cat([cls_token, tokens], dim=1)

            # build positional indices including CLS (0 reserved for CLS)
            cls_idx = torch.zeros((B, 1), dtype=torch.long, device=X.device)
            int_pos = torch.cat([cls_idx, Xp_], dim=1)  # (B, num_embeddings)

            # lookup positional embeddings -> (B, num_embeddings, d_model)
            pos_emb = self.pos_embedding(int_pos)

            # transformer forward -> (B, num_embeddings, d_model)
            transformer_out = self.transformer_encoder(tokens + pos_emb, split_sections)

            if split_sections is None:
                concat.append(transformer_out[:, 0])  # CLS vector (B, d_model)
            else:
                cls_indices = torch.arange(transformer_out.size(0), device=X.device)
                concat.append(transformer_out[cls_indices, 0])

        # If only one window, return directly
        if self.windows == 1:
            return self.cls_head(concat[0])

        # aggregate across windows (original code creates mean over windows but returns last transformer_out CLS)
        concat_agg = torch.stack(concat, dim=0)
        concat_agg = torch.mean(concat_agg, dim=0)  # (B, d_model)

        # NOTE: preserving original final return (as in supplied code).
        # The original author left an alternative (commented) return that used concat_agg.
        return self.cls_head(transformer_out[:, 0])

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
    dropout : float, optional
        Dropout probability applied to attention weights and residual projection.
    flash_att : bool, optional
        Whether to use `torch.nn.functional.scaled_dot_product_attention` when available.
    """

    def __init__(
        self,
        d_model: int,
        n_head: int,
        bias: bool,
        dropout: float = 0.0,
        flash_att: bool = True,
    ) -> None:
        super().__init__()

        assert d_model % n_head == 0, "d_model must be divisible by n_head"

        # qkv and output projection
        self.attn = nn.Linear(d_model, 3 * d_model, bias=bias)
        self.proj = nn.Linear(d_model, d_model, bias=bias)

        # dropout modules
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        self.n_head = n_head
        self.d_model = d_model
        self.dropout = dropout
        self.flash_att = flash_att

    def forward(self, x: torch.Tensor, split_sections=None) -> torch.Tensor:
        """Forward pass for MHSA.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, T, C) where C == d_model.
        split_sections : optional
            Optional integer list/tuple used to split heads along the head-dimension
            for memory/compute trade-offs.

        Returns
        -------
        torch.Tensor
            Output tensor of shape (B, T, C).
        """
        # Optional extra-dim insertion for split_sections path (kept from original logic)
        if split_sections is not None:
            x = torch.unsqueeze(input=x, dim=0)

        B, T, C = x.size()

        # project to q, k, v and reshape for multi-head attention
        q, k, v = self.attn(x).split(self.d_model, dim=2)

        # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        if split_sections is None:
            if self.flash_att:
                y = torch.nn.functional.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    attn_mask=None,
                    dropout_p=self.dropout if self.training else 0,
                    is_causal=False,
                )
            else:
                scores = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
                y = self.attn_dropout(F.softmax(scores, dim=-1)) @ v

            # re-assemble heads -> (B, T, C)
            y = y.transpose(1, 2).contiguous().view(B, T, C)
        else:
            # split along head-dimension for memory constrained attention
            q = torch.tensor_split(q, split_sections, dim=2)
            k = torch.tensor_split(k, split_sections, dim=2)
            v = torch.tensor_split(v, split_sections, dim=2)

            if self.flash_att:
                att_dropout = self.dropout if self.training else 0
                y = torch.cat(
                    [
                        torch.nn.functional.scaled_dot_product_attention(
                            qs,
                            ks,
                            vs,
                            attn_mask=None,
                            dropout_p=att_dropout,
                            is_causal=False,
                        )
                        for qs, ks, vs in zip(q, k, v)
                    ],
                    dim=2,
                )
            else:
                parts = [
                    self.attn_dropout(F.softmax((qs @ ks.transpose(-2, -1)) * (1.0 / math.sqrt(ks.size(-1))), dim=-1)) @ vs
                    for qs, ks, vs in zip(q, k, v)
                ]
                y = torch.cat(parts, dim=2)

            # re-assemble heads, restore removed dimension
            y = y.transpose(1, 2).contiguous().view(B, T, C).squeeze(dim=0)

        # final linear projection + residual dropout
        y = self.resid_dropout(self.proj(y))
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
    dropout : float, optional
        Dropout probability.
    bias : bool, optional
        Whether linear layers use bias.
    """

    def __init__(
        self,
        d_model: int,
        dim_feedforward: Optional[int] = None,
        dropout: float = 0.0,
        bias: bool = False,
    ) -> None:
        super().__init__()

        if dim_feedforward is None:
            raise ValueError("dim_feedforward must be provided")

        self.proj_in = nn.Linear(d_model, dim_feedforward, bias=bias)
        self.gelu = nn.GELU()
        self.proj = nn.Linear(dim_feedforward, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the feed-forward block."""
        x = self.proj_in(x)
        x = self.gelu(x)
        x = self.dropout1(x)
        x = self.proj(x)
        x = self.dropout(x)
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
        dropout: float = 0.0,
        dim_feedforward: Optional[int] = None,
        bias: bool = False,
    ) -> None:
        super().__init__()

        if dim_feedforward is None:
            dim_feedforward = 4 * d_model
            # note: preserve the original behaviour (print) from the provided code
            print(
                "dim_feedforward is set to 4*d_model, the default in Vaswani et al. (Attention is all you need)"
            )

        self.layer_norm_att = _LayerNorm(d_model, bias=bias)
        self.mhsa = _MHSA(d_model, n_head, bias, dropout=dropout, flash_att=True)
        self.layer_norm_ff = _LayerNorm(d_model, bias=bias)
        self.feed_forward = _FeedForward(
            d_model=d_model, dim_feedforward=dim_feedforward, dropout=dropout, bias=bias
        )

    def forward(self, x: torch.Tensor, split_sections=None) -> torch.Tensor:
        """Execute one encoder layer.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape (B, T, d_model).
        split_sections : optional
            Optional split sections forwarded to MHSA.

        Returns
        -------
        torch.Tensor
            Output of the same shape as input.
        """
        x = x + self.mhsa(self.layer_norm_att(x), split_sections)
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
    dropout : float
        Dropout probability.
    bias : bool
        Whether linear layers use bias.
    """

    def __init__(self, n_blocks: int, d_model: int, n_head: int, dropout: float, bias: bool) -> None:
        super().__init__()

        self.encoder_block = nn.ModuleList(
            [
                _TransformerEncoderLayer(
                    d_model=d_model,
                    n_head=n_head,
                    dropout=dropout,
                    dim_feedforward=None,
                    bias=bias,
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

    def forward(self, x: torch.Tensor, split_sections=None) -> torch.Tensor:
        """Forward through all encoder blocks sequentially."""
        for block in self.encoder_block:
            x = block(x, split_sections)
        return x


class _ChannelEncoding(EEGModuleMixin, nn.Sequential):
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
    device : str or torch.device, optional
        Device where the buffer will be allocated.
    """

    def __init__(
        self,
        n_chans: int = None,
        n_times: int = None,
        num_tokens_per_channel: int = 8,
        device: str = "cpu",
    ) -> None:
        super().__init__(n_chans=n_chans, n_times=n_times)

        x_pos_single = torch.zeros((n_chans, n_times), dtype=torch.long, device=device)

        for c in range(n_chans):
            start = c * num_tokens_per_channel + 1
            end = (c + 1) * num_tokens_per_channel + 1
            seq = torch.arange(start, end, device=device, dtype=torch.long)
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