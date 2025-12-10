"""
REVE (Representation for EEG with versatile embeddings) model.
Authors: Jonathan Lys (jonathan.lys@imt-atlantique.org)
License: BSD 3 clause
"""

import json
import math
from typing import Union

import requests
import torch
import torch.nn.functional as F
from einops import rearrange
from packaging import version
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel

from braindecode.models.base import EEGModuleMixin


class REVE(EEGModuleMixin, nn.Module):
    r"""
    REVE (Representation for EEG with versatile embeddings) model from El Ouahidi et al. (2025) [reve]_.

    :bdg-danger:`Large Brain Model`

    This implementation is based on the one available at https://huggingface.co/brain-bzh/reve-base (although it is gated).

    .. figure:: https://brain-bzh.github.io/reve/static/images/architecture.png
        :align: center
        :alt:  REVE Training pipeline overview
        :width: 680px

    REVE tokenizes EEG signal into latent patches that get fed into a Transformer encoder model.
    The model uses a 4D positional encoding to contextualize the patches in space and time.

    .. rubric:: Macro Components

    - ``REVE.tokenization`` **patch extraction**

      *Operations.* The EEG signal is split into overlapping patches along the time dimension,
      generating :math:`p = \left\lceil \frac{T - w}{w - o} \right\rceil + \mathbbold{1} \left[ (T - w) \bmod (w - o) \neq 0 \right]` patches of size :math:`w` with overlap :math:`o`, where :math:`T` is the length of the signal.

    - ``REVE.4DPE`` **4D positional embedding**

        *Operations.* The 4D positional embedding module combines a Fourier-based positional embedding and an MLP-based positional embedding.
        The Fourier-based embedding encodes the 3D spatial positions of the EEG channels along with the temporal position of each patch using sinusoidal functions across multiple frequencies.
        The MLP-based embedding processes the 4D positions through a small MLP to capture additional positional information.
        The outputs of both embeddings are summed and normalized to produce the final 4D positional embedding.

    - ``REVE.transformer`` **Transformer encoder**

        *Operations.* The Transformer encoder consists of multiple layers of multi-head self-attention and feed-forward neural networks.
        The chosen components are: GEGLU activation, RMSNorm normalization, and Flash Attention for efficient computation.
        If Flash Attention is not available, it falls back to a standard scaled dot-product attention implementation.

    - ``REVE.final_layer`` **Final prediction layer**

        *Operations.* A linear layer that maps the flattened output of the Transformer encoder to the desired output shape for the specific EEG task.
        We also prepend a Layer Norm for stability.


    Parameters
    ----------
    embed_dim : int
        Dimension of the embedding.
    depth : int
        Number of transformer layers.
    heads : int
        Number of attention heads.
    head_dim : int
        Dimension of each attention head.
    mlp_dim_ratio : float
        Ratio to compute the hidden dimension of the MLP from the embed_dim.
    use_geglu : bool
        Whether to use GEGLU activation in the MLP (True) or GELU (False).
    freqs : int
        Number of frequencies for the Fourier positional embedding.
    noise_ratio : float
        Ratio of noise to add to the input during training.
    patch_size : int
        Size of each patch for patch embedding.
    patch_overlap : int
        Overlap size between patches.

    References
    ----------
    .. [reve] El Ouahidi, Y., Lys, J., Thölke, P., Farrugia, N., Pasdeloup, B., Gripon, V., Jerbi, K. & Lioi, G. (2025).
        REVE: A Foundation Model for EEG - Adapting to Any Setup with Large-Scale Pretraining on 25,000 Subjects. NeurIPS.
        Online: `https://arxiv.org/abs/2510.21585`
    """

    def __init__(
        self,
        n_outputs=None,
        n_chans=None,
        chs_info=None,
        n_times=None,
        input_window_seconds=None,
        sfreq=None,
        # REVE specific parameters
        embed_dim=512,
        depth=22,
        heads=8,
        head_dim=64,
        mlp_dim_ratio=2.66,
        use_geglu=True,
        freqs=4,
        noise_ratio=0.0025,
        patch_size=200,
        patch_overlap=20,
        attention_pooling: bool = False,
    ):
        super().__init__(
            n_outputs=n_outputs,
            n_chans=n_chans,
            chs_info=chs_info,
            n_times=n_times,
            input_window_seconds=input_window_seconds,
            sfreq=sfreq,
        )

        self.embed_dim = embed_dim
        self.freqs = freqs
        self.patch_size = patch_size
        self.overlap_size = patch_overlap
        self.noise_ratio = noise_ratio
        self.depth = depth
        self.heads = heads
        self.head_dim = head_dim
        self.mlp_dim_ratio = mlp_dim_ratio
        self.use_geglu = use_geglu

        self.use_attention_pooling = attention_pooling

        self.to_patch_embedding = patch_embedding(self.embed_dim, self.patch_size)

        self.fourier4d = FourierEmb4D(self.embed_dim, freqs=self.freqs)
        self.mlp4d = mlp_pos_embedding(self.embed_dim)
        self.ln = nn.LayerNorm(self.embed_dim)  # 4DPE module layernorm

        self.transformer = TransformerBackbone(
            dim=self.embed_dim,
            depth=self.depth,
            heads=self.heads,
            head_dim=self.head_dim,
            mlp_dim=int(self.embed_dim * self.mlp_dim_ratio),
            geglu=self.use_geglu,
        )

        final_dim = self._get_flattened_output_dim()
        self.final_layer = nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(final_dim),
            nn.Linear(final_dim, self.n_outputs),
        )

        self.cls_query_token = nn.Parameter(torch.randn(1, 1, self.embed_dim))

        self._position_bank = RevePositionBank()

    def _get_flattened_output_dim(self) -> int:
        """Helper function to compute the flattened output dimension after the transformer."""

        if self.use_attention_pooling:
            return self.embed_dim

        n_patches = math.ceil(
            (self.n_times - self.patch_size) / (self.patch_size - self.overlap_size)
        )

        if (self.n_times - self.patch_size) % (
            self.patch_size - self.overlap_size
        ) == 0:
            n_patches += 1

        flat_dim = self.n_chans * n_patches * self.embed_dim
        return flat_dim

    def get_positions(self, channel_names: list[str]) -> torch.Tensor:
        """Fetch channel positions from the position bank.

        Downloads and caches the position bank on first call.

        Parameters
        ----------
        channel_names : list[str]
            List of channel names for which to fetch positions.

        Returns
        -------
        torch.Tensor
            Tensor of shape (num_channels, 3) containing the (x, y, z) positions of the channels.
        """

        return self._position_bank.forward(channel_names)

    def forward(
        self,
        eeg: torch.Tensor,
        pos: torch.Tensor,
        return_output: bool = False,
    ) -> Union[torch.Tensor, list[torch.Tensor]]:
        """
        Forward pass of the model.

        Goes through the following steps:
        1. Patch extraction from the EEG signal.
        2. 4D positional embedding computation.
        3. Transformer encoding.
        4. Final layer processing (if `return_output` is False).

        Parameters:
        ----------
        eeg : torch.Tensor
            Input EEG tensor of shape (batch_size, channels, sequence_length).
        pos : torch.Tensor
            Position tensor of shape (batch_size, channels, 3) representing (x, y, z) coordinates.
        return_output : bool, optional
            If True, returns the output from the transformer directly.
            If False, applies the final layer and returns the processed output. Default is False.

        Returns:
        -------
        Union[torch.Tensor, list[torch.Tensor]]
            The output tensor(s) from the model. If `return_output` is True,
            returns the transformer output; otherwise, returns the output after the final layer.
        """

        patches = eeg.unfold(
            dimension=2,
            size=self.patch_size,
            step=self.patch_size - self.overlap_size,
        )
        batch_size, channel, heights, _n_patches = patches.shape

        pos = FourierEmb4D.add_time_patch(pos, heights)
        pos_embed = self.ln(self.fourier4d(pos) + self.mlp4d(pos))

        x = (
            rearrange(
                self.to_patch_embedding(patches),
                "b c h e -> b (c h) e",
                c=channel,
                h=heights,
                e=self.embed_dim,
            )
            + pos_embed
        )
        x = self.transformer(x, return_output)

        if return_output:
            return x

        x = rearrange(
            x,
            "b (c h) e -> b c h e",
            b=batch_size,
            c=channel,
            h=heights,
            e=self.embed_dim,
        )

        if self.use_attention_pooling:
            x = self.attention_pooling(x)

        x = self.final_layer(x)
        return x

    def attention_pooling(self, x: torch.Tensor) -> torch.Tensor:
        """Apply attention pooling on the sequence dimension of x.

        Parameters:
        ----------
        x : torch.Tensor
            Input tensor of shape (B, C, S, E), where B is the batch size,
            C is the number of channels, S is the sequence length,
            and E is the embedding dimension. Typically the output from the transformer.
        Returns:
        -------
        torch.Tensor
            Output tensor of shape (B, E) after attention pooling.
        """

        batch_size, _channels, _seq_len, _embed_dim = x.shape
        x = rearrange(x, "b c s e -> b (c s) e")  # (B, C*S, E)
        query_output = self.cls_query_token.expand(batch_size, -1, -1)  # (B, 1, E)
        attention_scores = torch.matmul(query_output, x.transpose(-1, -2)) / (
            self.embed_dim**0.5
        )  # (B, 1, C*S)
        attention_weights = torch.softmax(attention_scores, dim=-1)  # (B, 1, C*S)
        out = torch.matmul(attention_weights, x).squeeze(1)  # (B, E)
        return out


#################################################################################
#                                  Layers                                       #
#################################################################################


class GEGLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gates = x.chunk(2, dim=-1)
        return F.gelu(gates) * x


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, geglu: bool):
        super().__init__()
        self.net = nn.Sequential(
            RMSNorm(dim),
            nn.Linear(dim, hidden_dim * 2 if geglu else hidden_dim, bias=False),
            GEGLU() if geglu else nn.GELU(),
            nn.Linear(hidden_dim, dim, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


#################################################################################
#                                  Attention                                    #
#################################################################################


class ClassicalAttention(nn.Module):
    def __init__(self, heads: int, use_sdpa: bool = True):
        super().__init__()
        self.use_sdpa = use_sdpa
        self.heads = heads
        if self.use_sdpa:
            assert version.parse(torch.__version__) >= version.parse("2.2.0"), (
                "in order to use sdpa, you must be using pytorch 2.2 or above"
            )

    def forward(self, qkv: torch.Tensor) -> torch.Tensor:
        q, k, v = qkv.chunk(3, dim=-1)
        q, k, v = (
            rearrange(t, "b n (h d) -> b h n d", h=self.heads) for t in (q, k, v)
        )

        if self.use_sdpa:  # SDPA Implementation
            with sdpa_kernel(
                [
                    SDPBackend.FLASH_ATTENTION,
                    SDPBackend.EFFICIENT_ATTENTION,
                    SDPBackend.MATH,
                ]
            ):
                out = F.scaled_dot_product_attention(q, k, v)
        else:  # Naive Implementation
            _, _, scale = q.shape[-2], q.device, q.shape[-1] ** -0.5
            dots = torch.matmul(q, k.transpose(-1, -2)) * scale
            attn = nn.Softmax(dim=-1)(dots)
            out = torch.matmul(attn, v)

        out = rearrange(out, "b h n d -> b n (h d)")
        return out


class Attention(nn.Module):
    """
    Multi-head self-attention layer with RMSNorm.
    """

    def __init__(self, dim: int, heads: int = 8, head_dim: int = 64):
        super().__init__()
        inner_dim = head_dim * heads
        self.heads = heads
        self.scale = head_dim**-0.5
        self.norm = RMSNorm(dim)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

        self.attend = ClassicalAttention(self.heads, use_sdpa=True)

    def forward(self, x):
        x = self.norm(x)
        qkv = self.to_qkv(x)
        out = self.attend(qkv)
        return self.to_out(out)


#################################################################################
#                                  Transformer                                  #
#################################################################################


class TransformerBackbone(nn.Module):
    """
    Transformer backbone consisting of multiple layers of attention and feed-forward networks.
    """

    def __init__(self, dim, depth, heads, head_dim, mlp_dim, geglu):
        super().__init__()
        self.dim = dim
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Attention(
                            self.dim,
                            heads=heads,
                            head_dim=head_dim,
                        ),
                        FeedForward(self.dim, mlp_dim, geglu),
                    ]
                )
            )

    def forward(
        self, x, return_out_layers=False
    ) -> Union[torch.Tensor, list[torch.Tensor]]:
        out_layers = [x] if return_out_layers else []
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
            if return_out_layers:
                out_layers.append(x)
        return out_layers if return_out_layers else x


##################################################################################
#                                       4D PE                                    #
##################################################################################


class FourierEmb4D(nn.Module):
    """
    Fourier positional embedding for 4D positions (x, y, z, t).
    This version allows for a reduced number of frequencies (n_freqs),
    and ensures the output embedding has the specified dimension.
    """

    def __init__(
        self, dimension: int, freqs: int, increment_time=0.1, margin: float = 0.4
    ):
        super().__init__()
        self.dimension = dimension
        self.freqs = freqs
        self.increment_time = increment_time
        self.margin = margin

    def forward(self, positions_: torch.Tensor) -> torch.Tensor:
        positions = positions_.clone()
        positions[:, :, -1] *= self.increment_time
        *U, _ = positions.shape

        freqs_w = torch.arange(self.freqs).to(positions)
        freqs_z = freqs_w[:, None]
        freqs_y = freqs_z[:, None]
        freqs_x = freqs_y[:, None]
        width = 1 + 2 * self.margin
        positions = positions + self.margin
        p_x = 2 * math.pi * freqs_x / width
        p_y = 2 * math.pi * freqs_y / width
        p_z = 2 * math.pi * freqs_z / width
        p_w = 2 * math.pi * freqs_w / width
        positions = positions[..., None, None, None, None, :]
        loc = (
            positions[..., 0] * p_x
            + positions[..., 1] * p_y
            + positions[..., 2] * p_z
            + positions[..., 3] * p_w
        ).view(*U, -1)
        if self.dimension != 512:  # noqa
            _, _, hd = loc.shape
            diff = hd - self.dimension // 2
            loc = loc[:, :, :-diff]
        emb = torch.cat([torch.cos(loc), torch.sin(loc)], dim=-1)
        return emb

    @classmethod
    def add_time_patch(cls, pos: torch.Tensor, num_patches: int) -> torch.Tensor:
        """
        Expand the position tensor by adding a time dimension, handling batched data.

        Parameters:
        ----------
        pos : torch.Tensor
            Input tensor of shape (B, C, 3), where B is the batch size,
            C is the number of channels, and 3 represents x, y, z.
        num_patches : int
            The number of time patches.

        Returns:
        -------
        torch.Tensor
            Output tensor of shape (B, C * num_patches, 4), where each position is repeated with each time value.
        """
        B, C, _ = pos.shape
        # Repeat each position for each time step
        pos_repeated = pos.unsqueeze(2).repeat(
            1, 1, num_patches, 1
        )  # Shape: (B, C, num_patches, 3)
        # Generate time values with the specified increment
        time_values = torch.arange(
            0, num_patches, 1, device=pos.device
        ).float()  # Shape: (num_patches,)
        time_values = time_values.view(1, 1, num_patches, 1).expand(
            B, C, num_patches, 1
        )  # (B, C, num_patches, 1)
        # Concatenate the repeated positions with the time values along the last dimension
        pos_with_time = torch.cat(
            (pos_repeated, time_values), dim=-1
        )  # Shape: (B, C, num_patches, 4)
        # Reshape to (B, C * num_patches, 4)
        pos_with_time = pos_with_time.view(B, C * num_patches, 4)

        return pos_with_time


def patch_embedding(embed_dim, patch_size):
    to_patch_embedding = nn.Sequential(nn.Linear(patch_size, embed_dim))
    return to_patch_embedding


def mlp_pos_embedding(embed_dim):
    mlp_pos_embedding = nn.Sequential(
        nn.Linear(4, embed_dim, bias=False), nn.GELU(), nn.LayerNorm(embed_dim)
    )
    return mlp_pos_embedding


class RevePositionBank(torch.nn.Module):
    def __init__(self):
        super().__init__()

        url = "https://huggingface.co/brain-bzh/reve-positions/resolve/main/positions.json"
        response = requests.get(url)
        config = json.loads(response.text)

        self.position_names = config.keys()
        self.mapping = {name: i for i, name in enumerate(self.position_names)}
        positions = torch.tensor(list(config.values()))
        self.register_buffer("embedding", positions)
        assert self.embedding.shape == (len(self.mapping), 3)

    def forward(self, channel_names: list[str]):
        indices = [self.mapping[q] for q in channel_names if q in self.mapping]

        if len(indices) < len(channel_names):
            print(
                f"Found {len(indices)} positions out of {len(channel_names)} channels"
            )

        indices = torch.tensor(indices, device=self.embedding.device)

        return self.embedding[indices]

    def get_all_positions(self):
        return self.position_names
