"""
REVE (Representation for EEG with versatile embeddings) model.
Authors: Jonathan Lys (jonathan.lys@imt-atlantique.org)
License: BSD 3 clause
"""

from torch import nn
from braindecode.models.base import EEGModuleMixin

import math
from typing import Union

import torch
import torch.nn.functional as F
from einops import rearrange
from packaging import version
from torch.nn.attention import SDPBackend, sdpa_kernel

try:
    import flash_attn  # type: ignore

    FLASH_AVALIABLE = True
except ImportError:
    FLASH_AVALIABLE = False
    print(
        "flash_attn not found, install it with `pip install flash_attn` if you want to use it"
    )


class REVE(EEGModuleMixin, nn.Module):
    """
    REVE (Representation for EEG with versatile embeddings).

    Model described in [elouahidi2025reve]_.

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
    .. [elouahidi2025reve] Yassine El Ouahidi · Jonathan Lys · Philipp Thölke · Nicolas Farrugia · Bastien Pasdeloup · Vincent Gripon · Karim Jerbi · Giulia Lioi (2025).
       REVE: A Foundation Model for EEG - Adapting to Any Setup with Large-Scale Pretraining on 25,000 Subjects.
       NeurIPS 2025.
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
    ):
        super().__init__(
            n_outputs=n_outputs,
            n_chans=n_chans,
            chs_info=chs_info,
            n_times=n_times,
            input_window_seconds=input_window_seconds,
            sfreq=sfreq,
        )
        self.config = _ReveConfig(
            embed_dim=embed_dim,
            depth=depth,
            heads=heads,
            head_dim=head_dim,
            mlp_dim_ratio=mlp_dim_ratio,
            use_geglu=use_geglu,
            freqs=freqs,
            noise_ratio=noise_ratio,
            patch_size=patch_size,
            patch_overlap=patch_overlap,
        )

        self.embed_dim = self.config.embed_dim
        self.freqs = self.config.freqs
        self.patch_size = self.config.patch_size
        self.overlap_size = self.config.patch_overlap
        self.noise_ratio = self.config.noise_ratio

        self.transformer = TransformerBackbone(
            dim=self.config.embed_dim,
            depth=self.config.depth,
            heads=self.config.heads,
            head_dim=self.config.head_dim,
            mlp_dim=int(self.config.embed_dim * self.config.mlp_dim_ratio),
            geglu=self.config.use_geglu,
        )

        self.to_patch_embedding = patch_embedding(self.embed_dim, self.patch_size)
        self.fourier4d = FourierEmb4D(self.embed_dim, freqs=self.freqs)
        self.mlp4d = mlp_pos_embedding(self.embed_dim)
        self.ln = nn.LayerNorm(self.embed_dim)

        self.final_layer = nn.Identity()

        self.cls_query_token = nn.Parameter(torch.randn(1, 1, self.embed_dim))

        self._position_bank = None

    def get_positions(self, channel_names: list[str]) -> torch.Tensor:
        """
        Get the 3D positions for the given channel names using the REVE position bank.

        Args:
            channel_names (list[str]): List of channel names for which to retrieve positions.
        Returns:
            torch.Tensor: Tensor of shape (num_channels, 3) containing the (x, y, z) positions.
        """

        if self._position_bank is None:
            try:
                from transformers import AutoModel

                self._position_bank = AutoModel.from_pretrained(
                    "brain-bzh/reve-positions", trust_remote_code=True
                )
            except ImportError:
                raise ImportError(
                    "Please install transformers to use the REVE position bank: pip install transformers"
                )

        return self._position_bank.forward(channel_names)

    def forward(
        self,
        eeg: torch.Tensor,
        pos: torch.Tensor,
        return_output: bool = False,
    ) -> Union[torch.Tensor, list[torch.Tensor]]:
        """
        Forward pass of the model.
        Args:
            eeg (torch.Tensor): Input EEG tensor of shape (batch_size, channels, sequence_length).
            pos (torch.Tensor): Position tensor of shape (batch_size, channels, 3) representing (x, y, z) coordinates.
            return_output (bool, optional): If True, returns the output from the transformer directly.
                If False, applies the final layer and returns the processed output. Default is False.
        Returns:
            Union[torch.Tensor, list[torch.Tensor]]: The output tensor(s) from the model. If `return_output` is True,
                returns the transformer output; otherwise, returns the output after the final layer.
        """

        eeg = eeg.float()
        patches = eeg.unfold(
            dimension=2, size=self.patch_size, step=self.patch_size - self.overlap_size
        )
        _b, c, h, _p = patches.shape

        pos = FourierEmb4D.add_time_patch(pos, h)
        pos_embed = self.ln(self.fourier4d(pos) + self.mlp4d(pos))

        x = (
            rearrange(
                self.to_patch_embedding(patches),
                "b c h e -> b (c h) e",
                c=c,
                h=h,
                e=self.embed_dim,
            )
            + pos_embed
        )
        x = self.transformer(x, return_output)
        if return_output:
            return x

        x = rearrange(x, "b (c h) e -> b c h e", b=_b, c=c, h=h, e=self.embed_dim)
        x = self.final_layer(x)
        return x

    def attention_pooling(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply attention pooling on the sequence dimension of x.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, S, E), where B is the batch size,
                              C is the number of channels, S is the sequence length,
                              and E is the embedding dimension.
        Returns:
            torch.Tensor: Output tensor of shape (B, E) after attention pooling.
        """

        b, c, s, e = x.shape
        x = rearrange(x, "b c s e -> b (c s) e")  # (B, C*S, E)
        query_output = self.cls_query_token.expand(b, -1, -1)  # (B, 1, E)
        attention_scores = torch.matmul(query_output, x.transpose(-1, -2)) / (
            self.embed_dim**0.5
        )  # (B, 1, C*S)
        attention_weights = torch.softmax(attention_scores, dim=-1)  # (B, 1, C*S)
        out = torch.matmul(attention_weights, x).squeeze(1)  # (B, E)
        return out


# Configuration class for REVE


class _ReveConfig:
    model_type = "reve"

    def __init__(
        self,
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
        **kwargs,
    ):
        self.embed_dim = embed_dim
        self.depth = depth
        self.heads = heads
        self.head_dim = head_dim
        self.mlp_dim_ratio = mlp_dim_ratio
        self.use_geglu = use_geglu
        self.freqs = freqs
        self.noise_ratio = noise_ratio
        self.patch_size = patch_size
        self.patch_overlap = patch_overlap


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


class FlashAttention(nn.Module):
    def __init__(self, num_heads: int):
        super().__init__()
        self.num_heads = num_heads

    def forward(self, qkv: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = qkv.shape[:2]

        qkv = rearrange(
            qkv, "b n (three h d) -> (b n) three h d", three=3, h=self.num_heads
        )
        cu_seqlens = torch.arange(
            0, (batch_size + 1) * seq_len, seq_len, dtype=torch.int32, device=qkv.device
        )

        out = flash_attn.flash_attn_varlen_qkvpacked_func(
            qkv,
            cu_seqlens,
            seq_len,  # max seq len
            0.0,
            causal=False,
        )

        out = rearrange(out, "(b n) h d -> b n (h d)", b=batch_size)
        return out


class Attention(nn.Module):
    """
    Common API for both classical and flash attention
    """

    def __init__(
        self, dim: int, heads: int = 8, head_dim: int = 64, use_flash: bool = True
    ):
        super().__init__()
        inner_dim = head_dim * heads
        self.heads = heads
        self.scale = head_dim**-0.5
        self.norm = RMSNorm(dim)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

        self.use_flash = use_flash
        self.attend = (
            FlashAttention(self.heads)
            if use_flash
            else ClassicalAttention(self.heads, use_sdpa=True)
        )

    def forward(self, x):
        x = self.norm(x)
        qkv = self.to_qkv(x)
        out = self.attend(qkv)
        return self.to_out(out)


#################################################################################
#                                  Transformer                                  #
#################################################################################


class TransformerBackbone(nn.Module):
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
                            use_flash=FLASH_AVALIABLE,
                        ),
                        FeedForward(self.dim, mlp_dim, geglu),
                    ]
                )
            )

    def forward(
        self, x, return_out_layers=False
    ) -> Union[torch.Tensor, list[torch.Tensor]]:
        out_layers = [x] if return_out_layers else None
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

        Args:
        - pos (Tensor): Input tensor of shape (B, C, 3), where B is the batch size,
        C is the number of channels, and 3 represents x, y, z.
        - num_patches (int): The number of time patches.

        Returns:
        - Tensor: Output tensor of shape (B, C * num_patches, 4), where each position is repeated with each time value.
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
