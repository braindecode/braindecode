"""
REVE (Representation for EEG with versatile embeddings) model.
Authors: Jonathan Lys (jonathan.lys@imt-atlantique.org)
License: BSD 3 clause
"""

import json
import math
from typing import Optional, Union

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
    **R**\ epresentation for **E**\ EG with **V**\ ersatile **E**\ mbeddings (REVE) from El Ouahidi et al. (2025) [reve]_.

    :bdg-danger:`Foundation Model` :bdg-info:`Attention/Transformer`

    .. figure:: https://brain-bzh.github.io/reve/static/images/architecture.png
        :align: center
        :alt:  REVE Training pipeline overview
        :width: 1000px

    Foundation models have transformed AI by reducing reliance on task-specific data through large-scale
    pretraining. While successful in language and vision, their adoption in EEG has lagged due to the
    heterogeneity of public datasets, collected under varying protocols, devices, and electrode configurations.
    Existing EEG foundation models struggle to generalize across these variations, often restricting pretraining
    to a single setup and resulting in suboptimal performance—particularly under linear probing.

    REVE is a pretrained model explicitly designed to generalize across diverse EEG signals. It introduces
    a **4D positional encoding** scheme that enables processing signals of arbitrary length and electrode
    arrangement. Using a masked autoencoding objective, REVE was pretrained on over **60,000 hours** of EEG
    data from **92 datasets** spanning **25,000 subjects**—the largest EEG pretraining effort to date.

    .. rubric:: Core Innovation: Setup-Agnostic Processing

    Prior EEG foundation models (:class:`~braindecode.models.Labram`, :class:`~braindecode.models.BIOT`) rely on 
    fixed positional embeddings, making direct transfer to unseen electrode layouts infeasible. CBraMod uses 
    convolution-based positional encoding that requires fine-tuning when adapting to new configurations. 
    As noted in the CBraMod paper: *"fixing the pre-trained parameters during training on downstream 
    datasets will lead to a very large performance decline. CBraMod cannot currently serve as a fixed-parameter feature extractor"*.

    REVE's 4D positional encoding jointly encodes spatial :math:`(x, y, z)` and temporal :math:`(t)` positions
    using Fourier embeddings, enabling true cross-configuration transfer without retraining. The fourier embedding
    have inspirion on brainmodule [brainmodule]_.

    .. rubric:: Linear Probing Performance

    A key advantage of REVE is producing high-quality latent spaces without heavy fine-tuning. Under linear
    probing (frozen encoder), REVE achieves an average balanced accuracy of **0.654** across 10 downstream
    tasks, compared to **0.501** for CBraMod under matched evaluation. This enables practical deployment
    in low-data scenarios where extensive fine-tuning is not feasible.

    .. rubric:: Architecture

    The model adopts modern Transformer components validated through ablation studies:

    - **Normalization**: RMSNorm outperforms LayerNorm (avg. 0.596 vs 0.579)
    - **Activation**: GEGLU outperforms GELU (avg. 0.596 vs 0.558)
    - **Attention**: Flash Attention via PyTorch's SDPA
    - **Masking ratio**: 55% optimal for spatio-temporal block masking

    These choices align with best practices from large language models and were empirically validated
    on EEG data.

    .. rubric:: Secondary Loss

    A secondary reconstruction objective using attention pooling across layers prevents over-specialization
    in the final layer. This pooling acts as an information bottleneck, forcing the model to distill key
    information from the entire sequence. Ablations show this loss is crucial for linear probing quality:
    removing it drops average performance from 0.665 to 0.558 under frozen evaluation.

    .. rubric:: Macro Components

    - ``REVE.to_patch_embedding`` **Patch Tokenization**

      The EEG signal is split into overlapping patches along the time dimension, generating
      :math:`p = \left\lceil \frac{T - w}{w - o} \right\rceil + \mathbf{1}[(T - w) \bmod (w - o) \neq 0]`
      patches of size :math:`w` with overlap :math:`o`, where :math:`T` is the signal length.
      Each patch is linearly projected to the embedding dimension.

    - ``REVE.fourier4d`` + ``REVE.mlp4d`` **4D Positional Embedding (4DPE)**

      The 4DPE encodes each token's 4D coordinates :math:`(x, y, z, t)` where :math:`(x, y, z)` are the
      3D spatial coordinates from a standardized electrode position bank, and :math:`t` is the temporal
      patch index. The encoding combines:

      1. **Fourier embedding**: Sinusoidal encoding across multiple frequencies for smooth interpolation
         to unseen positions
      2. **MLP embedding**: :class:`~torch.nn.Linear` (4 → embed_dim) → :class:`~torch.nn.GELU` → :class:`~torch.nn.LayerNorm` for learnable refinement

      Both components are summed and normalized. The 4DPE adds negligible computational overhead,
      scaling linearly with the number of tokens.

    - ``REVE.transformer`` **Transformer Encoder**

      Pre-LN Transformer with multi-head self-attention (:class:`~torch.nn.RMSNorm`), feed-forward networks (GEGLU
      activation), and residual connections. Default configuration: 22 layers, 8 heads, 512 embedding
      dimension (~72M parameters).

    - ``REVE.final_layer`` **Classification Head**

      Two modes: (1) Flatten all tokens → :class:`~torch.nn.LayerNorm` → :class:`~torch.nn.Linear`, or (2) Attention pooling with a
      learnable query token attending to all encoder outputs.

    .. rubric:: Known Limitations

    - **Sparse electrode setups**: Performance degrades with very few channels. On motor imagery,
      accuracy drops from 0.824 (64 channels) to 0.660 (1 channel). For tasks requiring broad
      spatial coverage (e.g., imagined speech), performance with <4 channels approaches chance level.
    - **Demographic bias**: The pretraining corpus aggregates publicly available datasets, most
      originating from North America and Europe, resulting in limited demographic diversity.
    - **Clinical validation**: While REVE achieves strong benchmark results, validation in
      real-world clinical settings remains future work.

    .. rubric:: Pretrained Weights

    Weights are available on `HuggingFace <https://huggingface.co/collections/brain-bzh/reve>`_,
    but you must agree to the data usage terms before downloading:

    - ``brain-bzh/reve-base``: 72M parameters, 512 embedding dim, 22 layers (~260 A100 GPU hours)
    - ``brain-bzh/reve-large``: Larger variant with 1250 embedding dim

    .. rubric:: Usage

    .. code-block:: python

        from braindecode.models import REVE

        model = REVE(
            n_outputs=4,  # e.g., 4-class motor imagery
            n_chans=22,
            n_times=1000,  # 5 seconds at 200 Hz
            sfreq=200,
            chs_info=[{'ch_name': 'C3'}, {'ch_name': 'C4'}, ...]  
        )

        # Forward pass: (batch, n_chans, n_times) -> (batch, n_outputs)
        output = model(eeg_data, pos=channel_positions)

    .. warning::

        Input data must be sampled at **200 Hz** to match pretraining. The model applies
        z-score normalization followed by clipping at 15 standard deviations internally
        during pretraining—users should apply similar preprocessing.

    Parameters
    ----------
    embed_dim : int, default=512
        Embedding dimension. Use 512 for REVE-Base, 1250 for REVE-Large.
    depth : int, default=22
        Number of Transformer layers.
    heads : int, default=8
        Number of attention heads.
    head_dim : int, default=64
        Dimension per attention head.
    mlp_dim_ratio : float, default=2.66
        FFN hidden dimension ratio: ``mlp_dim = embed_dim × mlp_dim_ratio``.
    use_geglu : bool, default=True
        Use GEGLU activation (recommended) or standard GELU.
    freqs : int, default=4
        Number of frequencies for Fourier positional embedding.
    patch_size : int, default=200
        Temporal patch size in samples (200 samples = 1 second at 200 Hz).
    patch_overlap : int, default=20
        Overlap between patches in samples.
    attention_pooling : bool, default=False
        Use attention-based pooling instead of flattening.

    References
    ----------
    .. [reve] El Ouahidi, Y., Lys, J., Thölke, P., Farrugia, N., Pasdeloup, B.,
       Gripon, V., Jerbi, K. & Lioi, G. (2025). REVE: A Foundation Model for EEG -
       Adapting to Any Setup with Large-Scale Pretraining on 25,000 Subjects.
       The Thirty-Ninth Annual Conference on Neural Information Processing Systems.
       https://openreview.net/forum?id=ZeFMtRBy4Z
    .. [brainmodule] Défossez, A., Caucheteux, C., Rapin, J., Kabeli, O., & King, J. R.
       (2023). Decoding speech perception from non-invasive brain recordings. Nature
       Machine Intelligence, 5(10), 1097-1107.

    Notes
    -----
    The position bank is downloaded from HuggingFace on first initialization, mapping
    standard 10-20/10-10/10-05 electrode names to 3D coordinates. This enables the
    4D positional encoding to generalize across electrode configurations without
    requiring matched layouts between pretraining and downstream tasks.
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
        embed_dim: int = 512,
        depth: int = 22,
        heads: int = 8,
        head_dim: int = 64,
        mlp_dim_ratio: float = 2.66,
        use_geglu: bool = True,
        freqs: int = 4,
        patch_size: int = 200,
        patch_overlap: int = 20,
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

        if self.use_attention_pooling:
            self.final_layer = nn.Sequential(
                nn.LayerNorm(self.embed_dim),
                nn.Linear(self.embed_dim, self.n_outputs),
            )
        else:
            self.final_layer = nn.Sequential(
                nn.Flatten(),
                nn.LayerNorm(final_dim),
                nn.Linear(final_dim, self.n_outputs),
            )

        self.cls_query_token = nn.Parameter(torch.randn(1, 1, self.embed_dim))

        self._position_bank = RevePositionBank()

        self.default_pos = None
        if chs_info is not None:
            self.default_pos = self.get_positions([ch["ch_name"] for ch in chs_info])

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
        """Fetch channel positions from the position bank. The position bank is downloaded when the model is instantiated.

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
        pos: Optional[torch.Tensor] = None,
        return_output: bool = False,
    ) -> Union[torch.Tensor, list[torch.Tensor]]:
        """
        Forward pass of the model.

        Goes through the following steps:
        1. Patch extraction from the EEG signal.
        2. 4D positional embedding computation.
        3. Transformer encoding.
        4. Final layer processing (if `return_output` is False).

        Parameters
        ----------
        eeg : torch.Tensor
            Input EEG tensor of shape (batch_size, channels, sequence_length).
        pos : torch.Tensor
            Position tensor of shape (batch_size, channels, 3) representing (x, y, z) coordinates.
        return_output : bool, optional
            If True, returns the output from the transformer directly.
            If False, applies the final layer and returns the processed output. Default is False.

        Returns
        -------
        Union[torch.Tensor, list[torch.Tensor]]
            - If `return_output` is False: Returns a single `torch.Tensor` (output after final layer).
            - If `return_output` is True: Returns a `list[torch.Tensor]` (outputs from transformer layers).

            The output tensor(s) from the model. If `return_output` is True,
            returns the transformer output; otherwise, returns the output after the final layer.
        """

        patches = eeg.unfold(
            dimension=2,
            size=self.patch_size,
            step=self.patch_size - self.overlap_size,
        )
        batch_size, channel, n_patches, _ = patches.shape

        if pos is None:
            if self.default_pos is None:
                raise ValueError(
                    "No positions provided and no default positions available. Please provide channel positions."
                )
            pos = self.default_pos.expand(batch_size, -1, -1).to(eeg.device)
        pos = FourierEmb4D.add_time_patch(pos, n_patches)
        pos_embed = self.ln(self.fourier4d(pos) + self.mlp4d(pos))

        # Patch embedding: (batch, channels, n_patches, patch_size) -> (batch, channels, n_patches, embed_dim)
        patch_embeddings = self.to_patch_embedding(patches)

        # Flatten spatial and temporal dimensions into a single sequence dimension
        # (batch, channels, n_patches, embed_dim) -> (batch, channels * n_patches, embed_dim)
        # This creates a sequence of tokens where each token represents one patch from one channel
        x = (
            rearrange(
                patch_embeddings,
                "batch chan patch emb -> batch (chan patch) emb",
                chan=channel,
                patch=n_patches,
                emb=self.embed_dim,
            )
            + pos_embed
        )
        x = self.transformer(x, return_output)

        if return_output:
            return x

        # Reshape back from flattened sequence to separate channel and temporal dimensions
        # (batch, channels * n_patches, embed_dim) -> (batch, channels, n_patches, embed_dim)
        # This recovers the spatio-temporal structure for downstream processing
        x = rearrange(
            x,
            "batch (chan patch) emb -> batch chan patch emb",
            batch=batch_size,
            chan=channel,
            patch=n_patches,
            emb=self.embed_dim,
        )

        if self.use_attention_pooling:
            x = self._attention_pooling(x)

        x = self.final_layer(x)
        return x

    def _attention_pooling(self, x: torch.Tensor) -> torch.Tensor:
        """Apply attention pooling on the sequence dimension of x.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, C, S, E), where B is the batch size,
            C is the number of channels, S is the sequence length,
            and E is the embedding dimension. Typically the output from the transformer.

        Returns
        -------
        torch.Tensor
            Output tensor of shape (B, E) after attention pooling.
        """

        batch_size, n_channels, seq_len, embed_dim = x.shape
        # Flatten channel and sequence dimensions for attention pooling
        # (batch, channels, seq_len, embed_dim) -> (batch, channels * seq_len, embed_dim)
        x = rearrange(
            x,
            "batch chan seq emb -> batch (chan seq) emb",
            chan=n_channels,
            seq=seq_len,
            emb=embed_dim,
        )
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
        # Split concatenated QKV into separate tensors
        # qkv shape: (batch, seq_len, 3 * heads * head_dim)
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape for multi-head attention: split last dim into (heads, head_dim)
        # (batch, seq_len, heads * head_dim) -> (batch, heads, seq_len, head_dim)
        q, k, v = (
            rearrange(
                t,
                "batch seq (heads dim) -> batch heads seq dim",
                heads=self.heads,
            )
            for t in (q, k, v)
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

        # Merge heads back: (batch, heads, seq_len, head_dim) -> (batch, seq_len, heads * head_dim)
        out = rearrange(out, "batch heads seq dim -> batch seq (heads dim)")
        return out


class Attention(nn.Module):
    """
    Multi-head self-attention layer with RMSNorm.
    """

    def __init__(self, dim: int, heads: int = 8, head_dim: int = 64):
        super().__init__()
        inner_dim = head_dim * heads
        self.heads = heads
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
        input_shape = positions.shape
        batch_dims = list(input_shape[:-1])

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
        )
        batch_dims.append(-1)
        loc = loc.view(batch_dims)

        half_dim = self.dimension // 2
        current_dim = loc.shape[-1]
        if current_dim != half_dim:
            if current_dim > half_dim:
                loc = loc[..., :half_dim]
            else:
                raise ValueError(
                    f"Input dimension ({current_dim}) is too small for target "
                    f"embedding dimension ({self.dimension}). Expected at least {half_dim}."
                )

        emb = torch.cat([torch.cos(loc), torch.sin(loc)], dim=-1)
        return emb

    @classmethod
    def add_time_patch(cls, pos: torch.Tensor, num_patches: int) -> torch.Tensor:
        """
        Expand the position tensor by adding a time dimension, handling batched data.

        Parameters
        ----------
        pos : torch.Tensor
            Input tensor of shape (B, C, 3), where B is the batch size,
            C is the number of channels, and 3 represents x, y, z.
        num_patches : int
            The number of time patches.

        Returns
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
    def __init__(
        self,
        url: str = "https://huggingface.co/brain-bzh/reve-positions/resolve/main/positions.json",
        timeout: int = 5,
    ):
        super().__init__()

        config = None

        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            config = json.loads(response.text)
        except (requests.RequestException, json.JSONDecodeError) as e:
            raise RuntimeError(
                f"Failed to download or parse the position bank from {url}: {e}"
            ) from e

        try:
            self.position_names = list(config.keys())
            self.mapping = {name: i for i, name in enumerate(self.position_names)}
            positions_data = list(config.values())
            positions = torch.tensor(positions_data, dtype=torch.float32)
            self.register_buffer("embedding", positions)

            # Validate shape (N, 3)
            if self.embedding.dim() != 2 or self.embedding.shape[1] != 3:
                raise ValueError(
                    f"Position data must have shape (N, 3), but got {self.embedding.shape}"
                )

            assert self.embedding.shape == (len(self.mapping), 3), (
                f"Expected embedding shape ({len(self.mapping)}, 3), but got {self.embedding.shape}"
            )

        except (ValueError, TypeError, AssertionError) as e:
            raise RuntimeError(
                f"Invalid position data format in the downloaded config: {e}"
            ) from e

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
