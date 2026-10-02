# Adapted from TFM-Tokenizer, Copyright (c) 2025 Jathurshan Pradeepkumar.
# The upstream implementation is MIT-licensed; see the license notice below.
# https://github.com/Jathurshan0330/TFM-Tokenizer
#
# MIT License
#
# Copyright (c) 2025 Jathurshan Pradeepkumar
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Time-frequency motif tokenizer for EEG signals."""

from typing import NamedTuple

import torch
import torch.nn.functional as F
from linear_attention_transformer import LinearAttentionTransformer
from torch import nn

from braindecode.models.base import EEGModuleMixin


class TFMTokenizerOutput(NamedTuple):
    """Outputs from :class:`TFMTokenizer`.

    Attributes
    ----------
    reconstruction : torch.Tensor
        Reconstructed magnitude spectra with shape ``(batch, channels, n_freqs,
        n_frames)``. This is the decoder output used by the tokenizer's
        reconstruction objective.
    token_ids : torch.LongTensor
        Quantized motif IDs with shape ``(batch, channels, n_frames)``.
    quantized : torch.Tensor
        Straight-through quantized embeddings with shape
        ``(batch * channels, n_frames, embed_dim)``.
    embeddings : torch.Tensor
        L2-normalized pre-quantization embeddings with the same shape as
        ``quantized``.
    quantization_loss : torch.Tensor
        Commitment loss for the exponential-moving-average codebook.
    target_spectrogram : torch.Tensor
        Unmasked target magnitude spectra with shape ``(batch, channels,
        n_freqs, n_frames)``.

    """

    reconstruction: torch.Tensor
    token_ids: torch.Tensor
    quantized: torch.Tensor
    embeddings: torch.Tensor
    quantization_loss: torch.Tensor
    target_spectrogram: torch.Tensor


class _EMAVectorQuantizer(nn.Module):
    """Vector quantizer whose codebook is updated with exponential averages."""

    def __init__(self, embed_dim: int, codebook_size: int, decay: float = 0.99):
        super().__init__()
        self.embed_dim = embed_dim
        self.codebook_size = codebook_size
        self.decay = decay
        self.eps = 1e-5

        self.embedding = nn.Embedding(codebook_size, embed_dim)
        nn.init.uniform_(self.embedding.weight, -1 / codebook_size, 1 / codebook_size)
        self.register_buffer("cluster_size", torch.zeros(codebook_size))
        self.register_buffer("ema_weight", self.embedding.weight.detach().clone())

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        flat_x = x.reshape(-1, self.embed_dim)
        # Read from a stable copy: the EMA update below mutates the registered
        # parameter before backward, while the dictionary loss still trains it.
        codebook = self.embedding.weight.clone()
        distances = (
            flat_x.square().sum(dim=1, keepdim=True)
            - 2 * flat_x @ codebook.T
            + codebook.square().sum(dim=1)
        )
        indices = distances.argmin(dim=1)
        quantized = F.embedding(indices, codebook).view_as(x)

        if self.training:
            with torch.no_grad():
                assignments = F.one_hot(indices, self.codebook_size).to(flat_x.dtype)
                self.cluster_size.mul_(self.decay).add_(
                    assignments.sum(dim=0), alpha=1 - self.decay
                )
                self.ema_weight.mul_(self.decay).add_(
                    assignments.T @ flat_x, alpha=1 - self.decay
                )

                total = self.cluster_size.sum()
                smoothed_size = (
                    (self.cluster_size + self.eps)
                    / (total + self.codebook_size * self.eps)
                    * total
                )
                self.embedding.weight.copy_(
                    self.ema_weight / smoothed_size.clamp_min(self.eps).unsqueeze(1)
                )

        return quantized, indices.view(x.shape[0], x.shape[1])


class TFMTokenizer(EEGModuleMixin, nn.Module, license="mit"):
    r"""Time-Frequency Motif (TFM) tokenizer for single-channel EEG motifs.

    The model follows Pradeepkumar et al., *Tokenizing Single-Channel EEG with
    Time-Frequency Motif Learning* (ICLR 2026). It embeds each EEG channel
    independently through parallel frequency and temporal paths, combines the
    features with a temporal transformer, and quantizes the resulting sequence
    into a learned vocabulary. Complementary time-frequency masks can be passed
    to the encoder while the decoder reconstructs the original spectrogram.

    Parameters
    ----------
    sfreq : int
        Sampling frequency in Hz. The reference tokenizer uses 200 Hz and a
        one-second STFT window with a half-second hop. ``sfreq`` must be even.
    embed_dim : int
        Joint embedding size. Must be divisible by 8 and by twice the number of
        frequency groups.
    codebook_size : int
        Number of discrete time-frequency motifs.
    freq_patch_size : int
        Width and stride of the frequency patches. For the reference settings,
        ``sfreq=200`` and ``freq_patch_size=5`` produce 100 bins and four groups.
    freq_encoder_depth : int
        Number of linear-attention blocks in the frequency encoder.
    temporal_encoder_depth : int
        Number of linear-attention blocks in the joint temporal encoder.
    decoder_depth : int
        Number of linear-attention blocks in the reconstruction decoder.
    max_seq_len : int
        Maximum temporal sequence length supported by the transformers.
    commitment_cost : float
        Weight of the vector-quantization commitment loss.
    activation : type[nn.Module]
        Activation module used in convolutional projection blocks.
    drop_prob : float
        Dropout probability used in attention blocks.

    Notes
    -----
    Input must be resampled to ``sfreq`` before calling the model. Each channel
    is tokenized independently; channel identity is not mixed in the tokenizer.
    For a signal window of length ``n_times``, the token count is
    ``1 + floor((n_times - sfreq) / (sfreq / 2))``. Windows shorter than one
    second are rejected.

    The upstream research implementation and its pretrained checkpoints are
    available at https://github.com/Jathurshan0330/TFM-Tokenizer and
    https://huggingface.co/Jathurshan/TFM-Tokenizer. This class does not claim
    checkpoint parity; it provides a library-native implementation of the
    tokenizer architecture and training outputs.

    Examples
    --------
    >>> import torch
    >>> from braindecode.models import TFMTokenizer
    >>> model = TFMTokenizer(sfreq=200, codebook_size=256)
    >>> out = model(torch.randn(2, 8, 1000))
    >>> out.token_ids.shape
    torch.Size([2, 8, 9])
    >>> x = torch.randn(2, 8, 1000)
    >>> spec = model.compute_spectrogram(x)
    >>> mask_a, mask_b = model.make_complementary_masks(spec)
    >>> out_a = model(x, spectrogram_mask=mask_a)
    >>> out_b = model(x, spectrogram_mask=mask_b)
    >>> loss = (out_a.reconstruction - spec).square().mean() + out_a.quantization_loss

    """

    def __init__(
        self,
        n_outputs: int | None = None,
        n_chans: int | None = None,
        chs_info=None,
        n_times: int | None = None,
        input_window_seconds: float | None = None,
        sfreq: float = 200,
        embed_dim: int = 64,
        codebook_size: int = 8192,
        freq_patch_size: int = 5,
        freq_encoder_depth: int = 2,
        temporal_encoder_depth: int = 2,
        decoder_depth: int = 8,
        max_seq_len: int = 1024,
        commitment_cost: float = 1.0,
        activation: type[nn.Module] = nn.GELU,
        drop_prob: float = 0.2,
    ):
        super().__init__(
            n_outputs=n_outputs,
            n_chans=n_chans,
            chs_info=chs_info,
            n_times=n_times,
            input_window_seconds=input_window_seconds,
            sfreq=sfreq,
        )
        sfreq = self.sfreq
        if sfreq <= 0 or not float(sfreq).is_integer() or int(sfreq) % 2:
            raise ValueError(f"sfreq must be a positive even integer, got {sfreq}.")
        if embed_dim < 8 or embed_dim % 8:
            raise ValueError(
                f"embed_dim must be a positive multiple of 8, got {embed_dim}."
            )
        if codebook_size < 2:
            raise ValueError("codebook_size must be at least 2.")
        if freq_patch_size < 1:
            raise ValueError("freq_patch_size must be positive.")
        if min(freq_encoder_depth, temporal_encoder_depth, decoder_depth) < 1:
            raise ValueError("All transformer depths must be positive.")
        if max_seq_len < 1:
            raise ValueError("max_seq_len must be positive.")
        if commitment_cost < 0:
            raise ValueError("commitment_cost must be non-negative.")
        if not 0 <= drop_prob < 1:
            raise ValueError("drop_prob must be in [0, 1).")

        self.window_size = int(round(self.sfreq))
        self.n_freqs = self.window_size // 2
        self.embed_dim = embed_dim
        self.codebook_size = codebook_size
        self.freq_patch_size = freq_patch_size
        self.max_seq_len = max_seq_len
        self.commitment_cost = commitment_cost
        self.activation = activation
        self.drop_prob = drop_prob

        patch_area = freq_patch_size**2
        if self.n_freqs % patch_area:
            raise ValueError(
                "sfreq // 2 must be divisible by freq_patch_size squared; "
                f"got {self.n_freqs} bins and patch size {freq_patch_size}."
            )
        self.n_freq_groups = self.n_freqs // patch_area
        if embed_dim % (2 * self.n_freq_groups):
            raise ValueError(
                "embed_dim must be divisible by twice the number of frequency "
                f"groups ({2 * self.n_freq_groups}), got {embed_dim}."
            )
        if self.n_freqs // freq_patch_size > max_seq_len:
            raise ValueError("max_seq_len is too short for the frequency encoder.")

        self.frequency_patch_embedding = nn.Sequential(
            nn.Conv1d(
                1, embed_dim, kernel_size=freq_patch_size, stride=freq_patch_size
            ),
            activation(),
            nn.GroupNorm(embed_dim // 4, embed_dim),
            nn.Conv1d(embed_dim, embed_dim, kernel_size=1),
            activation(),
            nn.GroupNorm(embed_dim // 4, embed_dim),
            nn.Conv1d(embed_dim, embed_dim, kernel_size=1),
            activation(),
            nn.GroupNorm(embed_dim // 4, embed_dim),
        )
        self.frequency_encoder = LinearAttentionTransformer(
            dim=embed_dim,
            heads=8,
            depth=freq_encoder_depth,
            max_seq_len=self.n_freqs // freq_patch_size,
            attn_layer_dropout=drop_prob,
            attn_dropout=drop_prob,
        )

        frequency_width = embed_dim // (2 * self.n_freq_groups)
        self.frequency_attention = nn.Sequential(
            nn.Conv1d(
                embed_dim,
                frequency_width,
                kernel_size=freq_patch_size,
                stride=freq_patch_size,
            ),
            nn.Sigmoid(),
        )
        self.frequency_projection = nn.Conv1d(
            embed_dim,
            frequency_width,
            kernel_size=freq_patch_size,
            stride=freq_patch_size,
        )

        self.temporal_patch_embedding = nn.Sequential(
            nn.Conv1d(
                1,
                embed_dim,
                kernel_size=self.window_size,
                stride=self.window_size // 2,
            ),
            activation(),
            nn.GroupNorm(embed_dim // 4, embed_dim),
            nn.Conv1d(embed_dim, embed_dim, kernel_size=1),
            activation(),
            nn.GroupNorm(embed_dim // 4, embed_dim),
            nn.Conv1d(embed_dim, embed_dim // 2, kernel_size=1),
            activation(),
            nn.GroupNorm(embed_dim // 4, embed_dim // 2),
        )
        self.temporal_encoder = LinearAttentionTransformer(
            dim=embed_dim,
            heads=8,
            depth=temporal_encoder_depth,
            max_seq_len=max_seq_len,
            attn_layer_dropout=drop_prob,
            attn_dropout=drop_prob,
        )
        self.quantizer = _EMAVectorQuantizer(embed_dim, codebook_size)
        self.decoder = LinearAttentionTransformer(
            dim=embed_dim,
            heads=8,
            depth=decoder_depth,
            max_seq_len=max_seq_len,
            attn_layer_dropout=drop_prob,
            attn_dropout=drop_prob,
        )
        # Braindecode's integration checks expect a task-output layer name.
        # For this non-classification model it projects tokens to spectrum bins.
        self.final_layer = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.Tanh(),
            nn.Linear(embed_dim, self.n_freqs),
        )
        self.register_buffer(
            "stft_window", torch.hann_window(self.window_size), persistent=False
        )

    def _compute_spectrogram(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, n_chans, n_times = x.shape
        flattened = x.reshape(batch_size * n_chans, n_times)
        window = self.stft_window.to(device=x.device, dtype=x.dtype)
        return torch.stft(
            flattened,
            n_fft=self.window_size,
            hop_length=self.window_size // 2,
            win_length=self.window_size,
            window=window,
            center=False,
            return_complex=True,
        ).abs()[:, : self.n_freqs]

    def compute_spectrogram(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the reference STFT magnitude representation.

        Parameters
        ----------
        x : torch.Tensor
            EEG tensor with shape ``(batch, channels, samples)``.

        Returns
        -------
        torch.Tensor
            Magnitude spectrogram with shape ``(batch, channels, n_freqs,
            n_frames)``.

        """
        if x.ndim != 3 or not x.is_floating_point():
            raise ValueError(
                "x must be a floating-point (batch, channels, samples) tensor."
            )
        if x.shape[-1] < self.window_size:
            raise ValueError(
                f"Input must contain at least one second ({self.window_size} samples), "
                f"got {x.shape[-1]} samples."
            )
        batch_size, n_chans, _ = x.shape
        spectrogram = self._compute_spectrogram(x)
        return spectrogram.reshape(batch_size, n_chans, self.n_freqs, -1)

    def _encode(self, x: torch.Tensor, spectrogram: torch.Tensor) -> torch.Tensor:
        batch_size, n_chans, _ = x.shape
        flattened = x.reshape(batch_size * n_chans, -1)
        n_frames = spectrogram.shape[-1]

        # Encode each frame's frequency profile independently, then aggregate
        # adjacent frequency groups into a half-width frequency representation.
        freq = spectrogram.transpose(1, 2).reshape(-1, 1, self.n_freqs)
        freq = self.frequency_patch_embedding(freq).transpose(1, 2)
        freq = self.frequency_encoder(freq).transpose(1, 2)
        freq = self.frequency_projection(freq) * self.frequency_attention(freq)
        freq = freq.flatten(1).reshape(
            batch_size * n_chans, n_frames, self.embed_dim // 2
        )

        temporal = self.temporal_patch_embedding(flattened.unsqueeze(1)).transpose(1, 2)
        if temporal.shape[1] != n_frames:
            raise RuntimeError(
                "The temporal convolution and STFT produced different frame counts: "
                f"{temporal.shape[1]} and {n_frames}."
            )
        embeddings = self.temporal_encoder(torch.cat((freq, temporal), dim=-1))
        embeddings = F.normalize(embeddings, p=2, dim=-1)
        return embeddings

    def make_complementary_masks(
        self,
        spectrogram: torch.Tensor,
        freq_mask_ratio: float = 0.5,
        freq_bin_size: int = 5,
        time_mask_ratio: float = 0.5,
        time_bin_size: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Create complementary time-frequency masks for paired-view training.

        The boolean masks have the same shape as ``spectrogram`` and are shared
        across batch elements and channels, matching the reference setup.
        """
        if spectrogram.ndim != 4:
            raise ValueError(
                "Expected spectrogram with shape (batch, channels, freqs, frames), "
                f"got {tuple(spectrogram.shape)}."
            )
        if not 0 <= freq_mask_ratio <= 1 or not 0 <= time_mask_ratio <= 1:
            raise ValueError("Mask ratios must be between 0 and 1.")
        if freq_bin_size < 1 or time_bin_size < 1:
            raise ValueError("Mask bin sizes must be positive.")

        n_freqs, n_frames = spectrogram.shape[-2:]
        if n_freqs % freq_bin_size:
            raise ValueError(
                f"{n_freqs} frequency bins must be divisible by freq_bin_size "
                f"({freq_bin_size})."
            )
        if n_frames % time_bin_size:
            raise ValueError(
                f"{n_frames} time frames must be divisible by time_bin_size "
                f"({time_bin_size})."
            )

        keep = torch.ones(
            (n_freqs, n_frames), dtype=torch.bool, device=spectrogram.device
        )
        n_freq_groups = n_freqs // freq_bin_size
        n_masked_freq_groups = int(n_freq_groups * freq_mask_ratio)
        if n_masked_freq_groups:
            groups = torch.randperm(n_freq_groups, device=spectrogram.device)[
                :n_masked_freq_groups
            ]
            freqs = (
                groups[:, None] * freq_bin_size
                + torch.arange(freq_bin_size, device=spectrogram.device)[None, :]
            ).reshape(-1)
            keep[freqs, :] = False

        n_time_groups = n_frames // time_bin_size
        n_masked_time_groups = int(n_time_groups * time_mask_ratio)
        if n_masked_time_groups:
            groups = torch.randperm(n_time_groups, device=spectrogram.device)[
                :n_masked_time_groups
            ]
            frames = (
                groups[:, None] * time_bin_size
                + torch.arange(time_bin_size, device=spectrogram.device)[None, :]
            ).reshape(-1)
            keep[:, frames] = False

        keep = keep.expand_as(spectrogram)
        return keep, ~keep

    def forward(
        self, x: torch.Tensor, spectrogram_mask: torch.Tensor | None = None
    ) -> TFMTokenizerOutput:
        """Encode EEG windows and reconstruct their unmasked magnitude spectra.

        Parameters
        ----------
        x : torch.Tensor
            Floating-point EEG tensor of shape ``(batch, channels, samples)``
            sampled at ``self.sfreq``.
        spectrogram_mask : torch.Tensor, optional
            Boolean mask matching the computed spectrogram shape. ``True`` bins
            are visible to the encoder. If omitted, all bins are visible.

        Returns
        -------
        TFMTokenizerOutput
            Reconstruction, discrete token IDs, straight-through quantized
            embeddings, normalized pre-quantization embeddings, commitment loss,
            and the unmasked spectrogram target.

        """
        if x.ndim != 3:
            raise ValueError(
                "Expected EEG input with shape (batch, channels, samples), "
                f"got {tuple(x.shape)}."
            )
        if not x.is_floating_point():
            raise TypeError("EEG input must have a floating-point dtype.")
        if x.shape[-1] < self.window_size:
            raise ValueError(
                f"Input must contain at least one second ({self.window_size} samples), "
                f"got {x.shape[-1]} samples."
            )

        batch_size, n_chans, _ = x.shape
        if batch_size < 1 or n_chans < 1:
            raise ValueError("EEG input must contain at least one trial and channel.")
        n_frames = 1 + (x.shape[-1] - self.window_size) // (self.window_size // 2)
        if n_frames > self.max_seq_len:
            raise ValueError(
                f"Input produces {n_frames} frames, exceeding max_seq_len="
                f"{self.max_seq_len}."
            )
        target_spectrogram = self.compute_spectrogram(x)
        expected_shape = target_spectrogram.shape
        if spectrogram_mask is None:
            input_spectrogram = target_spectrogram.reshape(
                batch_size * n_chans, self.n_freqs, -1
            )
        else:
            if spectrogram_mask.shape != expected_shape:
                raise ValueError(
                    "spectrogram_mask must match the computed spectrogram shape "
                    f"{expected_shape}, got {tuple(spectrogram_mask.shape)}."
                )
            input_spectrogram = (
                target_spectrogram
                * spectrogram_mask.to(
                    device=target_spectrogram.device, dtype=target_spectrogram.dtype
                )
            ).reshape(batch_size * n_chans, self.n_freqs, -1)

        embeddings = self._encode(x, input_spectrogram)
        codebook_vectors, token_ids = self.quantizer(embeddings)
        quantized = embeddings + (codebook_vectors - embeddings).detach()
        codebook_loss = F.mse_loss(codebook_vectors, embeddings.detach())
        commitment_loss = F.mse_loss(codebook_vectors.detach(), embeddings)
        quantization_loss = codebook_loss + self.commitment_cost * commitment_loss
        reconstruction = self.final_layer(self.decoder(quantized)).transpose(1, 2)
        n_frames = reconstruction.shape[-1]
        reconstruction = reconstruction.reshape(
            batch_size, n_chans, self.n_freqs, n_frames
        )
        token_ids = token_ids.reshape(batch_size, n_chans, n_frames)
        quantized = quantized.reshape(batch_size * n_chans, n_frames, self.embed_dim)
        embeddings = embeddings.reshape(batch_size * n_chans, n_frames, self.embed_dim)
        return TFMTokenizerOutput(
            reconstruction,
            token_ids,
            quantized,
            embeddings,
            quantization_loss,
            target_spectrogram,
        )
