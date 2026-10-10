# Authors: Jathurshan Pradeepkumar (original TFM-Tokenizer)
#
# License: MIT
# Adapted from https://github.com/Jathurshan0330/TFM-Tokenizer (MIT).

from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import nn

from braindecode.functional import spectral_input
from braindecode.models.base import EEGModuleMixin
from braindecode.modules.linear_attention import LinearAttentionTransformer
from braindecode.modules.quantization import EMACodebook


class TFMTokenizerOutput(NamedTuple):
    """Outputs of :meth:`TFMTokenizer.tokenize`.

    Attributes
    ----------
    reconstruction : torch.Tensor
        Reconstructed magnitude spectra, ``(batch, channels, n_freqs, n_frames)``.
    token_ids : torch.LongTensor
        Motif IDs, ``(batch, channels, n_frames)``.
    quantized : torch.Tensor
        Straight-through quantized embeddings, ``(batch * channels, n_frames,
        embed_dim)``.
    embeddings : torch.Tensor
        L2-normalized embeddings before quantization, same shape as ``quantized``.
    quantization_loss : torch.Tensor
        Codebook plus ``commitment_cost`` times commitment loss.
    target_spectrogram : torch.Tensor
        Unmasked magnitude spectra, same shape as ``reconstruction``.
    """

    reconstruction: torch.Tensor
    token_ids: torch.Tensor
    quantized: torch.Tensor
    embeddings: torch.Tensor
    quantization_loss: torch.Tensor
    target_spectrogram: torch.Tensor


class TFMTokenizer(EEGModuleMixin, nn.Module, license="mit"):
    r"""Time-Frequency Motif (TFM) tokenizer from Pradeepkumar et al. (2026) [tfm2026]_.

    :bdg-danger:`Foundation Model` :bdg-info:`Attention/Transformer`

    .. figure:: https://arxiv.org/html/2502.16060v5/Method_Overview_Figure_New.png
        :align: center
        :alt: TFM-Tokenizer overview (Pradeepkumar et al., 2026, Fig. 2).

    Each channel is tokenized independently. A frequency path (patch
    convolutions and a linear-attention transformer over the STFT magnitude of
    each frame) and a temporal path (strided convolution over the raw signal)
    are concatenated, contextualized by a temporal transformer and quantized by
    an EMA codebook. A decoder reconstructs the unmasked spectrogram from the
    quantized tokens. :meth:`tokenize` returns every output;
    :meth:`forward` returns the reconstruction only.

    Parameters
    ----------
    embed_dim : int, default=64
        Embedding size; a multiple of 8 and of twice the number of frequency
        groups.
    codebook_size : int, default=8192
        Number of motifs in the codebook.
    freq_patch_size : int, default=5
        Width and stride of the frequency patches; ``sfreq // 2`` must be
        divisible by its square.
    freq_encoder_depth : int, default=2
        Linear-attention blocks of the frequency encoder.
    temporal_encoder_depth : int, default=2
        Linear-attention blocks of the temporal encoder.
    decoder_depth : int, default=8
        Linear-attention blocks of the decoder.
    max_seq_len : int, default=1024
        Maximum number of frames.
    commitment_cost : float, default=1.0
        Weight of the commitment loss.
    activation : type[nn.Module], default=nn.GELU
        Activation of the convolutional blocks.
    drop_prob : float, default=0.2
        Dropout of the attention blocks.

    Notes
    -----
    The STFT uses a one-second Hann window and a half-second hop, so ``sfreq``
    must be an even integer (200 Hz for the released weights) and a window of
    ``n_times`` samples gives ``1 + (n_times - sfreq) // (sfreq // 2)`` tokens
    per channel.

    Training differs from the reference code in one place; inference does
    not. The codebook term of the quantization loss carries no gradient (the
    codebook moves by EMA only), so at ``commitment_cost=1`` the encoder still
    receives the commitment gradient (in the reference both terms use the
    straight-through tensor and their encoder gradients cancel).

    Tokens from this class are identical to the reference tokenizer's. With
    them, on CHB-MIT, the authors' released fine-tuned classifier gives a
    balanced accuracy of 0.619 and our retraining with the authors' code gives
    0.611 ± 0.034 (10 seeds); the paper reports 0.647 ± 0.015.

    .. important::
       **Pre-trained Weights Available**

       The authors' tokenizer weights (MIT) are on the Hugging Face Hub at
       `Jathurshan/TFM-Tokenizer <https://huggingface.co/Jathurshan/TFM-Tokenizer>`_
       under the reference module names; rename the keys to load them:

       .. code-block:: python

           import torch
           from huggingface_hub import hf_hub_download
           from braindecode.models import TFMTokenizer

           path = hf_hub_download(
               "Jathurshan/TFM-Tokenizer",
               "multiple_dataset_settings/Pretrained_tfm_tokenizer_2x2x8/"
               "tfm_tokenizer_last.pth",
           )
           renames = {
               "trans_freq_encoder.transformer.": "frequency_encoder.",
               "trans_temporal_encoder.transformer.": "temporal_encoder.",
               "trans_decoder.transformer.": "decoder.",
               "freq_patch_embedding_2_atten.": "frequency_attention.",
               "freq_patch_embedding_2.0.": "frequency_projection.",
               "freq_patch_embedding.": "frequency_patch_embedding.",
               "decoder.": "final_layer.",
               "quantizer.embedding.weight": "quantizer.embed",
               "quantizer.ema_w": "quantizer.embed_avg",
           }
           state = {
               next(
                   (new + k[len(old) :] for old, new in renames.items() if k.startswith(old)), k
               ): v
               for k, v in torch.load(path, map_location="cpu").items()
           }
           state["quantizer.inited"] = torch.ones(1)
           model = TFMTokenizer(sfreq=200)
           model.load_state_dict(state)

    .. versionadded:: 1.9

    Examples
    --------
    >>> import torch
    >>> from braindecode.models import TFMTokenizer
    >>> model = TFMTokenizer(sfreq=200, codebook_size=256)
    >>> x = torch.randn(2, 8, 1000)
    >>> spec = model.compute_spectrogram(x)
    >>> mask_a, mask_b = model.make_complementary_masks(spec)
    >>> out = model.tokenize(x, spectrogram_mask=mask_a)
    >>> out.token_ids.shape
    torch.Size([2, 8, 9])
    >>> loss = (out.reconstruction - spec).square().mean() + out.quantization_loss

    References
    ----------
    .. [tfm2026] Pradeepkumar, J., Piao, X., Chen, Z., Sun, J. (2026).
       Tokenizing Single-Channel EEG with Time-Frequency Motif Learning.
       ICLR 2026. https://openreview.net/forum?id=2sPmWHZ8Ir
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
        if self.sfreq <= 0 or not float(self.sfreq).is_integer() or self.sfreq % 2:
            raise ValueError(
                f"sfreq must be a positive even integer, got {self.sfreq}."
            )
        self.window_size = int(self.sfreq)
        self.n_freqs = self.window_size // 2
        n_freq_groups = self.n_freqs // freq_patch_size**2
        if self.n_freqs % freq_patch_size**2:
            raise ValueError(
                "sfreq // 2 must be divisible by freq_patch_size squared; "
                f"got {self.n_freqs} bins and patch size {freq_patch_size}."
            )
        if embed_dim % 8 or embed_dim % (2 * n_freq_groups):
            raise ValueError(
                "embed_dim must be a multiple of 8 and of twice the number of "
                f"frequency groups ({2 * n_freq_groups}), got {embed_dim}."
            )
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        self.commitment_cost = commitment_cost

        def lat(depth):
            return LinearAttentionTransformer(
                dim=embed_dim, heads=8, depth=depth, attn_layer_dropout=drop_prob
            )

        def conv_block(kernel_size, stride, out_dim):
            return nn.Sequential(
                nn.Conv1d(1, embed_dim, kernel_size=kernel_size, stride=stride),
                activation(),
                nn.GroupNorm(embed_dim // 4, embed_dim),
                nn.Conv1d(embed_dim, embed_dim, kernel_size=1),
                activation(),
                nn.GroupNorm(embed_dim // 4, embed_dim),
                nn.Conv1d(embed_dim, out_dim, kernel_size=1),
                activation(),
                nn.GroupNorm(embed_dim // 4, out_dim),
            )

        freq_width = embed_dim // (2 * n_freq_groups)
        self.frequency_patch_embedding = conv_block(
            freq_patch_size, freq_patch_size, embed_dim
        )
        self.frequency_encoder = lat(freq_encoder_depth)
        self.frequency_attention = nn.Sequential(
            nn.Conv1d(
                embed_dim,
                freq_width,
                kernel_size=freq_patch_size,
                stride=freq_patch_size,
            ),
            nn.Sigmoid(),
        )
        self.frequency_projection = nn.Conv1d(
            embed_dim, freq_width, kernel_size=freq_patch_size, stride=freq_patch_size
        )
        self.temporal_patch_embedding = conv_block(
            self.window_size, self.window_size // 2, embed_dim // 2
        )
        self.temporal_encoder = lat(temporal_encoder_depth)
        # The reference EMA codebook: no k-means init, no dead-code expiry.
        self.quantizer = EMACodebook(
            embed_dim,
            codebook_size,
            epsilon=1e-5,
            threshold_ema_dead_code=0,
            kmeans_init=False,
        )
        nn.init.uniform_(self.quantizer.embed, -1 / codebook_size, 1 / codebook_size)
        self.quantizer.embed_avg.copy_(self.quantizer.embed)
        self.decoder = lat(decoder_depth)
        # Reconstruction head (spectrum bins per token), named for the
        # braindecode head contract.
        self.final_layer = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.Tanh(),
            nn.Linear(embed_dim, self.n_freqs),
        )
        self.register_buffer(
            "stft_window", torch.hann_window(self.window_size), persistent=False
        )

    def compute_spectrogram(self, x: torch.Tensor) -> torch.Tensor:
        """STFT magnitude of ``x`` ``(batch, channels, samples)``.

        Returns a ``(batch, channels, n_freqs, n_frames)`` tensor.
        """
        if x.ndim != 3 or not x.is_floating_point():
            raise ValueError(
                "x must be a floating-point (batch, channels, samples) tensor."
            )
        if x.shape[-1] < self.window_size:
            raise ValueError(
                f"Input must contain at least one second ({self.window_size} "
                f"samples), got {x.shape[-1]} samples."
            )
        batch_size, n_chans, n_times = x.shape
        signal = spectral_input(x.reshape(batch_size * n_chans, n_times))
        spectrogram = torch.stft(
            signal,
            n_fft=self.window_size,
            hop_length=self.window_size // 2,
            win_length=self.window_size,
            window=self.stft_window.to(signal),
            center=False,
            return_complex=True,
        ).abs()[:, : self.n_freqs]
        return spectrogram.to(x).reshape(batch_size, n_chans, self.n_freqs, -1)

    def encode(self, x: torch.Tensor, spectrogram: torch.Tensor) -> torch.Tensor:
        """Embed ``x`` and its ``(batch * channels, n_freqs, n_frames)`` spectrogram."""
        batch_size, n_chans, _ = x.shape
        n_frames = spectrogram.shape[-1]
        freq = spectrogram.transpose(1, 2).reshape(-1, 1, self.n_freqs)
        freq = self.frequency_patch_embedding(freq).transpose(1, 2)
        freq = self.frequency_encoder(freq).transpose(1, 2)
        freq = self.frequency_projection(freq) * self.frequency_attention(freq)
        freq = freq.reshape(batch_size * n_chans, n_frames, self.embed_dim // 2)
        temporal = self.temporal_patch_embedding(
            x.reshape(batch_size * n_chans, 1, -1)
        ).transpose(1, 2)
        embeddings = self.temporal_encoder(torch.cat((freq, temporal), dim=-1))
        return F.normalize(embeddings, p=2.0, dim=-1)

    def make_complementary_masks(
        self,
        spectrogram: torch.Tensor,
        freq_mask_ratio: float = 0.5,
        freq_bin_size: int = 5,
        time_mask_ratio: float = 0.5,
        time_bin_size: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return a random time-frequency mask and its complement.

        Groups of ``freq_bin_size`` bins and ``time_bin_size`` frames are hidden
        at the given ratios. One mask, shaped like ``spectrogram``, is shared
        by every trial and channel, as in the reference.
        """
        keep = torch.ones(
            spectrogram.shape[-2:], dtype=torch.bool, device=spectrogram.device
        )
        for dim, bin_size, ratio in (
            (0, freq_bin_size, freq_mask_ratio),
            (1, time_bin_size, time_mask_ratio),
        ):
            if keep.shape[dim] % bin_size:
                raise ValueError(
                    f"{keep.shape[dim]} bins/frames must be divisible by the bin "
                    f"size ({bin_size})."
                )
            n_groups = keep.shape[dim] // bin_size
            n_masked = int(n_groups * ratio)
            if n_masked:
                groups = torch.randperm(n_groups, device=keep.device)[:n_masked]
                index = groups[:, None] * bin_size + torch.arange(
                    bin_size, device=keep.device
                )
                keep.index_fill_(dim, index.reshape(-1), False)
        keep = keep.expand_as(spectrogram)
        return keep, ~keep

    def tokenize(
        self, x: torch.Tensor, spectrogram_mask: torch.Tensor | None = None
    ) -> TFMTokenizerOutput:
        """Tokenize ``x`` and reconstruct its spectrogram.

        Parameters
        ----------
        x : torch.Tensor
            EEG ``(batch, channels, samples)`` sampled at ``sfreq``.
        spectrogram_mask : torch.Tensor, optional
            Boolean mask shaped like :meth:`compute_spectrogram` output;
            ``True`` bins are visible to the encoder. Default: all visible.

        Returns
        -------
        TFMTokenizerOutput
        """
        target = self.compute_spectrogram(x)
        batch_size, n_chans, _, n_frames = target.shape
        if n_frames > self.max_seq_len:
            raise ValueError(
                f"Input produces {n_frames} frames, exceeding max_seq_len="
                f"{self.max_seq_len}."
            )
        visible = target
        if spectrogram_mask is not None:
            if spectrogram_mask.shape != target.shape:
                raise ValueError(
                    "spectrogram_mask must match the computed spectrogram shape "
                    f"{list(target.shape)}, got {list(spectrogram_mask.shape)}."
                )
            visible = target * spectrogram_mask
        embeddings = self.encode(
            x, visible.reshape(batch_size * n_chans, self.n_freqs, -1)
        )
        codebook_vectors, token_ids = self.quantizer(embeddings)
        quantized = embeddings + (codebook_vectors - embeddings).detach()
        # The codebook term has the commitment term's value and no gradient
        # (see Notes).
        commitment_loss = F.mse_loss(codebook_vectors, embeddings)
        reconstruction = self.final_layer(self.decoder(quantized)).transpose(1, 2)
        return TFMTokenizerOutput(
            reconstruction.reshape(batch_size, n_chans, self.n_freqs, n_frames),
            token_ids.reshape(batch_size, n_chans, n_frames),
            quantized,
            embeddings,
            commitment_loss.detach() + self.commitment_cost * commitment_loss,
            target,
        )

    def forward(
        self, x: torch.Tensor, spectrogram_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Return the reconstructed spectrogram (see :meth:`tokenize`)."""
        return self.tokenize(x, spectrogram_mask=spectrogram_mask).reconstruction
