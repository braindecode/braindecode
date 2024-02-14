import math

import torch
import torch.nn as nn
import numpy as np
from linear_attention_transformer import LinearAttentionTransformer

from braindecode.models.base import EEGModuleMixin

class _PatchFrequencyEmbedding(nn.Module):
    def __init__(self, emb_size=256, n_freq=101):
        super().__init__()
        self.projection = nn.Linear(n_freq, emb_size)

    def forward(self, x):
        """
        x: (batch, freq, time)
        out: (batch, time, emb_size)
        """
        x = x.permute(0, 2, 1)
        x = self.projection(x)
        return x


class _ClassificationHead(nn.Sequential):
    def __init__(self, emb_size, n_classes):
        super().__init__()
        self.clshead = nn.Sequential(
            nn.ELU(),
            nn.Linear(emb_size, n_classes),
        )

    def forward(self, x):
        out = self.clshead(x)
        return out


class _PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 1000):
        super(_PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        """
        Args:
            x: `embeddings`, shape (batch, max_len, d_model)
        Returns:
            `encoder input`, shape (batch, max_len, d_model)
        """
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class _BIOTEncoder(nn.Module):
    """
    BIOT Encoder.



    """

    def __init__(
        self,
        emb_size=256,
        heads=8,
        depth=4,
        n_channels=16,
        n_fft=200,
        hop_length=100,
    ):
        super().__init__()

        self.n_fft = n_fft
        self.hop_length = hop_length

        self.patch_embedding = _PatchFrequencyEmbedding(
            emb_size=emb_size, n_freq=self.n_fft // 2 + 1
        )
        self.transformer = LinearAttentionTransformer(
            dim=emb_size,
            heads=heads,
            depth=depth,
            max_seq_len=1024,
            attn_layer_dropout=0.2,  # dropout right after self-attention layer
            attn_dropout=0.2,  # dropout post-attention
        )
        self.positional_encoding = _PositionalEncoding(emb_size)

        # channel token, N_channels >= your actual channels
        self.channel_tokens = nn.Embedding(n_channels, 256)
        self.index = nn.Parameter(
            torch.LongTensor(range(n_channels)), requires_grad=False
        )

    def stft(self, sample):
        spectral = torch.stft(
            input = sample.squeeze(1),
            n_fft = self.n_fft,
            hop_length = self.hop_length,
            center = False,
            onesided = True,
            return_complex = True,
        )
        return torch.abs(spectral)

    def forward(self, x, n_channel_offset=0, perturb=False):
        """
        x: [batch_size, channel, ts]
        output: [batch_size, emb_size]
        """
        emb_seq = []
        for i in range(x.shape[1]):
            channel_spec_emb = self.stft(x[:, i : i + 1, :])
            channel_spec_emb = self.patch_embedding(channel_spec_emb)
            batch_size, ts, _ = channel_spec_emb.shape
            # (batch_size, ts, emb)
            channel_token_emb = (
                self.channel_tokens(self.index[i + n_channel_offset])
                .unsqueeze(0)
                .unsqueeze(0)
                .repeat(batch_size, ts, 1)
            )
            # (batch_size, ts, emb)
            channel_emb = self.positional_encoding(channel_spec_emb + channel_token_emb)

            # perturb
            if perturb:
                ts = channel_emb.shape[1]
                ts_new = np.random.randint(ts // 2, ts)
                selected_ts = np.random.choice(range(ts), ts_new, replace=False)
                channel_emb = channel_emb[:, selected_ts]
            emb_seq.append(channel_emb)

        # (batch_size, 16 * ts, emb)
        emb = torch.cat(emb_seq, dim=1)
        # (batch_size, emb)
        emb = self.transformer(emb).mean(dim=1)
        return emb


class BIOT(EEGModuleMixin, nn.Module):
    """BIOT: Cross-data Biosignal Learning in the Wild.

    BIOT is a large language model for biosignal classification. It is
    a wrapper around the `BIOTEncoder` and `ClassificationHead` modules.

    It is designed for N-Dimensional biosignal data such as EEG, ECG, etc.
    The method was proposed by Yang et al. [Yang2023]_ and the code is
    available at [BioTCode]_.

    The model is trained with a contrastive loss on a large dataset of EEG
    TUH Abnormal EEG Corpus with 400K samples and Sleep Heart Health Study
    5M. Here, we only provide the model architecture, not the pre-trained
    weights or the contrastive loss training.

    The architecture is based on the `LinearAttentionTransformer` and
    `PatchFrequencyEmbedding` modules. The `BIOTEncoder` is a transformer
    that takes the input data and outputs a fixed-size representation
    of the input data. More details are present in the `BIOTEncoder` class.

    The `ClassificationHead` is an ELU activation layer, follow by a simple
    linear layer that takes the output of the `BIOTEncoder` and outputs the
    classification probabilities.

    .. versionadded:: 0.9

    Parameters
    ----------
    emb_size : int, optional
        The size of the embedding layer, by default 256
    att_num_heads : int, optional
        The number of attention heads, by default 8
    depth : int, optional
        The number of transformer layers, by default 4

    References
    ----------
    .. [Yang2023] Yang, C., Westover, M.B. and Sun, J., 2023, November.
    BIOT: Biosignal Transformer for Cross-data Learning in the Wild.
    In Thirty-seventh Conference on Neural Information Processing Systems
    NeurIPS.
    .. [BioTCode] Yang, C., Westover, M.B. and Sun, J., 2023.
    https://github.com/ycq091044/BIOT
    """

    def __init__(self,
                 emb_size=256,
                 att_num_heads=8,
                 depth=4,
                 n_outputs=None,
                 n_chans=None,
                 chs_info=None,
                 n_times=None,
                 sfreq=None):

        super().__init__(
            n_outputs=n_outputs,
            n_chans=n_chans,
            chs_info=chs_info,
            n_times=n_times,
            sfreq=sfreq,
        )
        del n_outputs, n_chans, chs_info, n_times, sfreq

        self.biot = _BIOTEncoder(emb_size=emb_size,
                                heads=att_num_heads,
                                depth=depth)
        self.classifier = _ClassificationHead(emb_size, self.n_outputs)

    def forward(self, x):
        x = self.biot(x)
        x = self.classifier(x)
        return x
