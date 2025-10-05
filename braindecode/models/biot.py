import math
from warnings import warn

import torch
import torch.nn as nn
from linear_attention_transformer import LinearAttentionTransformer

from braindecode.models.base import EEGModuleMixin


class BIOT(EEGModuleMixin, nn.Module):
    """BIOT from Yang et al. (2023) [Yang2023]_

    .. figure:: https://braindecode.org/dev/_static/model/biot.jpg
       :align: center
       :alt: BioT

    BIOT: Cross-data Biosignal Learning in the Wild.

    BIOT is a large brain model for biosignal classification. It is
    a wrapper around the `BIOTEncoder` and `ClassificationHead` modules.

    It is designed for N-dimensional biosignal data such as EEG, ECG, etc.
    The method was proposed by Yang et al. [Yang2023]_ and the code is
    available at [Code2023]_

    The model is trained with a contrastive loss on large EEG datasets
    TUH Abnormal EEG Corpus with 400K samples and Sleep Heart Health Study
    5M. Here, we only provide the model architecture, not the pre-trained
    weights or contrastive loss training.

    The architecture is based on the `LinearAttentionTransformer` and
    `PatchFrequencyEmbedding` modules.
    The `BIOTEncoder` is a transformer that takes the input data and outputs
    a fixed-size representation of the input data. More details are
    present in the `BIOTEncoder` class.

    The `ClassificationHead` is an ELU activation layer, followed by a simple
    linear layer that takes the output of the `BIOTEncoder` and outputs
    the classification probabilities.

    .. versionadded:: 0.9

    Parameters
    ----------
    emb_size : int, optional
        The size of the embedding layer, by default 256
    att_num_heads : int, optional
        The number of attention heads, by default 8
    n_layers : int, optional
        The number of transformer layers, by default 4
    activation: nn.Module, default=nn.ELU
        Activation function class to apply. Should be a PyTorch activation
        module class like ``nn.ReLU`` or ``nn.ELU``. Default is ``nn.ELU``.
    return_feature: bool, optional
        Changing the output for the neural network. Default is single tensor
        when return_feature is True, return embedding space too.
        Default is False.
    hop_length: int, optional
        The hop length for the torch.stft transformation in the
        encoder. The default is 100.
    sfreq: int, optional
        The sfreq parameter for the encoder. The default is 200

    References
    ----------
    .. [Yang2023] Yang, C., Westover, M.B. and Sun, J., 2023, November. BIOT:
       Biosignal Transformer for Cross-data Learning in the Wild. In Thirty-seventh
       Conference on Neural Information Processing Systems, NeurIPS.
    .. [Code2023] Yang, C., Westover, M.B. and Sun, J., 2023. BIOT
       Biosignal Transformer for Cross-data Learning in the Wild.
       GitHub https://github.com/ycq091044/BIOT (accessed 2024-02-13)
    """

    def __init__(
        self,
        emb_size=256,
        att_num_heads=8,
        n_layers=4,
        sfreq=200,
        hop_length=100,
        return_feature=False,
        n_outputs=None,
        n_chans=None,
        chs_info=None,
        n_times=None,
        input_window_seconds=None,
        activation: nn.Module = nn.ELU,
        drop_prob: float = 0.5,
        # Parameters for the encoder
        max_seq_len: int = 1024,
        attn_dropout=0.2,
        attn_layer_dropout=0.2,
    ):
        super().__init__(
            n_outputs=n_outputs,
            n_chans=n_chans,
            chs_info=chs_info,
            n_times=n_times,
            input_window_seconds=input_window_seconds,
            sfreq=sfreq,
        )
        del n_outputs, n_chans, chs_info, n_times, sfreq
        self.emb_size = emb_size
        self.hop_length = hop_length
        self.att_num_heads = att_num_heads
        self.n_layers = n_layers
        self.return_feature = return_feature
        if (self.sfreq != 200) & (self.sfreq is not None):
            warn(
                "This model has only been trained on a dataset with 200 Hz. "
                + "no guarantee to generalize well with the default parameters",
                UserWarning,
            )
        if self.n_chans > emb_size:
            warn(
                "The number of channels is larger than the embedding size. "
                + "This may cause overfitting. Consider using a larger "
                + "embedding size or a smaller number of channels.",
                UserWarning,
            )
        if self.hop_length > self.sfreq:
            warn(
                "The hop length is larger than the sampling frequency. "
                + "This may cause aliasing. Consider using a smaller "
                "hop length.",
                UserWarning,
            )
            hop_length = self.sfreq // 2

        if self.input_window_seconds < 1.0:
            warning_msg = (
                "The input window is less than 1 second, which may not be "
                "sufficient for the model to learn meaningful representations."
                "Changing the `n_fft` to `n_times`."
            )
            warn(warning_msg, UserWarning)
            self.n_fft = self.n_times
        else:
            self.n_fft = int(self.sfreq)

        self.encoder = _BIOTEncoder(
            emb_size=emb_size,
            att_num_heads=att_num_heads,
            n_layers=n_layers,
            n_chans=self.n_chans,
            n_fft=self.n_fft,
            hop_length=hop_length,
            drop_prob=drop_prob,
            max_seq_len=max_seq_len,
            attn_dropout=attn_dropout,
            attn_layer_dropout=attn_layer_dropout,
        )

        self.final_layer = _ClassificationHead(
            emb_size=emb_size,
            n_outputs=self.n_outputs,
            activation=activation,
        )

    def forward(self, x):
        """
        Pass the input through the BIOT encoder, and then through the
        classification head.

        Parameters
        ----------
        x: Tensor
            (batch_size, n_channels, n_times)

        Returns
        -------
        out: Tensor
            (batch_size, n_outputs)
        (out, emb): tuple Tensor
            (batch_size, n_outputs), (batch_size, emb_size)
        """
        emb = self.encoder(x)
        x = self.final_layer(emb)

        if self.return_feature:
            return x, emb
        else:
            return x


class _PatchFrequencyEmbedding(nn.Module):
    """
    Patch Frequency Embedding.

    A simple linear layer is used to learn some representation over the
    frequency domain with permutation in the frequency axis.

    Parameters
    ----------
    emb_size: int
        The size of the embedding layer
    n_freq: int
        The number of frequency points after the Fourier transform

    Returns
    -------
    out: Tensor
        (batch, time, emb_size)
    """

    def __init__(self, emb_size: int = 256, n_freq: int = 101):
        super().__init__()
        self.projection = nn.Linear(n_freq, emb_size)

    def forward(self, x):
        """
        Parameters
        ----------
        x: Tensor
            (batch, time, n_freq)

        Returns
        -------
        out: Tensor
            (batch, time, emb_size)

        """
        x = x.permute(0, 2, 1)
        x = self.projection(x)
        return x


class _ClassificationHead(nn.Sequential):
    """
    Classification head for the BIOT model.

    Simple linear layer with ELU activation function.

    Parameters
    ----------
    emb_size: int
        The size of the embedding layer
    n_outputs: int
        The number of classes
    activation: nn.Module, default=nn.ELU
        Activation function class to apply. Should be a PyTorch activation
        module class like ``nn.ReLU`` or ``nn.ELU``. Default is ``nn.ELU``.

    Returns
    -------
    out: Tensor
        (batch, n_outputs)
    """

    def __init__(self, emb_size: int, n_outputs: int, activation: nn.Module = nn.ELU):
        super().__init__()
        self.activation_layer = activation()
        self.classification_head = nn.Linear(emb_size, n_outputs)

    def forward(self, x):
        x = self.activation_layer(x)
        out = self.classification_head(x)
        return out


class _PositionalEncoding(nn.Module):
    """
    Positional Encoding.

    We first create a `pe` zero matrix of shape (max_len, d_model) where max_len is the
    the maximum length of the sequence, and emb_size is the size of the embedding.

    Then we create a `position` tensor of shape (max_len, 1) with indices from 0 to max_len
    and a `div_term` tensor of shape (d_model // 2) with the exponential of the
    multiplication of the indices from 0 to d_model by -log(10000.0) / d_model.

    For more details about the positional encoding, see the `Attention is All You Need` paper.

    Parameters
    ----------
    emb_size: int
        The size of the embedding layer
    dropout: float
        The dropout rate
    max_len: int
        The maximum length of the sequence
    drop_prob : float, default=0.5
        The dropout rate for regularization. Values should be between 0 and 1.

    Returns
    -------
    out: Tensor
        (batch, max_len, d_model)
    """

    def __init__(self, emb_size: int, drop_prob: float = 0.1, max_len: int = 1000):
        super(_PositionalEncoding, self).__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, emb_size)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, emb_size, 2).float() * -(math.log(10000.0) / emb_size)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        """
        Parameters
        ----------
        x: FloatTensor
            `embeddings,` shape (batch, max_len, d_model)

        Returns
        -------
        Tensor
            `encoder input`, shape (batch, max_len, d_model)
        """
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class _BIOTEncoder(nn.Module):
    """
    BIOT Encoder.

    The BIOT encoder is a transformer that takes the time series input data and
    return a fixed-size embedding representation of the input data.
    The architecture is based on the `LinearAttentionTransformer` and
    `PatchFrequencyEmbedding` modules.

    The input data is transformed into a spectrogram and then embedded using a
    "patch" embedding.
    The channel token is added to the patch embedding, and then
    positional encoding is applied (simple index positional).
    The resulting embeddings are concatenated
    and passed through a transformer layer. The mean across different channels
    embeddings is returned.

    Parameters
    ----------
    n_chans: int
        The number of channels
    emb_size: int
        The size of the embedding layer
    att_num_heads: int
        The number of attention heads
    n_layers: int
        The number of transformer layers
    n_fft: int
        The number of Fourier transform points
    hop_length: int (default 100)
        The distance between neighboring sliding window frames
    """

    def __init__(
        self,
        emb_size=256,  # The size of the embedding layer
        att_num_heads=8,  # The number of attention heads
        n_chans=16,  # The number of channels
        n_layers=4,  # The number of transformer layers
        n_fft=200,  # Related with the frequency resolution
        hop_length=100,
        drop_prob: float = 0.1,
        max_seq_len: int = 1024,  # The maximum sequence length
        attn_dropout=0.2,  # dropout post-attention
        attn_layer_dropout=0.2,  # dropout right after self-attention layer
    ):
        super().__init__()

        self.n_fft = int(n_fft)
        self.hop_length = hop_length

        self.patch_embedding = _PatchFrequencyEmbedding(
            emb_size=emb_size, n_freq=int(self.n_fft // 2 + 1)
        )
        self.transformer = LinearAttentionTransformer(
            dim=emb_size,
            heads=att_num_heads,
            depth=n_layers,
            max_seq_len=max_seq_len,
            attn_layer_dropout=attn_layer_dropout,
            attn_dropout=attn_dropout,
        )
        self.positional_encoding = _PositionalEncoding(emb_size, drop_prob=drop_prob)

        # channel token, N_channels >= your actual channels
        self.channel_tokens = nn.Embedding(
            num_embeddings=n_chans, embedding_dim=emb_size
        )
        self.index = nn.Parameter(torch.LongTensor(range(n_chans)), requires_grad=False)

    def stft(self, sample):
        """
        Short-time Fourier transform.
        For more details, see `torch.stft`.

        The size of the Fourier transform is obtained by `n_fft`, and the distance
        between neighboring sliding window frames `hop_length` defined in
        the __init__ functions.

        Parameters
        ----------
        sample: Tensor
            channel representation with size (batch_size, n_times)
        Returns
        -------
        spectral: Tensor
            Absolute value of the Fourier transform with size
            (batch_size, n_fft // 2 + 1, n_times // hop_length + 1)
        """
        spectral = torch.stft(
            input=sample.squeeze(1),
            n_fft=int(self.n_fft),
            hop_length=self.hop_length,
            center=False,
            onesided=True,
            return_complex=True,
        )
        return torch.abs(spectral)

    def forward(self, x, n_channel_offset=0, perturb=False):
        """
        Forward pass of the BIOT encoder.

        For each channel, the input is transformed into a spectrogram
        and then embedded using a patch embedding. The channel token
        is added to the patch embedding, and then positional encoding
        is applied. The resulting embeddings are concatenated and
        passed through a transformer layer. The mean of the resulting
        embeddings is returned.

        For each channel in channels, the channels are transformed into a
        spectrogram with STFT; The spectrogram representation is permuted
        and passed through a linear layer to learn some representation over
        the frequency domain.

        For each embedding in the sequence, the channel token is added to
        the patch embedding, and then positional encoding is applied.

        The resulting embeddings are concatenated and passed through a
        transformer layer. The mean of the resulting embeddings is returned.

        Parameters
        ----------
        x: Tensor
            (batch_size, n_channels, n_times)
        n_channel_offset: int (default 0)
            The offset term to be added to the channel tokens
        perturb: bool (default False)
            Randomly select a number of time steps and reduce the
            channel embedding to those time steps.

        Returns
        -------
        emb: Tensor
            (batch_size, emb_size)
        """
        emb_seq = []
        for i in range(x.shape[1]):
            # Getting the spectrogram
            channel_spec_emb = self.stft(x[:, i : i + 1, :])
            # Linear layer to learn some representation over the frequency domain
            # with permutation
            channel_spec_emb = self.patch_embedding(channel_spec_emb)
            batch_size, ts, _ = channel_spec_emb.shape
            # (batch_size, ts, emb)
            # Step by step the following lines do the following operations:
            #    - self.channel_tokens(self.index[i + n_channel_offset]):
            #    Fetches the embedding for a channel specified by i + n_channel_offset,
            #    where i is the current index and n_channel_offset adjusts
            #    which channel's embedding is retrieved.
            #    - .unsqueeze(0).unsqueeze(0):
            #    Adds two singleton dimensions to the embedding tensor.
            #    [emb_size] to [1, 1, emb_size] .
            #    - Repeat(batch_size, ts, 1):
            #    Replicates the embedding tensor to match the batch size and
            #    time steps (ts),
            channel_token_emb = (
                self.channel_tokens(self.index[i + n_channel_offset])
                .unsqueeze(0)
                .unsqueeze(0)
                .repeat(batch_size, ts, 1)
            )
            # (batch_size, ts, emb)
            # The positional embedding is explaining with more
            # detail in the _PositionalEncoding class.
            channel_emb = self.positional_encoding(channel_spec_emb + channel_token_emb)
            # In case of perturb, the time steps are randomly selected
            # and the channel embedding is reduced to a random number
            # of time steps.
            if perturb:
                ts = channel_emb.shape[1]
                ts_new = torch.randint(low=ts // 2, high=ts, size=(1,)).item()
                selected_ts = torch.randperm(ts)[:ts_new]
                channel_emb = channel_emb[:, selected_ts]
            emb_seq.append(channel_emb)

        # Concat and transformer
        # (batch_size, 16 * ts, emb)
        emb = torch.cat(emb_seq, dim=1)
        # (batch_size, emb)
        emb = self.transformer(emb).mean(dim=1)
        return emb
