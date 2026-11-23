# Authors: Raghav Rathi <44058192+raghav-rathi@users.noreply.github.com>
#
# License: BSD (3-clause)

from __future__ import annotations

import torch
import torch.nn as nn

from braindecode.functional import check_n_chans, pad_or_crop_to_patches, patch_count
from braindecode.models.base import EEGModuleMixin


class DeepSOZ(EEGModuleMixin, nn.Module):
    r"""DeepSOZ from Shama et al. (2023) [Shama2023]_.

    :bdg-info:`Attention/Transformer` :bdg-secondary:`Recurrent`

    .. versionadded:: 2.1

    DeepSOZ detects seizures second by second in multichannel scalp EEG and
    localizes their onset zone (SOZ) channel by channel [Shama2023]_.
    DeepSOZ-HEM [Niranjan2025]_, the same network retrained with hard example
    mining, came fourth in the 2025 seizure detection challenge, which was
    scored with the SzCORE framework [Dan2025]_. This class is written from the
    paper; with the DeepSOZ-HEM weights it gives the outputs of the authors'
    code.

    .. rubric:: Architecture Overview

    The one-second segments of each channel are the tokens, without a
    convolutional front end. One Transformer encoder layer mixes the channel
    tokens of each second with a learned global token, a bidirectional LSTM
    follows the global tokens over the seconds, and a linear layer gives the
    seizure logits of every second::

        (batch, n_chans, n_times)
        -> tokens + channel embedding (batch, n_patches, n_chans, patch_size)
        -> + global token (batch, n_patches, n_chans + 1, patch_size)
        -> encoder, each second alone (batch, n_patches, n_chans + 1, patch_size)
        -> lstm over the global tokens (batch, n_patches, 2 * lstm_hidden_size)
        -> final_layer (batch, n_outputs, n_patches)

    With ``return_soz=True``, the channel tokens are also scored and pooled
    over time into one logit per channel::

        encoded channel tokens (batch, n_patches, n_chans, patch_size)
        -> soz_layer (batch, n_patches, n_chans)
        -> mean over time weighted by the seizure logit (batch, n_chans)

    ``n_patches`` is ``n_times / patch_size``, rounded as ``on_non_divisible``
    says.

    .. rubric:: Macro Components

    ``DeepSOZ.channel_embedding``
        **Operations.** An :class:`~torch.nn.Embedding` with one row per
        channel, added to that channel's tokens, and a last row that is the
        global token of every second (``detector.pos_encoder`` in the
        reference). The paper writes it as a linear map of one-hot channel
        indices.

        **Role.** Tells the encoder which electrode a token comes from.

    ``DeepSOZ.encoder``
        **Operations.** A post-norm :class:`~torch.nn.TransformerEncoderLayer`
        of width ``patch_size``, with ``num_heads`` heads and a feed-forward
        block of width ``dim_feedforward``, applied to the ``n_chans + 1``
        tokens of each second (``detector.tx_encoder``).

        **Role.** Mixes the channels of each second into a global encoding and
        channel encodings.

    ``DeepSOZ.lstm``
        **Operations.** A one-layer bidirectional :class:`~torch.nn.LSTM` with
        ``lstm_hidden_size`` units per direction, over the sequence of global
        encodings (``detector.multi_lstm``).

        **Role.** Follows how the activity evolves from second to second.

    ``DeepSOZ.soz_layer``
        **Operations.** A :class:`~torch.nn.Linear` layer, shared by the
        channels, from each channel encoding to a score (``hc_linear``).

        **Role.** Scores each channel and second for the onset zone; used only
        with ``return_soz=True``.

    ``DeepSOZ.final_layer``
        **Operations.** A :class:`~torch.nn.Linear` layer from the LSTM outputs
        to ``n_outputs`` logits for every second (``detector.multi_linear``).

        **Role.** The seizure detection head, without the softmax of the
        reference.

    .. rubric:: Temporal, Spatial, and Spectral Encoding

    - **Temporal:** the raw samples of a channel within one second form its
      token; across seconds, the bidirectional LSTM reads the whole window in
      both directions, so the logit of a second also depends on the seconds
      after it.
    - **Spatial:** the encoder's self-attention mixes the channels of each
      second; the channel embedding fixes the channel order of training.
    - **Spectral:** not modelled; the authors band-pass the EEG between 1.6
      and 30 Hz before the model.

    .. rubric:: Additional Mechanisms

    - **Onset zone pooling.** The SOZ logit of channel :math:`c` is
      :math:`\sum_t w_t s_{t,c}`, with :math:`s_{t,c}` the ``soz_layer`` score
      of second :math:`t` and :math:`w = \mathrm{softmax}_t(z_t)` the softmax
      over time of the seizure logit :math:`z_t` (the last output). The
      reference code computes it this way; Equation 3 of the paper normalizes
      the seizure probabilities instead.
    - **Reference parity.** With the DeepSOZ-HEM weights, the logits, the SOZ
      probabilities and the input gradients equal those of the reference
      ``txlstm_szpool`` in float64 and float32 on CPU, and the parameter
      gradients differ by at most 2e-15 relative in float64. The differences
      are in the interface: logits with a channel axis instead of
      ``(batch, n_patches, 2)`` probabilities, SOZ logits instead of
      probabilities and only on request, sizes as parameters, and
      ``on_non_divisible`` for inputs that are not a whole number of tokens.

    .. rubric:: Pretrained Weights

    The authors released one trained model, DeepSOZ-HEM (GPL-3.0), as
    ``deepsoz-hem/src/deepsoz/deepsoz_fold4.pth_4.tar`` in [Niranjan2025]_. It
    uses the defaults with ``n_chans=19`` and ``n_outputs=2`` (the second
    output is the seizure class), was trained with a cross-entropy weighted
    0.185 and 0.815, and expects this input, which the model does not prepare:

    - 19 channels in the order Fp1, Fp2, F7, F3, Fz, F4, F8, T3, C3, Cz, C4,
      T4, T5, P3, Pz, P4, T6, O1, O2, average reference, sampled at 256 Hz,
      as in the SzCORE BIDS datasets such as
      :class:`~braindecode.datasets.SIENA`;
    - in microvolts (braindecode datasets hold volts), each channel filtered
      over the whole recording by a zero-phase fourth-order Butterworth
      low-pass at 30 Hz then high-pass at 1.6 Hz, and clipped to its mean plus
      or minus two standard deviations;
    - 10 min windows (``n_times=153600``) every 8 min, the last one padded
      with zeros.

    The authors keep the larger softmax seizure probability where two windows
    overlap, smooth it with a 25 s moving average and mark the seconds above
    0.65 as seizure. On :class:`~braindecode.datasets.SIENA`, which was part of
    its training data, this gives the eight SzCORE scores the challenge
    published for the model (event F1 0.820), in
    ``results/amruth1234-deepsoz-hem-v0-1-3`` of
    https://github.com/esl-epfl/szcore.

    The released model was trained for detection only: its ``soz_layer`` was
    not trained, and on the 47 seizures of Siena its top channel is in the
    annotated onset zone less often than a random channel. Train that layer
    before using the SOZ output, as in the paper's second stage: freeze the
    LSTM and minimize a binary cross-entropy on the sigmoid of the SOZ logits,
    plus 0.1 times the detection loss and an L1 penalty on the SOZ
    probabilities [Shama2023]_.

    Parameters
    ----------
    patch_size : int, default=256
        Samples per token, one second: the sampling rate (200 Hz in the paper,
        256 Hz for DeepSOZ-HEM). It is also the width of the Transformer.
    num_heads : int, default=8
        Attention heads of the encoder. Must divide ``patch_size``.
    dim_feedforward : int, default=256
        Width of the encoder's feed-forward block. With ``patch_size=200`` the
        defaults give the 510K parameters of the paper's model.
    lstm_hidden_size : int, default=100
        Units of each direction of the LSTM.
    drop_prob : float, default=0.25
        Dropout of the encoder (0.25 in DeepSOZ-HEM, 0.15 in the paper).
    on_non_divisible : {"pad", "crop", "error"}, default="pad"
        How an input that is not a whole number of tokens is handled: right
        zero-padding, cropping, or a ``ValueError``.
    activation : type[nn.Module], default=nn.ReLU
        Non-linearity of the encoder's feed-forward block.

    References
    ----------
    .. [Shama2023] Shama, D. M., Jing, J., & Venkataraman, A. (2023). DeepSOZ:
       A robust deep model for joint temporal and spatial seizure onset
       localization from multichannel EEG data. In Medical Image Computing and
       Computer Assisted Intervention (MICCAI 2023), LNCS 14222, 184-194.
       https://doi.org/10.1007/978-3-031-43993-3_18
    .. [Niranjan2025] Niranjan, A., Shama, D. M., & Venkataraman, A. (2025).
       DeepSOZ-HEM. Submission to the 2025 seizure detection challenge.
       https://github.com/amruth-sn/deepsoz-hem
    .. [Dan2025] Dan, J., Pale, U., Amirshahi, A., Cappelletti, W., Ingolfsson,
       T. M., Wang, X., Cossettini, A., Bernini, A., Benini, L., Beniczky, S.,
       Atienza, D., & Ryvlin, P. (2025). SzCORE: Seizure Community Open-Source
       Research Evaluation framework for the validation of
       electroencephalography-based automated seizure detection algorithms.
       Epilepsia, 66(S3), 14-24. https://doi.org/10.1111/epi.18113
    """

    _time_axis = "free"
    _channel_axis = "fixed"  # fixed: channel_embedding.weight

    def __init__(
        self,
        # braindecode parameters
        n_outputs=None,
        n_chans=None,
        chs_info=None,
        n_times=None,
        input_window_seconds=None,
        sfreq=None,
        *,
        # model-specific parameters
        patch_size: int = 256,
        num_heads: int = 8,
        dim_feedforward: int = 256,
        lstm_hidden_size: int = 100,
        drop_prob: float = 0.25,
        on_non_divisible: str = "pad",
        activation: type[nn.Module] = nn.ReLU,
    ):
        super().__init__(
            n_outputs=n_outputs,
            n_chans=n_chans,
            chs_info=chs_info,
            n_times=n_times,
            input_window_seconds=input_window_seconds,
            sfreq=sfreq,
        )
        del n_outputs, n_chans, chs_info, n_times, input_window_seconds, sfreq

        for name, value in (
            ("patch_size", patch_size),
            ("dim_feedforward", dim_feedforward),
            ("lstm_hidden_size", lstm_hidden_size),
        ):
            if value < 1:
                raise ValueError(f"{name} must be positive, got {value}.")
        if num_heads < 1 or patch_size % num_heads:
            raise ValueError(
                "num_heads must be a positive divisor of patch_size "
                f"({patch_size}), the width of the Transformer; got {num_heads}."
            )
        # Validates on_non_divisible, and that n_times, when given, can be cut
        # into tokens ("crop" needs one token, "error" a multiple of patch_size).
        patch_count(
            patch_size if self._n_times is None else self._n_times,
            patch_size,
            on_non_divisible,
        )
        self.patch_size = patch_size
        self.on_non_divisible = on_non_divisible

        self.channel_embedding = nn.Embedding(self.n_chans + 1, patch_size)
        self.encoder = nn.TransformerEncoderLayer(
            d_model=patch_size,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=drop_prob,
            activation=activation(),
            batch_first=True,
        )
        self.lstm = nn.LSTM(
            patch_size, lstm_hidden_size, batch_first=True, bidirectional=True
        )
        self.soz_layer = nn.Linear(patch_size, 1)
        self.final_layer = nn.Linear(2 * lstm_hidden_size, self.n_outputs)
        self._n_times_step = patch_size
        self._min_n_times = 1 if on_non_divisible == "pad" else patch_size

    def reset_head(self, n_outputs: int) -> None:
        """Replace the detection head for a new number of outputs."""
        self._set_n_outputs(n_outputs)
        old_head = self.final_layer
        self.final_layer = nn.Linear(old_head.in_features, n_outputs).to(
            old_head.weight
        )

    def forward(self, x: torch.Tensor, return_soz: bool = False):
        """Predict seizure logits for every segment of ``patch_size`` samples.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape ``(batch, n_chans, n_times)``.
        return_soz : bool, default=False
            Also return the seizure onset zone logits, pooled over time with a
            softmax of the last output (the seizure logit of the released
            model). A scripted model ignores this flag.

        Returns
        -------
        torch.Tensor or tuple of torch.Tensor
            Logits of shape ``(batch, n_outputs, n_patches)``, and with
            ``return_soz=True`` also the SOZ logits of shape
            ``(batch, n_chans)``.
        """
        if torch.jit.is_scripting():
            # The eager check lives in EEGModuleMixin.__call__ (scripting skips
            # it); without it embedding[n_chans] reads a channel row as the
            # global token.
            check_n_chans(
                x, self.n_chans, "DeepSOZ", hint="build it with that many n_chans."
            )
            # x: (batch, n_chans, n_patches * patch_size), whole tokens
            x = pad_or_crop_to_patches(x, self.patch_size, self.on_non_divisible)
        batch, n_chans, n_times = x.shape
        n_patches = n_times // self.patch_size
        # tokens: (batch, n_patches, n_chans, patch_size), one per channel and second
        tokens = x.reshape(batch, n_chans, n_patches, self.patch_size).transpose(1, 2)
        embedding = self.channel_embedding.weight
        # each channel's embedding added to its tokens, same shape
        tokens = tokens + embedding[:n_chans]
        # global token, the last embedding row: (batch, n_patches, 1, patch_size)
        global_token = embedding[n_chans].expand(batch, n_patches, 1, self.patch_size)
        # tokens: (batch, n_patches, n_chans + 1, patch_size)
        tokens = torch.cat([tokens, global_token], dim=2)
        # encoded: (batch, n_patches, n_chans + 1, patch_size), each second alone
        encoded = self.encoder(tokens.flatten(0, 1)).unflatten(0, (batch, n_patches))
        # features: (batch, n_patches, 2 * lstm_hidden_size), over the global tokens
        features, _ = self.lstm(encoded[:, :, n_chans])
        # logits: (batch, n_outputs, n_patches)
        logits = self.final_layer(features).transpose(1, 2)

        if return_soz and not torch.jit.is_scripting():
            # channel_scores: (batch, n_patches, n_chans)
            channel_scores = self.soz_layer(encoded[:, :, :n_chans]).squeeze(-1)
            # weights: (batch, n_patches), softmax over time of the seizure logit
            weights = torch.softmax(logits[:, -1], dim=-1)
            # soz_logits: (batch, n_chans), the scores averaged with these weights
            soz_logits = (weights.unsqueeze(-1) * channel_scores).sum(dim=1)
            return logits, soz_logits
        return logits
