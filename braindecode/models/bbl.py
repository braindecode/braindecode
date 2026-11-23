# Authors: Dulhan Jayalath and the-brains-bitter-lesson contributors
#          Bruno Aristimunha <b.aristimunha@gmail.com> (braindecode adaptation)
#
# License: MIT
#
# Adapted from https://github.com/neural-processing-lab/the-brains-bitter-lesson (MIT).
from __future__ import annotations

from typing import Dict, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange

from braindecode.functional import check_ch_mask
from braindecode.models.base import EEGModuleMixin
from braindecode.models.util import warn_if_sfreq_differs
from braindecode.modules import SEANetEncoder


class BBL(EEGModuleMixin, nn.Module, license="mit"):
    r"""BBL from Jayalath et al. (2025) [bbl]_.

    :bdg-danger:`Foundation Model` :bdg-success:`Convolution` :bdg-secondary:`Recurrent`

    .. figure:: ../_static/model/bbl_arch.png
       :align: center
       :alt: BBL architecture
       :width: 1000px

       Architecture overview with the dataset-conditional layer, the cortex encoder
       and the pre-training and fine-tuning heads, reproduced from [bbl]_ (CC BY 4.0).

    .. versionadded:: 2.1

    BBL is the MEG encoder of *The Brain's Bitter Lesson*, a self-supervised
    representation for speech decoding from MEG. It is pretrained on CamCAN with
    three pretext tasks (band, phase and amplitude). The implementation follows
    the authors' code, ``neural-processing-lab/the-brains-bitter-lesson`` at
    commit ``3965a2be``.

    .. rubric:: Architecture Overview

    A dataset-conditional convolutional projector, an EnCodec SEANet encoder
    averaged over time, and a learned subject embedding feed a linear head. With
    ``B`` the batch, ``C = n_chans``, ``T = n_times``,
    ``T' = ceil(T / downsample_stride)`` and ``T'' = ceil(T' / prod(ratios))``::

        x (B, C, T)
        -> zero-pad to n_sensors         (B, n_sensors, T)
        -> projector                     (B, n_datasets * model_dim, T')
        -> dataset gate                  (B, model_dim, T')
        -> SEANetEncoder                 (B, model_dim, T'')
        -> mean over time                (B, model_dim)
        -> concat subject embedding      (B, model_dim + subject_dim)
        -> final_layer                   (B, n_outputs)

    .. rubric:: Macro Components

    ``BBL.projector``
        **Operations.** ``Conv1d(n_sensors, n_datasets * model_dim,
        sensor_kernel_size) -> activation -> Conv1d(downsample_kernel_size,
        stride downsample_stride)``; the one-hot ``dataset_index`` keeps one
        ``model_dim`` slice. **Role.** Harmonizes sensor layouts across datasets.

    ``BBL.encoder``
        **Operations.** :class:`~braindecode.modules.SEANetEncoder` (residual
        convolutions, downsampling by ``prod(ratios)``, two-layer LSTM).
        **Role.** Self-supervised backbone (band, phase and amplitude pretext
        tasks in the paper).

    ``BBL.subject_embedding``
        **Operations.** ``Embedding(n_subjects, subject_dim)`` concatenated to
        the pooled features. **Role.** Subject conditioning: one learned vector
        per subject, so that the head can account for what varies between
        subjects (Section 3.1 of the paper).

    ``BBL.final_layer``
        **Operations.** ``Linear(model_dim + subject_dim, n_outputs)``, the
        paper's linear probe. **Role.** The task head, as in the authors'
        ``Prober``; it is not pretrained.

    .. rubric:: Temporal, Spatial, and Spectral Encoding

    - **Temporal:** strided convolutions and the LSTM, then a mean over time.
    - **Spatial:** the projector mixes all (zero-padded) sensors.
    - **Spectral:** learned by the band and phase pretext tasks.

    .. rubric:: Additional Mechanisms

    - **Dataset gate:** ``dataset_index`` picks one projector slice (slot 0 by
      default). ``subject_index`` is offset per dataset slot: subject ``s`` of
      slot ``d`` uses row ``d * 1024 + s``.
    - **Subject fallback:** without ``subject_index``, the mean embedding of
      the rows flagged in the ``trained_subjects`` buffer (the paper's
      novel-subject fallback). Only CamCAN's rows are trained and flagged in the
      release; set the buffer to the rows of your own training before relying on it.
    - **Temporal mask:** ``temporal_mask`` hides raw time samples before any
      convolution or LSTM, and ``return_features`` adds ``temporal_features``,
      the encoder output before pooling.

    .. rubric:: Pretrained Weights

    ``braindecode/bbl-pretrained`` (revision ``5a19a369``) holds the authors'
    CamCAN checkpoint (slot 0); the head is not pretrained::

        from braindecode.models import BBL
        model = BBL.from_pretrained("braindecode/bbl-pretrained", n_outputs=2)

    The checkpoint takes its 306 Elekta ``MEG*`` channels in FIF order,
    band-passed 0.1-125 Hz with 50 and 100 Hz notches, then resampled to 250 Hz
    (Section 4.1 of the paper), on 0.5 s windows (125 samples). Each window is
    scaled per channel by the recording's median and IQR, clipped to
    ``[-5, 5]`` and first 0.5 s mean-subtracted, so inputs are dimensionless
    (authors' ``data/preprocess.py`` and ``data/utils.py``; ``forward`` only
    does the zero-padding to 306 sensors). The paper's bad-channel interpolation
    is not in the authors' code. On the same inputs, the features and the
    gradients equal those of the authors' ``Backbone`` with these weights; the
    mean-subject fallback differs by float32 rounding only (:gh:`1553`).

    Parameters
    ----------
    n_sensors : int, default=306
        Sensor slots of the projector; inputs with fewer channels are
        zero-padded, as the reference collator does (306: Elekta MEG).
    n_datasets : int, default=4
        Dataset slots of the projector gate.
    n_subjects : int, default=4096
        Rows of the subject embedding (the release: 1024 per dataset slot).
    subject_dim : int, default=16
        Subject embedding dimension.
    model_dim : int, default=512
        Projector and encoder output dimension.
    sensor_kernel_size : int, default=3
        Kernel size of the projector's first convolution (3 in the release); odd,
        so that its padding ``sensor_kernel_size // 2`` keeps the length.
    downsample_kernel_size : int, default=3
        Kernel size of the projector's strided convolution (3 in the release);
        odd, padded by ``downsample_kernel_size // 2``.
    downsample_stride : int, default=2
        Stride of that convolution, so the projector outputs
        ``ceil(n_times / downsample_stride)`` steps (2 in the release).
    n_filters : int, default=128
        Channels after the first SEANet convolution.
    ratios : tuple of int, default=(4, 2, 2)
        SEANet downsampling ratios.
    kernel_size : int, default=7
        Kernel size of the first SEANet convolution.
    last_kernel_size : int, default=7
        Kernel size of the last SEANet convolution.
    activation : type[nn.Module], default=nn.GELU
        Activation of the projector (the SEANet keeps EnCodec's ELU).

    References
    ----------
    .. [bbl] Jayalath, D., Landau, G., Shillingford, B., Woolrich, M.,
       Parker Jones, O. (2025). The Brain's Bitter Lesson: Scaling Speech
       Decoding With Self-Supervised Learning. ICML 2025.
       https://arxiv.org/abs/2406.04328

    `License <https://github.com/neural-processing-lab/the-brains-bitter-lesson/blob/main/LICENSE>`_
    """

    _time_axis = "free"
    _channel_axis = "free"

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
        n_sensors: int = 306,
        n_datasets: int = 4,
        n_subjects: int = 4096,
        subject_dim: int = 16,
        model_dim: int = 512,
        sensor_kernel_size: int = 3,
        downsample_kernel_size: int = 3,
        downsample_stride: int = 2,
        n_filters: int = 128,
        ratios: tuple[int, ...] = (4, 2, 2),
        kernel_size: int = 7,
        last_kernel_size: int = 7,
        activation: type[nn.Module] = nn.GELU,
    ):
        super().__init__(
            n_outputs=n_outputs,
            n_chans=n_chans,
            chs_info=chs_info,
            n_times=n_times,
            input_window_seconds=input_window_seconds,
            sfreq=sfreq,
        )
        # The released CamCAN checkpoint runs at 250 Hz (authors' data/preprocess.py).
        warn_if_sfreq_differs("BBL", sfreq, 250.0)
        del n_outputs, n_chans, chs_info, n_times, input_window_seconds, sfreq
        if self.n_chans > n_sensors:
            raise ValueError(
                f"n_chans ({self.n_chans}) is larger than n_sensors ({n_sensors})."
            )
        if sensor_kernel_size % 2 == 0 or downsample_kernel_size % 2 == 0:
            raise ValueError(
                "sensor_kernel_size and downsample_kernel_size must be odd, got "
                f"{sensor_kernel_size} and {downsample_kernel_size}."
            )
        self.n_sensors = n_sensors
        self.n_datasets = n_datasets
        self.features_dim = model_dim + subject_dim

        # Paddings are kernel_size // 2 (an odd kernel keeps the length at stride 1):
        # (batch, n_sensors, n_times) -> (batch, n_datasets * model_dim, n_times)
        # -> (batch, n_datasets * model_dim, ceil(n_times / downsample_stride)).
        self.projector = nn.Sequential(
            nn.Conv1d(
                n_sensors,
                n_datasets * model_dim,
                sensor_kernel_size,
                padding=sensor_kernel_size // 2,
            ),
            activation(),
            nn.Conv1d(
                n_datasets * model_dim,
                n_datasets * model_dim,
                downsample_kernel_size,
                stride=downsample_stride,
                padding=downsample_kernel_size // 2,
            ),
        )
        self.split_datasets = Rearrange("b (d c) t -> b d c t", d=n_datasets)
        self.encoder = SEANetEncoder(
            model_dim,
            n_filters,
            ratios,
            kernel_size,
            last_kernel_size,
            in_channels=model_dim,
        )
        self.subject_embedding = nn.Embedding(n_subjects, subject_dim)
        # Rows averaged when no subject_index is given.
        self.register_buffer("trained_subjects", torch.ones(n_subjects))
        self.final_layer = nn.Linear(self.features_dim, self.n_outputs)
        self._min_n_times = 1

    def forward(
        self,
        x: torch.Tensor,
        subject_index: Optional[torch.Tensor] = None,
        dataset_index: Optional[torch.Tensor] = None,
        return_features: bool = False,
        ch_mask: Optional[torch.Tensor] = None,
        temporal_mask: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Dict[str, Optional[torch.Tensor]]]:
        """Classify ``x`` of shape ``(batch, n_chans, n_times)``.

        ``subject_index`` and ``dataset_index`` are ``(batch,)`` integer
        tensors; ``dataset_index`` defaults to slot 0. ``ch_mask`` is an
        optional ``(batch, n_chans)`` mask of real channels, see
        :func:`~braindecode.functional.check_ch_mask`: padded channels are set
        to zero, the empty sensor slots of the reference collator, so a
        recording padded in a batch gives its output alone. The projector
        mixes every sensor slot into each feature, so the model has no
        per-channel ``tokens``. Shorter recordings occupy the first sensor
        slots; interior missing sensors must retain their original slots.

        ``temporal_mask`` is an optional boolean ``(batch, n_times)`` mask:
        **True hides** a raw time sample on every channel, unlike ``ch_mask``
        where True means real. Hidden samples (including NaNs) become zero
        before any convolution or LSTM and receive no input gradient.
        All-hidden windows are allowed. Keep clean targets and mask sampling
        downstream; this is input corruption, not the paper's band/amp/phase
        objectives.

        ``return_features=True`` returns ``"features"``
        ``(batch, model_dim + subject_dim)``, ``"cls_token"`` (None) and
        ``"temporal_features"``, the encoder output before pooling,
        ``(batch, model_dim, n_steps)`` with
        ``n_steps = ceil(n_times / (downsample_stride * prod(ratios)))``. The
        steps are not independent receptive fields: the LSTM carries prefix
        context and the noncausal/reflect convolutions mix neighboring samples.
        """
        ch_mask = check_ch_mask(ch_mask, x)
        batch, n_times = x.shape[0], x.shape[-1]
        if temporal_mask is not None:
            shape = [batch, n_times]
            if temporal_mask.dtype != torch.bool or list(temporal_mask.shape) != shape:
                raise ValueError(
                    "temporal_mask must be boolean with shape (batch, n_times)."
                )
            temporal_mask = temporal_mask.to(device=x.device)  # (batch, n_times)
            # (batch, n_chans, n_times), hidden raw samples zeroed BEFORE context
            x = x.masked_fill(temporal_mask[:, None, :], 0.0)
        if ch_mask is not None:
            # (batch, n_chans, n_times), padded channels zeroed
            x = x.masked_fill(~ch_mask[:, :, None], 0.0)
        # (batch, n_chans, n_times) -> (batch, n_sensors, n_times); a wider
        # input than n_sensors is not cropped, the projector rejects it.
        x = F.pad(x, (0, 0, 0, max(self.n_sensors - x.shape[1], 0)))
        # (batch, n_sensors, n_times) -> (batch, n_datasets * model_dim, n_half),
        # n_half = ceil(n_times / downsample_stride)
        x = self.projector(x)
        # -> (batch, n_datasets, model_dim, n_half)
        x = self.split_datasets(x)
        if dataset_index is None:
            # (batch,)
            dataset_index = torch.zeros(batch, dtype=torch.long, device=x.device)
        # Compare with arange instead of F.one_hot, which reads the index range
        # on the host.
        slots = torch.arange(self.n_datasets, device=x.device)  # (n_datasets,)
        gate = (dataset_index[:, None] == slots).to(x.dtype)  # (batch, n_datasets)
        # (batch, n_datasets, model_dim, n_half) -> (batch, model_dim, n_half)
        x = (x * gate[:, :, None, None]).sum(dim=1)
        # -> (batch, model_dim, n_steps), n_steps = ceil(n_half / prod(ratios))
        x = self.encoder(x)
        temporal_features = x  # (batch, model_dim, n_steps)
        x = x.mean(dim=-1)  # (batch, model_dim)
        if subject_index is None:
            # (n_subjects,)
            weights = self.trained_subjects / self.trained_subjects.sum()
            subject = weights @ self.subject_embedding.weight  # (subject_dim,)
            subject = subject.expand(batch, -1)  # (batch, subject_dim)
        else:
            subject = self.subject_embedding(subject_index)  # (batch, subject_dim)
        feat = torch.cat([x, subject], dim=1)  # (batch, model_dim + subject_dim)
        if return_features:
            out: Dict[str, Optional[torch.Tensor]] = {
                "features": feat,
                "cls_token": None,  # nosec B105
                "temporal_features": temporal_features,
            }
            return out
        return self.final_layer(feat)  # (batch, n_outputs)

    def reset_head(self, n_outputs: int) -> None:
        """Replace the linear probe for ``n_outputs`` classes."""
        self._set_n_outputs(n_outputs)
        self.final_layer = nn.Linear(self.features_dim, n_outputs)
