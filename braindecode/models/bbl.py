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

from braindecode.models.base import EEGModuleMixin
from braindecode.models.util import warn_if_sfreq_differs
from braindecode.modules import SEANetEncoder


class BBL(EEGModuleMixin, nn.Module, license="mit"):
    r"""BBL, the MEG encoder of *The Brain's Bitter Lesson* from Jayalath et al. (2025) [bbl]_.

    :bdg-danger:`Foundation Model` :bdg-success:`Convolution` :bdg-secondary:`Recurrent`

    .. rubric:: Architecture Overview

    A dataset-conditional convolutional projector, an EnCodec SEANet encoder
    averaged over time, and a learned subject embedding feed a linear head::

        (batch, n_chans, n_times) -> zero-pad to n_sensors -> projector ->
        dataset gate -> SEANetEncoder -> mean over time -> concat subject
        embedding -> (batch, n_outputs)

    .. rubric:: Macro Components

    ``BBL.projector``
        **Operations.** ``Conv1d(n_sensors, n_datasets * model_dim, 3) -> activation ->
        Conv1d(stride 2)``; the one-hot ``dataset_index`` keeps one
        ``model_dim`` slice. **Role.** Harmonizes sensor layouts across datasets.

    ``BBL.encoder``
        **Operations.** :class:`~braindecode.modules.SEANetEncoder` (residual
        convolutions, downsampling by ``prod(ratios)``, two-layer LSTM).
        **Role.** Self-supervised backbone (band, phase and amplitude pretext
        tasks in the paper).

    ``BBL.subject_embedding``
        **Operations.** ``Embedding(n_subjects, subject_dim)`` concatenated to
        the pooled features. Without ``subject_index``, the mean embedding of
        the rows flagged in the ``trained_subjects`` buffer (the paper's
        novel-subject fallback).

    ``BBL.final_layer``
        **Operations.** ``Linear(model_dim + subject_dim, n_outputs)``, the
        paper's linear probe.

    .. rubric:: Temporal, Spatial, and Spectral Encoding

    - **Temporal:** strided convolutions and the LSTM, then a mean over time.
    - **Spatial:** the projector mixes all (zero-padded) sensors.
    - **Spectral:** learned by the band and phase pretext tasks.

    .. important::

        The released checkpoint was pretrained on CamCAN (dataset slot 0),
        band-passed 0.1-125 Hz with 50 and 100 Hz notches, then resampled to
        250 Hz, on 0.5 s windows (125 samples). Each window is scaled per
        channel by the median and interquartile range fitted on its recording,
        clipped to ``[-5, 5]`` and has its first 0.5 s mean subtracted, so
        inputs are dimensionless (authors' ``data/preprocess.py`` and
        ``data/utils.py``; ``forward`` only does the zero-padding to 306
        sensors). Its projector expects CamCAN's 306 Elekta ``MEG*``
        channels in FIF order. ``subject_index`` is offset per dataset slot:
        subject ``s`` of slot ``d`` uses row ``d * 1024 + s``. Only CamCAN
        subject rows of ``subject_embedding`` are trained, and
        ``trained_subjects`` flags them; the head is not pretrained. Set
        ``trained_subjects`` to the rows your own training used before relying
        on the mean-subject fallback.

    .. versionadded:: 2.1

    Parameters
    ----------
    n_sensors : int
        Sensor slots of the projector; inputs with fewer channels are
        zero-padded, as the reference collator does (306: Elekta MEG).
    n_datasets : int
        Dataset slots of the projector gate.
    n_subjects : int
        Rows of the subject embedding (the release: 1024 per dataset slot).
    subject_dim : int
        Subject embedding dimension.
    model_dim : int
        Projector and encoder output dimension.
    n_filters : int
        Channels after the first SEANet convolution.
    ratios : tuple of int
        SEANet downsampling ratios.
    kernel_size : int
        Kernel size of the first SEANet convolution.
    last_kernel_size : int
        Kernel size of the last SEANet convolution.
    activation : type[nn.Module]
        Activation of the projector (the SEANet keeps EnCodec's ELU).

    References
    ----------
    .. [bbl] Jayalath, D., Landau, G., Shillingford, B., Woolrich, M.,
       Parker Jones, O. (2025). The Brain's Bitter Lesson: Scaling Speech
       Decoding With Self-Supervised Learning. ICML 2025.
       https://arxiv.org/abs/2406.04328

    `License <https://github.com/neural-processing-lab/the-brains-bitter-lesson/blob/main/LICENSE>`_
    """

    def __init__(
        self,
        n_outputs=None,
        n_chans=None,
        chs_info=None,
        n_times=None,
        input_window_seconds=None,
        sfreq=None,
        *,
        n_sensors: int = 306,
        n_datasets: int = 4,
        n_subjects: int = 4096,
        subject_dim: int = 16,
        model_dim: int = 512,
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
        warn_if_sfreq_differs("BBL", sfreq, 250.0)
        del n_outputs, n_chans, chs_info, n_times, input_window_seconds, sfreq
        if self.n_chans > n_sensors:
            raise ValueError(
                f"n_chans ({self.n_chans}) is larger than n_sensors ({n_sensors})."
            )
        self.n_sensors = n_sensors
        self.n_datasets = n_datasets
        self.features_dim = model_dim + subject_dim

        self.projector = nn.Sequential(
            nn.Conv1d(n_sensors, n_datasets * model_dim, 3, padding=1),
            activation(),
            nn.Conv1d(
                n_datasets * model_dim, n_datasets * model_dim, 3, stride=2, padding=1
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

    def forward(
        self,
        x: torch.Tensor,
        subject_index: Optional[torch.Tensor] = None,
        dataset_index: Optional[torch.Tensor] = None,
        return_features: bool = False,
    ) -> Union[torch.Tensor, Dict[str, Optional[torch.Tensor]]]:
        """Classify ``x`` of shape ``(batch, n_chans, n_times)``.

        ``subject_index`` and ``dataset_index`` are ``(batch,)`` integer
        tensors; ``dataset_index`` defaults to slot 0.
        """
        batch = x.shape[0]
        x = F.pad(x, (0, 0, 0, self.n_sensors - self.n_chans))
        x = self.projector(x)
        x = self.split_datasets(x)
        if dataset_index is None:
            dataset_index = torch.zeros(batch, dtype=torch.long, device=x.device)
        # Compare with arange instead of F.one_hot, which reads the index range
        # on the host.
        slots = torch.arange(self.n_datasets, device=x.device)
        gate = (dataset_index[:, None] == slots).to(x.dtype)
        x = (x * gate[:, :, None, None]).sum(dim=1)
        x = self.encoder(x)
        x = x.mean(dim=-1)
        if subject_index is None:
            weights = self.trained_subjects / self.trained_subjects.sum()
            subject = weights @ self.subject_embedding.weight
            subject = subject.expand(batch, -1)
        else:
            subject = self.subject_embedding(subject_index)
        feat = torch.cat([x, subject], dim=1)
        if return_features:
            out: Dict[str, Optional[torch.Tensor]] = {
                "features": feat,
                "cls_token": None,  # nosec B105
            }
            return out
        return self.final_layer(feat)

    def reset_head(self, n_outputs: int) -> None:
        """Replace the linear probe for ``n_outputs`` classes."""
        self._set_n_outputs(n_outputs)
        reference = self.final_layer.weight
        self.final_layer = nn.Linear(self.features_dim, n_outputs).to(
            device=reference.device, dtype=reference.dtype
        )
