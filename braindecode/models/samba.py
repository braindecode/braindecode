# Authors: Jiazhen Hong
#
# License: Apache-2.0
# Adapted from https://github.com/Jiazhen-Hong/SAMBA, with the Mamba-2 block of
# https://github.com/state-spaces/mamba (Apache-2.0) rewritten in PyTorch.

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from braindecode.functional import check_n_chans
from braindecode.models.base import EEGModuleMixin
from braindecode.models.util import extract_channel_locations_from_chs_info
from braindecode.modules import MLP


class SAMBA(EEGModuleMixin, nn.Module, license="apache-2.0"):
    r"""SAMBA from Hong et al. (2026) [samba]_.

    :bdg-success:`Convolution` :bdg-secondary:`Recurrent` :bdg-dark-line:`Channel` :bdg-danger:`Foundation Model`

    .. figure:: ../_static/model/samba_arch.png
       :align: center
       :alt: SAMBA architecture
       :width: 1000px

       SAMBA pre-training with the coordinate-driven spatial embedding, the
       multi-head differential Mamba and the encoder-mediator-decoder, reproduced
       from [samba]_ (CC BY 4.0).

    .. versionadded:: 2.1

    .. rubric:: Architecture Overview

    The encoder of SAMBA, a U-shaped Mamba autoencoder pretrained to rebuild
    masked EEG; the reconstruction decoder is not part of this class. A
    coordinate-driven spatial embedding (CDSE) first projects the electrodes
    onto a fixed target montage from their 3D positions, so the layers behind it
    never see the recording montage. Mamba-2 state space blocks [mamba2]_, with
    max-pooling and convolutions in between, then read the time axis in linear
    time (``p`` is ``pool_size``, 2 by default)::

        (batch, n_chans, n_times)
        -> CDSE                       (batch, n_targets, n_times)
        -> multi-branch convolution   (batch, base_channels, n_times)
        -> Linear, residual Mamba-2   (batch, n_times, base_channels)
        -> pool, convolution          (batch, base_channels, n_times // p)
        -> pool, convolution          (batch, bottleneck_channels, n_times // p**2)
        -> pool                       (batch, bottleneck_channels, n_times // p**3)
        -> Linear, differential Mamba-2, Linear
                                      (batch, n_times // p**3, bottleneck_channels)
        -> mean and std over time     (batch, 2 * bottleneck_channels)
        -> final_layer                (batch, n_outputs)

    .. rubric:: Macro Components

    ``SAMBA.encoder.cdse``
        **Operations.** An MLP ``Linear(3, cdse_hidden_dim) -> ReLU ->
        Linear(cdse_hidden_dim, 1)`` scores the offset from each electrode to
        each target position; a softmax over the electrodes turns the scores into
        weights, and each target channel is the weighted sum of the electrodes.
        **Role.** Montage-agnostic input: any layout with 3D positions reaches
        the same layers.

    ``SAMBA.encoder.input_embedding``
        **Operations.** One ``Conv1d`` per entry of ``branch_kernel_sizes`` (1,
        3 and 7 by default) over the target channels, concatenated and fused by
        a pointwise convolution.
        **Role.** Temporal features at several scales on the target montage.

    ``SAMBA.encoder.encoder1_mamba``
        **Operations.** After a ``Linear``, a residual Mamba-2 block:
        input-dependent step sizes, a causal depthwise convolution, a scan over
        the time axis in chunks of ``chunk_size`` steps and a gated RMS norm.
        The scan is a pure PyTorch recurrence, with no CUDA dependency; its
        results need not be bitwise identical to the authors' fused CUDA
        kernels.
        **Role.** Models the signal at the input rate.

    ``SAMBA.encoder.encoder2``, ``SAMBA.encoder.encoder3``
        **Operations.** ``Conv1d`` of kernel size ``stage_kernel_size`` behind a
        max-pooling by ``pool_size`` each, with no activation in between, as in
        the authors' code.
        **Role.** Divide the time axis by ``pool_size`` at each stage and widen
        the features to ``bottleneck_channels``.

    ``SAMBA.encoder.bottleneck_mamba``
        **Operations.** The channels are split into heads of ``head_dim``. Two
        Mamba-2 blocks read each head, and ``first - lam * second`` followed by
        a GroupNorm cancels what the two readings share, with ``lam`` a learned
        scale per channel. A ``Linear`` mixes the heads, with a residual
        connection.
        **Role.** The multi-head differential Mamba of the paper, at
        ``1 / pool_size ** 3`` of the input rate (one eighth by default).

    ``SAMBA.final_layer``
        **Operations.** ``Linear`` on the mean and the population standard
        deviation (``correction=0``) of the encoder output over time. Not
        pretrained.
        **Role.** The task head; :meth:`reset_head` replaces it.

    .. rubric:: Temporal, Spatial, and Spectral Encoding

    - **Temporal:** Mamba-2 state space blocks at the input rate and at the
      rate left by three max-poolings by ``pool_size``; the output has
      ``n_times // pool_size ** 3`` steps (``n_times // 8`` by default).
    - **Spatial:** electrodes enter only through their 3D positions, so any
      montage with positions can be read.
    - **Spectral:** no explicit spectral transform; the convolutions and the
      state space blocks learn it.

    .. rubric:: Additional Mechanisms

    - **Coordinates.** Positions are read from ``chs_info[i]["loc"][:3]``, in
      metres, and must be in the frame of ``target_positions``. The target
      montage of the authors' released weights is in the frame of the
      ``colin27_1005`` template of MNE (``mne.channels.make_standard_montage``;
      ``standard_1005`` in older versions, deprecated in MNE 1.13), not in the
      head frame that ``raw.set_montage`` writes into ``chs_info``, which is a
      few centimetres away.
    - **Input scale.** The model does not normalise its input. The authors'
      recipe uses EEG at 128 Hz and divides each channel of each window by the
      95th percentile of its absolute value.
    - **Not causal.** The convolutions and the GroupNorm look at the whole
      window, so the encoder is not a streaming model. The Mamba-2 blocks alone
      are causal.
    - **Mamba-2 defaults.** ``encoder1_head_dim``, ``dt_min``, ``dt_max``,
      ``dt_init_floor``, ``a_init_range`` and ``norm_eps`` default to the values
      of the reference ``Mamba2`` of state-spaces/mamba.

    .. rubric:: Pretrained Weights

    The encoders of the authors' SAMBA-2s and SAMBA-100s are on the Hugging
    Face Hub at ``braindecode/samba-2s-pretrained`` and
    ``braindecode/samba-100s-pretrained``, with the authors' values (their files
    are `Dcode-AI/SAMBA-2s <https://huggingface.co/Dcode-AI/SAMBA-2s>`_ at
    ``e553287`` and `Dcode-AI/SAMBA-100s
    <https://huggingface.co/Dcode-AI/SAMBA-100s>`_ at ``3a235a1``). The authors
    retrained both after the checkpoints of the paper were lost, so they are
    not the paper's SAMBA-E and SAMBA-T. SAMBA-2s was pretrained on 2 s windows
    (256 samples) of Emotiv, HBN, Muse and DICE recordings, SAMBA-100s on 100 s
    windows (12,800 samples) of TUAB, so a TUAB task does not test transfer for
    it. Both expect EEG at 128 Hz, each channel of each window divided by the
    95th percentile of its absolute value, with positions in metres in the
    frame of the ``colin27_1005`` template (see Coordinates and Input scale).
    The head is not pretrained: ``from_pretrained`` builds it for your
    ``n_outputs`` and takes the 94 target electrodes of the weights as
    ``chs_info`` unless you pass yours. On a fixed input, the first and last
    five features of both encoders match those of the authors' files, loaded
    with the port of :gh:`1689`, to within 1e-5 (:gh:`1739`); they were not
    compared with the authors' fused CUDA kernels.

    .. code-block:: python

        from braindecode.models import SAMBA

        model = SAMBA.from_pretrained("braindecode/samba-2s-pretrained", n_outputs=2)
        # x: (batch, 94, n_times) -> features: (batch, 256)
        features = model(x, return_features=True)["features"]

    Parameters
    ----------
    target_positions : list of list of float or None, default=None
        Positions ``(n_targets, 3)``, in metres, of the montage onto which the
        CDSE projects the electrodes. ``None`` takes the positions of
        ``chs_info``; the saved config records them, so the trained model
        loads on another montage.
    base_channels : int, default=64
        Width of the multi-branch embedding and of the first Mamba-2 stage.
    bottleneck_channels : int, default=128
        Width of the differential Mamba-2 bottleneck; the head reads twice this
        many features.
    d_state : int, default=16
        State size of every Mamba-2 block.
    d_conv : int, default=2
        Kernel size of the causal depthwise convolution of every Mamba-2 block.
    expand : int, default=8
        Inner width of every Mamba-2 block, as a multiple of its input width; it
        is split into heads. ``base_channels * expand`` must be divisible by
        ``encoder1_head_dim``.
    head_dim : int, default=16
        Channels per head of the differential bottleneck; divides
        ``bottleneck_channels``.
    cdse_hidden_dim : int, default=32
        Hidden width of the MLP of the CDSE.
    branch_kernel_sizes : tuple of int, default=(1, 3, 7)
        Kernel size of each branch of the multi-branch embedding, as in the
        authors' code. Odd, so that the padding of ``k // 2`` keeps the steps.
    pool_size : int, default=2
        Kernel size and stride of the three max-poolings: each divides the
        number of steps by ``pool_size``, so ``n_times >= pool_size ** 3``.
    stage_kernel_size : int, default=3
        Kernel size of the two convolutions behind the first two poolings. Odd,
        so that the padding of ``k // 2`` keeps the steps.
    encoder1_head_dim : int, default=64
        Channels per head of the Mamba-2 of the first stage (``headdim`` in the
        reference ``Mamba2``). ``head_dim`` is that of the bottleneck.
    chunk_size : int, default=64
        Steps per chunk of the Mamba-2 scan; bounds the size of its quadratic
        form.
    dt_min : float, default=1e-3
        Lower end of the log-uniform range of the initial step sizes of the
        Mamba-2 blocks.
    dt_max : float, default=1e-1
        Upper end of that range.
    dt_init_floor : float, default=1e-4
        Floor of the initial step sizes.
    a_init_range : tuple of float, default=(1.0, 16.0)
        ``(low, high)`` of the uniform draw of the initial decay rates ``A`` of
        the Mamba-2 blocks (``A_init_range`` in the reference ``Mamba2``).
    norm_eps : float, default=1e-5
        Epsilon of the RMS norm that follows the gate of every Mamba-2 block.
    channel_strategy : str, default="native"
        How another montage reaches the encoder; see
        :class:`~braindecode.models.base.EEGModuleMixin`. The model reads the
        montage of ``chs_info`` natively, so the default ``"native"`` is usually
        what you want.
    channel_strategy_kwargs : dict or None, default=None
        Options of ``channel_strategy``.

    References
    ----------
    .. [samba] Hong, J., Mackellar, G., Ghane, S. (2026). SAMBA: Toward an EEG
       Foundation Model via Differential Mamba and Coordinate-Driven Spatial
       Embedding. NeurReps 2026 (Proceedings Track). arXiv:2511.18571.
       https://github.com/Jiazhen-Hong/SAMBA
    .. [mamba2] Dao, T., Gu, A. (2024). Transformers are SSMs: Generalized
       Models and Efficient Algorithms Through Structured State Space Duality.
       ICML. arXiv:2405.21060.
    """

    _time_axis = "free"
    _channel_axis = "fixed"  # fixed: encoder.cdse.target_positions

    def __init__(
        self,
        # Braindecode parameters
        n_outputs: int | None = None,
        n_chans: int | None = None,
        chs_info: list[dict] | None = None,
        n_times: int | None = None,
        input_window_seconds: float | None = None,
        sfreq: float | None = None,
        *,
        # Model-specific parameters
        target_positions: list[list[float]] | None = None,
        base_channels: int = 64,
        bottleneck_channels: int = 128,
        d_state: int = 16,
        d_conv: int = 2,
        expand: int = 8,
        head_dim: int = 16,
        cdse_hidden_dim: int = 32,
        branch_kernel_sizes: tuple[int, ...] = (1, 3, 7),
        pool_size: int = 2,
        stage_kernel_size: int = 3,
        encoder1_head_dim: int = 64,
        chunk_size: int = 64,
        dt_min: float = 1e-3,
        dt_max: float = 1e-1,
        dt_init_floor: float = 1e-4,
        a_init_range: tuple[float, float] = (1.0, 16.0),
        norm_eps: float = 1e-5,
        channel_strategy: str = "native",
        channel_strategy_kwargs: dict | None = None,
    ):
        super().__init__(
            n_outputs=n_outputs,
            n_chans=n_chans,
            chs_info=chs_info,
            n_times=n_times,
            input_window_seconds=input_window_seconds,
            sfreq=sfreq,
            channel_strategy=channel_strategy,
            channel_strategy_kwargs=channel_strategy_kwargs,
        )
        del n_outputs, n_chans, chs_info, n_times, input_window_seconds, sfreq

        # Electrode positions of the input channels, read from chs_info (the
        # montage of the encoder under a channel_strategy). A missing or an
        # all-zero loc comes back as NaN.
        # (n_chans, 3)
        positions = torch.as_tensor(
            extract_channel_locations_from_chs_info(self.chs_info, fill_missing=True),
            dtype=torch.get_default_dtype(),
        )
        if not positions.isfinite().all():
            raise ValueError(
                "SAMBA reads the electrode positions from chs_info: every channel "
                "needs a finite loc[:3]; call raw.set_montage(...)."
            )

        # Montage the CDSE projects onto: a copy of the input montage to train from
        # scratch, since loading weights overwrites the targets, not the electrodes
        # (n_targets, 3)
        targets = (
            positions.clone()
            if target_positions is None
            else torch.tensor(target_positions, dtype=torch.get_default_dtype())
        )
        if targets.ndim != 2 or targets.shape[1] != 3 or not targets.isfinite().all():
            raise ValueError(
                "target_positions must be finite with shape (n_targets, 3), got "
                f"shape {tuple(targets.shape)}."
            )
        # Saved configs rebuild these targets, whatever montage loads the model
        self._update_init_kwargs(target_positions=targets.tolist())

        # Every Mamba-2 head takes a whole number of channels: the first stage
        # reads base_channels * expand of them in heads of encoder1_head_dim, the
        # bottleneck splits bottleneck_channels into heads of head_dim
        if base_channels * expand % encoder1_head_dim:
            raise ValueError(
                f"base_channels * expand ({base_channels * expand}) must be "
                f"divisible by encoder1_head_dim ({encoder1_head_dim}), the head "
                "width of the first Mamba-2 block."
            )
        if bottleneck_channels % head_dim:
            raise ValueError(
                f"bottleneck_channels ({bottleneck_channels}) must be divisible "
                f"by head_dim ({head_dim}): a head takes bottleneck_channels / "
                "head_dim channels."
            )
        # A padding of half the kernel keeps the number of steps for odd kernels only
        if any(k % 2 == 0 for k in (*branch_kernel_sizes, stage_kernel_size)):
            raise ValueError(
                "branch_kernel_sizes and stage_kernel_size must be odd, so that a "
                "padding of half the kernel keeps the number of steps; got "
                f"{tuple(branch_kernel_sizes)} and {stage_kernel_size}."
            )

        # Shortest window: the three max-poolings by pool_size leave
        # n_times // pool_size ** 3 steps, and at least one is needed
        self.min_n_times = pool_size**3
        # Derived from chs_info, so not saved with the weights
        self.register_buffer("input_positions", positions, persistent=False)
        self.encoder = _SambaEncoder(
            target_positions=targets,
            base_channels=base_channels,
            bottleneck_channels=bottleneck_channels,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            head_dim=head_dim,
            cdse_hidden_dim=cdse_hidden_dim,
            branch_kernel_sizes=branch_kernel_sizes,
            pool_size=pool_size,
            stage_kernel_size=stage_kernel_size,
            encoder1_head_dim=encoder1_head_dim,
            chunk_size=chunk_size,
            dt_min=dt_min,
            dt_max=dt_max,
            dt_init_floor=dt_init_floor,
            a_init_range=a_init_range,
            norm_eps=norm_eps,
        )
        # Head on the mean and the standard deviation of the encoder output
        self.final_layer = nn.Linear(2 * bottleneck_channels, self.n_outputs)
        self._min_n_times = self.min_n_times

    def forward(
        self, x: torch.Tensor, return_features: bool = False
    ) -> torch.Tensor | dict[str, torch.Tensor | None]:
        """Encode EEG windows and score them.

        Parameters
        ----------
        x : torch.Tensor
            EEG of shape ``(batch, n_chans, n_times)``, in the channel order of
            ``chs_info``, with ``n_times >= pool_size ** 3`` (8 by default).
        return_features : bool, default=False
            Return the features that ``final_layer`` reads instead of its
            outputs.

        Returns
        -------
        torch.Tensor or dict
            Outputs of shape ``(batch, n_outputs)``. With
            ``return_features=True``, ``{"features": Tensor, "cls_token":
            None}`` with the ``(batch, 2 * bottleneck_channels)`` input of
            ``final_layer``.
        """
        if torch.jit.is_scripting():
            # The eager checks live in EEGModuleMixin.__call__ (scripting skips it).
            check_n_chans(x, self.n_chans, "SAMBA")
            if x.shape[-1] < self.min_n_times:
                raise ValueError(
                    f"SAMBA needs at least {self.min_n_times} samples, one for "
                    f"each step it keeps after its max-poolings; got "
                    f"{x.shape[-1]}."
                )
        # Encoder: n_steps = n_times // pool_size ** 3 steps of bottleneck_channels
        # features
        # (batch, n_chans, n_times) -> (batch, n_steps, bottleneck_channels)
        hidden = self.encoder(x, self.input_positions)

        # Mean and population standard deviation over time, side by side
        # (batch, n_steps, bottleneck_channels) -> (batch, 2 * bottleneck_channels)
        features = torch.cat([hidden.mean(dim=1), hidden.std(dim=1, correction=0)], -1)
        if return_features:
            return {"features": features, "cls_token": None}

        # (batch, 2 * bottleneck_channels) -> (batch, n_outputs)
        return self.final_layer(features)

    def reset_head(self, n_outputs: int) -> None:
        """Replace ``final_layer`` by a new one with ``n_outputs`` outputs.

        Parameters
        ----------
        n_outputs : int
            New number of outputs.
        """
        self._set_n_outputs(n_outputs)
        # The features do not change, so the head keeps its input width
        self.final_layer = nn.Linear(self.final_layer.in_features, self.n_outputs)


# -----------------------------------------------------------------------------
# Blocks of SAMBA, below the model class
# -----------------------------------------------------------------------------


class _SambaEncoder(nn.Module):
    """Encoder of SAMBA: the electrodes onto the target montage, then Mamba-2
    stages and convolutions that shorten the time axis ``pool_size ** 3``-fold.

    For ``x`` of shape ``(batch, n_chans, n_times)`` and the electrode
    ``positions`` ``(n_chans, 3)``, with ``n_steps = n_times // pool_size ** 3``:

    1. CDSE: ``(batch, n_targets, n_times)``;
    2. multi-branch convolution: ``(batch, base_channels, n_times)``;
    3. ``Linear`` and a residual Mamba-2 block over time;
    4. max-pooling by ``pool_size`` and a convolution, twice, widening the
       features to ``bottleneck_channels``, then a last max-pooling by
       ``pool_size``: ``(batch, bottleneck_channels, n_steps)``;
    5. ``Linear``, differential Mamba-2 and ``Linear`` over time:
       ``(batch, n_steps, bottleneck_channels)``.

    Parameters
    ----------
    target_positions : torch.Tensor
        Positions of the target montage, ``(n_targets, 3)``.
    base_channels : int
        Width of the embedding and of the first stage.
    bottleneck_channels : int
        Width of the bottleneck.
    d_state, d_conv, expand : int
        Parameters of the Mamba-2 blocks, see ``SAMBA``.
    head_dim : int
        Channels per head of the differential bottleneck.
    cdse_hidden_dim : int
        Hidden width of the MLP of the CDSE.
    branch_kernel_sizes : tuple of int
        Kernel size of each branch of the multi-branch embedding.
    pool_size : int
        Kernel size and stride of the max-poolings.
    stage_kernel_size : int
        Kernel size of the convolutions of the second and third stages.
    encoder1_head_dim : int
        Channels per head of the first-stage Mamba-2 block.
    chunk_size : int
        Steps per chunk of the Mamba-2 scan, see ``SAMBA``.
    dt_min, dt_max, dt_init_floor : float
        Parameters of the Mamba-2 blocks, see ``SAMBA``.
    a_init_range : tuple of float
        Parameter of the Mamba-2 blocks, see ``SAMBA``.
    norm_eps : float
        Parameter of the Mamba-2 blocks, see ``SAMBA``.
    """

    def __init__(
        self,
        target_positions: torch.Tensor,
        base_channels: int,
        bottleneck_channels: int,
        d_state: int,
        d_conv: int,
        expand: int,
        head_dim: int,
        cdse_hidden_dim: int,
        branch_kernel_sizes: tuple[int, ...],
        pool_size: int,
        stage_kernel_size: int,
        encoder1_head_dim: int,
        chunk_size: int,
        dt_min: float,
        dt_max: float,
        dt_init_floor: float,
        a_init_range: tuple[float, float],
        norm_eps: float,
    ):
        super().__init__()
        self.cdse = _CoordinateDrivenSpatialEmbedding(target_positions, cdse_hidden_dim)
        self.input_embedding = _MultiBranchEmbedding(
            len(target_positions), base_channels, branch_kernel_sizes
        )

        # First stage, at the input rate
        self.encoder1_linear = nn.Linear(base_channels, base_channels)
        self.encoder1_mamba = _Mamba2(
            base_channels,
            d_state,
            d_conv,
            expand,
            encoder1_head_dim,
            chunk_size=chunk_size,
            dt_min=dt_min,
            dt_max=dt_max,
            dt_init_floor=dt_init_floor,
            a_init_range=a_init_range,
            norm_eps=norm_eps,
        )
        # Second and third stages, behind a max-pooling each; a padding of half
        # the kernel keeps the number of steps
        self.pool = nn.MaxPool1d(kernel_size=pool_size)
        padding = stage_kernel_size // 2
        self.encoder2 = nn.Conv1d(
            base_channels, base_channels, kernel_size=stage_kernel_size, padding=padding
        )
        self.encoder3 = nn.Conv1d(
            base_channels,
            bottleneck_channels,
            kernel_size=stage_kernel_size,
            padding=padding,
        )

        # Bottleneck, at 1 / pool_size ** 3 of the input rate
        self.bottleneck_in = nn.Linear(bottleneck_channels, bottleneck_channels)
        self.bottleneck_mamba = _DifferentialMamba(
            bottleneck_channels,
            head_dim,
            d_state,
            d_conv,
            expand,
            chunk_size=chunk_size,
            dt_min=dt_min,
            dt_max=dt_max,
            dt_init_floor=dt_init_floor,
            a_init_range=a_init_range,
            norm_eps=norm_eps,
        )
        self.bottleneck_out = nn.Linear(bottleneck_channels, bottleneck_channels)

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """``(batch, n_chans, n_times)`` -> ``(batch, n_steps, bottleneck)``."""
        # Electrodes onto the target montage
        # (batch, n_chans, n_times) -> (batch, n_targets, n_times)
        x = self.cdse(x, positions)
        # (batch, n_targets, n_times) -> (batch, base_channels, n_times)
        x = self.input_embedding(x)

        # First stage: the sequence layers take (batch, n_times, features)
        # (batch, base_channels, n_times) -> (batch, n_times, base_channels)
        x = x.transpose(1, 2)
        # Pointwise mixing of the features; shape unchanged
        x = self.encoder1_linear(x)
        # Residual Mamba-2 block; shape unchanged
        x = x + self.encoder1_mamba(x)

        # Second and third stages: each max-pooling divides the steps by pool_size
        # (batch, n_times, base_channels)
        # -> (batch, base_channels, n_times // pool_size)
        x = self.pool(x.transpose(1, 2))
        x = self.encoder2(x)
        # (batch, base_channels, n_times // pool_size)
        # -> (batch, bottleneck, n_times // pool_size ** 2)
        x = self.encoder3(self.pool(x))

        # Bottleneck: the steps are divided by pool_size once more, back to
        # (batch, steps, features)
        # (batch, bottleneck, n_times // pool_size ** 2) -> (batch, n_steps, bottleneck)
        x = self.pool(x).transpose(1, 2)
        # Pointwise mixing into the bottleneck; shape unchanged
        x = self.bottleneck_in(x)
        # Differential Mamba-2 with its residual; shape unchanged
        x = self.bottleneck_mamba(x)
        # Pointwise mixing out of the bottleneck; shape unchanged
        return self.bottleneck_out(x)


class _CoordinateDrivenSpatialEmbedding(nn.Module):
    """Project the electrodes onto a fixed target montage from their positions.

    For ``x`` of shape ``(batch, n_chans, n_times)`` and the electrode
    ``positions`` ``(n_chans, 3)``, the same for every window of the batch:

    1. offset from every electrode to every target,
       ``(n_targets, n_chans, 3)``;
    2. an MLP scores each offset, ``(n_targets, n_chans)``;
    3. a softmax over the electrodes gives the weights of each target;
    4. each target channel is the weighted sum of the electrodes,
       ``(batch, n_targets, n_times)``.

    Parameters
    ----------
    target_positions : torch.Tensor
        Positions of the target montage, ``(n_targets, 3)``. Saved with the
        weights, since they belong to the pretrained montage.
    hidden_dim : int
        Hidden width of the MLP.
    """

    def __init__(self, target_positions: torch.Tensor, hidden_dim: int):
        super().__init__()
        self.register_buffer("target_positions", target_positions)
        # Linear, ReLU, Linear: (..., 3) -> (..., hidden_dim) -> (..., 1)
        self.score = MLP(
            in_features=3,
            hidden_features=(hidden_dim,),
            out_features=1,
            activation=nn.ReLU,
        )

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """``(batch, n_chans, n_times)`` -> ``(batch, n_targets, n_times)``."""
        # Offset from every electrode to every target
        # (n_targets, 1, 3) - (1, n_chans, 3) -> (n_targets, n_chans, 3)
        offsets = self.target_positions[:, None, :] - positions[None, :, :]
        # Score of every electrode for every target, normalised over electrodes
        # (n_targets, n_chans, 3) -> (n_targets, n_chans)
        weights = self.score(offsets).squeeze(-1).softmax(dim=-1)
        # Weighted sum of the electrodes
        # (n_targets, n_chans) x (batch, n_chans, n_times) -> (batch, n_targets, n_times)
        return torch.einsum("oc,bct->bot", weights, x)


class _MultiBranchEmbedding(nn.Module):
    """Multi-scale temporal convolution.

    For ``x`` of shape ``(batch, in_channels, n_times)``:

    1. one ``Conv1d`` per entry of ``kernel_sizes``, each
       ``(batch, out_channels, n_times)``;
    2. the branches side by side,
       ``(batch, len(kernel_sizes) * out_channels, n_times)``;
    3. a pointwise convolution fuses them, ``(batch, out_channels, n_times)``.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of channels of each branch and of the output.
    kernel_sizes : tuple of int
        Kernel size of each branch.
    """

    def __init__(
        self, in_channels: int, out_channels: int, kernel_sizes: tuple[int, ...]
    ):
        super().__init__()
        # Padding of half the kernel keeps the number of steps
        self.branches = nn.ModuleList(
            [
                nn.Conv1d(in_channels, out_channels, kernel, padding=kernel // 2)
                for kernel in kernel_sizes
            ]
        )
        self.fuse = nn.Conv1d(
            len(kernel_sizes) * out_channels, out_channels, kernel_size=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(batch, in_channels, n_times)`` -> ``(batch, out_channels, n_times)``."""
        # (batch, in_channels, n_times) -> n_branches x (batch, out_channels, n_times)
        branches = [branch(x) for branch in self.branches]
        # (batch, n_branches * out_channels, n_times) -> (batch, out_channels, n_times)
        return self.fuse(torch.cat(branches, dim=1))


class _DifferentialMamba(nn.Module):
    """Multi-head differential Mamba-2, with a residual connection.

    For ``x`` of shape ``(batch, n_steps, width)``, for each of the
    ``width // head_dim`` heads, a slice of ``head_dim`` channels:

    1. two Mamba-2 blocks read the head, each ``(batch, n_steps, head_dim)``;
    2. the difference ``first - lam * second``, with ``lam`` a learned scale per
       channel;
    3. a GroupNorm with one group, over the channels and the steps of each
       window;

    then the heads side by side, ``(batch, n_steps, width)``, a ``Linear`` that
    mixes them, and the sum with ``x``.

    Parameters
    ----------
    width : int
        Number of channels of ``x``.
    head_dim : int
        Channels per head; divides ``width``.
    d_state, d_conv, expand : int
        Parameters of the Mamba-2 blocks, see ``SAMBA``.
    chunk_size : int
        Steps per chunk of the Mamba-2 scan, see ``SAMBA``.
    dt_min, dt_max, dt_init_floor : float
        Parameters of the Mamba-2 blocks, see ``SAMBA``.
    a_init_range : tuple of float
        Parameter of the Mamba-2 blocks, see ``SAMBA``.
    norm_eps : float
        Parameter of the Mamba-2 blocks, see ``SAMBA``.
    """

    def __init__(
        self,
        width: int,
        head_dim: int,
        d_state: int,
        d_conv: int,
        expand: int,
        *,
        chunk_size: int,
        dt_min: float,
        dt_max: float,
        dt_init_floor: float,
        a_init_range: tuple[float, float],
        norm_eps: float,
    ):
        super().__init__()
        n_heads = width // head_dim
        self.head_dim = head_dim

        def mamba2() -> _Mamba2:
            # Each block's heads are as wide as the head it reads
            return _Mamba2(
                head_dim,
                d_state,
                d_conv,
                expand,
                head_dim,
                chunk_size=chunk_size,
                dt_min=dt_min,
                dt_max=dt_max,
                dt_init_floor=dt_init_floor,
                a_init_range=a_init_range,
                norm_eps=norm_eps,
            )

        # Two Mamba-2 blocks per head
        self.first = nn.ModuleList([mamba2() for _ in range(n_heads)])
        self.second = nn.ModuleList([mamba2() for _ in range(n_heads)])
        self.lam = nn.Parameter(torch.ones(width))
        self.norms = nn.ModuleList(
            [nn.GroupNorm(num_groups=1, num_channels=head_dim) for _ in range(n_heads)]
        )
        self.projection = nn.Linear(width, width)

    def reset_parameters(self) -> None:
        # The two readings start with equal weight
        nn.init.ones_(self.lam)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(batch, n_steps, width)`` -> ``(batch, n_steps, width)``."""
        heads = []
        for index, (first, second, norm) in enumerate(
            zip(self.first, self.second, self.norms)
        ):
            start = index * self.head_dim
            # This head: (batch, n_steps, head_dim)
            head = x[..., start : start + self.head_dim]
            lam = self.lam[start : start + self.head_dim]
            # Difference of the two readings; shape unchanged
            difference = first(head) - lam * second(head)
            # GroupNorm takes (batch, channels, steps)
            heads.append(norm(difference.transpose(1, 2)).transpose(1, 2))
        # Heads side by side, mixed, plus the residual
        # (batch, n_steps, width)
        return x + self.projection(torch.cat(heads, dim=-1))


class _Mamba2(nn.Module):
    """Mamba-2 block: a selective state space layer over the time axis.

    For ``u`` of shape ``(batch, n_steps, d_model)``:

    1. ``in_proj`` gives the gate ``z``, the convolution input ``xbc`` and the
       raw step size ``dt`` of each head;
    2. a causal depthwise convolution and a SiLU turn ``xbc`` into the input
       ``x`` of the heads and the state projections ``b`` and ``c``;
    3. ``_ssd_scan`` runs the recurrence of each head over the steps,
       ``(batch, n_steps, d_inner)``;
    4. the output is gated by ``z``, RMS-normalised and projected back to
       ``d_model``.

    Parameters
    ----------
    d_model : int
        Width of the input and of the output.
    d_state : int
        State size of each head.
    d_conv : int
        Kernel size of the causal depthwise convolution.
    expand : int
        ``d_inner = expand * d_model``, the total width of the heads.
    head_dim : int
        Width of one head; divides ``d_inner``.
    chunk_size : int
        Steps per chunk of the scan: bounds the size of the quadratic form.
    dt_min, dt_max : float
        The initial step size is log-uniform in ``[dt_min, dt_max]``.
    dt_init_floor : float
        Floor of the initial step size.
    a_init_range : tuple of float
        The initial decay rate ``A`` is uniform in ``a_init_range``.
    norm_eps : float
        Epsilon of the RMS norm that follows the gate.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int,
        d_conv: int,
        expand: int,
        head_dim: int,
        *,
        chunk_size: int,
        dt_min: float,
        dt_max: float,
        dt_init_floor: float,
        a_init_range: tuple[float, float],
        norm_eps: float,
    ):
        super().__init__()
        d_inner = expand * d_model
        self.d_inner = d_inner
        self.d_state = d_state
        self.head_dim = head_dim
        self.n_heads = d_inner // head_dim
        self.chunk_size = chunk_size
        # Read by reset_parameters, which the model also calls without arguments
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.dt_init_floor = dt_init_floor
        self.a_init_range = a_init_range

        # Width of the convolved sequence: the input x of the heads, and the
        # projections b and c of the state, shared by all heads
        conv_dim = d_inner + 2 * d_state
        # z | x b c | dt
        self.in_proj = nn.Linear(d_model, d_inner + conv_dim + self.n_heads, bias=False)
        self.conv1d = nn.Conv1d(
            conv_dim, conv_dim, d_conv, groups=conv_dim, padding=d_conv - 1
        )
        # Per head: step size bias, log of the decay rate, skip connection
        self.dt_bias = nn.Parameter(torch.empty(self.n_heads))
        self.A_log = nn.Parameter(torch.empty(self.n_heads))
        self.D = nn.Parameter(torch.empty(self.n_heads))
        self.norm = nn.RMSNorm(d_inner, eps=norm_eps)
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Draw the step sizes, decay rates and skip as the reference Mamba-2."""
        with torch.no_grad():
            # Step sizes log-uniform in [dt_min, dt_max], at least dt_init_floor;
            # dt_bias is their inverse softplus, so that softplus(dt_bias) starts
            # at dt
            log_dt = torch.empty_like(self.dt_bias)
            log_dt.uniform_(math.log(self.dt_min), math.log(self.dt_max))
            dt = log_dt.exp().clamp(min=self.dt_init_floor)
            self.dt_bias.copy_(dt + torch.log(-torch.expm1(-dt)))
            # Decay rates A uniform in a_init_range, kept as log(A)
            decay = torch.empty_like(self.A_log).uniform_(*self.a_init_range)
            self.A_log.copy_(decay.log())
            self.D.fill_(1.0)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """``(batch, n_steps, d_model)`` -> ``(batch, n_steps, d_model)``."""
        n_steps = u.shape[1]

        # Gate, convolution input and raw step sizes
        # (batch, n_steps, d_model) -> (batch, n_steps, d_inner),
        # (batch, n_steps, d_inner + 2 * d_state) and (batch, n_steps, n_heads)
        z, xbc, dt = self.in_proj(u).split(
            [self.d_inner, self.d_inner + 2 * self.d_state, self.n_heads], dim=-1
        )
        # Causal depthwise convolution over time: the padding added on the right
        # is cut, so a step sees only the past
        # (batch, n_steps, d_inner + 2 * d_state)
        xbc = F.silu(self.conv1d(xbc.transpose(1, 2))[..., :n_steps].transpose(1, 2))
        # Input of the heads, and the state projections shared by all heads
        # (batch, n_steps, d_inner), 2 x (batch, n_steps, d_state)
        x, b, c = xbc.split([self.d_inner, self.d_state, self.d_state], dim=-1)

        # The scan accumulates decays over many steps: float32 at least
        dtype = torch.promote_types(u.dtype, torch.float32)
        # Positive step sizes, (batch, n_steps, n_heads); negative decay rates,
        # (n_heads,)
        dt = F.softplus(dt.to(dtype) + self.dt_bias.to(dtype))
        a = -self.A_log.to(dtype).exp()
        # Heads of the input: (batch, n_steps, n_heads, head_dim)
        x = x.to(dtype).unflatten(-1, (self.n_heads, self.head_dim))
        y = _ssd_scan(
            x, dt, a, b.to(dtype), c.to(dtype), self.D.to(dtype), self.chunk_size
        )

        # Heads side by side, back in the dtype of the model
        # (batch, n_steps, n_heads, head_dim) -> (batch, n_steps, d_inner)
        y = y.flatten(2).to(u.dtype)
        # Gate, RMS norm, project back
        # (batch, n_steps, d_inner) -> (batch, n_steps, d_model)
        return self.out_proj(self.norm(y * F.silu(z)))


def _ssd_scan(
    x: torch.Tensor,
    dt: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    d: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    """Selective state space scan of Mamba-2, in chunks of ``chunk_size`` steps.

    For every head, with a state ``h`` of shape ``(head_dim, d_state)``::

        h_t = exp(dt_t * a) * h_{t-1} + dt_t * x_t (outer) b_t
        y_t = h_t @ c_t + d * x_t

    Inside a chunk the recurrence is one matrix product (the quadratic, dual
    form of Mamba-2); the state ``h`` is carried from chunk to chunk by the
    recurrence. No tensor of ``n_steps x n_steps`` is formed.

    Parameters
    ----------
    x : torch.Tensor
        Input of the heads, ``(batch, n_steps, n_heads, head_dim)``.
    dt : torch.Tensor
        Positive step sizes, ``(batch, n_steps, n_heads)``.
    a : torch.Tensor
        Negative decay rates, ``(n_heads,)``.
    b, c : torch.Tensor
        Input and output projections of the state, shared by all heads,
        ``(batch, n_steps, d_state)``.
    d : torch.Tensor
        Skip connection of each head, ``(n_heads,)``.
    chunk_size : int
        Steps per chunk.

    Returns
    -------
    torch.Tensor
        Output of the heads, ``(batch, n_steps, n_heads, head_dim)``.
    """
    batch_size, n_steps, n_heads, head_dim = x.shape
    # State carried between chunks
    # (batch, n_heads, head_dim, d_state)
    state = x.new_zeros(batch_size, n_heads, head_dim, b.shape[-1])

    outputs = []
    for start in range(0, n_steps, chunk_size):
        # The steps of this chunk, n_chunk <= chunk_size of them
        x_chunk = x[:, start : start + chunk_size]
        dt_chunk = dt[:, start : start + chunk_size]
        b_chunk = b[:, start : start + chunk_size]
        c_chunk = c[:, start : start + chunk_size]
        n_chunk = x_chunk.shape[1]

        # Log of the decay accumulated up to each step of the chunk
        # (batch, n_chunk, n_heads) -> (batch, n_heads, n_chunk)
        log_decay = (dt_chunk * a).cumsum(dim=1).transpose(1, 2)

        # Decay from step s to step t >= s; zero for a future step s > t
        # (batch, n_heads, n_chunk, n_chunk), entry [t, s]
        log_between = log_decay[..., :, None] - log_decay[..., None, :]
        future = torch.ones(n_chunk, n_chunk, dtype=torch.bool, device=x.device)
        decay = log_between.masked_fill(future.triu(1), float("-inf")).exp()

        # Weight of the input of step s in the output of step t: decay, c_t . b_s
        # and the step size of s
        # (batch, n_heads, n_chunk, n_chunk)
        cb = torch.matmul(c_chunk, b_chunk.transpose(1, 2))[:, None]
        weight = decay * cb * dt_chunk.transpose(1, 2)[..., None, :]
        # Output from the inputs of the chunk
        # (batch, n_heads, n_chunk, n_chunk) x (batch, n_heads, n_chunk, head_dim)
        # -> (batch, n_chunk, n_heads, head_dim)
        y = torch.matmul(weight, x_chunk.transpose(1, 2)).transpose(1, 2)

        # Output from the state carried in, decayed up to each step
        # (batch, n_chunk, n_heads, head_dim)
        carried = torch.einsum("bhpn,btn->bthp", state, c_chunk)
        y = y + carried * log_decay.transpose(1, 2).exp()[..., None]

        # State at the end of the chunk: the carried state decayed over the
        # chunk, plus the input of each step decayed up to the end
        # (batch, n_chunk, n_heads)
        decay_to_end = (log_decay[..., -1:] - log_decay).exp().transpose(1, 2)
        state = state * log_decay[..., -1].exp()[..., None, None] + torch.einsum(
            "bth,bthp,btn->bhpn", decay_to_end * dt_chunk, x_chunk, b_chunk
        )

        # Skip connection of each head
        outputs.append(y + x_chunk * d[:, None])

    # (batch, n_steps, n_heads, head_dim)
    return torch.cat(outputs, dim=1)
