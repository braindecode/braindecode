.. _channel-strategies:

####################
 Channel strategies
####################

A pretrained model expects the channels it was trained on. To run it on your recording,
give it your montage and a ``channel_strategy``:

.. code-block:: python

    from braindecode.models import EEGPT

    model = EEGPT.from_pretrained(
        "braindecode/eegpt-pretrained",
        chs_info=raw.info["chs"],  # your montage
        channel_strategy="spline",  # how to map it
    )
    y = model(x)  # x: (batch, len(chs_info), n_times)
    y = model(x_other, chs_info=other_chs)  # another montage, per call

Any strategy other than ``"native"`` puts a :class:`~braindecode.modules.ChannelLayer`
in front of the backbone. The layer maps the montage of ``chs_info``, or the one given
to the call, onto the channels of the backbone. :class:`~braindecode.models.BENDR`,
:class:`~braindecode.models.BIOT`, :class:`~braindecode.models.CodeBrain`,
:class:`~braindecode.models.EEGDINO`, :class:`~braindecode.models.EEGPT`,
:class:`~braindecode.models.Labram`, :class:`~braindecode.models.MIRepNet` and the
:class:`~braindecode.models.SignalJEPA` models build their backbone on their
pre-training montage. The other models build it on the montage given at construction.

``"native"`` is the default. It adds no layer, so the released checkpoints give the same
outputs (max-abs difference 0.0) with the same ``state_dict``. The config saves
``channel_strategy``, ``channel_strategy_kwargs`` and the input montage, so
``from_config`` rebuilds the same model.

.. note::

    **When to attach an external strategy vs. keep "native".** Across five pretrained
    EEG foundation models and five downstream tasks, :footcite:t:`kokate2026channel`
    report a regime-dependent trade-off: rigid-montage models (such as
    :class:`~braindecode.models.BENDR`) require an external adapter on mismatched caps,
    whereas montage-flexible models (such as :class:`~braindecode.models.EEGPT` and
    :class:`~braindecode.models.CBraMod`) benefit from external interpolation under
    frozen-encoder linear probing but match or exceed it under ``"native"`` during full
    fine-tuning.

****************
 The strategies
****************

Input channels that match a target channel are copied (see
:ref:`channel-strategies-mechanics`). The strategy fills the targets that are left.

.. list-table::
    :header-rows: 1
    :widths: 14 50 20 16

    - - Strategy
      - How a missing target channel is filled
      - What it needs
      - Trainable
    - - ``"native"`` (default)
      - No layer. The model reads its input as it is.
      - The model's own channels
      - no
    - - ``"exact"``
      - Not filled. A missing target that has a position raises a ``ValueError``.
      - Nothing
      - no
    - - ``"zero"``
      - Zeros.
      - Nothing
      - no
    - - ``"nearest"``
      - Copy of the nearest input electrode.
      - One input channel with a position
      - no
    - - ``"idw"``
      - Mean of the inputs, weighted by inverse distance (``p=2``).
      - One input channel with a position
      - no
    - - ``"spline"``
      - MNE spherical spline (``reg=1e-3``).
      - :bdg-warning:`4+ positions` four input channels with a position
      - no
    - - ``"field"``
      - MNE field mapping (``reg=0``).
      - :bdg-warning:`4+ positions` four input channels with a position
      - no
    - - ``"source"``
      - Minimum-norm inverse on a template sphere head (``n_parcels=64``), projected
        back onto the targets.
      - :bdg-warning:`4+ positions` four input channels with a position;
        :bdg-danger:`scalp EEG only` no intracranial channels
      - :bdg-info:`optional` ``trainable=True`` adds a parcel mixing that starts at
        zero, so training starts from the physics.
    - - ``"region"``
      - Mean of the inputs within ``radius``, by default 1.5 times the median spacing
        between neighbouring input electrodes. With no input in range, the target is
        zero.
      - Two input channels with a position
      - no
    - - ``"wiener"``
      - Linear minimum mean square error estimate from a spatial covariance fitted on
        dense recordings (``noise=0.01``).
      - :bdg-primary:`fit() first` ``model.channel_layer.fit``; one input channel with a
        position
      - :bdg-secondary:`fitted` the covariance is saved in the ``state_dict``
    - - ``"latent"``
      - Learned cross-attention, as in POYO. The position of each missing target queries
        the positions and signal statistics of the inputs.
      - One input channel with a position
      - :bdg-success:`trainable` attention weights

Options go in ``channel_strategy_kwargs``, for example ``{"reg": 1e-2}`` for
``"spline"``.

***************************************
 Models that read only their own input
***************************************

The models below have released weights but no channel layer, at least for some of their
inputs. Those that take no ``channel_strategy`` raise a ``TypeError`` that says so. The
others take it and raise a ``ValueError`` that names the input the layer cannot map.

.. list-table::
    :header-rows: 1
    :widths: 30 42 28

    - - Model
      - Its input
      - With a ``channel_strategy``
    - - :class:`~braindecode.models.BBL`, :class:`~braindecode.models.MEGXL`
      - MEG sensors
      - :bdg-danger:`TypeError` not a parameter
    - - :class:`~braindecode.models.MEGGPT`
      - MEG source parcels
      - :bdg-danger:`TypeError` not a parameter
    - - :class:`~braindecode.models.GeoNIRS`
      - fNIRS channels placed by their optode positions
      - :bdg-danger:`TypeError` not a parameter
    - - :class:`~braindecode.models.MtReNIRS`
      - fNIRS HbO/HbR pairs with their optode positions
      - :bdg-danger:`TypeError` not a parameter
    - - :class:`~braindecode.models.JungMAE`
      - fNIRS channel patches
      - :bdg-danger:`TypeError` not a parameter
    - - :class:`~braindecode.models.TinyMyo`
      - sEMG channel slots
      - :bdg-danger:`TypeError` not a parameter
    - - :class:`~braindecode.models.NeuroPose`, :class:`~braindecode.models.VEMG2Pose`
      - sEMG, for hand-pose regression
      - :bdg-danger:`TypeError` not a parameter
    - - :class:`~braindecode.models.CORTEG`
      - iEEG, with a second per-channel input, ``x_hi``, that the layer does not map
      - :bdg-danger:`TypeError` not a parameter
    - - :class:`~braindecode.models.NICE`
      - EEG. It is not a foundation model.
      - :bdg-danger:`TypeError` not a parameter
    - - :class:`~braindecode.models.SleepFM`, :class:`~braindecode.models.SleepFMStager`
      - Polysomnography grouped by modality (EEG, EOG, ECG, EMG, respiration)
      - :bdg-warning:`ValueError` for every strategy but ``"native"``
    - - :class:`~braindecode.models.BrainOmni`,
        :class:`~braindecode.models.BrainTokenizer`
      - EEG and MEG
      - :bdg-warning:`ValueError` when ``chs_info`` has MEG sensors
    - - :class:`~braindecode.models.NeuroRVQ`,
        :class:`~braindecode.models.NeuroRVQTokenizer`
      - EEG, or the modality set by ``modality``
      - :bdg-warning:`ValueError` when ``modality`` is not EEG
    - - :class:`~braindecode.models.MAPA`
      - iEEG; spectrograms with ``normalization="session"``
      - :bdg-warning:`ValueError` with ``normalization="session"``. Under a strategy,
        ``forward`` also refuses ``sensor_indices``.

.. _channel-strategies-mechanics:

******************************
 How the layer maps a montage
******************************

Positions and copies
====================

The layer is one matrix per montage, built the first time it sees that montage. A target
channel found in the input is copied. An input channel matches by name, ignoring case,
or by a legacy name (``T3`` is ``T7``, ``T5`` is ``P7``). An input channel whose name
MNE does not know matches a target within 15 mm. Positions come from ``loc``, else from
``standard_1005`` by name. A target without a position, such as the ``SCALE`` channel of
BENDR, is zero unless the input has it. A map whose row gain exceeds 2 warns, because it
amplifies noise.

Derived channels
================

A channel named ``A-B`` is ``V(A) - V(B)``. A target that is a combination of measured
channels is computed exactly under every strategy, ``"exact"`` included. For example,
``F3-C3`` comes from ``C3-F3`` (weight -1), ``F3-P3`` from ``F3-C3`` and ``C3-P3``, and
``F3-C3`` from ``F3`` and ``C3`` or from ``F3-REF`` and ``C3-REF``. No reference is
guessed. ``F3-C3`` is not exact from ``F3-R1`` and ``C3-R2``, and ``F3`` is not exact
from bipolar channels alone. The same algebra is in
:func:`braindecode.functional.derivation_matrix` and
:func:`~braindecode.functional.exact_derivation`.

Other channel types
===================

The layer maps EEG, sEEG, ECoG and DBS electrodes. Any other channel raises a
``ValueError``, unless ``channel_strategy_kwargs={"drop_non_eeg": True}`` makes the
layer ignore it. ``"source"`` uses a scalp-EEG head and also refuses intracranial
channels.

One montage per sample
======================

The layer also maps a batch with one montage per sample. Pass ``chs_info`` as a list of
montages, ``x`` padded to the largest one, and ``ch_mask`` (by keyword) for the real
channels. This works for the strategies without trained state, so not for ``"latent"``
or ``trainable=True``. Batches of :func:`~braindecode.datasets.pad_channels_collate`,
with ``pos`` and ``ch_mask``, work as they are. The layer reads the montage of each
sample from its positions.

Saved state and TorchScript
===========================

For ``"wiener"``, ``model.channel_layer.fit(X, chs_info_dense)`` takes ``X`` of shape
``(n_samples, n_dense_channels)``. Each montage is matched to the dense one by position,
within 15 mm. An unfitted map raises. The state of the layer (the latent attention, the
parcel mixing, the fitted covariance) lives under ``channel_layer.*``, so a backbone
checkpoint without these keys still loads. The layer runs in eager mode, and TorchScript
works under ``"native"`` only.

.. _channel-strategies-forward-arguments:

****************************************
 Mask and position arguments of forward
****************************************

Every model takes its signal as ``x``. A channel mask is always ``ch_mask``, of shape
``(batch, n_chans)``, ``True`` for real channels (see
:func:`~braindecode.functional.check_ch_mask`); these models take it: BBL, BIOT,
BaRISTA, BrainBERT, Brant, CBraMod, CORTEG, DIVER1, EpiNT, GeoNIRS, LUNA, Labram, MAPA,
MEGGPT, MVPFormer, MtReNIRS, PopulationTransformer, REVE, SleepFM and SleepFMStager.
Electrode positions in metres are always ``pos``. The other arguments describe something
else, so they keep their own name:

.. list-table::
    :header-rows: 1
    :widths: 24 22 54

    - - Model
      - Argument
      - Shape and meaning
    - - CORTEG, LUNA, REVE
      - ``pos``
      - ``(batch, n_chans, 3)`` electrode positions in metres (CORTEG also takes
        ``(n_chans, 3)``), as :func:`~braindecode.datasets.pad_channels_collate` gives
    - - MtReNIRS
      - ``coordinates``
      - ``(n_chans, 4)`` or ``(batch, n_chans, 4)``: standardized optode coordinates
        (x, y, z) and the chromophore (0 HbO, 1 HbR)
    - - GeoNIRS
      - ``geometry``
      - ``(n_pairs, 6)`` or ``(batch, n_pairs, 6)`` geometry vector of each pair
    - - PopulationTransformer
      - ``coords``
      - ``(batch, n_chans, 3)`` integer voxel coordinates
    - - BaRISTA
      - ``spatial_indices``
      - embedding indices of the montage, ``(n_chans, 3)`` or ``(n_chans,)``, with an
        optional batch axis
    - - DIVER1
      - ``chan_metadata``
      - ``(n_chans, 5)`` or ``(batch, n_chans, 5)``: x, y, z, modality, sub-modality
    - - MAPA
      - ``sensor_indices``
      - ``(n_chans, 3)`` or ``(batch, n_chans, 3)``: array, contact, region slot
    - - Labram
      - ``ch_names``
      - list of channel names, matched to ``LABRAM_CHANNEL_ORDER``
    - - LUNA
      - ``channel_names``
      - channel ids for the reconstruction head (``n_outputs=0``)
    - - CBraMod, CSBrain, CodeBrain
      - ``mask``
      - ``(batch, n_chans, n_patches)`` patches replaced by the mask encoding where ``1``
    - - LUNA
      - ``mask``
      - shaped like ``x``; a patch with a masked sample becomes the mask token
    - - NICE
      - ``mask``
      - boolean ``(batch, n_chans, n_times)``; ``True`` samples are set to zero
    - - BBL
      - ``temporal_mask``
      - boolean ``(batch, n_times)``; ``True`` samples are set to zero on every channel
    - - SleepFMStager
      - ``temporal_mask``
      - ``(batch, n_patches)``, ``True`` for padded patches
    - - MEGGPT
      - ``token_mask``
      - boolean ``(batch, n_chans, n_times)``; ``True`` hides the token
    - - TFMTokenizer
      - ``spectrogram_mask``
      - boolean, shaped like the spectrogram; ``True`` bins are visible

The old names ``X``, ``input``, ``inputs``, ``eeg`` and ``input_tensor`` for ``x``, and
LUNA's ``channel_locations`` for ``pos``, still work by keyword with a ``FutureWarning``
until braindecode 2.2. SleepFM's ``channel_mask`` is the inverse of ``ch_mask`` and is
removed in 2.2 too.

***************************
 Reconstruct a bad channel
***************************

To reconstruct a bad EEG channel, keep the full montage as the target. Select the good
EEG channels before extracting the signal and its input montage:

.. doctest::

    >>> import mne
    >>> import numpy as np
    >>> import torch
    >>> from braindecode.modules import ChannelLayer
    >>> info = mne.create_info(["C3", "Cz", "C4"], sfreq=100, ch_types="eeg")
    >>> signals = np.array([[1.0, 2.0], [9.0, 9.0], [3.0, 4.0]]) * 1e-6
    >>> raw = mne.io.RawArray(signals, info, verbose=False)
    >>> raw.info["bads"] = ["Cz"]
    >>> good = raw.copy().pick("eeg", exclude="bads")
    >>> layer = ChannelLayer(raw.info["chs"], "nearest", good.info["chs"])
    >>> x = torch.tensor(good.get_data(), dtype=torch.float32).unsqueeze(0)
    >>> mapped, observed = layer(x)
    >>> observed.tolist()
    [True, False, True]
    >>> torch.equal(mapped[:, observed], x)
    True

``observed`` marks the targets that are measured or exactly derived. Here ``Cz`` takes
the signal of its nearest good electrode, and ``C3`` and ``C4`` are copied as they are.

**********************
 Adapting a built model
**********************

A model that is already instantiated (including native CNNs without a ``channel_strategy``
constructor parameter, such as :class:`~braindecode.models.EEGNet`,
:class:`~braindecode.models.ShallowFBCSPNet` and :class:`~braindecode.models.Deep4Net`)
can be adapted in place with :meth:`~braindecode.models.base.EEGModuleMixin.adapt_input`.
Calling ``adapt_input`` again cleanly replaces the active adaptation while keeping all
``state_dict`` keys identical.

.. code-block:: python

    from braindecode.models import EEGNet

    model = EEGNet(chs_info=train_chs, n_outputs=4, n_times=1000, sfreq=250.0)
    model.adapt_input(chs_info=eval_chs, sfreq=500.0)
    logits = model(x_eval_500hz)  # resampled to 250 Hz, mapped to train_chs

Montage and mixed-montage batches
=================================

Passing ``chs_info`` maps recordings from that montage onto the build montage using
``channel_strategy`` (``"nearest"`` by default). Any EEG, iEEG, or ECoG model built with
``chs_info`` supports montage adaptation, whereas MEG, fNIRS, sEMG, and polysomnography
models refuse with ``ValueError``. For mixed-montage batches collated with
:func:`~braindecode.datasets.pad_channels_collate` and ``set_return_ch_pos(True)``, pass
``channel_strategy`` with ``chs_info=None`` and call ``model(X, pos=pos, ch_mask=ch_mask)``:

.. code-block:: python

    model = EEGNet(chs_info=train_chs, n_outputs=2, n_times=1000, sfreq=250.0)
    model.adapt_input(channel_strategy="spline")
    logits = model(X, pos=pos, ch_mask=ch_mask)

Stateless strategies only
=========================

``adapt_input`` accepts only the stateless strategies ``("exact", "zero", "nearest",
"idw", "spline", "field", "source", "region")`` (with ``trainable=False`` enforced on
``"source"``), because strategies with learned or fitted parameters (``"wiener"``,
``"latent"``, or ``"source"`` with ``trainable=True``) add keys to ``state_dict``.

Sampling rate and window length
===============================

Passing ``sfreq`` installs a non-persistent :class:`torchaudio.transforms.Resample` layer
to convert integer-rate inputs to ``model.sfreq`` (with a warning), and passing
``sfreq=model.sfreq`` clears it. Free and dense models accept any compatible ``n_times``
directly. Supported fixed-time models adaptively pool features right before their
classification head when ``n_times`` is set (exact identity at the build length), while
fixed-time models whose internal layers depend on build length refuse with a ``ValueError``
naming the fixed parameter.

Configuration persistence
=========================

Whenever an input adaptation is active, :meth:`~braindecode.models.base.EEGModuleMixin.get_config`
and :meth:`~braindecode.models.base.EEGModuleMixin.save_pretrained` store it under
``"input_adaptation"`` while keeping ``"chs_info"`` equal to the build montage.
:meth:`~braindecode.models.base.EEGModuleMixin.from_config` and
:meth:`~braindecode.models.base.EEGModuleMixin.from_pretrained` re-apply the saved
adaptation after loading weights, while unadapted models omit ``"input_adaptation"``.

**********************
 See it in a tutorial
**********************

.. grid:: 1 2 3 3
    :gutter: 3

    .. grid-item-card:: |share-nodes| Compare the strategies
        :link: channel-interpolation-tutorial
        :link-type: ref
        :shadow: sm

        Runs BENDR on a synthetic 32-channel recording that lacks three of its
        channels, and plots what each strategy reconstructs.

    .. grid-item-card:: |layer-group| Train across montages
        :link: heterogeneous-montage-training
        :link-type: ref
        :shadow: sm

        Trains one model on recordings with different channels. Each window carries
        its electrode positions, and ``pad_channels_collate`` pads every batch.

    .. grid-item-card:: |cloud-download-alt| Load a pretrained model
        :link: load-pretrained-models
        :link-type: ref
        :shadow: sm

        Loads foundation models from the Hugging Face Hub, adapts them to a new task
        and saves them with their configuration.

.. rubric:: Examples that use the channel layer

.. minigallery:: braindecode.modules.ChannelLayer

.. rubric:: References

.. footbibliography::
