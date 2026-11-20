.. _channel-strategies:

####################################
 Channel strategies: any montage in
####################################

The pretrained EEG models take ``channel_strategy=`` to map your montage onto the
channels their backbone was trained on:

.. code-block:: python

    from braindecode.models import EEGPT

    model = EEGPT.from_pretrained(
        "braindecode/eegpt-pretrained",
        chs_info=raw.info["chs"],  # your montage
        channel_strategy="spline",  # how to map it
    )
    y = model(x)  # x: (batch, len(chs_info), n_times)
    y = model(x_other, chs_info=other_chs)  # another montage, per call

Models that read their input natively take no ``channel_strategy`` and say so in a
``TypeError``: the MEG models (:class:`~braindecode.models.BBL`,
:class:`~braindecode.models.MEGGPT`, :class:`~braindecode.models.MEGXL`), the fNIRS
models (:class:`~braindecode.models.GeoNIRS`, :class:`~braindecode.models.JungMAE`,
:class:`~braindecode.models.MtReNIRS`), :class:`~braindecode.models.TinyMyo` (sEMG) and
:class:`~braindecode.models.CORTEG` (its second input, ``x_hi``).
:class:`~braindecode.models.SleepFM` and :class:`~braindecode.models.SleepFMStager` take
polysomnography grouped by modality, so they accept only ``"native"`` and say why in a
``ValueError``. The same holds, for some inputs only, for
:class:`~braindecode.models.BrainOmni` and :class:`~braindecode.models.BrainTokenizer`
with MEG sensors, :class:`~braindecode.models.NeuroRVQ` and
:class:`~braindecode.models.NeuroRVQTokenizer` with a non-EEG ``modality``, and
:class:`~braindecode.models.MAPA` with ``normalization="session"``.

``"native"`` (the default) leaves the model as it is: the released checkpoints give the
same outputs (max-abs difference 0.0) with the same ``state_dict``. Any other strategy
puts a :class:`~braindecode.modules.ChannelLayer` in front of the backbone, which is
then built on its pre-training montage (LaBraM, EEGPT, BENDR, BIOT, CodeBrain, EEG-DINO,
MIRepNet) or on the montage given at construction (the others). ``channel_strategy`` and
``channel_strategy_kwargs`` are saved in the config.

The layer is one matrix per montage. Channels found in the input are copied: same name
(case-insensitive), a legacy name (``T3`` = ``T7``), or a channel whose name MNE does
not know within 15 mm. The strategy fills the other targets:

- ``exact``: nothing; a missing channel is a ``ValueError``.
- ``zero``, ``nearest``, ``idw``: zeros, nearest electrode, inverse-distance mean
  (``p=2``).
- ``spline``: MNE spherical spline (``reg=1e-3``). ``field``: MNE field mapping.
- ``source``: minimum-norm inverse on a template sphere head (``n_parcels=64``),
  projected back; scalp EEG only. ``trainable=True`` adds a parcel mixing initialised at
  zero, so training starts from the physics.
- ``region``: mean of the inputs within ``radius`` (default 1.5 x the median spacing of
  the input montage); none in range gives a zero row.
- ``wiener``: linear MMSE from a spatial covariance fitted on dense recordings with
  ``model.channel_layer.fit(X, chs_info_dense)`` (``X``: samples x dense channels,
  matched by position within 15 mm; ``noise=0.01``). The fit is saved in the
  ``state_dict``; an unfitted map raises.
- ``latent``: learned (POYO-style) cross-attention: each missing target's position
  queries the inputs' positions and signal statistics. Its weights live under
  ``channel_layer.*``, so a backbone checkpoint still loads.

Positions come from ``loc``, else from ``standard_1005`` by name. ``spline``, ``field``
and ``source`` need at least four positioned channels; a map whose row gain exceeds 2
warns. A target without a position (BENDR's ``SCALE``) is zero unless the input has it;
a channel named ``A-B`` is ``V(A) - V(B)``. A target that is a combination of the
measured channels is computed exactly, under every strategy including ``"exact"``:
``F3-C3`` from ``C3-F3`` (weight -1), ``F3-P3`` from ``F3-C3`` and ``C3-P3``, ``F3-C3``
from ``F3`` and ``C3`` or from ``F3-REF`` and ``C3-REF``. The same algebra is in
:func:`braindecode.functional.derivation_matrix` and
:func:`~braindecode.functional.exact_derivation`. No reference is guessed: ``F3-C3`` is
not exact from ``F3-R1`` and ``C3-R2``, nor ``F3`` from bipolar channels only. Non-EEG
channels raise unless ``channel_strategy_kwargs={"drop_non_eeg": True}``. The layer runs
in eager mode; TorchScript works under ``native`` only.

``spline``, ``wiener`` and a kernel passed as ``channel_strategy`` (any ``kernel(pos_a,
pos_b)`` covariance, such as :func:`~braindecode.functional.spherical_spline_kernel`)
are one estimate, :func:`~braindecode.functional.linear_channel_map`: inputs ``y = A
v``, targets ``t = B v``, a prior covariance of the potentials ``v`` and a flat prior on
the unknown recording reference. It also gives each target's posterior variance
(``model.channel_layer.variance``); ``match_gain=True`` restores the amplitude the
estimate shrinks. ``channel_strategy_kwargs={"reference": "average"}`` (or an electrode
name) gives the backbone its pretraining reference.

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

The layer also takes one montage per sample: ``chs_info`` a list of montages, ``x``
padded to the largest and ``ch_mask`` the real channels. Batches of
:func:`~braindecode.datasets.pad_channels_collate` (``pos`` and ``ch_mask``) work as
they are: each sample's montage is read from its positions.

The tutorial :ref:`channel-interpolation-tutorial` compares strategies on a recording.
