.. _channel-strategies:

####################################
 Channel strategies: any montage in
####################################

The 19 pretrained models take ``channel_strategy=`` to map your montage onto the
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

Positions come from ``loc``, else from ``standard_1005`` by name. ``spline``, ``field``
and ``source`` need at least four positioned channels; a map whose row gain exceeds 2
warns. A target without a position (BENDR's ``SCALE``) is zero unless the input has it;
a target named ``A-B`` (BIOT) is ``V(A) - V(B)``. Non-EEG channels raise unless
``channel_strategy_kwargs={"drop_non_eeg": True}``. The layer runs in eager mode;
TorchScript works under ``native`` only.

The tutorial :ref:`channel-interpolation-tutorial` compares strategies on a recording.
