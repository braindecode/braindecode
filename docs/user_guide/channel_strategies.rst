.. _channel-strategies:

####################################
 Channel strategies: any montage in
####################################

.. currentmodule:: braindecode.modules.channels

Every pretrained model has a channel layer in front of its backbone that maps your
montage onto what the backbone was trained on. You pick how with one argument:

.. code-block:: python

    from braindecode.models import EEGPT

    model = EEGPT.from_pretrained(
        "braindecode/eegpt-pretrained",
        chs_info=raw.info["chs"],  # your montage
        channel_strategy="spline",  # how to map it
    )
    y = model(x)  # x: (batch, len(chs_info), n_times)
    y = model(x_other, chs_info=other_chs)  # another montage, per call

The default ``"native"`` is the model's own behaviour: on its canonical montage the
released checkpoint gives the same outputs (max-abs difference 0.0) and the same
``state_dict``. ``channel_strategy`` and ``channel_strategy_kwargs`` are saved in the
model configuration.

The layer (:class:`ChannelTokenizer`) resolves the montage (case-insensitive names, then
the aliases ``T3``/``T7``, ``T4``/``T8``, ``T5``/``P7``, ``T6``/``P8``, ``A1``/``M1``,
``A2``/``M2``; positions from ``loc``, else ``standard_1005`` by name; non-EEG channels
raise unless ``drop_non_eeg=True``), builds a map once per montage (cached, on the
model's device and dtype) and hands the backbone a :class:`ChannelEncoding`. Target
channels found in the input (by name, or within 15 mm for coordinate-only channels) are
copied; the others are filled by the strategy and marked ``observed=False``, which masks
them in the attention of LaBraM, EEGPT, LUNA, REVE, PopT and CBraMod.

************
 Strategies
************

- ``exact``: reorder or select by name; a missing channel is a ``ValueError``.
- ``zero``, ``nearest``, ``idw``: zeros, nearest electrode, inverse-distance mean.
- ``spline``: MNE spherical spline, ``reg=1e-3``. ``field``: MNE field mapping.
- ``source``: minimum-norm inverse on a template sphere head (64 parcels), projected
  back; scalp EEG only; ``trainable=True`` adds a gated correction, zero at init.
- ``wiener``: linear MMSE from a covariance fitted with ``model.channel_tokenizer.fit(X,
  dense_chs_info)``.
- ``region``: mean of the inputs within 40 mm (dense input only).
- ``latent``: learned cross-attention over the input positions and signals.

``spline``, ``field`` and ``source`` need at least four positioned channels. On ``free``
backbones (CBraMod, Brant, BrainBERT, MVPFormer) sensor strategies pass the input
through, ``source`` feeds its parcels and ``latent`` its latents. Models whose read-out
does not depend on the channel set (LUNA, PopulationTransformer, mean- or
attention-pooled REVE, BaRISTA, DIVER1) take the input channels with their positions
under every strategy.

Trainable strategies (``latent``, ``source`` with ``trainable=True``) and a fitted
``wiener`` keep their state under ``channel_tokenizer.*``; loading a checkpoint without
it warns once and loads the backbone as strictly as asked.

*********
 Caveats
*********

- The accuracy ranking (``wiener`` > ``source`` > ``idw`` ≈ ``field`` > ``spline`` with
  8 of 19 electrodes) comes from simulated fields, not held-out real electrodes.
- A map whose row gain exceeds 2 warns once per montage: it amplifies noise.
- BIOT, CodeBrain and MIRepNet under ``native`` feed a non-canonical montage unchecked
  and emit a ``FutureWarning``. Under a strategy BIOT takes monopolar input and forms
  its bipolar derivations itself.
- The layer runs in eager mode; TorchScript works under ``native`` only.

A new strategy is one registered :class:`ChannelStrategy` subclass implementing
``_fill`` (rows for the missing targets):

.. code-block:: python

    import numpy as np

    from braindecode.modules.channels import ChannelStrategy, register_channel_strategy


    @register_channel_strategy("mean")
    class MeanStrategy(ChannelStrategy):
        def _fill(self, src, use, tgt_pos):
            rows = np.zeros((len(tgt_pos), len(src.names)))
            rows[:, use] = 1 / use.sum()
            return rows

The tutorial :ref:`channel-interpolation-tutorial` compares strategies on a recording.
