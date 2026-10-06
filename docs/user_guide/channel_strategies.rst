.. _channel-strategies:

####################################
 Channel strategies: any montage in
####################################

.. currentmodule:: braindecode.modules.channels

A pretrained model was trained on one set of electrodes. Your recording rarely has the
same ones, in the same order, with the same names. Every pretrained model in braindecode
has a channel layer in front of its backbone that maps your montage onto what the
backbone consumes. You choose how with one argument:

.. code-block:: python

    from braindecode.models import EEGPT

    model = EEGPT.from_pretrained(
        "braindecode/eegpt-pretrained",
        chs_info=raw.info["chs"],  # your montage
        channel_strategy="spline",  # how to map it
    )
    y = model(x)  # x: (batch, len(chs_info), n_times)
    y = model(x_other, chs_info=other_chs)  # another montage, per call

The default, ``channel_strategy="native"``, is the model's own behaviour: on its
canonical montage the released checkpoint gives exactly the same outputs as before
(maximum absolute difference 0.0), and its ``state_dict`` is unchanged. The strategy and
its options (``channel_strategy_kwargs``) are saved in the model configuration, so
``save_pretrained`` / ``from_pretrained`` keep them.

*********************
 What the layer does
*********************

1. **Resolve** the montage (:func:`resolve_montage`). Names are matched without case,
   exact names first, then legacy aliases (``T3``/``T7``, ``T4``/``T8``, ``T5``/``P7``,
   ``T6``/``P8``, ``A1``/``M1``, ``A2``/``M2``). Positions come from each channel's
   ``loc``; when that is missing they are filled from ``standard_1005`` by name. Non-EEG
   channels (EOG, ECG, stim, ...) raise a ``ValueError`` unless ``drop_non_eeg=True``;
   duplicated names raise.
2. **Map** it with the strategy onto the model's :class:`ChannelTarget`, once per
   montage (maps are cached, eight montages at most, and follow the model's device and
   dtype).
3. **Encode**: the backbone receives a :class:`ChannelEncoding`, with the mapped signal,
   the channel ids or positions it needs, and an ``observed`` mask. Where the backbone
   has an attention mask over channels (LaBraM, EEGPT with ``chan_proj_type="none"``,
   LUNA, REVE, PopT), reconstructed channels are hidden as keys; CBraMod masks their
   patches.

******************
 Model interfaces
******************

Each model declares one interface: what its backbone consumes.

.. list-table::
    :header-rows: 1
    :widths: 15 40 45

    - - Interface
      - Models
      - What the backbone gets
    - - ``montage``
      - BENDR, BIOT, CodeBrain, MIRepNet
      - its training channels, in its fixed order
    - - ``ids``
      - Labram, EEGPT, STEEGFormer, SignalJEPA, SignalJEPA_Contextual
      - channels plus their ids in the model's channel vocabulary
    - - ``positions``
      - LUNA, REVE, ZUNA, BaRISTA, DIVER1, PopulationTransformer
      - channels plus their 3D positions
    - - ``slots``
      - EEGDINO
      - its first ``n_slots`` training channels, in order
    - - ``free``
      - CBraMod, Brant, BrainBERT, MVPFormer
      - any number of channels, without identity

*************************
 Strategies by interface
*************************

"Fill" means: target channels found in your montage (by name, or by position within 15
mm for coordinate-only channels) are copied; the others are filled as the strategy says
and marked ``observed=False``.

.. list-table::
    :header-rows: 1
    :widths: 12 30 22 18 18

    - - Strategy
      - Fills a missing channel with
      - ``montage`` / ``slots`` / ``ids``
      - ``positions``
      - ``free``
    - - ``native``
      - the model's own behaviour (default)
      - unchanged
      - unchanged
      - unchanged
    - - ``exact``
      - nothing: a missing target is a ``ValueError``. For ``ids``, each input channel
        is mapped to its id (no reconstruction).
      - reorder / select
      - reorder / select
      - pass-through
    - - ``zero``
      - zeros (baseline)
      - fill
      - fill
      - pass-through
    - - ``nearest``
      - the nearest input electrode
      - fill
      - fill
      - pass-through
    - - ``idw``
      - inverse-distance weighted mean
      - fill
      - fill
      - pass-through
    - - ``spline``
      - MNE spherical spline, ``reg=1e-3`` (needs 4 positioned inputs)
      - fill
      - fill
      - pass-through
    - - ``field``
      - MNE minimum-norm field mapping (needs 4 positioned inputs)
      - fill
      - fill
      - pass-through
    - - ``source``
      - a minimum-norm inverse on a template sphere head (64 parcels), projected back to
        the electrode (needs 4 positioned inputs; scalp EEG only; ``trainable=True``
        adds a gated correction, zero at initialisation)
      - fill
      - fill
      - 64 parcels
    - - ``wiener``
      - linear MMSE from a covariance you fit on dense recordings
        (``model.channel_tokenizer.fit(X, dense_chs_info)``)
      - fill
      - fill
      - pass-through
    - - ``region``
      - the mean of the inputs within 40 mm (useful for dense input only)
      - fill
      - fill
      - pass-through
    - - ``latent``
      - learned cross-attention from the input positions and signals (trainable)
      - fill
      - fill
      - 64 latents

For ``positions`` models the target is their training montage only when their read-out
is tied to a channel set: REVE with its flattened head, ZUNA, BaRISTA with
``pooling="learned"`` and DIVER1 with ``pooling="flatten"`` map every montage onto the
``chs_info`` they were built with. The others (LUNA, REVE with
``attention_pooling=True``, BaRISTA and DIVER1 with mean pooling, PopulationTransformer)
have no training montage: every strategy passes your channels through with their
positions.

***************
 How to choose
***************

- **Your montage is the canonical one** (same names, maybe another order): ``native``,
  or ``exact`` to reorder or select a superset by name.
- **Some training channels are missing**: ``spline`` is the default choice for scalp
  EEG; ``source`` was the most accurate physics method in our simulations with an
  electrode reference (see below); ``idw`` and ``nearest`` are cheap and never amplify.
  Give at least four channels with positions.
- **You have dense recordings with the same kind of data**: fit ``wiener`` on them; it
  was the most accurate in simulations, but only within its fitted montage (electrodes
  more than 15 mm away from a fitted site raise).
- **You fine-tune anyway**: ``latent`` or ``source`` with ``trainable=True`` add
  parameters under ``channel_tokenizer.*``. A checkpoint without them loads, with one
  warning listing the freshly initialised keys.
- **Intracranial data**: the positions models and MVPFormer, Brant and BrainBERT accept
  sEEG, ECoG and DBS contacts (``kinds=ELECTRODE_KINDS``); use a sensor strategy.

*********
 Caveats
*********

- **Simulations, not held-out electrodes.** The accuracy figures behind these
  recommendations come from synthetic fields: 8 of BENDR's 19 electrodes kept, 10 %
  noise, 20 montages. With an electrode reference and dipoles in a mismatched head, the
  median relative error (1.0 = zero fill) was 0.76 for ``source``, 0.83 for ``idw``,
  0.84 for ``field`` and 0.91 for ``spline``; on smooth fields, ``wiener`` reached 0.59.
  They have not been checked on real recordings with held-out electrodes yet.
- **source is scalp EEG only.** Its head model is a three-shell sphere; it raises a
  ``ValueError`` on intracranial channels, and BaRISTA and PopulationTransformer refuse
  it.
- **positions models without a training montage pass channels through.** For LUNA,
  PopulationTransformer and the mean- or attention-pooled variants, a reconstructing
  strategy changes nothing (a warning says so); build a fixed-read-out model with the
  montage you want instead.
- **Few channels amplify noise.** A map whose row gain exceeds 2, or targets more than
  about 21 mm from any input, trigger a warning once per montage.
- **Rigid montage models.** BIOT, CodeBrain and MIRepNet under ``native`` feed any
  montage to the backbone unchecked; a non-canonical ``chs_info`` now gives a
  ``FutureWarning``, and will raise in the next release. Pass a strategy.
- **BIOT bipolar input is native only.** Under a strategy, BIOT takes the 18 monopolar
  electrodes and forms its bipolar derivations itself.
- **TorchScript.** The channel layer runs in eager mode; a scripted model works under
  ``native`` only.

The tutorial :ref:`channel-interpolation-tutorial` drops channels from a recording and
compares the reconstructions of several strategies.

*******************
 Your own strategy
*******************

Subclass :class:`ChannelStrategy`, implement ``_fill`` (or ``build``), and register it;
the name is then valid for ``channel_strategy=`` in every model:

.. code-block:: python

    import numpy as np

    from braindecode.modules.channels import ChannelStrategy, register_channel_strategy


    @register_channel_strategy("mean")
    class MeanStrategy(ChannelStrategy):
        def _fill(self, src, use, tgt_pos):
            rows = np.zeros((len(tgt_pos), len(src.names)))
            rows[:, use] = 1 / use.sum()
            return rows
