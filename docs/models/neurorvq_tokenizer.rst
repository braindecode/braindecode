NeuroRVQTokenizer
=================

``NeuroRVQTokenizer`` ports the released NeuroRVQ EEG tokenizer: four temporal
scales are encoded independently, quantized with residual EMA codebooks, and
decoded into amplitude and phase components for signal reconstruction. The
same model can return discrete codes for downstream generative models.

The released checkpoint and the adapted implementation are licensed under
CC BY-NC 4.0. Use this model only under those terms.

Input requirements
------------------

Inputs have shape ``(batch, channels, samples)`` and must be sampled at 200 Hz.
Each window must contain an integer number of 200-sample patches, up to 256
patches. Provide ordered channel names from the released 104-channel montage.
The model does not resample, filter, clip, or reorder input channels; prepare
the signal and channel order before calling it.

Pretrained token extraction
---------------------------

.. code-block:: python

    from braindecode.models import NeuroRVQTokenizer

    model = NeuroRVQTokenizer(
        n_chans=3,
        n_times=400,
        sfreq=200,
        channel_names=("f3", "f4", "cz"),
    )
    model.load_pretrained_weights()
    codes = model.tokenize(eeg)
    target, reconstruction = model(eeg)

``codes`` has shape ``(4, 8, batch, channels * patches)``. Reconstruction
returns standardized target and reconstructed patches with shape
``(batch, channels * patches, 200)``. Loading the released checkpoint requires
``huggingface_hub``; a local checkpoint path can be passed to
``load_pretrained_weights``.

Reference parity
----------------

The optional script ``scripts/validate_neurorvq_tokenizer_parity.py`` compares
the port with the released implementation using the pinned Hugging Face
checkpoint. It checks eval-mode reconstructions and codes, then checks one
training step, representative gradients, and EMA state. Run it from the
Braindecode repository root after cloning the reference repository and
installing its Python dependencies:

.. code-block:: shell

    python scripts/validate_neurorvq_tokenizer_parity.py \
        --neurorvq-source /path/to/NeuroRVQ

The script downloads the pinned 304 MB checkpoint unless ``--checkpoint`` is
provided. On the deterministic CPU input in the script, the port and reference
matched exactly for eval outputs, discrete codes, train outputs, the selected
encoder gradient, and all quantizer state after one update.

.. currentmodule:: braindecode.models

.. autosummary::
    :toctree: generated/

    NeuroRVQTokenizer
