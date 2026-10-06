###################
 NeuroRVQTokenizer
###################

``NeuroRVQTokenizer`` ports the released NeuroRVQ EEG tokenizer: four temporal scales
are encoded independently, quantized with residual EMA codebooks, and decoded into
amplitude and phase components for signal reconstruction. The same model can return
discrete codes for downstream generative models.

The released checkpoint and the adapted implementation are licensed under CC BY-NC 4.0.
Use this model only under those terms.

********************
 Input requirements
********************

Inputs have shape ``(batch, channels, samples)`` and must be sampled at 200 Hz. Each
window must contain an integer number of 200-sample patches, up to 256 patches. Provide
ordered channel names from the released 104-channel montage. The model does not
resample, filter, clip, or reorder input channels; prepare the signal and channel order
before calling it. The released example (``preprocessing/preprocessing_eeg_example.py``
in the source repository) applies notch filters at 50, 60 and 100 Hz, a 0.5-44.5 Hz
Butterworth band-pass, clipping at 500 uV and resampling to 200 Hz.

*****************************
 Pretrained token extraction
*****************************

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

``codes`` has shape ``(4, 8, batch, channels * patches)``. Reconstruction returns
standardized target and reconstructed patches with shape ``(batch, channels * patches,
200)``. Both are z-scored per window over channels, patches and samples, so the
time-domain reconstruction error of a batch is ``(target -
reconstruction).square().mean()``. Loading the released checkpoint requires
``huggingface_hub``; a local checkpoint path can be passed to
``load_pretrained_weights``.

******************
 Reference parity
******************

With the pinned Hugging Face checkpoint loaded into both the port and the released
implementation, eval-mode discrete codes are identical and reconstructions agree to
float32 precision. The Transformer blocks, temporal convolution and channel list are
shared with :class:`NeuroRVQ`.

.. currentmodule:: braindecode.models

.. autosummary::
    :toctree: generated/

    NeuroRVQTokenizer
