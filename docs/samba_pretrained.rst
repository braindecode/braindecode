###########################
 SAMBA pretrained encoders
###########################

SAMBA is a self-supervised EEG representation learning model based on Differential
Mamba-2. It uses coordinate-driven spatial projection followed by convolutions and
sequence modeling. The Braindecode model provides pooled representations and a newly
initialized linear task head. Reconstruction decoders are not loaded.

Download ``config.json`` and ``model.safetensors`` from either ``Dcode-AI/SAMBA-2s`` or
``Dcode-AI/SAMBA-100s`` on Hugging Face. Downloads are explicit; model construction does
not fetch files or import CUDA extensions. The two checkpoints share the encoder
architecture but have different pretraining corpora and window lengths. Do not treat
them as interchangeable training results. Consult each model card for data exposure and
provenance.

The model table has one entry for the ``SAMBA`` class. The two released checkpoints use
the same encoder architecture:

.. list-table:: Released checkpoints
    :header-rows: 1
    :widths: 40 20 20 20

    - - Checkpoint
      - Sampling rate
      - Pretraining window
      - ``n_times``
    - - ``Dcode-AI/SAMBA-2s``
      - 128 Hz
      - 2 seconds
      - 256
    - - ``Dcode-AI/SAMBA-100s``
      - 128 Hz
      - 100 seconds
      - 12800

``n_times`` counts input samples, not seconds. ``n_outputs`` sets the size of the newly
initialized downstream head; it is not a property of the pretraining checkpoint. For
example, four-class classification uses ``n_outputs=4``, while scalar regression uses
``n_outputs=1``.

The summary table's 392733 parameters refer to its example with 94 target positions and
``n_outputs=4``. Changing the output count changes the head's parameter count. Changing
only the input window length does not change the parameter count. The table lists
architecture categories; self-supervised pretraining and Differential Mamba-2 are
described here explicitly.

The loading example additionally requires ``pip install safetensors``.

.. code-block:: python

    import json
    from pathlib import Path
    from safetensors.torch import load_file
    from braindecode.models import SAMBA

    variant = "SAMBA-2s"  # or "SAMBA-100s"
    n_times = {"SAMBA-2s": 256, "SAMBA-100s": 12800}[variant]
    root = Path(variant)
    config = json.loads((root / "config.json").read_text())
    architecture = {
        key: config[key]
        for key in (
            "target_positions",
            "base_channels",
            "bottleneck_channels",
            "d_state",
            "d_conv",
            "expand",
            "head_dim",
            "cdse_hidden_dim",
        )
    }
    # chs_info must contain loc[:3] in the checkpoint coordinate frame/units,
    # in the same order as the signal channels. Do not silently substitute
    # MNE head-frame positions for another coordinate frame.
    # This example creates a new two-class head.
    model = SAMBA(
        chs_info=chs_info, n_outputs=2, n_times=n_times, sfreq=128, **architecture
    )
    model.load_pretrained_encoder(load_file(str(root / "model.safetensors")))
    logits = model(normalized_eeg)  # (batch, channels, time) -> (batch, 2)
    # Ordinary torch losses/backward or EEGClassifier/EEGRegressor can be used.

Resample signals to 128 Hz and divide each channel/window by its absolute 95th
percentile, as specified in the model cards. Handle invalid or flat channels explicitly
in preprocessing. The encoder does not silently normalize inputs. SAMBA-2s was
pretrained with 256 samples; SAMBA-100s with 12800 samples. Other lengths are supported
architecturally, but their task performance needs validation.

The portable backend evaluates the same ngroups=1 Mamba-2 recurrence in chunks of 64
tokens. It avoids full-sequence quadratic matrices. Floating-point outputs need not be
bitwise identical to fused kernels. This implementation supports gradients, but CPU
runtime depends on window length, batch size and hardware. The model operates on a
complete supplied window; spatial/temporal normalization and convolutions do not provide
a causal within-window stream.

Validation of implementation agreement does not reproduce the paper's benchmark results.
This contribution does not include a new downstream benchmark study.
