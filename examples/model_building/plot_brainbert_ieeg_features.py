# type: ignore
""".. _brainbert-ieeg-features:

BrainBERT: Frozen Spectrogram Features for Intracranial Signals
==============================================================

:class:`~braindecode.models.BrainBERT` is a self-supervised foundation model for
**intracranial** recordings (sEEG/ECoG). It is not trained on the raw waveform:
it is trained by masking patches of a **spectrogram** and reconstructing them.
The spectrogram is therefore not a preprocessing detail you may swap at will —
it is the interface the pretrained weights were fitted against, and every one of
its knobs (window length, overlap, number of frequency bins, normalisation,
sampling rate) is part of the checkpoint.

That is what makes BrainBERT a good tutorial subject, and it is what this example
is really about. Three things silently change what the pretrained encoder sees,
and none of them raise an error on their own:

1. **the sampling rate**, because an STFT bin is a fraction of ``sfreq``, not a
   frequency in Hz;
2. **the normalisation order**, because upstream ships two different recipes;
3. **the window length**, because the published protocol pools a fixed number of
   frames centred on the window.

We work through all three on real ECoG, then reproduce the downstream protocol of
the paper — a **frozen** encoder and a bare linear probe — and finish with the
fine-tuning alternative.

.. contents:: This example covers:
   :local:
   :depth: 1
"""

# Authors: Adam Mounir <am91ris@gmail.com>
#
# License: BSD (3-clause)

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from braindecode.models import BrainBERT

torch.manual_seed(20200220)  # the untrained control below must be reproducible

######################################################################
# The spectrogram is the model's input
# ------------------------------------
#
# In braindecode, a model takes ``(batch, n_chans, n_times)`` — the signal, not a
# spectrogram. The port therefore moves the short-time Fourier transform *inside*
# ``forward``, as :class:`~braindecode.modules.brainbert_modules._STFTSpectrogram`,
# so ``BrainBERT`` keeps the same signature as every other braindecode model. The
# upstream reference instead consumes a spectrogram computed offline.
#
# The consequence is that the STFT parameters are now constructor arguments, and
# their defaults are the ones behind the published checkpoint:
# ``nperseg=400``, ``noverlap=350`` and ``idx_freq_cutoff=40`` low-frequency bins.

model = BrainBERT(n_chans=1, n_outputs=2, n_times=2048, sfreq=2048.0)
stft = model.spectrogram
print(
    f"nperseg={stft.nperseg}, noverlap={stft.noverlap}, bins kept={stft.idx_freq_cutoff}"
)
print(
    f"a 2048-sample window becomes {model.seq_len} frames of {stft.idx_freq_cutoff} bins"
)

######################################################################
# Sampling rate: a bin is a fraction of ``sfreq``, not a frequency
# ---------------------------------------------------------------
#
# ``idx_freq_cutoff=40`` keeps the first 40 bins. A bin is ``sfreq / nperseg``
# wide, so what those 40 bins *physically cover* depends entirely on the
# sampling rate:
#
# .. math::
#
#    f_{\max} = \texttt{idx\_freq\_cutoff} \times \frac{\texttt{sfreq}}{\texttt{nperseg}}
#
# The checkpoint was pretrained at 2048 Hz, where 40 bins reach ~205 Hz — the
# high-gamma range that carries most of the decodable information in intracranial
# signals. Feed the same model 1000 Hz data and the very same 40 bins now stop at
# 100 Hz: the encoder receives a different slice of the spectrum than the one it
# was fitted on, and nothing warns you.

for sfreq in (1000.0, 2048.0):
    bin_hz = sfreq / 400
    print(
        f"sfreq={sfreq:>6.0f} Hz -> {bin_hz:5.2f} Hz/bin, 40 bins reach {40 * bin_hz:6.1f} Hz"
    )

######################################################################
# So the first thing to do with a recording that is not at 2048 Hz is to
# **resample it**, not to reconfigure the model. Resampling upwards adds no
# information; it simply restores the correspondence between bin index and
# frequency that the pretrained weights expect.

######################################################################
# Loading real intracranial data
# ------------------------------
#
# We use BCI Competition IV dataset 4 [1]_, the intracranial dataset shipped with
# braindecode: ECoG from three patients flexing their fingers, sampled at
# 1000 Hz, with the flexion of each finger recorded as a target channel at 25 Hz.
#
# It is not the corpus BrainBERT was pretrained on (that is the Brain Treebank),
# which is precisely the situation this tutorial is for: a pretrained
# intracranial encoder applied to someone else's electrodes.

from braindecode.datasets import BCICompetitionIVDataset4  # noqa: E402
from braindecode.preprocessing import Preprocessor, preprocess  # noqa: E402

dataset = BCICompetitionIVDataset4(subject_ids=[1])
train_set = dataset.split("session")["train"]
raw = train_set.datasets[0].raw

######################################################################
# The finger flexions live in this dataset as ``misc`` channels carried at the
# signal's own 1000 Hz but **filled only every 40th sample** — they were recorded
# at 25 Hz, and the gaps are ``NaN``. A sparse channel like this cannot be
# resampled: any interpolation kernel that touches a ``NaN`` returns ``NaN``, and
# the whole trace collapses. So we read the targets out at their native rate
# *before* touching the sampling frequency, and resample only the ECoG.

FLEXION_SFREQ = 25
thumb = raw.get_data(picks=["target_0"])[0][:: int(raw.info["sfreq"] // FLEXION_SFREQ)]
assert np.isfinite(thumb).all()

SFREQ = 2048.0
preprocess(train_set, [Preprocessor("resample", sfreq=SFREQ)])
raw = train_set.datasets[0].raw
print(
    f"signal resampled to {raw.info['sfreq']:.0f} Hz; "
    f"thumb trace kept at {FLEXION_SFREQ} Hz ({thumb.size} points)"
)

######################################################################
# We keep a handful of ECoG channels.

ecog_names = [ch for ch, t in zip(raw.ch_names, raw.get_channel_types()) if t == "ecog"]
picked = ecog_names[:8]
data = raw.get_data(picks=picked)
print(f"{len(ecog_names)} ECoG channels, using {len(picked)}; signal {data.shape}")

######################################################################
# Window length: the protocol pools frames, so it has a floor
# -----------------------------------------------------------
#
# The published downstream protocol does not average the whole sequence. It
# averages the ``pool_n_frames=10`` encoder outputs **centred** on the window
# (``preprocessors/spec_pretrained.py`` upstream); averaging every frame appears
# in that file only as a commented-out alternative. The port keeps the centred
# average, and adds a mean over channels on top of it — which is the identity for
# ``n_chans=1``, so a single-channel ``BrainBERT`` reproduces the upstream
# feature exactly.
#
# A window must therefore be long enough to *have* 10 frames left after the
# boundary frames are trimmed. Shorter windows are refused at construction rather
# than silently pooled differently:

try:
    BrainBERT(n_chans=1, n_outputs=2, n_times=1024, sfreq=SFREQ)
except ValueError as err:
    print(f"ValueError: {err}")

######################################################################
# The floor is 1410 samples, about 688 ms at 2048 Hz. We use 1-second windows,
# which leave a comfortable margin.

WINDOW = 2048  # 1 s at 2048 Hz
STRIDE = 1024  # 50 % overlap

starts = np.arange(0, data.shape[1] - WINDOW, STRIDE)
X = np.stack([data[:, s : s + WINDOW] for s in starts]).astype(np.float32)

# Each window spans [s, s + WINDOW) samples at SFREQ, i.e. the same seconds on
# the 25 Hz thumb trace. Summarise the flexion over that span by its median.
spans = [
    (int(s / SFREQ * FLEXION_SFREQ), int((s + WINDOW) / SFREQ * FLEXION_SFREQ))
    for s in starts
]
flexion = np.array([np.median(thumb[a:b]) for a, b in spans])

# Binarise into "thumb flexed" vs "thumb at rest" at the median, so the classes
# are balanced by construction and chance level is exactly 0.5. This is a
# deliberately simple stand-in for the binary tasks of the BrainBERT paper
# (sound onset, speech, pitch, volume), which are defined on another corpus.
y = (flexion > np.median(flexion)).astype(int)
print(f"{len(X)} windows of {WINDOW} samples, {y.mean():.2f} positive")

######################################################################
# Two upstream STFT recipes, and which one is the checkpoint's
# -----------------------------------------------------------
#
# Upstream ships the z-score and the boundary trim in **two different orders**:
#
# * ``preprocessors/stft.py`` — z-score, *then* trim 10 frames per side. This is
#   the one ``conf/preprocessor/stft_pretrained.yaml`` wires up, so it is the one
#   behind the checkpoint's published numbers.
# * ``notebooks/demo.ipynb`` — trim 5 frames per side, *then* z-score.
#
# They are close but not equal, and they do not even return the same number of
# frames. The port defaults to the first and exposes the second:

recipes = {
    "checkpoint (stft.py)": dict(stft_clip=10, stft_zscore_before_clip=True),
    "demo notebook": dict(stft_clip=5, stft_zscore_before_clip=False),
}

wav = torch.tensor(X[0, :1][None], dtype=torch.float32)  # (1, 1, n_times)
specs = {}
for name, kwargs in recipes.items():
    m = BrainBERT(n_chans=1, n_outputs=2, n_times=WINDOW, sfreq=SFREQ, **kwargs)
    with torch.no_grad():
        specs[name] = m.spectrogram(wav)[0, 0].numpy()
    print(f"{name:>22}: {specs[name].shape[0]} frames")

fig, axes = plt.subplots(1, 2, figsize=(10, 3.2), sharey=True, constrained_layout=True)
for ax, (name, s) in zip(axes, specs.items()):
    ax.imshow(s.T, aspect="auto", origin="lower", cmap="magma")
    ax.set_title(name, fontsize=10)
    ax.set_xlabel("frame")
axes[0].set_ylabel("frequency bin")
fig.suptitle("The same ECoG second, under the two upstream recipes", fontsize=11)

######################################################################
# The two are highly correlated — which is exactly why the discrepancy survives
# unnoticed — but they are not interchangeable, and *how far apart they are
# depends on the data*. On filtered noise the unit tests see a correlation of
# 0.9992; on this ECoG window it drops to about 0.99, with individual bins
# differing by more than half a z-unit. A frozen encoder has no way to tell you
# it is being fed the other recipe, so choose one deliberately.

# Mind the alignment: one recipe trims 10 frames per side and the other 5, so
# frame *i* of one is frame *i + 5* of the other. Comparing them index by index
# without that shift compares different moments in time and reports nonsense.
ref = specs["checkpoint (stft.py)"]
shift = 10 - 5
alt = specs["demo notebook"][shift : shift + len(ref)]
a, b = ref.ravel(), alt.ravel()
print(
    f"mean |delta| = {np.abs(a - b).mean():.4f} z-units, "
    f"max = {np.abs(a - b).max():.4f}, r = {np.corrcoef(a, b)[0, 1]:.6f}"
)

######################################################################
# The published protocol: frozen encoder, linear probe
# ----------------------------------------------------
#
# BrainBERT's downstream evaluation does not fine-tune. It freezes the encoder,
# takes the pooled representation, and fits a **bare linear probe** on top — the
# upstream ``models/linear_wav_baseline.py`` is literally ``nn.Linear(dim, 1)``
# and nothing else. Reproducing the protocol means reproducing that austerity:
# whatever accuracy comes out is attributable to the representation, not to a
# decoder trained on top of it.
#
# ``return_features=True`` returns the pooled embedding under the braindecode
# foundation-model convention, ``{"features": ..., "cls_token": ...}``.
# BrainBERT has no class token, so ``cls_token`` is ``None``.

device = "cuda" if torch.cuda.is_available() else "cpu"


def extract(net):
    """Pooled embedding of every window, with the encoder frozen."""
    net.eval().to(device)
    out = []
    with torch.no_grad():
        for i in range(0, len(X), 32):
            batch = torch.from_numpy(X[i : i + 32]).to(device)
            out.append(net(batch, return_features=True)["features"].cpu().numpy())
    return np.concatenate(out)


pretrained = BrainBERT.from_pretrained(
    "braindecode/brainbert-pretrained",
    n_chans=len(picked),
    n_outputs=2,
    n_times=WINDOW,
    sfreq=SFREQ,
)
features = extract(pretrained)
print(f"frozen features: {features.shape}")

######################################################################
# A chronological split, never a random one: consecutive windows overlap and are
# autocorrelated, so shuffling would leak a window's own neighbours into the
# training set and inflate the score.

cut = int(0.75 * len(features))


def probe_auc(feats):
    scaler = StandardScaler().fit(feats[:cut])
    probe = LogisticRegression(max_iter=2000)
    probe.fit(scaler.transform(feats[:cut]), y[:cut])
    scores = probe.predict_proba(scaler.transform(feats[cut:]))[:, 1]
    return roc_auc_score(y[cut:], scores)


print(f"frozen BrainBERT + linear probe: ROC-AUC = {probe_auc(features):.3f}")

######################################################################
# Above chance — but a number above chance means nothing on its own. The
# architecture alone could be doing the work: a random projection of a
# spectrogram is already a usable feature. The control that separates *the
# pretraining* from *the architecture* is the same encoder with its weights
# untrained, everything else held fixed.

random_init = BrainBERT(
    hidden_dim=768,
    ffn_dim=3072,
    n_layers=6,
    n_heads=12,
    n_chans=len(picked),
    n_outputs=2,
    n_times=WINDOW,
    sfreq=SFREQ,
)
auc_random = probe_auc(extract(random_init))
print(f"same architecture, untrained:    ROC-AUC = {auc_random:.3f}")
print("chance:                          ROC-AUC = 0.500")

######################################################################
# The untrained encoder already lands well above chance, and the pretrained one
# beats it by a smaller margin than it beats chance. Read plainly: on this
# dataset most of the signal is recovered by the spectrogram and the random
# projection, and pretraining adds a modest amount on top.
#
# That is the informative outcome, and it is why the control belongs in the
# tutorial. Without it, the first number alone would have read as a much
# stronger claim than the data supports.
#
# .. warning::
#    Do not read these three numbers as a measurement of BrainBERT. They come
#    from one patient, eight electrodes, one chronological split and a task we
#    invented by thresholding a thumb trace — there is no error bar here, and a
#    single split of 200 correlated windows cannot carry one. The setting is
#    also deliberately out of domain: BrainBERT was pretrained on sEEG depth
#    electrodes at 2048 Hz, and this is subdural ECoG recorded at 1000 Hz and
#    upsampled. Treat the section as a demonstration of the protocol, and see
#    the paper [2]_ for the evaluation itself.

######################################################################
# Fine-tuning instead of probing
# ------------------------------
#
# If you want the encoder to move, the checkpoint is an ordinary braindecode
# model: :meth:`~braindecode.models.BrainBERT.reset_head` swaps the
# classification head for a new number of outputs, and
# :class:`~braindecode.EEGClassifier` trains it like any other.
#
# Fitting 43M parameters on a few hundred correlated windows from one patient is
# not something this example can evaluate honestly, so the code below is shown
# rather than run, and no claim is made about how it would compare to the probe:
#
# .. code-block:: python
#
#     from skorch.helper import predefined_split
#     from braindecode import EEGClassifier
#
#     model = BrainBERT.from_pretrained(
#         "braindecode/brainbert-pretrained",
#         n_chans=len(picked), n_outputs=2,
#         n_times=WINDOW, sfreq=SFREQ,
#     )
#     clf = EEGClassifier(
#         model,
#         optimizer=torch.optim.AdamW,
#         optimizer__lr=1e-5,          # small: the encoder is already trained
#         train_split=predefined_split(valid_set),
#         batch_size=16,
#     )
#     clf.fit(train_set, y=None, epochs=10)

######################################################################
# What to take away
# -----------------
#
# * The spectrogram *is* the checkpoint's input. Resample to 2048 Hz rather than
#   retuning ``nperseg`` or ``idx_freq_cutoff``, so that bin indices keep the
#   frequencies the weights were fitted on.
# * Pick a recipe on purpose. ``stft_clip=10`` with
#   ``stft_zscore_before_clip=True`` is the one behind the published numbers.
# * Windows shorter than 1410 samples at 2048 Hz are refused, because the
#   protocol pools 10 centred frames.
# * ``n_chans=1`` reproduces the upstream feature exactly; more channels is the
#   braindecode-native generalisation.
# * Always probe an untrained copy of the same architecture alongside the
#   pretrained one. A spectrogram pushed through a random Transformer is already
#   a decent feature, so "above chance" does not mean "the pretraining
#   transferred".
#
# .. topic:: Licensing
#
#    The upstream BrainBERT repository ships **no LICENSE file**. The re-hosted
#    weights therefore declare their licence as ``unknown`` rather than assume a
#    permissive one, and the model card records the SHA-256 and retrieval date of
#    the source archive. Check with the authors before any use beyond research.
#
# References
# ----------
#
# .. [1] Miller, K.J., 2019. A library of human electrocorticographic data and
#        analyses. Nature Human Behaviour, 3(11), pp.1225-1235.
#
# .. [2] Wang, C., Subramaniam, V., Yaari, A.U., Kreiman, G., Katz, B., Cases, I.
#        and Barbu, A., 2023. BrainBERT: Self-supervised representation learning
#        for intracranial recordings. ICLR.
