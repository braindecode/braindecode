# MIRepNet: original-author BNCI2014004 reproduction

Recorded 2026-09-29. **These measurements use the original-author implementation
and released checkpoint, not the Braindecode port.** They support implementation
provenance; they do not demonstrate end-to-end real-data performance of this PR.
No model or training algorithm changes were made for this evidence update.

## Results

Each seed mean is the unweighted mean of nine last-epoch subject accuracies.
The headline is the mean of three seed means; SD is across those three means,
not across 27 individual results. SD units are percentage points (pp).

| Protocol | Mean accuracy (%) | Population SD (pp) | Sample SD (pp) | Difference from paper (pp) |
|---|---:|---:|---:|---:|
| Baseline, 8–32 Hz | 80.313051 | 0.898893 | 1.100914 | -2.046949 |
| Single fixed control, 8–30 Hz | 80.555556 | 1.408435 | 1.724974 | -1.804444 |
| Paper, arXiv v1 Tables IV/VII | 82.36 | — | — | — |

The paper reports **82.36 ± 0.10%** without specifying the SD convention here.
**The paper value was not matched.** Changing only the upper band edge increased
the mean by 0.242504 pp; it did not close the gap. Neither band is established as
the authors' historical setting. Baseline 8–32 Hz remains the headline rather
than selecting whichever result is closer to the paper.

[Machine-readable results](bnci2014004.csv) contain all 54 observations, including
integer correct counts and test-set sizes. Subject identifiers below are zero-based.

### 8-32 Hz: last epoch of ten

| Subject | Seed 666 | Seed 667 | Seed 668 | Three-seed mean |
|---|---:|---:|---:|---:|
| 0 | 88.392857 | 80.357143 | 86.607143 | 85.119048 |
| 1 | 50.000000 | 54.761905 | 58.333333 | 54.365079 |
| 2 | 64.285714 | 49.107143 | 57.142857 | 56.845238 |
| 3 | 97.321429 | 99.107143 | 99.107143 | 98.511905 |
| 4 | 91.071429 | 87.500000 | 84.821429 | 87.797619 |
| 5 | 86.607143 | 84.821429 | 82.142857 | 84.523810 |
| 6 | 78.571429 | 83.928571 | 80.357143 | 80.952381 |
| 7 | 87.500000 | 85.714286 | 86.607143 | 86.607143 |
| 8 | 87.500000 | 86.607143 | 90.178571 | 88.095238 |
| Unweighted subject mean | 81.250000 | 79.100529 | 80.588624 | 80.313051 |

### 8-30 Hz: last epoch of ten

| Subject | Seed 666 | Seed 667 | Seed 668 | Three-seed mean |
|---|---:|---:|---:|---:|
| 0 | 87.500000 | 83.035714 | 84.821429 | 85.119048 |
| 1 | 55.952381 | 55.952381 | 55.952381 | 55.952381 |
| 2 | 65.178571 | 49.107143 | 56.250000 | 56.845238 |
| 3 | 97.321429 | 99.107143 | 99.107143 | 98.511905 |
| 4 | 91.071429 | 87.500000 | 83.928571 | 87.500000 |
| 5 | 87.500000 | 83.035714 | 81.250000 | 83.928571 |
| 6 | 81.250000 | 83.928571 | 80.357143 | 81.845238 |
| 7 | 87.500000 | 85.714286 | 86.607143 | 86.607143 |
| 8 | 89.285714 | 85.714286 | 91.071429 | 88.690476 |
| Unweighted subject mean | 82.506614 | 79.232804 | 79.927249 | 80.555556 |

## Reconstructed protocol

- BNCI2014004: all nine subjects, using the first test session. The full ordered
  construction has 6,520 trials. All 1,400 selected trial identities were checked
  against the author's positional blocks, not just against an expected count.
- Filter continuous EEG at native 250 Hz with MOABB 0.4.6 / MNE 1.6.1,
  fourth-order Butterworth IIR, forward/backward zero phase. Epoch interval
  [3, 7.5] s gives 1,126 samples in microvolts; original `dataset.py` takes the
  first 1,000 samples. Baseline 8–32 Hz follows the author-referenced generator's
  defaults; 8–30 Hz is the one fixed paper-band control, not a filter sweep.
- Unstratified `train_test_split(test_size=0.7, random_state=seed)` within each
  subject: 48 train / 112 test trials, except subject 1 with 36 / 84. Trial IDs,
  labels and ordered split metadata are identical between the two filter runs.
- Euclidean alignment is computed **separately on native-channel train and test
  EEG, before interpolation to 45 channels** with the upstream inverse-distance
  method. This is transductive: unlabeled test EEG contributes to its own
  covariance reference; no test labels are used for training. This is not a
  strictly inductive evaluation.
- Original `run_experiment` / `train_subject`, batch size 8, ten epochs, Adam
  learning rate 0.001, weight decay 1e-6, cosine scheduler `T_max=10`, four
  DataLoader workers, all parameters trainable, random two-class head.
- Seeds 666/667/668 are set once per full sequential nine-subject run, with no
  per-subject RNG reset. Observed Torch RNG fingerprints verify the sequential
  chain. Report the last epoch, never the best test epoch. No seed search.
- Original fine-tuning loads **108 of 110 downstream tensors** exactly from the
  checkpoint; two head tensors have intentionally different shapes. This is
  distinct from the PR's **109 inference-tensor** mapping/port-parity checks.

## Source and environment provenance

- [Original code](https://github.com/staraink/MIRepNet/tree/b35d113cd7b1629b1e0f35a5f2495ee15c69c291),
  revision `b35d113cd7b1629b1e0f35a5f2495ee15c69c291`.
- [Released checkpoint](https://huggingface.co/starself/MIRepNet),
  `MIRepNet.pth`, SHA256
  `432288958007e344a5a84a9ffe9d0e5e5c0cb616aef86c85522375a3f4da9aaf`.
- [Paper, arXiv v1](https://arxiv.org/abs/2507.20254v1),
  [author preprocessing discussion](https://github.com/staraink/MIRepNet/issues/5),
  and [separate reported reproduction](https://github.com/staraink/MIRepNet/issues/4).
  The latter's 82.61% is not a measurement from this work.
- Reconstructed CPU environment: Python 3.10.21, Torch 2.2.0+cpu (four compute
  threads, one interop thread), NumPy 1.24.4, SciPy 1.10.1, pandas 1.5.3,
  scikit-learn 1.3.2, MNE 1.6.1, MOABB 0.4.6, einops 0.8.1,
  pyriemann 0.3, PyYAML 6.0.2 and wandb 0.17.9 (import only).
- Compatibility adaptations: CPU checkpoint `map_location`, preparation-only
  restoration of archived NumPy aliases, SciPy pin compatible with NumPy 1.24.4,
  and PyYAML 6.0.2 instead of the archived package's failed build. No replacement
  model, optimizer or training loop; all 12 copied upstream Python files were
  hash-checked against the pinned source.

SHA256 fingerprints of reconstructed artifacts (not the unavailable historical
prepared arrays):

| Artifact | SHA256 |
|---|---|
| Baseline `X.npy` | `6fe24909d2865a58ab90e3520769822062af40168e55832b7b6be99002b1f8a3` |
| Control `X.npy` | `1306d67a15ea4fd961675599aa5848e4a762b44710ff075f6a7c426b6d0c79b8` |
| Shared `labels.npy` | `d406bcb5a144a4e17c80d1092cbef59f4f55d6832a4971c04abca9d6927b7ddb` |
| Shared ordered trial IDs | `3051979053fddee06b9dfc5997e7a1ec865ba1871f72ff57daf0b2a06eba1350` |
| Shared ordered splits | `26931ed3746c5f669061b2c53118ea87cb2fdc263764156a2f8e380bc2a8ffa6` |

## Verification and limits

Before publication, all 27 observations per filter were independently recomputed
from integer correct counts, cross-checked against numeric evaluation records,
and checked for complete seed/subject coverage, ten epochs and sequential RNG
continuity. Means and both SD definitions were independently recomputed.

Recorded terminal job and container receipts were inspected: baseline preparation,
smoke and evaluation succeeded (exit 0); four preceding prerequisite attempts
failed (exit 1: two archived NumPy import issues, missing wandb, and CUDA-only
checkpoint loading). Both control jobs succeeded (exit 0). All nine recorded jobs
are terminal. Baseline evaluation finished 2026-09-29 10:29:23 UTC; control
container evaluation finished 10:48:24 UTC. Raw logs, infrastructure identifiers,
EEG, checkpoints and environments are not included in this repository report.

The original prepared arrays, exact historical seeds and complete dependency /
hardware environment are unavailable. CPU and CUDA RNG streams and numerics are
not interchangeable. Therefore this is practical, code-faithful reconstruction
of the author implementation, **not exact historical replication**, and not a
real-data benchmark of `braindecode.models.MIRepNet`. The remaining accuracy gap
cannot be assigned to a particular missing historical detail from these runs.
No statistical significance or equivalence claim is made from three seeds.
