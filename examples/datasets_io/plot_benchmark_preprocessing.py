""".. _benchmark-preprocess-parallel-serial:

Benchmarking preprocessing with parallelization and serialization
=================================================================

In this example, we compare the execution time and memory requirements of
preprocessing data with the parallelization and serialization functionalities
available in :func:`braindecode.preprocessing.preprocess`.

We compare 4 cases:

1. Sequential, no serialization
2. Sequential, with serialization
3. Parallel, no serialization
4. Parallel, with serialization

Case 1 is the simplest approach, in which all recordings in a
:class:`braindecode.datasets.BaseConcatDataset` are preprocessed one after the
other. In this scenario, :func:`braindecode.preprocessing.preprocess` acts
inplace, which means memory usage will likely stay stable (depending on the
preprocessing operations) if recordings have been preloaded. However, two
potential issues arise when working with large datasets: (1) if recordings have
not been preloaded before preprocessing, ``preprocess()`` will need to load them
and keep them in memory, in which case memory can become a bottleneck, and (2)
sequential preprocessing can take a considerable amount of time to run when
working with many recordings.

A solution to the first issue (memory usage) is to save the preprocessed data
to a file so it can be cleared from memory before moving on to the next
recording (case 2). The recordings can then be reloaded with ``preload=False``
once they have all been saved to disk. This enables using the lazy loading
capabilities of :class:`braindecode.datasets.BaseConcatDataset` and avoids
potential memory bottlenecks. The downside is that the writing to disk can take
some time and of course requires disk space.

A solution to the second issue (slow preprocessing) is to parallelize the
preprocessing over multiple cores whenever possible (case 3). This can speed up
preprocessing significantly. However, this approach will increase memory usage
because of the way parallelization is implemented internally (with
`joblib`, copies of (part of) the data must be made when sending arguments to
parallel processes).

Finally, case 4 (combining parallelization and serialization) is likely to be
both fast and memory efficient. As shown in this example, this remains a
tradeoff though, and the selected configuration should depend on the size of
the dataset and the specific operations applied to the recordings.
"""

# Authors: Hubert Banville <hubert.jbanville@gmail.com>
#          Bruno Aristimunha <b.aristimunha@gmail.com>
#
# License: BSD (3-clause)

import gc
import json
import subprocess
import sys
import tempfile
import textwrap
from itertools import product

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import scale

from braindecode.datasets import SleepPhysionet
from braindecode.preprocessing import (
    Preprocessor,
    create_fixed_length_windows,
    preprocess,
)

###############################################################################
# We create a function that goes through the usual three steps of data
# preparation: (1) data loading, (2) continuous data preprocessing,
# (3) windowing and (4) windowed data preprocessing. We use the
# :class:`braindecode.datasets.SleepPhysionet` dataset for testing purposes.


def prepare_data(n_recs, save, preload, n_jobs):
    tmp_dir = tempfile.TemporaryDirectory() if save else None
    save_dir = tmp_dir.name if tmp_dir is not None else None
    try:
        # (1) Load the data
        concat_ds = SleepPhysionet(
            subject_ids=range(n_recs),
            recording_ids=[1],
            crop_wake_mins=30,
            preload=preload,
        )
        sfreq = concat_ds.datasets[0].raw.info["sfreq"]

        # (2) Preprocess the continuous data
        preprocessors = [
            Preprocessor("crop", tmin=10),
            Preprocessor("filter", l_freq=None, h_freq=30),
        ]
        preprocess(
            concat_ds, preprocessors, save_dir=save_dir, overwrite=True, n_jobs=n_jobs
        )

        # (3) Window the data
        windows_ds = create_fixed_length_windows(
            concat_ds,
            0,
            None,
            int(30 * sfreq),
            int(30 * sfreq),
            on_last_window="drop",
            preload=preload,
            n_jobs=n_jobs,
        )

        # (4) Preprocess the windowed data
        preprocessors = [Preprocessor(scale, channel_wise=True)]
        preprocess(
            windows_ds, preprocessors, save_dir=save_dir, overwrite=True, n_jobs=n_jobs
        )
    finally:
        if tmp_dir is not None:
            tmp_dir.cleanup()


###############################################################################
# Next, we measure run time and peak memory overhead for each of the 4 cases
# above inside an isolated Python subprocess. Running each configuration in its
# own ``subprocess`` works cleanly across Linux, macOS ``spawn``, Windows, and
# parallel Sphinx-Gallery workers (where ``multiprocessing``-based profilers
# fail inside daemonic worker processes), while warming ``joblib``'s worker pool
# once per configuration separates preprocessing cost from one-time Python
# import startup.
#
# .. note::
#   To better characterize the run time vs. memory usage tradeoff for your
#   specific configuration (as this will differ based on available hardware,
#   data size and preprocessing operations), we recommend adapting this example
#   to your use case and running it on your machine.

_CHILD_SCRIPT = textwrap.dedent(
    """
    import gc
    import json
    import resource
    import sys
    import tempfile
    import threading
    import time
    import numpy as np
    from joblib import Parallel, delayed
    from sklearn.preprocessing import scale
    from braindecode.datasets import SleepPhysionet
    from braindecode.preprocessing import (
        Preprocessor,
        create_fixed_length_windows,
        exponential_moving_demean,
        preprocess,
    )

    try:
        import psutil
    except ImportError:
        psutil = None

    n_recs = int(sys.argv[1])
    n_jobs = int(sys.argv[2])
    n_repets = int(sys.argv[3])

    def _tree_rss_mib():
        if psutil is not None:
            proc = psutil.Process()
            rss = proc.memory_info().rss
            for child in proc.children(recursive=True):
                try:
                    rss += child.memory_info().rss
                except psutil.Error:
                    pass
            return rss / (1024 ** 2)
        ru_self = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        ru_child = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
        scale_factor = 1024 ** 2 if sys.platform == "darwin" else 1024
        return (ru_self + ru_child) / scale_factor

    def _run_once(save):
        tmp_dir = tempfile.TemporaryDirectory() if save else None
        save_dir = tmp_dir.name if tmp_dir is not None else None
        try:
            concat_ds = SleepPhysionet(
                subject_ids=range(n_recs),
                recording_ids=[1],
                crop_wake_mins=30,
                preload=False,
            )
            sfreq = concat_ds.datasets[0].raw.info["sfreq"]
            preprocess(
                concat_ds,
                [
                    Preprocessor("crop", tmin=10),
                    Preprocessor("filter", l_freq=None, h_freq=30),
                ],
                save_dir=save_dir,
                overwrite=True,
                n_jobs=n_jobs,
            )
            windows_ds = create_fixed_length_windows(
                concat_ds,
                0,
                None,
                int(30 * sfreq),
                int(30 * sfreq),
                on_last_window="drop",
                preload=False,
                n_jobs=n_jobs,
            )
            preprocess(
                windows_ds,
                [Preprocessor(scale, channel_wise=True)],
                save_dir=save_dir,
                overwrite=True,
                n_jobs=n_jobs,
            )
        finally:
            if tmp_dir is not None:
                tmp_dir.cleanup()

    if n_jobs > 1:
        # Import braindecode inside workers before recording baseline RSS/time.
        Parallel(n_jobs=n_jobs)(
            delayed(exponential_moving_demean)(np.zeros((1, 2)))
            for _ in range(n_jobs)
        )

    gc.collect()
    idle_mib = _tree_rss_mib()
    out = []
    # Run serialized (lower-memory) mode before in-memory mode so OS allocator
    # high-water marks do not mask the smaller footprint of save=True.
    for save in (True, False):
        for _ in range(n_repets):
            gc.collect()
            peak = [_tree_rss_mib()]
            stop_evt = threading.Event()

            def _sampler(peak_ref=peak, evt=stop_evt):
                while not evt.wait(0.015):
                    cur = _tree_rss_mib()
                    if cur > peak_ref[0]:
                        peak_ref[0] = cur

            t_sampler = threading.Thread(target=_sampler, daemon=True)
            t_sampler.start()
            t0 = time.perf_counter()
            try:
                _run_once(save)
            finally:
                elapsed = time.perf_counter() - t0
                stop_evt.set()
                t_sampler.join()
            peak_mib = max(peak[0], _tree_rss_mib())
            out.append(
                {
                    "save": save,
                    "time": elapsed,
                    "max_mem": max(0.0, peak_mib - idle_mib),
                }
            )

    print("__BENCHMARK_JSON__" + json.dumps(out))
    """
)

n_repets = 2  # Number of repetitions
all_n_recs = 2  # Number of recordings to load and preprocess
all_n_jobs = [1, 2]  # Number of parallel processes

# Ensure the SleepPhysionet files are cached once before timing subprocesses.
_ = SleepPhysionet(
    subject_ids=range(all_n_recs), recording_ids=[1], crop_wake_mins=30, preload=False
)
gc.collect()

results = []
for n_recs, n_jobs in product([all_n_recs], all_n_jobs):
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            _CHILD_SCRIPT,
            str(n_recs),
            str(n_jobs),
            str(n_repets),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    marker_line = next(
        line
        for line in completed.stdout.splitlines()
        if line.startswith("__BENCHMARK_JSON__")
    )
    for run in json.loads(marker_line.removeprefix("__BENCHMARK_JSON__")):
        results.append(
            {
                "n_recs": n_recs,
                "max_mem": run["max_mem"],
                "save": run["save"],
                "n_jobs": n_jobs,
                "time": run["time"],
            }
        )


###############################################################################
# Finally, we can plot the results (two repetitions per configuration, with the
# configuration mean shown as a larger marker):

df = pd.DataFrame(results)

fig, ax = plt.subplots(figsize=(6.4, 4.0), facecolor="white")
ax.set_facecolor("white")
colors = {True: "#D55E00", False: "#0072B2"}
markers = {1: "o", 2: "s"}
labels = {
    (False, 1): "Sequential, in-memory (save=False, n_jobs=1)",
    (True, 1): "Sequential, serialized (save=True, n_jobs=1)",
    (False, 2): "Parallel, in-memory (save=False, n_jobs=2)",
    (True, 2): "Parallel, serialized (save=True, n_jobs=2)",
}

for (save, n_jobs), sub_df in df.groupby(["save", "n_jobs"], sort=False):
    ax.scatter(
        sub_df["time"],
        sub_df["max_mem"],
        color=colors[save],
        marker=markers[n_jobs],
        s=32,
        alpha=0.45,
    )
    ax.scatter(
        [sub_df["time"].mean()],
        [sub_df["max_mem"].mean()],
        color=colors[save],
        marker=markers[n_jobs],
        s=90,
        edgecolor="white",
        linewidth=0.8,
        label=labels[(save, n_jobs)],
        zorder=3,
    )

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_linewidth(0.8)
ax.spines["bottom"].set_linewidth(0.8)
ax.tick_params(direction="out", width=0.8)
ax.grid(axis="y", color="#EAEAEA", linewidth=0.7)
ax.set_axisbelow(True)
ax.legend(frameon=False, fontsize=8.5)
ax.set_xlabel("Execution time (s)")
ax.set_ylabel("Peak memory overhead (MiB)")
fig.tight_layout()
plt.show()


###############################################################################
# We see that parallel preprocessing without serialization (blue squares) is
# faster than simple sequential processing (blue circles), however it uses more
# memory because multiple workers hold and return preprocessed data at once.
#
# Combining parallel preprocessing and serialization (vermillion squares)
# lowers peak memory overhead because each worker writes its preprocessed
# recording directly to disk and the main process reloads headers lazily
# (``preload=False``), at the cost of extra disk I/O time. Depending on
# available resources (e.g. in memory-constrained settings with many
# recordings), combining parallelization and serialization can provide a
# practical balance between speed and memory footprint.
