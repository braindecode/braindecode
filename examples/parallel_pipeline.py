"""
Example for parallel preprocessing and serialization pipeline.

1. Load data
2. Preprocess data
3. Window data
4. Preprocess windows

Cases:
1. Sequential processing, no serialization
2. Sequential processing, with serialization
3. Parallel processing, no serialization
4. Parallel processing, with serialization

"""

import time
from itertools import product

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from memory_profiler import memory_usage

from braindecode.datasets.sleep_physionet import SleepPhysionet
from braindecode.preprocessing import (
    preprocess, Preprocessor, create_fixed_length_windows, zscore)


SAVE_DIR = '/home/hubert/Data/testing/braindecode_tests/'
preload = False


def run(n_recs, save, n_jobs):
    save_dir = SAVE_DIR if save else None

    concat_ds = SleepPhysionet(
        subject_ids=range(n_recs), recording_ids=[1], crop_wake_mins=30,
        preload=preload)
    sfreq = concat_ds.datasets[0].raw.info['sfreq']

    # Preprocess the data
    preprocessors = [
        Preprocessor(lambda x: x * 1e6),
        Preprocessor('filter', l_freq=None, h_freq=30)
    ]
    preprocess(concat_ds, preprocessors, save_dir=save_dir, overwrite=True,
               n_jobs=n_jobs)

    if save_dir is not None:
        assert all([not ds.raw.preload for ds in concat_ds.datasets])  # preload=False

    windows_ds = create_fixed_length_windows(
            concat_ds, 0, 0, int(30 * sfreq), int(30 * sfreq), True,
            preload=preload, n_jobs=n_jobs)

    if save_dir is not None:
        assert all([not ds.windows.preload for ds in windows_ds.datasets])  # preload=False

    # Preprocess the windows
    preprocessors = [Preprocessor(zscore)]
    preprocess(windows_ds, preprocessors, save_dir=save_dir, overwrite=True,
               n_jobs=n_jobs)

    if save_dir is not None:
        assert all([not ds.windows.preload for ds in windows_ds.datasets])  # preload=False

    print(f'Took {time.time() - start:.2f} s.')


if __name__ == '__main__':

    # parser = argparse.ArgumentParser()
    # parser.add_argument('--n_jobs', type=int)
    # parser.add_argument('--save', type=bool)

    # args = parser.parse_args()
    # run(args)

    # Test with memory_profiler:
    # >>> python -m memory_profiler examples/parallel_pipeline.py --n_jobs 1
    # >>> python -m memory_profiler examples/parallel_pipeline.py --n_jobs 1 --save True
    # >>> python -m memory_profiler examples/parallel_pipeline.py --n_jobs 8
    # >>> python -m memory_profiler examples/parallel_pipeline.py --n_jobs 8 --save True

    results = list()
    for _, n_recs, save, n_jobs in product(range(5), [8], [True, False], [1, 8]):
        start = time.time()
        # run(n_recs, save, n_jobs)
        mem = max(memory_usage(
            proc=(run, [n_recs, save, n_jobs], {})))  # , max_usage=True))
        time_taken = time.time() - start
        results.append({
            'n_recs': n_recs,
            'max_mem': mem,
            'save': save,
            'n_jobs': n_jobs,
            'time': time_taken
        })

    df = pd.DataFrame(results)
    ax = sns.scatterplot(
        data=df, x='time', y='max_mem', hue='n_jobs', style='save', palette='colorblind')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Memory usage (MiB)')
    ax.set_title('Loading and preprocessing 8 recordings from Sleep Physionet')
    plt.show()
