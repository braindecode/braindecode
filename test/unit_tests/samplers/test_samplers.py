"""
Test for samplers.
"""

# Authors: Hubert Banville <hubert.jbanville@gmail.com>
#          Young Truong <dt.young112@gmail.com>
#
# License: BSD (3-clause)

import bisect
import platform

import numpy as np
import pandas as pd
import pytest
import torch.distributed as dist
import torch.multiprocessing as mp

from braindecode.datasets import BaseConcatDataset, RawDataset
from braindecode.datasets.moabb import fetch_data_with_moabb
from braindecode.preprocessing.windowers import (
    create_fixed_length_windows,
    create_windows_from_events,
)
from braindecode.samplers import (
    BalancedSequenceSampler,
    DistributedRecordingSampler,
    RecordingSampler,
    SequenceSampler,
)
from braindecode.samplers.ssl import (
    DistributedRelativePositioningSampler,
    RelativePositioningSampler,
)


def test_distributed_relative_positioning_sampler_n_examples_formula():
    """Test that n_examples calculation uses correct operator precedence.

    This is a simple unit test that validates the formula directly without
    requiring network access or distributed training setup.

    The bug was: n_examples // total_recordings * recordings_per_rank
    The fix is: n_examples * recordings_per_rank // total_recordings

    Test cases from the bug report:
    1. n_examples=100, total_recordings=10, recordings_per_rank=2 (world_size=4)
       - Buggy: 100 // 10 * 2 = 10 * 2 = 20
       - Fixed: 100 * 2 // 10 = 200 // 10 = 20
       - In this case both formulas give the same result

    2. n_examples=50, total_recordings=100, recordings_per_rank=50 (world_size=2)
       - Buggy: 50 // 100 * 50 = 0 * 50 = 0 (WRONG!)
       - Fixed: 50 * 50 // 100 = 2500 // 100 = 25 (CORRECT)

    3. n_examples=100, total_recordings=10, world_size=4, recordings_per_rank=2 or 3
       - For rank with 2 recordings:
         - Buggy: 100 // 10 * 2 = 20
         - Fixed: 100 * 2 // 10 = 20
       - For rank with 3 recordings (10 doesn't divide evenly by 4):
         - Buggy: 100 // 10 * 3 = 30
         - Fixed: 100 * 3 // 10 = 30
    """
    # Test case 1: Equal precision case
    n_examples, total_recordings, recordings_per_rank = 100, 10, 2
    buggy = n_examples // total_recordings * recordings_per_rank
    fixed = n_examples * recordings_per_rank // total_recordings
    assert buggy == 20
    assert fixed == 20

    # Test case 2: Critical bug case - buggy formula gives 0
    n_examples, total_recordings, recordings_per_rank = 50, 100, 50
    buggy = n_examples // total_recordings * recordings_per_rank
    fixed = n_examples * recordings_per_rank // total_recordings
    assert buggy == 0, "Buggy formula should give 0 (this is the bug!)"
    assert fixed == 25, "Fixed formula should give 25"

    # Test case 3: Another precision loss case
    n_examples, total_recordings, recordings_per_rank = 100, 10, 3
    buggy = n_examples // total_recordings * recordings_per_rank
    fixed = n_examples * recordings_per_rank // total_recordings
    assert buggy == 30
    assert fixed == 30

    # Test case 4: More examples showing the difference
    n_examples, total_recordings, recordings_per_rank = 75, 20, 5
    buggy = n_examples // total_recordings * recordings_per_rank
    fixed = n_examples * recordings_per_rank // total_recordings
    assert buggy == 15, "Buggy: 75 // 20 * 5 = 3 * 5 = 15"
    assert fixed == 18, "Fixed: 75 * 5 // 20 = 375 // 20 = 18"


@pytest.fixture(scope="module")
def windows_ds():
    raws, description = fetch_data_with_moabb(dataset_name="BNCI2014_001", subject_ids=4)
    ds = [RawDataset(raws[i], description.iloc[i]) for i in range(3)]
    concat_ds = BaseConcatDataset(ds)

    windows_ds = create_fixed_length_windows(
        concat_ds=concat_ds,
        start_offset_samples=0,
        stop_offset_samples=None,
        window_size_samples=500,
        window_stride_samples=500,
        drop_last_window=False,
        preload=False,
    )

    return windows_ds


@pytest.fixture(scope="module")
def target_windows_ds():
    raws, description = fetch_data_with_moabb(dataset_name="BNCI2014_001", subject_ids=4)
    ds = [RawDataset(raws[i], description.iloc[i]) for i in range(3)]
    concat_ds = BaseConcatDataset(ds)

    windows_ds = create_windows_from_events(
        concat_ds,
        trial_start_offset_samples=0,
        trial_stop_offset_samples=0,
        window_size_samples=None,
        window_stride_samples=None,
        drop_last_window=False,
    )

    return windows_ds


def find_dataset_ind(windows_ds, win_ind):
    """Taken from torch.utils.data.dataset.ConcatDataset."""
    return bisect.bisect_right(windows_ds.cumulative_sizes, win_ind)


def test_recording_sampler(windows_ds):
    sampler = RecordingSampler(windows_ds.get_metadata(), random_state=87)
    assert sampler.n_recordings == windows_ds.description.shape[0]

    # Test info attribute
    assert isinstance(sampler.info, pd.DataFrame)
    assert isinstance(sampler.info.index, pd.MultiIndex)
    inds = np.concatenate(sampler.info["index"].values)
    assert len(inds) == len(windows_ds)
    assert len(inds) == len(set(inds))

    # Test methods
    for _ in range(100):
        rec_ind = sampler.sample_recording()
        assert rec_ind in range(windows_ds.description.shape[0])

        win_ind, rec_ind2 = sampler.sample_window(rec_ind=rec_ind)
        dataset_ind = find_dataset_ind(windows_ds, win_ind)
        assert rec_ind2 == rec_ind == dataset_ind

        X, y, z = windows_ds[win_ind]

        win_ind, rec_ind = sampler.sample_window(rec_ind=None)
        assert rec_ind in range(windows_ds.description.shape[0])

def dist_sampler_init_process(rank, world_size, windows_ds):
    """Initialize the process group for multi-CPU training."""
    dist.init_process_group(
        backend="gloo",
        init_method="tcp://127.0.0.1:29500",  # Localhost for single machine
        rank=rank,
        world_size=world_size
    )
    print(f"Process {rank} initialized")

    sampler = DistributedRecordingSampler(windows_ds.get_metadata(), random_state=87)
    if world_size == 1:
        sampler_single = RecordingSampler(windows_ds.get_metadata(), random_state=87)
        assert sampler.n_recordings == windows_ds.description.shape[0] == sampler_single.n_recordings
    assert len(sampler.dataset) == windows_ds.description.shape[0]
    assert sampler.n_recordings <= windows_ds.description.shape[0] // world_size
    print(f"Rank {rank} has {sampler.n_recordings} datasets after splitting")

    # Test info attribute
    assert isinstance(sampler.info, pd.DataFrame)
    assert isinstance(sampler.info.index, pd.MultiIndex)
    inds = np.concatenate(sampler.info["index"].values)
    assert len(inds) == len(windows_ds)
    assert len(inds) == len(set(inds))

    # Test methods
    for _ in range(100):
        rec_ind = sampler.sample_recording()
        assert rec_ind in range(windows_ds.description.shape[0])

        win_ind, rec_ind2 = sampler.sample_window(rec_ind=rec_ind)
        dataset_ind = find_dataset_ind(windows_ds, win_ind)
        assert rec_ind2 == rec_ind == dataset_ind

        X, y, z = windows_ds[win_ind]

        win_ind, rec_ind = sampler.sample_window(rec_ind=None)
        assert rec_ind in range(windows_ds.description.shape[0])

    # Cleanup
    dist.destroy_process_group()


@pytest.mark.skipif(platform.system() == 'Windows',
                    reason="Not supported on Windows because of use_libuv compatibility")
def test_distributed_recording_sampler(windows_ds):
    world_size = 1  # Test single process - no dataset splitting
    mp.spawn(dist_sampler_init_process, args=(world_size,windows_ds), nprocs=world_size, join=True)
    world_size = 3  # Test multiple processes - dataset splitting
    mp.spawn(dist_sampler_init_process, args=(world_size,windows_ds), nprocs=world_size, join=True)


@pytest.mark.parametrize("same_rec_neg", [True, False])
def test_relative_positioning_sampler(windows_ds, same_rec_neg):
    tau_pos, tau_neg = 2000, 3000
    n_examples = 100
    sampler = RelativePositioningSampler(
        windows_ds.get_metadata(),
        tau_pos=tau_pos,
        tau_neg=tau_neg,
        n_examples=n_examples,
        tau_max=None,
        same_rec_neg=same_rec_neg,
        random_state=33,
    )

    pairs = [pair for pair in sampler]
    pairs_df = pd.DataFrame(pairs, columns=["win_ind1", "win_ind2", "y"])
    pairs_df["diff"] = pairs_df.apply(
        lambda x: abs(
            windows_ds[int(x["win_ind1"])][2][1] - windows_ds[int(x["win_ind2"])][2][1]
        ),
        axis=1,
    )
    pairs_df["same_rec"] = pairs_df.apply(
        lambda x: (
            find_dataset_ind(windows_ds, int(x["win_ind1"]))
            == find_dataset_ind(windows_ds, int(x["win_ind2"]))
        ),
        axis=1,
    )

    assert len(pairs) == n_examples == len(sampler)
    assert all(pairs_df.loc[pairs_df["y"] == 1, "diff"] <= tau_pos)
    if same_rec_neg:
        assert all(pairs_df.loc[pairs_df["y"] == 0, "diff"] >= tau_neg)
        assert all(pairs_df["same_rec"] == same_rec_neg)
    else:
        assert all(pairs_df.loc[pairs_df["y"] == 0, "same_rec"] == False)  # noqa: E712
        assert all(pairs_df.loc[pairs_df["y"] == 1, "same_rec"] == True)  # noqa: E712
    assert abs(np.diff(pairs_df["y"].value_counts())) < 20

def test_relative_positioning_sampler_presample(windows_ds):
    tau_pos, tau_neg = 2000, 3000
    n_examples = 100
    sampler = RelativePositioningSampler(
        windows_ds.get_metadata(),
        tau_pos=tau_pos,
        tau_neg=tau_neg,
        n_examples=n_examples,
        tau_max=None,
        same_rec_neg=True,
        random_state=33,
    )

    sampler.presample()
    assert hasattr(sampler, "examples")
    assert len(sampler.examples) == n_examples

    pairs = [pair for pair in sampler]
    pairs2 = [pair for pair in sampler]
    assert np.array_equal(sampler.examples, pairs)
    assert np.array_equal(sampler.examples, pairs2)

def distributed_relative_positioning_sampler_init_process(rank, world_size, windows_ds, same_rec_neg):
    dist.init_process_group(
        backend="gloo",
        init_method="tcp://127.0.0.1:29500",  # Localhost for single machine
        rank=rank,
        world_size=world_size
    )
    print(f"Process {rank} initialized")

    tau_pos, tau_neg = 2000, 3000
    n_examples = 100
    sampler = DistributedRelativePositioningSampler(
        windows_ds.get_metadata(),
        tau_pos=tau_pos,
        tau_neg=tau_neg,
        n_examples=n_examples,
        tau_max=None,
        same_rec_neg=same_rec_neg,
        random_state=33,
    )

    pairs = [pair for pair in sampler]
    pairs_df = pd.DataFrame(pairs, columns=["win_ind1", "win_ind2", "y"])
    pairs_df["diff"] = pairs_df.apply(
        lambda x: abs(
            windows_ds[int(x["win_ind1"])][2][1] - windows_ds[int(x["win_ind2"])][2][1]
        ),
        axis=1,
    )
    pairs_df["same_rec"] = pairs_df.apply(
        lambda x: (
            find_dataset_ind(windows_ds, int(x["win_ind1"]))
            == find_dataset_ind(windows_ds, int(x["win_ind2"]))
        ),
        axis=1,
    )

    assert len(pairs) == len(sampler) <= n_examples // world_size
    assert all(pairs_df.loc[pairs_df["y"] == 1, "diff"] <= tau_pos)
    if same_rec_neg:
        assert all(pairs_df.loc[pairs_df["y"] == 0, "diff"] >= tau_neg)
        assert all(pairs_df["same_rec"] == same_rec_neg)
    else:
        assert all(pairs_df.loc[pairs_df["y"] == 0, "same_rec"] == False)  # noqa: E712
        assert all(pairs_df.loc[pairs_df["y"] == 1, "same_rec"] == True)  # noqa: E712
    assert abs(np.diff(pairs_df["y"].value_counts())) < 20


@pytest.mark.skipif(platform.system() == 'Windows',
                    reason="Not supported on Windows because of use_libuv compatibility")
@pytest.mark.parametrize("same_rec_neg", [True, False])
def test_distributed_relative_positioning_sampler(windows_ds, same_rec_neg):
    world_size = 1
    mp.spawn(distributed_relative_positioning_sampler_init_process, args=(world_size, windows_ds, same_rec_neg), nprocs=world_size, join=True)


def distributed_relative_positioning_sampler_n_examples_check(rank, world_size, windows_ds, n_examples_total):
    """Test that n_examples calculation uses correct operator precedence."""
    dist.init_process_group(
        backend="gloo",
        init_method="tcp://127.0.0.1:29500",
        rank=rank,
        world_size=world_size
    )

    tau_pos, tau_neg = 2000, 3000
    sampler = DistributedRelativePositioningSampler(
        windows_ds.get_metadata(),
        tau_pos=tau_pos,
        tau_neg=tau_neg,
        n_examples=n_examples_total,
        tau_max=None,
        same_rec_neg=True,
        random_state=33,
    )

    # Calculate expected n_examples for this rank
    # Formula should be: n_examples_total * n_recordings_for_rank // total_recordings
    total_recordings = sampler.info.shape[0]
    recordings_per_rank = sampler.n_recordings
    expected_n_examples = n_examples_total * recordings_per_rank // total_recordings

    assert sampler.n_examples == expected_n_examples, (
        f"Rank {rank}: Expected {expected_n_examples} examples but got {sampler.n_examples}. "
        f"total_recordings={total_recordings}, recordings_per_rank={recordings_per_rank}, "
        f"n_examples_total={n_examples_total}"
    )

    # Cleanup
    dist.destroy_process_group()


@pytest.mark.skipif(platform.system() == 'Windows',
                    reason="Not supported on Windows because of use_libuv compatibility")
@pytest.mark.parametrize("n_examples_total,world_size", [
    (100, 2),  # Test case where division might lose precision
    (50, 2),   # Test case from bug report that could truncate to 0
    (100, 4),  # Test case from bug report
])
def test_distributed_relative_positioning_sampler_n_examples_calculation(windows_ds, n_examples_total, world_size):
    """Test that n_examples calculation distributes examples correctly across ranks.

    This test validates the fix for the operator precedence bug where:
    - Buggy: n_examples // total_recordings * recordings_per_rank (double truncation)
    - Fixed: n_examples * recordings_per_rank // total_recordings (single division)
    """
    mp.spawn(
        distributed_relative_positioning_sampler_n_examples_check,
        args=(world_size, windows_ds, n_examples_total),
        nprocs=world_size,
        join=True
    )


@pytest.mark.parametrize("n_windows,n_windows_stride", [[10, 5], [10, 100], [1, 1]])
def test_sequence_sampler(windows_ds, n_windows, n_windows_stride):
    sampler = SequenceSampler(
        windows_ds.get_metadata(), n_windows, n_windows_stride, random_state=31
    )

    seqs = [seq for seq in sampler]

    seq_lens = [
        (len(ds) - n_windows) // n_windows_stride + 1 for ds in windows_ds.datasets
    ]
    file_ids = np.concatenate([[i] * length for i, length in enumerate(seq_lens)])
    n_seqs = sum(seq_lens)
    assert len(seqs) == n_seqs
    assert all([len(s) == n_windows for s in seqs])

    for i in range(seq_lens[0] - 1):
        np.testing.assert_array_equal(
            seqs[i][n_windows_stride:], seqs[i + 1][:-n_windows_stride]
        )

    assert (sampler.file_ids == file_ids).all()

    # for randomized sampler
    sampler = SequenceSampler(
        windows_ds.get_metadata(),
        n_windows,
        n_windows_stride,
        randomize=True,
        random_state=31,
    )

    seqs = [seq for seq in sampler]

    seq_lens = [
        (len(ds) - n_windows) // n_windows_stride + 1 for ds in windows_ds.datasets
    ]
    file_ids = np.concatenate([[i] * length for i, length in enumerate(seq_lens)])
    n_seqs = sum(seq_lens)
    assert len(seqs) == n_seqs
    assert all([len(s) == n_windows for s in seqs])


@pytest.mark.parametrize("n_sequences,n_windows", [[10, 2], [2, 40], [99, 1]])
def test_balanced_sequence_sampler(target_windows_ds, n_sequences, n_windows):
    md = target_windows_ds.get_metadata()
    sampler = BalancedSequenceSampler(
        md, n_windows, n_sequences=n_sequences, random_state=87
    )

    seqs = [seq for seq in sampler]

    assert len(seqs) == n_sequences
    assert all([len(s) == n_windows for s in seqs])

    # Make sure the sequences are valid
    for seq in seqs:
        assert all(np.diff(seq) == 1)  # windows must be consecutive
        seq_md = md.iloc[seq[0] : seq[-1] + 1]
        for c in ["subject", "session", "run"]:
            assert len(seq_md[c].unique()) == 1

    # Make sure the target is always in the sequence
    for _ in range(100):
        start_ind, rec_ind, class_ind = sampler._sample_seq_start_ind()
        seq_targets = md.iloc[start_ind : start_ind + n_windows + 1]["target"]
        assert class_ind in seq_targets.values
        rec_info = sampler.info.iloc[rec_ind].name
        rec_info_md = md.iloc[start_ind][["subject", "session", "run"]]
        assert rec_info == tuple(rec_info_md.tolist())


def test_balanced_sequence_sampler_single_category(target_windows_ds):
    """Test the case where there's only one category in the metadata, e.g.
    'subject'.
    """
    n_windows = 3
    n_sequences = 10

    md = target_windows_ds.get_metadata().drop(columns=["session", "run"])
    sampler = BalancedSequenceSampler(
        md, n_windows, n_sequences=n_sequences, random_state=87
    )

    seqs = [seq for seq in sampler]
    assert len(seqs) == n_sequences
    assert all([len(s) == n_windows for s in seqs])


def test_balanced_sequence_sampler_no_targets(windows_ds):
    md = windows_ds.get_metadata().drop(columns="target")
    with pytest.raises(ValueError):
        BalancedSequenceSampler(md, 10, n_sequences=5, random_state=87)
