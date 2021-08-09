"""
Test for samplers.
"""

# Authors: Hubert Banville <hubert.jbanville@gmail.com>
#
# License: BSD (3-clause)

import bisect

import pytest
import numpy as np
import pandas as pd

from braindecode.samplers import (
    RecordingSampler, SequenceSampler, BalancedSequenceSampler)
from braindecode.samplers.ssl import RelativePositioningSampler
from braindecode.datasets import BaseDataset, BaseConcatDataset
from braindecode.datasets.moabb import fetch_data_with_moabb
from braindecode.preprocessing.windowers import (
    create_fixed_length_windows, create_windows_from_events)


@pytest.fixture(scope='module')
def windows_ds():
    raws, description = fetch_data_with_moabb(
        dataset_name='BNCI2014001', subject_ids=4)
    ds = [BaseDataset(raws[i], description.iloc[i]) for i in range(3)]
    concat_ds = BaseConcatDataset(ds)

    windows_ds = create_fixed_length_windows(
        concat_ds=concat_ds, start_offset_samples=0, stop_offset_samples=None,
        window_size_samples=500, window_stride_samples=500,
        drop_last_window=False, preload=False)

    return windows_ds


@pytest.fixture(scope='module')
def target_windows_ds():
    raws, description = fetch_data_with_moabb(
        dataset_name='BNCI2014001', subject_ids=4)
    ds = [BaseDataset(raws[i], description.iloc[i]) for i in range(3)]
    concat_ds = BaseConcatDataset(ds)

    windows_ds = create_windows_from_events(
        concat_ds, trial_start_offset_samples=0, trial_stop_offset_samples=0,
        window_size_samples=None, window_stride_samples=None,
        drop_last_window=False)

    return windows_ds


def find_dataset_ind(windows_ds, win_ind):
    """Taken from torch.utils.data.dataset.ConcatDataset.
    """
    return bisect.bisect_right(windows_ds.cumulative_sizes, win_ind)


def test_recording_sampler(windows_ds):
    sampler = RecordingSampler(windows_ds.get_metadata(), random_state=87)
    assert sampler.n_recordings == windows_ds.description.shape[0]

    # Test info attribute
    assert isinstance(sampler.info, pd.DataFrame)
    assert isinstance(sampler.info.index, pd.MultiIndex)
    inds = np.concatenate(sampler.info['index'].values)
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


@pytest.mark.parametrize('same_rec_neg', [True, False])
def test_relative_positioning_sampler(windows_ds, same_rec_neg):
    tau_pos, tau_neg = 2000, 3000
    n_examples = 100
    sampler = RelativePositioningSampler(
        windows_ds.get_metadata(), tau_pos=tau_pos, tau_neg=tau_neg,
        n_examples=n_examples, tau_max=None, same_rec_neg=same_rec_neg,
        random_state=33)

    pairs = [pair for pair in sampler]
    pairs_df = pd.DataFrame(pairs, columns=['win_ind1', 'win_ind2', 'y'])
    pairs_df['diff'] = pairs_df.apply(
        lambda x: abs(windows_ds[int(x['win_ind1'])][2][1] -
                      windows_ds[int(x['win_ind2'])][2][1]), axis=1)
    pairs_df['same_rec'] = pairs_df.apply(
        lambda x: (find_dataset_ind(windows_ds, int(x['win_ind1'])) ==
                   find_dataset_ind(windows_ds, int(x['win_ind2']))), axis=1)

    assert len(pairs) == n_examples == len(sampler)
    assert all(pairs_df.loc[pairs_df['y'] == 1, 'diff'] <= tau_pos)
    if same_rec_neg:
        assert all(pairs_df.loc[pairs_df['y'] == 0, 'diff'] >= tau_neg)
        assert all(pairs_df['same_rec'] == same_rec_neg)
    else:
        assert all(pairs_df.loc[pairs_df['y'] == 0, 'same_rec'] == False)  # noqa: E712
        assert all(pairs_df.loc[pairs_df['y'] == 1, 'same_rec'] == True)  # noqa: E712
    assert abs(np.diff(pairs_df['y'].value_counts())) < 20


def test_relative_positioning_sampler_presample(windows_ds):
    tau_pos, tau_neg = 2000, 3000
    n_examples = 100
    sampler = RelativePositioningSampler(
        windows_ds.get_metadata(), tau_pos=tau_pos, tau_neg=tau_neg,
        n_examples=n_examples, tau_max=None, same_rec_neg=True,
        random_state=33)

    sampler.presample()
    assert hasattr(sampler, 'examples')
    assert len(sampler.examples) == n_examples

    pairs = [pair for pair in sampler]
    pairs2 = [pair for pair in sampler]
    assert np.array_equal(sampler.examples, pairs)
    assert np.array_equal(sampler.examples, pairs2)


@pytest.mark.parametrize('n_windows,n_windows_stride',
                         [[10, 5], [10, 100], [1, 1]])
def test_sequence_sampler(windows_ds, n_windows, n_windows_stride):
    sampler = SequenceSampler(
        windows_ds.get_metadata(), n_windows, n_windows_stride,
        random_state=31)

    seqs = [seq for seq in sampler]

    seq_lens = [(len(ds) - n_windows) // n_windows_stride + 1
                for ds in windows_ds.datasets]
    file_ids = np.concatenate([[i] * l for i, l in enumerate(seq_lens)])
    n_seqs = sum(seq_lens)
    assert len(seqs) == n_seqs
    assert all([len(s) == n_windows for s in seqs])

    for i in range(seq_lens[0] - 1):
        np.testing.assert_array_equal(
            seqs[i][n_windows_stride:], seqs[i + 1][:-n_windows_stride])

    assert (sampler.file_ids == file_ids).all()


@pytest.mark.parametrize('n_sequences,n_windows', [[10, 2], [2, 40], [99, 1]])
def test_balanced_sequence_sampler(target_windows_ds, n_sequences, n_windows):
    md = target_windows_ds.get_metadata()
    sampler = BalancedSequenceSampler(
        md, n_windows, n_sequences=n_sequences, random_state=87)

    seqs = [seq for seq in sampler]

    assert len(seqs) == n_sequences
    assert all([len(s) == n_windows for s in seqs])

    # Make sure the sequences are valid
    for seq in seqs:
        assert all(np.diff(seq) == 1)  # windows must be consecutive
        seq_md = md.iloc[seq[0]:seq[-1] + 1]
        for c in ['subject', 'session', 'run']:
            assert len(seq_md[c].unique()) == 1

    # Make sure the target is always in the sequence
    for _ in range(100):
        start_ind, rec_ind, class_ind = sampler._sample_seq_start_ind()
        seq_targets = md.iloc[start_ind:start_ind + n_windows + 1]['target']
        assert class_ind in seq_targets.values
        rec_info = sampler.info.iloc[rec_ind].name
        rec_info_md = md.iloc[start_ind][['subject', 'session', 'run']]
        assert rec_info == tuple(rec_info_md.tolist())


def test_balanced_sequence_sampler_single_category(target_windows_ds):
    """Test the case where there's only one category in the metadata, e.g.
    'subject'.
    """
    n_windows = 3
    n_sequences = 10

    md = target_windows_ds.get_metadata().drop(columns=['session', 'run'])
    sampler = BalancedSequenceSampler(
        md, n_windows, n_sequences=n_sequences, random_state=87)

    seqs = [seq for seq in sampler]
    assert len(seqs) == n_sequences
    assert all([len(s) == n_windows for s in seqs])


def test_balanced_sequence_sampler_no_targets(windows_ds):
    md = windows_ds.get_metadata().drop(columns='target')
    with pytest.raises(ValueError):
        BalancedSequenceSampler(md, 10, n_sequences=5, random_state=87)
