# Authors: Maciej Sliwowski <maciek.sliwowski@gmail.com>
#
# License: BSD (3-clause)

import unittest

import mne
import numpy as np
from braindecode.datautil import FixedLengthWindower


class TestFixedLengthWindower(unittest.TestCase):
    # TODO: add tests for case with drop_last_sample==False
    def setUp(self):
        rng = np.random.RandomState(42)
        self.sfreq = 50
        info = mne.create_info(ch_names=['0', '1'],
                               sfreq=self.sfreq, ch_types='eeg')
        self.data = rng.randn(2, 1000)
        self.raw = mne.io.RawArray(data=self.data, info=info)

        # test case:
        # (window_size_samples, overlap_size_samples, drop_last_samples,
        # tmin, n_windows)
        self.test_cases = [
            (100, 10, True, 0., 11),
            # (100, 10, True, -0.5, 11),  # TODO: does using tmin have sense?
            (100, 50, True, 0., 19),
            (None, 50, True, 0., 1)
        ]

    def test_windower_results(self):
        for test_case in self.test_cases:
            window_size, overlap_size, drop_last_samples, tmin, n_windows = test_case
            windower = FixedLengthWindower(
                window_size_samples=window_size,
                overlap_size_samples=overlap_size,
                drop_last_samples=drop_last_samples,
                tmin=tmin)

            epochs = windower(self.raw)
            epochs_data = epochs.get_data()
            if window_size is None:
                window_size = self.data.shape[1]
            idxs = np.arange(0,
                             self.data.shape[1] - window_size + 1,
                             window_size - overlap_size)

            self.assertEqual(len(idxs), epochs_data.shape[0])
            self.assertEqual(window_size, epochs_data.shape[2])
            for i, idx in enumerate(idxs):
                np.testing.assert_allclose(
                    self.data[:, idx: idx + window_size], epochs_data[i, :]
                )


if __name__ == '__main__':
    unittest.main()
