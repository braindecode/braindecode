# Authors: Maciej Sliwowski <maciek.sliwowski@gmail.com>
#
# License: BSD (3-clause)

import unittest

import mne
import numpy as np
import pandas as pd

from braindecode.datasets import WindowsDataset


class TestWindowsDataset(unittest.TestCase):
    # TODO: add test for transformers and case when subject_info is used
    def setUp(self):
        # super(TestWindowsDataset, self).__init__()
        rng = np.random.RandomState(42)
        info = mne.create_info(ch_names=['0', '1'],
                               sfreq=50, ch_types='eeg')
        self.raw = mne.io.RawArray(data=rng.randn(2, 1000), info=info)
        self.events = np.array([[100, 0, 1],
                                [200, 0, 2],
                                [300, 0, 1],
                                [400, 0, 4],
                                [500, 0, 3]])
        metadata = pd.DataFrame(data=self.events,
                                columns=['sample', 'x', 'target'])
        self.mne_epochs = mne.Epochs(raw=self.raw, events=self.events,
                                     metadata=metadata)
        self.epochs_data = self.mne_epochs.get_data()

        self.windows_dataset = WindowsDataset(self.mne_epochs, target="target")

    def test_get_item(self):
        for i in range(len(self.epochs_data)):
            x, y = self.windows_dataset[i]
            np.testing.assert_allclose(self.epochs_data[i], x)
            self.assertEqual(self.events[i, 2] - 1, y,
                             msg=f'Not equal for epoch {i}')

    def test_len(self):
        self.assertEqual(len(self.epochs_data), len(self.windows_dataset))

    def test_target_subject_info_is_none(self):
        with self.assertRaises(AssertionError):
            WindowsDataset(self.mne_epochs, target='is_none')

    def test_target_in_subject_info(self):
        with self.assertRaises(AssertionError):
            WindowsDataset(self.mne_epochs, target='does_not_exist')


if __name__ == '__main__':
    unittest.main()
