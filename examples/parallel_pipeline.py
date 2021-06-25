"""
Example for parallel preprocessing and serialization pipeline.

1. Load data
2. Preprocess data
3. Window data
4. Preprocess windows
"""

from braindecode.datasets.sleep_physionet import SleepPhysionet
from braindecode.preprocessing.preprocess import preprocess, Preprocessor


dataset = SleepPhysionet(
    subject_ids=[0, 1], recording_ids=[1], crop_wake_mins=30, preload=False)


# Preprocess the data
preprocessors = [
    Preprocessor(lambda x: x * 1e6),
    Preprocessor('filter', l_freq=None, h_freq=30)
]

save_dir = '/home/hubert/Data/testing/braindecode_tests/'
dataset = preprocess(dataset, preprocessors, save_dir=save_dir, overwrite=True,
                     reload=True, n_jobs=1)

assert all([not ds.raw.preload for ds in dataset.datasets])  # preload=False

breakpoint()

# ds2.load_data()  # put data in memory
# windows_ds = create_windows(ds, save_dir='my_save_dir', n_jobs=-1)
# assert windows_ds.preload == False
