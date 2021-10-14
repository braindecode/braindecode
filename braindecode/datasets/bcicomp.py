# Authors: Maciej Sliwowski <maciek.sliwowski@gmail.com>
#          Mohammed Fattouh <mo.fattouh@gmail.com>
#
# License: BSD (3-clause)

import glob
import os
import os.path as osp
from os import remove
from shutil import unpack_archive

import mne
import numpy as np
from mne.utils import verbose
from scipy.io import loadmat

from braindecode.datasets import BaseDataset, BaseConcatDataset

DATASET_URL = 'https://stacks.stanford.edu/file/druid:zk881ps0522/' \
              'BCI_Competion4_dataset4_data_fingerflexions.zip'


class BCICompetitionIVDataset4(BaseConcatDataset):
    """BCI competition IV dataset 4.

    Contains ECoG recordings for three patients moving fingers during the experiment.
    Targets correspond to the time courses of the flexion of each of five fingers.
    See http://www.bbci.de/competition/iv/desc_4.pdf and
    http://www.bbci.de/competition/iv/ for the dataset and competition description.
    ECoG library containing the dataset: https://searchworks.stanford.edu/view/zk881ps0522

    Notes
    -----
    When using this dataset please cite [1]_ .

    Parameters
    ----------
    subject_ids : list(int) | int | None
        (list of) int of subject(s) to be loaded. If None, load all available
        subjects. Should be in range 1-3.

    References
    ----------
    .. [1] Miller, Kai J. "A library of human electrocorticographic data and analyses."
    Nature human behaviour 3, no. 11 (2019): 1225-1235. https://doi.org/10.1038/s41562-019-0678-3
    """
    possible_subjects = [1, 2, 3]

    def __init__(self, subject_ids=None):
        data_path = self.download()
        if isinstance(subject_ids, int):
            subject_ids = [subject_ids]
        if subject_ids is None:
            subject_ids = self.possible_subjects
        self._validate_subjects(subject_ids)
        files_list = [f'{data_path}/sub{i}_comp.mat' for i in subject_ids]
        datasets = []
        for file_path in files_list:
            raw_train, raw_test = self._load_data_to_mne(file_path)
            desc_train = dict(
                subject=file_path.split('/')[-1].split('sub')[1][0],
                file_name=file_path.split('/')[-1],
                session='train'
            )
            desc_test = dict(
                subject=file_path.split('/')[-1].split('sub')[1][0],
                file_name=file_path.split('/')[-1],
                session='test'
            )
            datasets.append(BaseDataset(raw_train, description=desc_train))
            datasets.append(BaseDataset(raw_test, description=desc_test))
        super().__init__(datasets)

    @staticmethod
    def download(path=None, force_update=False, verbose=None):
        """Download the dataset.

        Parameters
        ----------
        path  (None | str) – Location of where to look for the data storing location.
        If None, the environment variable or config parameter
        MNE_DATASETS_(dataset)_PATH is used. If it doesn’t exist, the “~/mne_data”
        directory is used. If the dataset is not found under the given path, the data
        will be automatically downloaded to the specified folder.
        force_update (bool) – Force update of the dataset even if a local copy exists.
        verbose (bool, str, int, or None) – If not None, override default verbose level
        (see mne.verbose())

        Returns
        -------

        """
        signature = 'BCICompetitionIVDataset4'
        folder_name = 'BCI_Competion4_dataset4_data_fingerflexions'
        # Check if the dataset already exists (unpacked). We have to do that manually
        # because we are removing .zip file from disk to save disk space.

        from moabb.datasets.download import get_dataset_path  # keep soft depenency
        path = get_dataset_path(signature, path)
        key_dest = "MNE-{:s}-data".format(signature.lower())
        # We do not use mne _url_to_local_path due to ':' in the url that causes problems on Windows
        destination = osp.join(path, key_dest, folder_name)
        if len(list(glob.glob(osp.join(destination, '*.mat')))) == 6:
            return destination
        data_path = _data_dl(DATASET_URL, osp.join(destination, folder_name, signature),
                             force_update=force_update)
        unpack_archive(data_path, osp.dirname(destination))
        # removes .zip file that the data was unpacked from
        remove(data_path)
        return destination

    @staticmethod
    def _prepare_targets(upsampled_targets, targets_stride):
        original_targets = np.full_like(upsampled_targets, np.nan)
        original_targets[::targets_stride] = upsampled_targets[::targets_stride]
        return original_targets

    def _load_data_to_mne(self, file_path):
        data = loadmat(file_path)
        test_labels = loadmat(file_path.replace('comp.mat', 'testlabels.mat'))
        train_data = data['train_data']
        test_data = data['test_data']
        upsampled_train_targets = data['train_dg']
        upsampled_test_targets = test_labels['test_dg']

        signal_sfreq = 1000
        original_target_sfreq = 25
        targets_stride = int(signal_sfreq / original_target_sfreq)

        original_targets = self._prepare_targets(upsampled_train_targets, targets_stride)
        original_test_targets = self._prepare_targets(upsampled_test_targets, targets_stride)

        ch_names = [f'{i}' for i in range(train_data.shape[1])]
        ch_names += [f'target_{i}' for i in range(original_targets.shape[1])]
        ch_types = ['ecog' for _ in range(train_data.shape[1])]
        ch_types += ['misc' for _ in range(original_targets.shape[1])]

        info = mne.create_info(sfreq=signal_sfreq, ch_names=ch_names, ch_types=ch_types)
        info['target_sfreq'] = original_target_sfreq
        train_data = np.concatenate([train_data, original_targets], axis=1)
        test_data = np.concatenate([test_data, original_test_targets], axis=1)

        raw_train = mne.io.RawArray(train_data.T, info=info)
        raw_test = mne.io.RawArray(test_data.T, info=info)
        # TODO: show how to resample targets
        return raw_train, raw_test

    def _validate_subjects(self, subject_ids):
        if isinstance(subject_ids, (list, tuple)):
            if not all((subject in self.possible_subjects for subject in subject_ids)):
                raise ValueError(
                    f'Wrong subject_ids parameter. Possible values: {self.possible_subjects}. '
                    f'Provided {subject_ids}.'
                )
        else:
            raise ValueError(
                'Wrong subject_ids format. Expected types: None, list, tuple, int.'
            )


@verbose
def _data_dl(url, destination, force_update=False, verbose=None):
    # Code taken from moabb due to problem with ':' occurring in path
    # On Windows ':' is a forbidden in folder name
    # moabb/datasets/download.py

    from pooch import file_hash, retrieve  # keep soft depenency
    if not osp.isfile(destination) or force_update:
        if osp.isfile(destination):
            os.remove(destination)
        if not osp.isdir(osp.dirname(destination)):
            os.makedirs(osp.dirname(destination))
        known_hash = None
    else:
        known_hash = file_hash(destination)
    data_path = retrieve(
        url, known_hash, fname=osp.basename(url), path=osp.dirname(destination)
    )
    return data_path
