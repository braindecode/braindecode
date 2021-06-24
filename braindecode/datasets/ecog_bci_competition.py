import mne
import numpy as np
import scipy.io

from braindecode.datasets import BaseDataset, BaseConcatDataset


class EcogBCICompetition4(BaseConcatDataset):
    """BCI competition IV ECoG dataset with finger movements.

    Contains ECoG recordings for three patients moving fingers during the experiment.
    Targets correspond to the time courses of the flexion of each of five fingers.
    See http://www.bbci.de/competition/iv/desc_4.pdf for the dataset description.

    Test labels can be downloaded from:
    http://www.bbci.de/competition/iv/results/ds4/true_labels.zip

    Data can be downloaded from (Dataset 4):
    http://www.bbci.de/competition/iv/#dataset4

    Notes
    -----
    When using this dataset in publications please cite [1]_ .

    Parameters
    ----------
    path : str
        Path to the folder with BCI competition IV dataset 4. All .mat files are
        expected to be in this directory.
    subject_ids: list(int) | int | None
        (list of) int of subject(s) to be loaded. If None, load all available
        subjects. Should be in range 1-3.

    References
    ----------
    .. [1] Schalk, G., Kubanek, J., Miller, K.J., Anderson, N.R., Leuthardt, E.C.,
        Ojemann, J.G., Limbrick, D., Moran, D.W., Gerhardt, L.A., and Wolpaw, J.R.
        Decoding Two Dimensional Movement Trajectories Using Electrocorticographic Signals
        in Humans, J Neural Eng, 4: 264-275, 2007.
    """
    possible_subjects = [1, 2, 3]

    def __init__(self, path, subject_ids=None):
        if isinstance(subject_ids, int):
            subject_ids = [subject_ids]
        if subject_ids is None:
            subject_ids = self.possible_subjects
        self._validate_subjects(subject_ids)
        files_list = [f'{path}/sub{i}_comp.mat' for i in subject_ids]
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
    def _prepare_targets(upsampled_targets, targets_stride):
        original_targets = np.full_like(upsampled_targets, np.nan)
        original_targets[::targets_stride] = upsampled_targets[::targets_stride]
        return original_targets

    def _load_data_to_mne(self, file_path):
        data = scipy.io.loadmat(file_path)
        test_labels = scipy.io.loadmat(file_path.replace('comp.mat', 'testlabels.mat'))
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
        info.target_sfreq = original_target_sfreq
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
                    f'Wrong subject_ids parameter. Possible values: {subject_ids}. '
                    f'Provided {subject_ids}.'
                )
        else:
            raise ValueError(
                f'Wrong subject_ids format. Expected types: None, list, tuple, int.'
            )
