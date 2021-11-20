# Authors: Cédric Rommel <cedric.rommel@inria.fr>
#
# License: BSD (3-clause)

import mne
import numpy as np
import pytest
import torch
from sklearn.utils import check_random_state

from braindecode.augmentation.base import AugmentedDataLoader
from braindecode.augmentation.base import Compose
from braindecode.augmentation.base import Transform
from braindecode.datautil import create_from_mne_epochs


def dummy_k_operation(X, y, k):
    return torch.ones_like(X) * k, y


class DummyTransform(Transform):
    operation = staticmethod(dummy_k_operation)

    def __init__(self, probability=1.0, random_state=None, k=None):
        if k is None:
            self.k = np.random.randint(10)
        else:
            self.k = k
        super().__init__(probability=probability, random_state=random_state)

    def get_params(self, X, y):
        return {"k": self.k}


@pytest.fixture
def dummy_transform():
    return DummyTransform()


def common_tranform_assertions(
    input_batch,
    output_batch,
    expected_X=None,
    diff_param=None,
):
    """ Assert whether shapes and devices are conserved. Also, (optional)
    checks whether the expected features matrix is produced.

    Parameters
    ----------
    input_batch : tuple
        The batch given to the transform containing a tensor X, of shape
        (batch_sizze, n_channels, sequence_len), and a tensor  y of shape
        (batch_size).
    output_batch : tuple
        The batch output by the transform. Should have two elements: the
        transformed X and y.
    expected_X : torch.Tensor, optional
        The expected first element of output_batch, which will be compared to
        it. By default None.
    diff_param : torch.Tensor | None, optional
        Parameter which should have grads.
    """
    X, y = input_batch
    tr_X, tr_y = output_batch
    assert tr_X.shape == X.shape
    assert tr_X.shape[0] == tr_y.shape[0]
    assert torch.equal(tr_y, y)
    assert X.device == tr_X.device
    if expected_X is not None:
        assert torch.equal(tr_X, expected_X)
    if diff_param is not None:
        loss = (tr_X - X).sum()
        loss.backward()
        assert diff_param.grad is not None


def test_transform_call_with_no_label(random_batch, dummy_transform):
    X, y = random_batch
    tr_X1, _ = dummy_transform(X, y)
    tr_X2 = dummy_transform(X)
    assert torch.equal(tr_X1, tr_X2)


@pytest.mark.parametrize("k1,k2,expected,p1,p2", [
    (1, 0, 0, 1, 1),  # replace by 1s with p=1, then 0s with p=1 -> 0s
    (0, 1, 1, 1, 1),  # replace by 0s with p=1, then 1s with p=1 -> 1s
    (1, 0, 1, 1, 0),  # replace by 1s with p=1, then 1s with p=0 -> 1s
    (0, 1, 0, 1, 0),  # replace by 0s with p=1, then 0s with p=0 -> 0s
    (1, 0, 0, 0, 1),  # replace by 1s with p=0, then 0s with p=1 -> 0s
    (0, 1, 1, 0, 1),  # replace by 0s with p=0, then 1s with p=1 -> 1s
])
def test_transform_composition(random_batch, k1, k2, expected, p1, p2):
    X, y = random_batch
    dummy_transform1 = DummyTransform(k=k1, probability=p1)
    dummy_transform2 = DummyTransform(k=k2, probability=p2)
    concat_transform = Compose([dummy_transform1, dummy_transform2])
    expected_tensor = torch.ones(
        X.shape,
        device=X.device
    ) * expected

    common_tranform_assertions(
        random_batch,
        concat_transform(X, y),
        expected_tensor
    )


def test_transform_proba_exception(rng_seed, dummy_transform):
    rng = check_random_state(rng_seed)
    with pytest.raises(AssertionError):
        DummyTransform(
            probability='a',
            random_state=rng,
        )


@pytest.fixture(scope="session")
def concat_windows_dataset():
    """Generates a small BaseConcatDataset out of WindowDatasets extracted
    from the physionet database.
    """
    subject_id = 22
    event_codes = [5, 6, 9, 10, 13, 14]
    physionet_paths = mne.datasets.eegbci.load_data(
        subject_id, event_codes, update_path=False)

    parts = [mne.io.read_raw_edf(path, preload=True, stim_channel='auto')
             for path in physionet_paths]
    list_of_epochs = [mne.Epochs(raw, [[0, 0, 0]], tmin=0, baseline=None)
                      for raw in parts]
    windows_datasets = create_from_mne_epochs(
        list_of_epochs,
        window_size_samples=50,
        window_stride_samples=50,
        drop_last_window=False
    )

    return windows_datasets


# test AugmentedDataLoader with 0, 1 and 2 composed transforms
@pytest.mark.parametrize("nb_transforms,no_list", [
    (0, False), (1, False), (1, True), (2, False)
])
def test_data_loader(dummy_transform, concat_windows_dataset, nb_transforms,
                     no_list):
    transforms = [dummy_transform for _ in range(nb_transforms)]
    if no_list:
        transforms = transforms[0]
    data_loader = AugmentedDataLoader(
        concat_windows_dataset,
        transforms=transforms,
        batch_size=128)
    for idx_batch, _ in enumerate(data_loader):
        if idx_batch >= 3:
            break


def test_data_loader_exception(concat_windows_dataset):
    with pytest.raises(TypeError):
        AugmentedDataLoader(
            concat_windows_dataset,
            transforms='a',
            batch_size=128
        )


def test_dataset_with_transform(concat_windows_dataset):
    factor = 10
    transform = DummyTransform(k=factor)
    concat_windows_dataset.transform = transform
    transformed_X = concat_windows_dataset[0][0]
    assert torch.all(transformed_X == factor)
