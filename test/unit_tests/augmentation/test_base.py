# Authors: Cédric Rommel <cedric.rommel@inria.fr>
#          Sarthak Tayal <sarthaktayal2@gmail.com>
#
# License: BSD (3-clause)

import mne
import numpy as np
import pytest
import torch
from sklearn.utils import check_random_state

from braindecode import EEGRegressor
from braindecode.augmentation.base import (
    AugmentedDataLoader,
    Compose,
    IdentityTransform,
    Transform,
    _AugmentationCollate,
)
from braindecode.augmentation.transforms import (
    GaussianNoise,
    LabelSmoothing,
    Mixup,
    SmoothTimeMask,
)
from braindecode.datasets import create_from_mne_epochs
from braindecode.training.losses import mixup_criterion


def dummy_k_operation(X, y, k):
    return torch.ones_like(X) * k, y


class DummyTransform(Transform):
    operation = staticmethod(dummy_k_operation)  # type: ignore[assignment]

    def __init__(self, probability=1.0, random_state=None, k=None):
        if k is None:
            self.k = np.random.randint(10)
        else:
            self.k = k
        super().__init__(probability=probability, random_state=random_state)

    def get_augmentation_params(self, X, y):
        return {"k": self.k}


class _TinyMixupRegressor(torch.nn.Module):
    def __init__(self, n_chans=2, n_times=20, n_outputs=1):
        super().__init__()
        self.linear = torch.nn.Linear(n_chans * n_times, n_outputs)

    def forward(self, x):
        return self.linear(x.flatten(start_dim=1))


class _RecordingMixupMSELoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.seen_targets = []

    def forward(self, preds, target):
        y_a, y_b, lam = target
        self.seen_targets.append(
            tuple(part.detach().cpu().clone() for part in (y_a, y_b, lam))
        )
        lam = lam.reshape(lam.shape[0], *([1] * (preds.ndim - 1)))
        return (
            lam * torch.square(preds - y_a) + (1 - lam) * torch.square(preds - y_b)
        ).mean()


@pytest.fixture
def dummy_transform():
    return DummyTransform()


def common_transform_assertions(
        input_batch,
        output_batch,
        expected_X=None,
        diff_param=None,
):
    """Assert whether shapes and devices are conserved. Also, (optional)
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


@pytest.mark.parametrize(
    "k1,k2,expected,p1,p2",
    [
        (1, 0, 0, 1, 1),  # replace by 1s with p=1, then 0s with p=1 -> 0s
        (0, 1, 1, 1, 1),  # replace by 0s with p=1, then 1s with p=1 -> 1s
        (1, 0, 1, 1, 0),  # replace by 1s with p=1, then 1s with p=0 -> 1s
        (0, 1, 0, 1, 0),  # replace by 0s with p=1, then 0s with p=0 -> 0s
        (1, 0, 0, 0, 1),  # replace by 1s with p=0, then 0s with p=1 -> 0s
        (0, 1, 1, 0, 1),  # replace by 0s with p=0, then 1s with p=1 -> 1s
    ],
)
def test_transform_composition(random_batch, k1, k2, expected, p1, p2):
    X, y = random_batch
    dummy_transform1 = DummyTransform(k=k1, probability=p1)
    dummy_transform2 = DummyTransform(k=k2, probability=p2)
    concat_transform = Compose([dummy_transform1, dummy_transform2])
    expected_tensor = torch.ones(X.shape, device=X.device) * expected

    common_transform_assertions(random_batch, concat_transform(X, y), expected_tensor)


def test_transform_proba_exception(rng_seed, dummy_transform):
    rng = check_random_state(rng_seed)
    with pytest.raises(AssertionError):
        DummyTransform(
            probability="a",
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
        subject_id, event_codes, update_path=False
    )

    parts = [
        mne.io.read_raw_edf(path, preload=True, stim_channel="auto")
        for path in physionet_paths
    ]
    list_of_epochs = [
        mne.Epochs(raw, [[0, 0, 0]], tmin=0, baseline=None) for raw in parts
    ]
    windows_datasets = create_from_mne_epochs(
        list_of_epochs,
        window_size_samples=50,
        window_stride_samples=50,
        drop_last_window=False,
    )

    return windows_datasets


# test AugmentedDataLoader with 0, 1 and 2 composed transforms
@pytest.mark.parametrize(
    "nb_transforms,no_list,dummy",
    [
        (0, False, True),
        (1, False, True),
        (1, True, True),
        (2, False, True),
        (1, False, False),
    ],
)
@pytest.mark.network
def test_data_loader(
        dummy_transform, concat_windows_dataset, nb_transforms, no_list, dummy
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    transform = dummy_transform if dummy else SmoothTimeMask(0.5)
    transforms = [transform for _ in range(nb_transforms)]
    if no_list:
        transforms = transforms[0]
    data_loader = AugmentedDataLoader(
        concat_windows_dataset, transforms=transforms, batch_size=128, device=device
    )
    for idx_batch, _ in enumerate(data_loader):
        if idx_batch >= 3:
            break


@pytest.mark.network
def test_data_loader_exception(concat_windows_dataset):
    with pytest.raises(TypeError):
        AugmentedDataLoader(concat_windows_dataset, transforms="a", batch_size=128)


@pytest.mark.network
def test_dataset_with_transform(concat_windows_dataset):
    factor = 10
    transform = DummyTransform(k=factor)
    concat_windows_dataset.transform = transform
    transformed_X = concat_windows_dataset[0][0]
    assert torch.all(transformed_X == factor)


def test_augmented_data_loader_n_augmentation():
    """``n_augmentation`` keeps the clean originals and appends augmented copies."""
    from torch.utils.data import TensorDataset

    n_samples, n_chans, n_times, batch_size = 8, 4, 20, 8
    X = torch.randn(n_samples, n_chans, n_times)
    y = torch.arange(n_samples)
    k = 7.0

    loader = AugmentedDataLoader(
        TensorDataset(X, y),
        transforms=DummyTransform(k=k, probability=1.0),
        batch_size=batch_size,
        n_augmentation=5,
        shuffle=False,
    )
    batch_X, batch_y = next(iter(loader))

    # 1 clean + 5 augmented copies -> 6x expansion
    assert batch_X.shape == (6 * batch_size, n_chans, n_times)
    assert batch_y.shape == (6 * batch_size,)
    # The first block are the untouched originals (the point of the feature).
    assert torch.equal(batch_X[:batch_size], X)
    # Every appended copy was augmented (DummyTransform replaces X by all-k).
    assert torch.all(batch_X[batch_size:] == k)
    # Labels are tiled across the clean + augmented copies.
    assert torch.equal(batch_y, y.repeat(6))


def _epochs_of_noise(random_state, num_workers, n_epochs=2):
    from torch.utils.data import TensorDataset

    dataset = TensorDataset(torch.zeros(8, 2, 8), torch.zeros(8, dtype=torch.long))
    transform = GaussianNoise(1.0, std=1.0, random_state=random_state)
    loader = AugmentedDataLoader(
        dataset, transforms=transform, batch_size=4, num_workers=num_workers
    )
    return [torch.cat([X for X, _ in loader]) for _ in range(n_epochs)]


@pytest.mark.parametrize("random_state", [0, None])
def test_augmented_data_loader_workers_draw_new_augmentations(random_state):
    """Transforms in a worker draw new augmentations each epoch.

    The worker used to get a copy of the transform with the generator in the
    state of the main process (also a pickled copy of the global numpy
    generator for ``random_state=None`` with spawned workers), so every epoch
    drew the same noise.
    """
    np.random.seed(0)
    epoch_1, epoch_2 = _epochs_of_noise(random_state, num_workers=1)
    assert not torch.equal(epoch_1, epoch_2)
    # still reproducible from the seed
    np.random.seed(0)
    again_1, again_2 = _epochs_of_noise(random_state, num_workers=1)
    assert torch.equal(epoch_1, again_1) and torch.equal(epoch_2, again_2)


@pytest.mark.parametrize("container", ["single", "list", "nested"])
def test_explicit_worker_seeds_leave_global_numpy_rng_unchanged(container):
    from torch.utils.data import TensorDataset

    transform = GaussianNoise(1.0, std=1.0, random_state=0)
    if container == "list":
        transform = [transform]
    elif container == "nested":
        transform = Compose([Compose([transform])])
    loader = AugmentedDataLoader(
        TensorDataset(torch.zeros(4, 2, 8), torch.zeros(4, dtype=torch.long)),
        transforms=transform,
        num_workers=1,
    )
    before = np.random.get_state()
    loader.collated_tr.draw_worker_seeds()
    after = np.random.get_state()
    assert before[0] == after[0]
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]


def test_compose_keeps_global_rng_used_by_child():
    from braindecode.augmentation.base import _transform_rngs

    noise = GaussianNoise(1.0, std=1.0, random_state=None)
    rngs = _transform_rngs(Compose([Compose([noise])]))
    assert len(rngs) == 1
    assert rngs[0] is noise.rng


def test_custom_compose_keeps_its_own_rng(monkeypatch):
    from types import SimpleNamespace

    import braindecode.augmentation.base as base

    class RandomCompose(Compose):
        def forward(self, X, y=None):
            return super().forward(X + self.rng.normal(), y)

    child = GaussianNoise(1.0, std=1.0, random_state=17)
    transform = RandomCompose([child])
    transform.rng = np.random.RandomState(11)
    rngs = base._transform_rngs(transform)
    assert len(rngs) == 2
    assert rngs[0] is transform.rng and rngs[1] is child.rng
    expected = [int(np.random.RandomState(seed).randint(2**31)) for seed in (11, 17)]
    collate = base._AugmentationCollate(transform)
    collate.draw_worker_seeds()
    assert collate.worker_seeds == expected
    monkeypatch.setattr(base, "get_worker_info", lambda: SimpleNamespace(id=2))
    collate._seed_worker_rngs()
    for rng, seed in zip(rngs, expected):
        reference = np.random.RandomState(
            np.random.SeedSequence([seed, 2]).generate_state(1)
        )
        np.testing.assert_array_equal(rng.get_state()[1], reference.get_state()[1])
        assert rng.get_state()[2:] == reference.get_state()[2:]


def test_augmentation_collate_reseeds_per_worker(monkeypatch):
    """Two workers of one epoch do not draw the same augmentations."""
    import copy
    from types import SimpleNamespace

    import braindecode.augmentation.base as base

    noise = GaussianNoise(1.0, std=1.0, random_state=0)
    collate = _AugmentationCollate(Compose([noise, SmoothTimeMask(0.5)]))
    collate.draw_worker_seeds()
    batch = [(torch.zeros(2, 8), 0)] * 4
    outputs = []
    for worker_id in (0, 1):
        # what a worker receives: a copy, generators in the main-process state
        worker_collate = copy.deepcopy(collate)
        monkeypatch.setattr(
            base, "get_worker_info", lambda: SimpleNamespace(id=worker_id)
        )
        outputs.append(worker_collate(batch)[0])
    assert not torch.equal(outputs[0], outputs[1])


def test_augmented_data_loader_default_no_expansion():
    """Default (``n_augmentation=0``) keeps the batch size unchanged."""
    from torch.utils.data import TensorDataset

    batch_size = 8
    X = torch.randn(batch_size, 4, 20)
    y = torch.arange(batch_size)

    loader = AugmentedDataLoader(
        TensorDataset(X, y),
        transforms=DummyTransform(k=1.0, probability=1.0),
        batch_size=batch_size,
        shuffle=False,
    )
    batch_X, _ = next(iter(loader))
    assert batch_X.shape[0] == batch_size


def test_augmented_data_loader_negative_n_augmentation():
    """A negative ``n_augmentation`` is rejected."""
    from torch.utils.data import TensorDataset

    dataset = TensorDataset(torch.randn(4, 4, 20), torch.arange(4))
    with pytest.raises(ValueError, match="n_augmentation"):
        AugmentedDataLoader(dataset, n_augmentation=-1, batch_size=4)


@pytest.mark.parametrize("beta_per_sample", [False, True])
def test_augmented_data_loader_n_augmentation_mixup(beta_per_sample):
    """``n_augmentation`` keeps the ``(y_a, y_b, lam)`` triple Mixup returns."""
    from torch.utils.data import TensorDataset

    n_samples, n_chans, n_times, batch_size = 8, 4, 20, 8
    X = torch.randn(n_samples, n_chans, n_times)
    y = torch.arange(n_samples) % 2

    loader = AugmentedDataLoader(
        TensorDataset(X, y),
        transforms=Mixup(alpha=0.5, beta_per_sample=beta_per_sample, random_state=0),
        batch_size=batch_size,
        n_augmentation=2,
        shuffle=False,
    )
    batch_X, batch_y = next(iter(loader))

    assert batch_X.shape == (3 * batch_size, n_chans, n_times)
    assert isinstance(batch_y, tuple) and len(batch_y) == 3
    y_a, y_b, lam = batch_y
    assert y_a.shape == y_b.shape == lam.shape == (3 * batch_size,)
    # The clean block is untouched and mixed with itself with a coefficient of one.
    assert torch.equal(batch_X[:batch_size], X)
    assert torch.equal(y_a[:batch_size], y)
    assert torch.equal(y_b[:batch_size], y)
    assert torch.all(lam[:batch_size] == 1.0)
    # The loss can consume the batch, which needs one single dtype for lam.
    preds = torch.log_softmax(torch.randn(3 * batch_size, 2), dim=1)
    assert torch.isfinite(mixup_criterion(preds, batch_y))


def test_augmentation_collate_accepts_list_mixed_target():
    """List-form mixed targets follow the same expansion path as tuples."""
    batch_size = 4
    X = torch.randn(batch_size, 2, 10)
    y = torch.arange(batch_size)

    def list_mixup(batch_X, batch_y):
        lam = torch.full((batch_X.shape[0],), 0.5, dtype=batch_X.dtype)
        return batch_X, [batch_y, batch_y.flip(0), lam]

    batch_X, batch_y = _AugmentationCollate(list_mixup, n_augmentation=1)(
        list(zip(X, y))
    )

    assert batch_X.shape[0] == 2 * batch_size
    assert isinstance(batch_y, tuple) and len(batch_y) == 3
    assert all(part.shape == (2 * batch_size,) for part in batch_y)


@pytest.mark.parametrize("beta_per_sample", [False, True])
def test_mixup_lam_follows_batch_dtype(beta_per_sample):
    """``lam`` stays on the dtype of the batch so the loss is not upcast."""
    X = torch.randn(6, 3, 20)
    y = torch.arange(6) % 2

    _, out_y = Mixup(alpha=0.5, beta_per_sample=beta_per_sample, random_state=0)(X, y)
    assert out_y[2].dtype == X.dtype

    preds = torch.log_softmax(torch.randn(6, 2), dim=1)
    assert mixup_criterion(preds, out_y).dtype == preds.dtype


@pytest.mark.parametrize("beta_per_sample", [False, True])
@pytest.mark.parametrize("n_augmentation", [0, 2])
def test_eegregressor_mixup_preserves_fractional_targets(
    beta_per_sample, n_augmentation
):
    """The public fit boundary keeps fractional ``(B, 1)`` mixed targets."""
    batch_size, n_chans, n_times = 6, 2, 20
    X = torch.randn(batch_size, n_chans, n_times)
    y = torch.tensor(
        [0.25, 1.75, -2.5, 3.125, 4.875, -5.25], dtype=torch.float64
    ).reshape(batch_size, 1)

    regressor = EEGRegressor(
        _TinyMixupRegressor,
        criterion=_RecordingMixupMSELoss,
        optimizer=torch.optim.SGD,
        lr=0.01,
        iterator_train=AugmentedDataLoader,
        iterator_train__transforms=Mixup(
            alpha=0.5,
            beta_per_sample=beta_per_sample,
            random_state=0,
        ),
        iterator_train__n_augmentation=n_augmentation,
        iterator_train__shuffle=False,
        iterator_train__drop_last=False,
        batch_size=batch_size,
        max_epochs=1,
        train_split=None,
        verbose=0,
    )

    regressor.fit(X, y)

    assert len(regressor.criterion_.seen_targets) == 1
    y_a, y_b, lam = regressor.criterion_.seen_targets[0]
    n_copies = n_augmentation + 1
    expected_y = y.to(torch.float32)
    expected_y_a = expected_y.repeat(n_copies, 1)
    assert y_a.shape == y_b.shape == (n_copies * batch_size, 1)
    assert y_a.dtype == y_b.dtype == torch.float32
    assert torch.equal(y_a, expected_y_a)

    expected_sorted = expected_y.sort(dim=0).values
    for target_block in y_b.split(batch_size):
        assert torch.equal(target_block.sort(dim=0).values, expected_sorted)

    assert lam.shape == (n_copies * batch_size,)
    assert lam.dtype == torch.float32
    if n_augmentation:
        assert torch.equal(lam[:batch_size], torch.ones(batch_size))
    assert np.isfinite(regressor.history[-1, "train_loss"])


@pytest.mark.network
def test_single_input_aug(concat_windows_dataset):
    # Create single input without the batch dimension, just (channels, time)
    X, y, _ = concat_windows_dataset[0]
    assert len(X.shape) == 2
    assert isinstance(y, int)

    # Create dummy data augmentation
    factor = 10
    transform = DummyTransform(k=factor, probability=1.0)

    # Check that the transformation forward works and outputs an augmented
    # input with the batch dimension
    transformed_X = transform(X)
    assert transformed_X.shape == X.shape

    # Same check, when also passing y
    transformed_X, transformed_y = transform(X, y)
    assert len(transformed_y.shape) == 1
    assert transformed_X.shape == X.shape


@pytest.mark.parametrize("batch_size", [2, 3])
def test_compose_without_target(batch_size):
    X = torch.randn(batch_size, 3, 10)
    composed = Compose([IdentityTransform(), IdentityTransform()])
    for out in (composed(X), composed(X, None)):
        assert isinstance(out, torch.Tensor)
        assert torch.equal(out, X)
    y = torch.arange(batch_size)
    out_X, out_y = composed(X, y)
    assert torch.equal(out_X, X) and torch.equal(out_y, y)


def test_augmentation_collate_rejects_reshaped_y():
    """Fixed expansion rejects a transform reshaping ``y`` with a clear error."""
    batch_size = 4
    X = torch.randn(batch_size, 2, 10)
    y = torch.arange(batch_size) % 2

    def to_distribution(batch_X, batch_y):
        return batch_X, torch.eye(2)[batch_y]

    with pytest.raises(ValueError, match="n_augmentation = 0"):
        _AugmentationCollate(to_distribution, n_augmentation=1)(list(zip(X, y)))

    # The same transform keeps working in the in-place mode.
    out_X, out_y = _AugmentationCollate(to_distribution)(list(zip(X, y)))
    assert out_y.shape == (batch_size, 2)


def test_augmentation_collate_rejects_reshaped_mixed_y():
    """The guard also covers a reshaped ``(y_a, y_b, lam)`` triple."""
    batch_size = 4
    X = torch.randn(batch_size, 2, 10)
    y = torch.arange(batch_size) % 2

    def mixup_distributions(batch_X, batch_y):
        dist = torch.eye(2)[batch_y]
        lam = torch.full((batch_X.shape[0],), 0.5)
        return batch_X, (dist, dist, lam)

    with pytest.raises(ValueError, match="changes the shape of y"):
        _AugmentationCollate(mixup_distributions, n_augmentation=1)(
            list(zip(X, y))
        )


def test_augmentation_collate_rejects_reshaped_X():
    """The guard is symmetric for a transform that reshapes ``X``."""
    batch_size = 4
    X = torch.randn(batch_size, 2, 10)
    y = torch.arange(batch_size) % 2

    def flatten_signal(batch_X, batch_y):
        return batch_X.reshape(batch_X.shape[0], -1), batch_y

    with pytest.raises(ValueError, match="changes the shape of X"):
        _AugmentationCollate(flatten_signal, n_augmentation=1)(list(zip(X, y)))


def test_augmented_data_loader_rejects_label_smoothing():
    """``LabelSmoothing`` in fixed expansion fails with the documented error.

    The smoothed targets are distributions while the clean originals kept by
    fixed expansion are class indices, so the combination is rejected up front
    instead of failing inside ``torch.cat`` (as noted in :gh:`1401`).
    """
    from torch.utils.data import TensorDataset

    n_samples, n_chans, n_times, batch_size = 8, 2, 20, 8
    X = torch.randn(n_samples, n_chans, n_times)
    y = torch.arange(n_samples) % 2

    loader = AugmentedDataLoader(
        TensorDataset(X, y),
        transforms=LabelSmoothing(epsilon=0.1, num_classes=2),
        batch_size=batch_size,
        n_augmentation=2,
        shuffle=False,
    )
    with pytest.raises(ValueError, match="n_augmentation = 0"):
        next(iter(loader))
