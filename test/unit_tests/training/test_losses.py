# Authors: Simon Brandt <simonbrandt@protonmail.com>
#          Maciej Sliwowski <maciek.sliwowski@gmail.com>
#          Sarthak Tayal <sarthaktayal2@gmail.com>
#
# License: BSD-3 (3-clause)

import numpy as np
import pytest
import torch

from braindecode.models import DANCE
from braindecode.training.losses import (
    CroppedLoss,
    DanceLoss,
    TimeSeriesLoss,
    mixup_criterion,
)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_dance_loss_backward_with_empty_tokens(dtype):
    torch.manual_seed(0)
    model = DANCE(
        n_outputs=3,
        n_chans=2,
        n_times=32,
        sfreq=128,
        use_channel_merger=False,
        n_queries=2,
        embed_dim=8,
        conv_hidden=8,
        conv_depth=1,
        conv_kernel_size=3,
        conv_initial_linear=4,
        num_latents=8,
        perceiver_depth=1,
        cross_attn_heads=1,
        latent_attn_heads=1,
        cross_dim_head=4,
        latent_dim_head=4,
        decoder_dim=8,
        decoder_depth=1,
        decoder_heads=1,
        n_virtual_channels=2,
        drop_prob=0.0,
        conv_drop_prob=0.0,
    ).to(dtype)
    x = torch.randn(2, 2, 32).to(dtype)
    preds = model.detect(x)
    assert all(torch.isfinite(value).all() for value in preds.values())
    targets = {
        "class": torch.tensor([[1, 0], [0, 0]]),
        "start": torch.tensor([[0.2, 0.0], [0.0, 0.0]], dtype=dtype),
        "end": torch.tensor([[0.4, 0.0], [0.0, 0.0]], dtype=dtype),
    }
    criterion = DanceLoss(num_latents=8)
    loss, details = criterion(preds, targets)
    with torch.no_grad():
        _, expected = criterion(
            {name: value.float() for name, value in preds.items()}, targets
        )
    assert details["consistency_loss"] == pytest.approx(expected["consistency_loss"])
    assert torch.isfinite(loss)
    loss.backward()
    for head in (
        model.final_layer,
        model.decoder.class_head,
        model.decoder.start_head,
        model.decoder.end_head,
    ):
        assert torch.isfinite(head.weight.grad).all()
    # The spans do not overlap the target, so only these heads get a nonzero grad.
    for head in (model.final_layer, model.decoder.class_head):
        assert torch.count_nonzero(head.weight.grad)


def test_mixup_criterion():
    n_classes = 2
    n_samples = 5
    y_a = torch.zeros(n_samples, dtype=torch.int64)
    y_b = torch.ones(n_samples, dtype=torch.int64)
    lam = torch.arange(0.1, 1, 1 / n_samples)

    preds = torch.Tensor(np.random.RandomState(42).randn(n_samples, n_classes))
    log_probs = preds.log_softmax(dim=1)

    target = (y_a, y_b, lam)
    loss = mixup_criterion(preds, target)
    expected = -(lam * log_probs[:, 0] + (1 - lam) * log_probs[:, 1]).mean()
    assert loss == pytest.approx(expected)

    target = y_a
    loss = mixup_criterion(preds, target)
    expected = -log_probs[:, 0].mean()
    assert loss == pytest.approx(expected)


@pytest.mark.parametrize("n_samples", [1, 2, 3, 4])
def test_mixup_criterion_plain_target(n_samples):
    n_classes = 2
    preds = torch.Tensor(np.random.RandomState(42).randn(n_samples, n_classes))
    target = torch.zeros(n_samples, dtype=torch.int64)

    loss = mixup_criterion(preds, target)
    expected = -preds.log_softmax(dim=1)[:, 0].mean()
    assert loss == pytest.approx(expected)


@pytest.mark.parametrize("soft", [False, True])
def test_mixup_criterion_weights_dense_targets_per_trial(soft):
    logits = torch.randn(3, 4, 5)
    # Targets of the two mixed trials: class indices (3, 5) or probabilities (3, 4, 5).
    first, second = torch.randint(4, (2, 3, 5))
    if soft:
        first, second = torch.randn(2, 3, 4, 5).softmax(dim=2)
    weights = torch.tensor([0.2, 0.5, 0.8])
    ce = torch.nn.functional.cross_entropy
    expected = torch.stack(
        [
            w * ce(p[None], a[None]) + (1 - w) * ce(p[None], b[None])
            for p, a, b, w in zip(logits, first, second, weights)
        ]
    ).mean()
    torch.testing.assert_close(
        mixup_criterion(logits, (first, second, weights)), expected
    )


def test_mixup_criterion_accepts_list_target():
    n_samples = 3
    y_a = torch.zeros(n_samples, dtype=torch.int64)
    y_b = torch.ones(n_samples, dtype=torch.int64)
    lam = torch.arange(0.1, 1, 1 / n_samples)
    preds = torch.Tensor(np.random.RandomState(42).randn(n_samples, 2))
    log_probs = preds.log_softmax(dim=1)

    loss = mixup_criterion(preds, [y_a, y_b, lam])
    expected = -(lam * log_probs[:, 0] + (1 - lam) * log_probs[:, 1]).mean()
    assert loss == pytest.approx(expected)


def test_mixup_criterion_takes_logits_or_log_probabilities():
    # braindecode models output logits; a log-softmax output gives the same loss
    rng = np.random.RandomState(42)
    logits = torch.as_tensor(10 * rng.randn(6, 4), dtype=torch.float32)
    y_a = torch.as_tensor([0, 1, 2, 3, 0, 1])
    y_b = torch.as_tensor([3, 2, 1, 0, 2, 3])
    lam = torch.as_tensor(rng.uniform(size=6), dtype=torch.float32)
    for target in [(y_a, y_b, lam), y_a]:
        loss = mixup_criterion(logits, target)
        assert loss >= 0
        assert loss == pytest.approx(
            mixup_criterion(logits.log_softmax(dim=1), target).item(), rel=1e-5
        )


def test_cropped_loss_accepts_mixup_target():
    # Cropped training with Mixup hands CroppedLoss a (y_a, y_b, lam) list.
    n_samples = 3
    y_a = torch.zeros(n_samples, dtype=torch.int64)
    y_b = torch.ones(n_samples, dtype=torch.int64)
    lam = torch.arange(0.1, 1, 1 / n_samples)
    preds = torch.Tensor(np.random.RandomState(42).randn(n_samples, 2, 4))

    loss = CroppedLoss(mixup_criterion)(preds, [y_a, y_b, lam])
    expected = mixup_criterion(preds.mean(dim=2), [y_a, y_b, lam])
    assert loss == pytest.approx(expected)


def test_time_series_loss():
    targets = torch.Tensor(
        np.array(
            [
                [
                    [np.nan, 0.2, np.nan, 0.3, np.nan, 0.1, np.nan, 0.9],
                    [np.nan, 0.8, np.nan, 0.2, np.nan, 0.9, np.nan, 0.2],
                ],
                [
                    [np.nan, 0.3, np.nan, 0.2, np.nan, 0.5, np.nan, 0.1],
                    [np.nan, 0.1, np.nan, 0.2, np.nan, 0.3, np.nan, 0.2],
                ],
            ]
        )
    )
    targets_expected = torch.Tensor(
        np.array([[[0.1, 0.9], [0.9, 0.2]], [[0.5, 0.1], [0.3, 0.2]]])
    )

    preds = torch.Tensor(
        np.array(
            [
                [[0.4, 0.9, 0.4], [0.1, 0.3, 0.8]],
                [[0.2, 0.5, 0.3], [0.2, 0.1, 0.5]],
            ]
        )
    )
    preds_expected = torch.Tensor(
        np.array([[[0.4, 0.4], [0.1, 0.8]], [[0.2, 0.3], [0.2, 0.5]]])
    )

    time_series_loss = TimeSeriesLoss(lambda *args: args)

    preds_out, targets_out = time_series_loss(preds, targets)

    torch.testing.assert_allclose(preds_out, preds_expected.flatten())
    torch.testing.assert_allclose(targets_out, targets_expected.flatten())
