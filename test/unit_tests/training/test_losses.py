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
    info_nce,
    mixup_criterion,
)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_dance_loss_trains_every_head(dtype):
    """``DanceLoss`` on ``detect`` gives every head a finite, nonzero gradient.

    Every query predicts the span (sigmoid(-1), sigmoid(1)) = (0.27, 0.73), which
    overlaps the targets, so both bounds train; tokens 0, 4 and 5 of 6 lie in no
    span, which made the consistency term NaN in float16.
    """
    torch.manual_seed(0)
    model = DANCE(
        n_outputs=3,
        n_chans=2,
        n_times=32,
        sfreq=128,
        use_channel_merger=False,
        n_queries=4,
        num_latents=6,
        embed_dim=8,
        conv_initial_linear=8,
        conv_depth=1,
        perceiver_depth=1,
        cross_dim_head=4,
        latent_dim_head=4,
        decoder_dim=8,
        decoder_depth=1,
        decoder_heads=2,
        drop_prob=0.0,
    ).to(dtype)
    with torch.no_grad():
        for head, bias in ((model.decoder.start_head, -1), (model.decoder.end_head, 1)):
            head.weight.zero_()
            head.bias.fill_(bias)
    x = torch.randn(2, 2, 32).to(dtype).requires_grad_()
    preds = model.detect(x)
    assert preds["class"].shape == (2, 4, 3) and preds["dense"].shape == (2, 6, 3)
    torch.testing.assert_close(preds["dense"], model(x), rtol=0, atol=0)

    targets = {
        "class": torch.tensor([[1], [2]]),
        "start": torch.tensor([[0.2], [0.3]], dtype=dtype),
        "end": torch.tensor([[0.7], [0.8]], dtype=dtype),
    }
    criterion = DanceLoss(num_latents=model.num_latents)
    loss, details = criterion(preds, targets)
    with torch.no_grad():  # the consistency term does not depend on the precision
        _, expected = criterion({k: v.float() for k, v in preds.items()}, targets)
    assert details["consistency_loss"] == pytest.approx(expected["consistency_loss"])
    loss.backward()
    heads = (
        model.final_layer,
        model.decoder.class_head,
        model.decoder.start_head,
        model.decoder.end_head,
    )
    for grad in (x.grad, model.decoder.query_embed.grad, *(h.weight.grad for h in heads)):
        assert torch.isfinite(grad).all() and torch.count_nonzero(grad)


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


def test_info_nce_value_for_orthogonal_pairs():
    z = torch.eye(4)
    # positive logit 1 / 0.1 = 10, six negatives at 0
    expected = torch.log1p(6 * torch.exp(torch.tensor(-10.0)))
    torch.testing.assert_close(info_nce(z, z, temperature=0.1), expected)


def test_info_nce_is_symmetric():
    torch.manual_seed(0)
    z1, z2 = torch.randn(8, 16), torch.randn(8, 16)
    torch.testing.assert_close(info_nce(z1, z2), info_nce(z2, z1))


def test_info_nce_refuses_mismatched_views():
    with pytest.raises(ValueError, match=r"same shape, got \(4, 8\) and \(4, 9\)"):
        info_nce(torch.randn(4, 8), torch.randn(4, 9))


@pytest.mark.parametrize(
    "bad_shape",
    [(4,), (2, 4, 8), (0, 8), (4, 0)],
)
def test_info_nce_refuses_invalid_shapes(bad_shape):
    z = torch.randn(bad_shape)
    with pytest.raises(ValueError):
        info_nce(z, z)


@pytest.mark.parametrize("temperature", [0.0, -0.1, float("nan"), float("inf")])
def test_info_nce_refuses_invalid_temperature(temperature):
    z1, z2 = torch.randn(4, 8), torch.randn(4, 8)
    with pytest.raises(ValueError, match=r"temperature must be positive and finite"):
        info_nce(z1, z2, temperature=temperature)


def test_info_nce_backward_produces_finite_gradients():
    torch.manual_seed(0)
    z1 = torch.randn(6, 12, requires_grad=True)
    z2 = torch.randn(6, 12, requires_grad=True)
    loss = info_nce(z1, z2, temperature=0.1)
    assert torch.isfinite(loss)
    loss.backward()
    for grad in (z1.grad, z2.grad):
        assert grad is not None
        assert torch.isfinite(grad).all()
        assert torch.count_nonzero(grad) > 0


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_info_nce_low_precision_matches_float32(dtype):
    torch.manual_seed(0)
    z1_f32 = torch.randn(6, 12)
    z2_f32 = torch.randn(6, 12)
    z1 = z1_f32.to(dtype).detach().requires_grad_(True)
    z2 = z2_f32.to(dtype).detach().requires_grad_(True)

    loss = info_nce(z1, z2, temperature=0.1)
    expected = info_nce(z1.float(), z2.float(), temperature=0.1)
    assert torch.isfinite(loss)
    torch.testing.assert_close(loss.float(), expected, rtol=1e-2, atol=1e-2)
    loss.backward()
    for grad in (z1.grad, z2.grad):
        assert grad is not None
        assert torch.isfinite(grad).all()
        assert torch.count_nonzero(grad) > 0


def test_info_nce_exported_from_training():
    import braindecode.training as training

    assert "info_nce" in training.__all__
    assert training.info_nce is info_nce


def test_info_nce_single_sample_batch():
    z1 = torch.randn(1, 8, requires_grad=True)
    z2 = torch.randn(1, 8, requires_grad=True)
    loss = info_nce(z1, z2, temperature=0.1)
    assert torch.isfinite(loss)
    torch.testing.assert_close(loss, torch.zeros(()))
    loss.backward()
    assert z1.grad is not None and torch.isfinite(z1.grad).all()
    assert z2.grad is not None and torch.isfinite(z2.grad).all()
