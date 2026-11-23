# Authors: Robin Schirrmeister <robintibor@gmail.com>
#          Maciej Sliwowski <maciek.sliwowski@gmail.com>
#          Mohammed Fattouh <mo.fattouh@gmail.com>
#          Sarthak Tayal <sarthaktayal2@gmail.com>
#
# License: BSD (3-clause)

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

from braindecode.functional import iou_1d, pairwise_iou_1d


class CroppedLoss(nn.Module):
    """Compute Loss after averaging predictions across time.
    Assumes predictions are in shape:
    n_batch size x n_classes x n_predictions (in time)"""

    def __init__(self, loss_function):
        super().__init__()
        self.loss_function = loss_function

    def forward(self, preds, targets):
        """Forward pass.

        Parameters
        ----------
        preds: torch.Tensor
            Model's prediction with shape (batch_size, n_classes, n_times).
        targets: torch.Tensor
            Targets with shape (batch_size,), or (batch_size, n_classes) for
            regression. Mixup's ``(y_a, y_b, lam)`` is passed to
            ``loss_function`` as is.
        """
        avg_preds = torch.mean(preds, dim=2)
        # 2-D targets already match the (batch_size, n_outputs) predictions.
        # Everything else is squeezed as before, including Mixup's list target.
        if not (isinstance(targets, torch.Tensor) and targets.ndim == 2):
            avg_preds = avg_preds.squeeze(dim=1)
        return self.loss_function(avg_preds, targets)


class TimeSeriesLoss(nn.Module):
    """Compute Loss between timeseries targets and predictions.
    Assumes predictions are in shape:
    n_batch size x n_classes x n_predictions (in time)
    Assumes targets are in shape:
    n_batch size x n_classes x window_len (in time)
    If the targets contain NaNs, the NaNs will be masked out and the loss will be only computed for
    predictions valid corresponding to valid target values.

    Notes
    -----
    :class:`~braindecode.models.NeuroPose`, :class:`~braindecode.models.VEMG2Pose`
    and :class:`~braindecode.models.SensingDynamics` return ``(batch, time, joints)``,
    with time before joints: their docstrings show how to train them with this loss.
    """

    def __init__(self, loss_function):
        super().__init__()
        self.loss_function = loss_function

    def forward(self, preds, targets):
        """Forward pass.

        Parameters
        ----------
        preds: torch.Tensor
            Model's prediction with shape (batch_size, n_classes, n_times).
        targets: torch.Tensor
            Target labels with shape (batch_size, n_classes, n_times).
        """
        n_preds = preds.shape[-1]
        # slice the targets to fit preds shape
        targets = targets[:, :, -n_preds:]
        # create valid targets mask
        mask = ~torch.isnan(targets)
        # select valid targets that have a matching predictions
        masked_targets = targets[mask]
        masked_preds = preds[mask]
        return self.loss_function(masked_preds, masked_targets)


def mixup_criterion(preds, target):
    """Implements loss for Mixup for EEG data. See [1]_.

    Implementation based on [2]_.

    Parameters
    ----------
    preds : torch.Tensor
        Predictions from the model: logits, as braindecode models output, or
        log-probabilities of each window, which give the same loss.
    target : torch.Tensor | tuple of torch.Tensor | list of torch.Tensor
        For predictions without mixup, the targets as a tensor. If mixup has
        been applied, a tuple or list containing the targets of the two mixed
        samples and the mixing coefficients as tensors. A plain tensor is
        always read as targets, whatever the batch size is. Targets can be
        class indices or class probabilities, per trial or dense (one per
        prediction in time); for dense targets, each trial's loss is averaged
        over its predictions before the mixing coefficient weights it.

    Returns
    -------
    loss : float
        The loss value.

    References
    ----------
    .. [1] Hongyi Zhang, Moustapha Cisse, Yann N. Dauphin, David Lopez-Paz
       mixup: Beyond Empirical Risk Minimization
       Online: https://arxiv.org/abs/1710.09412
    .. [2] https://github.com/facebookresearch/mixup-cifar10/blob/master/train.py
    """
    # only a mixup target is a container, a plain target tensor of length 3 is not
    if isinstance(target, (tuple, list)) and len(target) == 3:
        # unpack target
        y_a, y_b, lam = target
        # compute loss per sample
        loss_a = torch.nn.functional.cross_entropy(preds, y_a, reduction="none")
        loss_b = torch.nn.functional.cross_entropy(preds, y_b, reduction="none")
        # Each coefficient weights its trial, including all dense predictions.
        if loss_a.ndim > 1:
            loss_a = loss_a.flatten(start_dim=1).mean(dim=1)
            loss_b = loss_b.flatten(start_dim=1).mean(dim=1)
        return (lam * loss_a + (1 - lam) * loss_b).mean()
    return torch.nn.functional.cross_entropy(preds, target)


def info_nce(
    z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.1
) -> torch.Tensor:
    r"""InfoNCE (NT-Xent) contrastive loss between two views of a batch [1]_.

    Row ``i`` of ``z1`` and row ``i`` of ``z2`` embed two views of the same
    sample. Each of the :math:`2N` rows of ``cat([z1, z2])`` is an anchor whose
    positive is the other view of its sample and whose negatives are the other
    :math:`2N - 2` rows:

    .. math::

        \ell_i = -\log \frac{\exp(\mathrm{sim}(z_i, z_{i^+}) / \tau)}
        {\sum_{k \neq i} \exp(\mathrm{sim}(z_i, z_k) / \tau)}

    with cosine similarity :math:`\mathrm{sim}` and temperature :math:`\tau`.
    The loss is the mean of :math:`\ell_i` over the :math:`2N` anchors.

    Parameters
    ----------
    z1, z2 : torch.Tensor
        Embeddings of the two views, both of shape ``(batch, dim)``, for
        example ``model(x, return_features=True)["embedding"]`` passed through
        a projection head.
    temperature : float
        Temperature :math:`\tau` dividing the cosine similarities. Must be
        positive.

    Returns
    -------
    torch.Tensor
        The scalar loss.

    References
    ----------
    .. [1] Chen, T., Kornblith, S., Norouzi, M., Hinton, G. (2020).
       A Simple Framework for Contrastive Learning of Visual Representations.
       ICML. https://arxiv.org/abs/2002.05709
    """
    if z1.ndim != 2 or z1.shape != z2.shape:
        raise ValueError(
            "info_nce takes two (batch, dim) tensors of the same shape, got "
            f"{tuple(z1.shape)} and {tuple(z2.shape)}."
        )
    if z1.shape[0] < 2:
        raise ValueError("info_nce needs a batch size of at least 2.")
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}.")
    n = z1.shape[0]
    z = torch.nn.functional.normalize(torch.cat([z1, z2]), dim=1)
    logits = (z @ z.T / temperature).fill_diagonal_(float("-inf"))
    # the positive of anchor i is the other view of the same sample
    targets = torch.arange(2 * n, device=z.device).roll(n)
    return torch.nn.functional.cross_entropy(logits, targets)


class HungarianMatcher(nn.Module):
    """Hungarian one-to-one matching between queries and ground-truth events.

    Class id 0 = no-object/padding (CLASS-0 CONTRACT): targets are kept via
    ``(tgt_class != 0)`` and unmatched query slots stay class 0.

    Cost = ``weight_class * CE + 1 * (1 - IoU)``. NOTE: ``weight_iou`` is stored
    but NOT applied inside the matcher cost (matches upstream exactly); it scales
    only the DETR IoU loss term in :class:`DanceLoss`.
    """

    def __init__(self, weight_class: float = 1.0, weight_iou: float = 5.0):
        super().__init__()
        self.weight_class = weight_class
        self.weight_iou = weight_iou

    @torch.no_grad()
    def forward(self, outputs, targets):
        cls = outputs["class"]  # (B, Q, n_classes)
        b, q, _ = cls.shape
        # Buffers on the preds' device so GPU targets assign cleanly.
        ms = torch.zeros(b, q, device=cls.device)
        me = torch.zeros(b, q, device=cls.device)
        mc = torch.zeros(b, q, dtype=torch.long, device=cls.device)
        for bi in range(b):
            tgt_cls = targets["class"][bi]
            keep = (tgt_cls != 0).nonzero(as_tuple=True)[0]
            if keep.numel() == 0:
                continue
            ts = targets["start"][bi][keep]
            te = targets["end"][bi][keep]
            tc = tgt_cls[keep]
            log_p = torch.log_softmax(cls[bi], dim=-1)  # (Q, n_classes)
            cls_cost = self.weight_class * (-log_p[:, tc])  # (Q, n_targets)
            # PAIRWISE IoU: (Q,) preds x (n_targets,) targets -> (Q, n_targets).
            iou = pairwise_iou_1d(outputs["start"][bi], outputs["end"][bi], ts, te)
            loc_cost = 1.0 - iou  # (Q, n_targets)
            total = cls_cost + loc_cost  # IoU NOT weighted here
            # NumPy has no bfloat16 dtype for SciPy's assignment solver.
            cost = total.detach().to(
                device="cpu", dtype=torch.promote_types(total.dtype, torch.float32)
            )
            cost_np = np.nan_to_num(cost.numpy(), nan=1e6, posinf=1e6, neginf=1e6)
            q_idx, t_idx = linear_sum_assignment(cost_np)
            for qi, ti in zip(q_idx, t_idx):
                ms[bi, qi] = ts[ti]
                me[bi, qi] = te[ti]
                mc[bi, qi] = tc[ti]
        # mc/ms/me are already on cls.device; unmatched slots stay class 0. The
        # caller reads preds straight from ``outputs`` (the matcher only sorts
        # targets onto query slots), so only the matched targets are returned.
        return {"class": mc, "start": ms, "end": me}


class DanceLoss(nn.Module):
    """DANCE criterion: matched DETR (CE + IoU) + dense CE + consistency KL."""

    def __init__(
        self,
        weight_class: float = 1.0,
        weight_iou: float = 5.0,
        weight_dense: float = 1.0,
        weight_consistency: float = 0.5,
        num_latents: int = 256,
    ):
        super().__init__()
        self.matcher = HungarianMatcher(weight_class, weight_iou)
        self.weight_class = weight_class
        self.weight_iou = weight_iou
        self.weight_dense = weight_dense
        self.weight_consistency = weight_consistency
        self.num_latents = num_latents
        self.ce = nn.CrossEntropyLoss()
        self.kl = nn.KLDivLoss(reduction="none")

    def _dense_target(self, targets, n_classes, device):
        start = targets["start"]
        end = targets["end"]
        cls = targets["class"]
        b, max_e = cls.shape
        t = self.num_latents
        dense = torch.zeros(b, t, dtype=torch.long, device=device)
        s_tok = (start * t).clamp(0, t).long()
        e_tok = (end * t).clamp(0, t).long()
        for bi in range(b):
            for i in range(max_e):
                c = int(cls[bi, i])
                if c == 0:
                    continue
                s, e = int(s_tok[bi, i]), int(e_tok[bi, i])
                if s < e:
                    dense[bi, s:e] = c
        return dense

    @staticmethod
    def _detr_to_dense_probs(preds, num_latents, n_classes):
        """Project DETR query predictions onto a dense per-token distribution.

        ``T == num_latents`` (passed directly, NOT derived from
        duration*frequency), so the output ``(B, num_latents, n_classes)`` can
        never silently mismatch the dense head. Training-side only (Python loop;
        the matcher already breaks the graph, so this is never on the export
        path).
        """
        cls = preds["class"]
        b, q, _ = cls.shape
        t = int(num_latents)
        # The probability floor for uncovered tokens is smaller than float16 can hold.
        probs = torch.softmax(
            cls, dim=-1, dtype=torch.promote_types(cls.dtype, torch.float32)
        )
        out = []
        for bi in range(b):
            mask = probs.new_zeros((t, n_classes))
            for qi in range(q):
                s = float(preds["start"][bi, qi])
                e = float(preds["end"][bi, qi])
                start = int(max(0, s * t))
                end = int(min(t, e * t))
                if start < end:
                    mask[start:end] += probs[bi, qi]
            out.append(mask / (mask.sum(-1, keepdim=True) + 1e-8))
        return torch.stack(out)

    def forward(self, detect_output, targets, duration=None):
        # `duration` kept for API symmetry but UNUSED: the consistency KL is
        # built on the num_latents token grid directly (see _detr_to_dense_probs).
        n_classes = detect_output["class"].shape[-1]
        device = detect_output["class"].device
        # CLASS-0 CONTRACT: matcher keeps tgt_class != 0; unmatched slots = 0.
        mt = self.matcher(detect_output, targets)
        logits = detect_output["class"].reshape(-1, n_classes)
        labels = mt["class"].reshape(-1).long()
        cls_term = self.weight_class * self.ce(logits, labels)
        # IoU loss over the Hungarian-matched spans ONLY. Unmatched/no-object
        # queries carry a (0, 0) target, so averaging over all Q would let them
        # each contribute (1 - 0) = 1 and flatten the localization signal; we
        # normalize by the number of matches instead (matches dance/losses.py).
        matched = (mt["class"] != 0).float()
        iou = iou_1d(
            detect_output["start"], detect_output["end"], mt["start"], mt["end"]
        )
        iou_term = self.weight_iou * (
            ((1.0 - iou) * matched).sum() / matched.sum().clamp(min=1.0)
        )

        # Dense head time dim MUST equal num_latents (defensive guard; raise
        # rather than assert so it survives ``python -O``).
        if detect_output["dense"].shape[1] != self.num_latents:
            raise ValueError(
                f"dense time dim {detect_output['dense'].shape[1]} != "
                f"num_latents {self.num_latents}"
            )
        dense_logits = detect_output["dense"].reshape(-1, n_classes)
        if "dense" in targets and targets["dense"] is not None:
            dense_t = targets["dense"].reshape(-1).long().to(device)
        else:
            dense_t = self._dense_target(targets, n_classes, device).reshape(-1)
        dense_term = self.weight_dense * self.ce(dense_logits, dense_t)

        dense = detect_output["dense"]
        dense_probs = torch.softmax(
            dense, dim=-1, dtype=torch.promote_types(dense.dtype, torch.float32)
        ).clamp(min=1e-8)
        detr_probs = self._detr_to_dense_probs(
            detect_output, self.num_latents, n_classes
        ).clamp(min=1e-8)  # (B, num_latents, n_classes) == dense_probs
        cons_term = self.weight_consistency * (
            self.kl(detr_probs.log(), dense_probs).sum(dim=-1).mean()
        )
        loss = cls_term + iou_term + dense_term + cons_term
        details = {
            "class_loss": float(cls_term),
            "iou_loss": float(iou_term),
            "dense_loss": float(dense_term),
            "consistency_loss": float(cons_term),
        }
        return loss, details
