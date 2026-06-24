# Authors: Robin Schirrmeister <robintibor@gmail.com>
#          Maciej Sliwowski <maciek.sliwowski@gmail.com>
#          Mohammed Fattouh <mo.fattouh@gmail.com>
#
# License: BSD (3-clause)

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

from braindecode.functional import detr_to_dense_probs, iou_1d, pairwise_iou_1d


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
            Target labels with shape (batch_size, n_classes, n_times).
        """
        avg_preds = torch.mean(preds, dim=2)
        avg_preds = avg_preds.squeeze(dim=1)
        return self.loss_function(avg_preds, targets)


class TimeSeriesLoss(nn.Module):
    """Compute Loss between timeseries targets and predictions.
    Assumes predictions are in shape:
    n_batch size x n_classes x n_predictions (in time)
    Assumes targets are in shape:
    n_batch size x n_classes x window_len (in time)
    If the targets contain NaNs, the NaNs will be masked out and the loss will be only computed for
    predictions valid corresponding to valid target values."""

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
        Predictions from the model.
    target : torch.Tensor | list of torch.Tensor
        For predictions without mixup, the targets as a tensor. If mixup has
        been applied, a list containing the targets of the two mixed
        samples and the mixing coefficients as tensors.

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
    if len(target) == 3:
        # unpack target
        y_a, y_b, lam = target
        # compute loss per sample
        loss_a = torch.nn.functional.nll_loss(preds, y_a, reduction="none")
        loss_b = torch.nn.functional.nll_loss(preds, y_b, reduction="none")
        # compute weighted mean
        ret = torch.mul(lam, loss_a) + torch.mul(1 - lam, loss_b)
        return ret.mean()
    else:
        return torch.nn.functional.nll_loss(preds, target)


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
    def __call__(self, outputs, targets):
        cls = outputs["class"]  # (B, Q, n_classes)
        b, q, _ = cls.shape
        ms = torch.zeros(b, q)
        me = torch.zeros(b, q)
        mc = torch.zeros(b, q, dtype=torch.long)
        matches = []
        for bi in range(b):
            tgt_cls = targets["class"][bi]
            keep = (tgt_cls != 0).nonzero(as_tuple=True)[0]
            if keep.numel() == 0:
                matches.append(([], []))
                continue
            ts = targets["start"][bi][keep]
            te = targets["end"][bi][keep]
            tc = tgt_cls[keep]
            log_p = torch.log_softmax(cls[bi], dim=-1)  # (Q, n_classes)
            cls_cost = self.weight_class * (-log_p[:, tc])  # (Q, n_targets)
            # PAIRWISE IoU: (Q,) preds x (n_targets,) targets -> (Q, n_targets).
            iou = pairwise_iou_1d(
                outputs["start"][bi], outputs["end"][bi], ts, te
            )
            loc_cost = 1.0 - iou  # (Q, n_targets)
            total = cls_cost + loc_cost  # IoU NOT weighted here
            cost_np = np.nan_to_num(
                total.cpu().numpy(), nan=1e6, posinf=1e6, neginf=1e6
            )
            q_idx, t_idx = linear_sum_assignment(cost_np)
            for qi, ti in zip(q_idx, t_idx):
                ms[bi, qi] = ts[ti]
                me[bi, qi] = te[ti]
                mc[bi, qi] = tc[ti]
            matches.append((list(q_idx), list(t_idx)))
        matched_preds = {
            "class": outputs["class"], "start": outputs["start"],
            "end": outputs["end"],
        }
        matched_targets = {"class": mc.to(cls.device), "start": ms.to(cls.device),
                           "end": me.to(cls.device)}
        return matched_preds, matched_targets, matches


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
        start = targets["start"]; end = targets["end"]; cls = targets["class"]
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

    def forward(self, detect_output, targets, duration=None):
        # `duration` kept for API symmetry but UNUSED: the consistency KL is
        # built on the num_latents token grid directly, so it can never
        # silently shape-mismatch the dense head (see detr_to_dense_probs).
        n_classes = detect_output["class"].shape[-1]
        device = detect_output["class"].device
        # CLASS-0 CONTRACT: matcher keeps tgt_class != 0; unmatched slots = 0.
        mp, mt, _ = self.matcher(detect_output, targets)
        logits = mp["class"].reshape(-1, n_classes)
        labels = mt["class"].reshape(-1).long()
        cls_term = self.weight_class * self.ce(logits, labels)
        # ELEMENTWISE IoU over the matched (B, Q) spans.
        iou = iou_1d(mp["start"], mp["end"], mt["start"], mt["end"])
        # Averages over ALL Q queries: unmatched/no-object slots contribute
        # (1 - 0) = 1. Documented loose-port choice (not upstream's
        # matched-only normalization); kept for parity with the dense head.
        iou_term = self.weight_iou * (1.0 - iou).mean()

        # Dense head time dim MUST equal num_latents (defensive guard).
        assert detect_output["dense"].shape[1] == self.num_latents, (
            f"dense time dim {detect_output['dense'].shape[1]} != "
            f"num_latents {self.num_latents}"
        )
        dense_logits = detect_output["dense"].reshape(-1, n_classes)
        if "dense" in targets and targets["dense"] is not None:
            dense_t = targets["dense"].reshape(-1).long().to(device)
        else:
            dense_t = self._dense_target(targets, n_classes, device).reshape(-1)
        dense_term = self.weight_dense * self.ce(dense_logits, dense_t)

        dense_probs = torch.softmax(detect_output["dense"], -1).clamp(min=1e-8)
        detr_probs = detr_to_dense_probs(
            detect_output, self.num_latents, n_classes
        ).clamp(min=1e-8).to(device)  # (B, num_latents, n_classes) == dense_probs
        cons_term = self.weight_consistency * (
            self.kl(detr_probs.log(), dense_probs).sum(dim=-1).mean()
        )
        loss = cls_term + iou_term + dense_term + cons_term
        details = {
            "class_loss": float(cls_term), "iou_loss": float(iou_term),
            "dense_loss": float(dense_term), "consistency_loss": float(cons_term),
        }
        return loss, details
