import math
import torch
from torch.optim.optimizer import Optimizer


class AdamW(Optimizer):
    """Implements Adam algorithm with weight decay fixed as in [AdamW]_ .

    Parameters
    ----------
    params: iterable
        Iterable of parameters to optimize or dicts defining parameter groups
    lr: float, optional
        Learning rate.
    betas: Tuple[float, float], optional
        Coefficients used for computing running averages of gradient and its square
    eps: float, optional
        Term added to the denominator to improve numerical stability
    weight_decay: float, optional
        The "fixed" weight decay.
    
    References
    ----------
        
    .. [AdamW] Loshchilov, I. & Hutter, F. (2017).
       Fixing Weight Decay Regularization in Adam.
       arXiv preprint arXiv:1711.05101.
       Online: https://arxiv.org/abs/1711.05101
      
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(AdamW, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = grad.new().resize_as_(grad).zero_()
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = grad.new().resize_as_(grad).zero_()

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                denom = exp_avg_sq.sqrt().add_(group["eps"])

                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]
                step_size = group["lr"] * math.sqrt(bias_correction2) / bias_correction1

                p.data.addcdiv_(-step_size, exp_avg, denom)
                if group["weight_decay"] != 0:
                    p.data.add_(-group["weight_decay"], p.data)

        return loss
