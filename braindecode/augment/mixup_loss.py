import torch


class general_mixup_criterion:

    def __init__(self, loss=torch.nn.functional.nll_loss):
        self.loss = loss

    def __call__(self, preds, target):
        return self.loss_function(preds, target)

    def loss_function(self, preds, target):
        ret = None
        for label in target.keys():
            prop = target[label]
            loss_val = self.loss(preds, label, reduction='none')
            if ret is None:
                ret = torch.mul(prop, loss_val)
            else:
                ret += torch.mul(prop, loss_val)
        return ret.mean()
