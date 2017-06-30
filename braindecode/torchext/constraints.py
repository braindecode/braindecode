import torch as th


class MaxNormDefaultConstraint(object):
    def apply(self, model):
        last_weight = None
        for name, module in list(model.named_children()):
            if hasattr(module, 'weight') and (
                    not module.__class__.__name__.startswith('BatchNorm')):
                module.weight.data = th.renorm(module.weight.data,2,0,maxnorm=2)
                last_weight = module.weight
        if last_weight is not None:
            last_weight.data = th.renorm(last_weight.data,2,0,maxnorm=0.5)