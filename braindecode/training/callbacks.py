# Authors: Robin Schirrmeister <robintibor@gmail.com>
#
# License: BSD (3-clause)

import torch
from skorch.callbacks import Callback

from braindecode.util import _warn_orphaned_api


class MaxNormConstraintCallback(Callback):
    """Constrain the weight norms of the top-level layers after each batch.

    .. deprecated:: 2.1
        It has no user in braindecode and will be removed in 2.2.
    """

    def __init__(self):
        super().__init__()
        _warn_orphaned_api("MaxNormConstraintCallback")

    def on_batch_end(self, net, training, *args, **kwargs):
        if training:
            model = net.module_
            last_weight = None
            for name, module in list(model.named_children()):
                if hasattr(module, "weight") and (
                    not module.__class__.__name__.startswith("BatchNorm")
                ):
                    module.weight.data = torch.renorm(
                        module.weight.data, 2, 0, maxnorm=2
                    )
                    last_weight = module.weight
            if last_weight is not None:
                last_weight.data = torch.renorm(last_weight.data, 2, 0, maxnorm=0.5)
