from skorch.callbacks import ProgressBar, Callback
import torch


class MaxNormConstraintCallback(Callback):
    def on_batch_end(self, net, training, *args, **kwargs):
        """
        Renames end of batch.

        Args:
            self: (todo): write your description
            net: (todo): write your description
            training: (todo): write your description
        """
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
                last_weight.data = torch.renorm(
                    last_weight.data, 2, 0, maxnorm=0.5
                )
