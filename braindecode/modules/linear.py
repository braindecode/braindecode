import torch
from torch import nn
from torch import Tensor


class MaxNormLinear(nn.Linear):
    """Linear layer with MaxNorm constraining on weights.

    Equivalent of Keras tf.keras.Dense(..., kernel_constraint=max_norm())
    [1, 2]_. Implemented as advised in [3]_.

    Parameters
    ----------
    in_features: int
        Size of each input sample.
    out_features: int
        Size of each output sample.
    bias: bool, optional
        If set to ``False``, the layer will not learn an additive bias.
        Default: ``True``.

    References
    ----------
    .. [1] https://keras.io/api/layers/core_layers/dense/#dense-class
    .. [2] https://www.tensorflow.org/api_docs/python/tf/keras/constraints/
           MaxNorm
    .. [3] https://discuss.pytorch.org/t/how-to-correctly-implement-in-place-
           max-norm-constraint/96769
    """

    def __init__(
        self, in_features, out_features, bias=True, max_norm_val=2, eps=1e-5, **kwargs
    ):
        super().__init__(
            in_features=in_features, out_features=out_features, bias=bias, **kwargs
        )
        self._max_norm_val = max_norm_val
        self._eps = eps

    def forward(self, X: Tensor) -> Tensor:
        self._max_norm()
        return super().forward(X)

    def _max_norm(self):
        with torch.no_grad():
            norm = self.weight.norm(2, dim=0, keepdim=True).clamp(
                min=self._max_norm_val / 2
            )
            desired = torch.clamp(norm, max=self._max_norm_val)
            self.weight *= desired / (self._eps + norm)


class LinearWithConstraint(nn.Linear):
    """Linear layer with max-norm constraint on the weights."""

    def __init__(self, *args, max_norm=1.0, **kwargs):
        super(LinearWithConstraint, self).__init__(*args, **kwargs)
        self.max_norm = max_norm

    def forward(self, x: Tensor) -> Tensor:
        with torch.no_grad():
            # In PyTorch, the weight matrix of nn.Linear is of shape (out_features, in_features),
            # which is the transpose of TensorFlow's typical kernel shape.
            #
            # The torch.renorm function applies a re-normalization to slices of the tensor:
            # - 'p=2' specifies that we are using the Euclidean (L2) norm.
            # - 'dim=0' indicates that the tensor will be split along the first dimension.
            #   This corresponds to each "row" in the weight matrix, which in this context
            #   represents a weight vector for each output neuron.
            # - 'maxnorm=self.max_norm' sets the maximum allowed norm for each of these sub-tensors.
            #
            # Note: In TensorFlow's max_norm constraint, the axis parameter determines along which
            # dimension the norm is computed. Here, due to the difference in kernel shape and axis
            # interpretation, we use torch.renorm with dim=0 to match TensorFlow's behavior.
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        return super(LinearWithConstraint, self).forward(x)
