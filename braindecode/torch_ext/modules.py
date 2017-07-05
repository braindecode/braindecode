import torch


class Expression(torch.nn.Module):
    """
    Compute given expression on forward pass.
    
    Parameters
    ----------
    expression_fn: function
        Should accept variable number of objects of type
        `torch.autograd.Variable` to compute its output.
    """
    def __init__(self, expression_fn):
        super(Expression, self).__init__()
        self.expression_fn = expression_fn

    def forward(self, *x):
        return self.expression_fn(*x)

    def __repr__(self):
        if (hasattr(self.expression_fn, 'func') and
                  hasattr(self.expression_fn, 'kwargs')):
                expression_str = "{:s} {:s}".format(
                    self.expression_fn.func.__name__,
                    str(self.expression_fn.kwargs))
        else:
            expression_str = self.expression_fn.__name__
        return (self.__class__.__name__ + '(' +
                'expression=' + str(expression_str) + ')')