import torch
from torch import Tensor, nn

import braindecode.functional as F


class SafeLog(nn.Module):
    r"""
    Safe logarithm activation function module.

    :math:`\text{SafeLog}(x) = \log\left(\max(x, \epsilon)\right)`

    Parameters
    ----------
    epsilon : float, optional
        A small value to clamp the input tensor to prevent computing log(0) or log of negative numbers.
        Default is 1e-6.

    Examples
    --------
    >>> import torch
    >>> from braindecode.modules import SafeLog
    >>> module = SafeLog(epsilon=1e-6)
    >>> inputs = torch.rand(2, 3)
    >>> outputs = module(inputs)
    >>> outputs.shape
    torch.Size([2, 3])

    """

    def __init__(self, epsilon: float = 1e-6):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, x) -> Tensor:
        """
        Forward pass of the SafeLog module.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor after applying safe logarithm.
        """
        return F.safe_log(x=x, eps=self.epsilon)

    def extra_repr(self) -> str:
        eps_str = f"eps={self.epsilon}"
        return eps_str


class LogActivation(nn.Module):
    """Logarithm activation function.

    Parameters
    ----------
    epsilon : float, default=1e-6
        Small float to adjust the activation.

    Examples
    --------
    >>> import torch
    >>> from braindecode.modules import LogActivation
    >>> module = LogActivation(epsilon=1e-6)
    >>> inputs = torch.rand(2, 3)
    >>> outputs = module(inputs)
    >>> outputs.shape
    torch.Size([2, 3])
    """

    def __init__(self, epsilon: float = 1e-6, *args, **kwargs):
        """
        Parameters
        ----------
        epsilon : float
            Small float to adjust the activation.
        """
        super().__init__(*args, **kwargs)
        self.epsilon = epsilon

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.log(x + self.epsilon)  # Adding epsilon to prevent log(0)
