import torch
from torch import Tensor, nn

import braindecode.functional as F


class Square(nn.Module):
    r"""Element-wise square activation.

    :math:`\text{Square}(x) = x^2`

    Examples
    --------
    >>> import torch
    >>> from braindecode.modules import Square
    >>> module = Square()
    >>> inputs = torch.rand(2, 3)
    >>> outputs = module(inputs)
    >>> outputs.shape
    torch.Size([2, 3])
    """

    def forward(self, x) -> Tensor:
        return x * x


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


class GatedLinearUnit(nn.Module):
    r"""Generalized gated linear unit (GLU family).

    Splits the last dimension in half into a ``value`` and a ``gate`` and
    returns :math:`\text{value} \otimes \text{activation}(\text{gate})`. With the
    default ``activation=nn.GELU`` this is **GEGLU** (Shazeer, 2020); ``nn.SiLU``
    gives SwiGLU and ``nn.Sigmoid`` the original GLU. Unlike
    :class:`torch.nn.GLU`, the gate nonlinearity is configurable (``torch.nn.GLU``
    is hard-wired to the sigmoid).

    Parameters
    ----------
    activation : type[nn.Module], default=nn.GELU
        Constructor of the gate activation. The default yields GEGLU.

    Examples
    --------
    >>> import torch
    >>> from braindecode.modules import GatedLinearUnit
    >>> module = GatedLinearUnit()
    >>> outputs = module(torch.randn(2, 10, 16))
    >>> outputs.shape
    torch.Size([2, 10, 8])
    """

    def __init__(self, activation: type[nn.Module] = nn.GELU):
        super().__init__()
        self.activation = activation()

    def forward(self, x: Tensor) -> Tensor:
        value, gate = x.chunk(2, dim=-1)
        return value * self.activation(gate)
