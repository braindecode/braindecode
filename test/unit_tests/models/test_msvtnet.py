# Authors: Bruno Aristimunha <b.aristimunha@gmail.com>
#
# License: BSD (3-clause)
import copy

import pytest
import torch
from torch import nn

from braindecode.models import MSVTNet
from braindecode.models.msvtnet import _TSConv


def test_msvtnet_branch_is_its_layers_in_order():
    torch.manual_seed(0)
    model = MSVTNet(n_chans=5, n_outputs=2, n_times=300).double()
    branch = model.mstsconv[0][0].eval()
    x = torch.randn(3, 1, 5, 300, dtype=torch.float64)
    params = list(branch.parameters())
    out = branch(x)
    out.square().sum().backward()
    grads = [p.grad.clone() for p in params]
    branch.zero_grad()
    ref = nn.Sequential.forward(branch, x)
    ref.square().sum().backward()
    torch.testing.assert_close(out, ref)
    for g, p in zip(grads, params):
        torch.testing.assert_close(g, p.grad)


@pytest.mark.parametrize(
    "n_times, kernel, momentum", [(300, 15, 0.1), (300, 16, 0.1), (20, 31, None)]
)
def test_msvtnet_spatial_first_is_layers_0_to_2(n_times, kernel, momentum):
    torch.manual_seed(0)
    a = _TSConv(5, 3, kernel, 15, 2, 8, 7, 0.3).double()
    a[1].momentum = momentum
    nn.init.normal_(a[1].weight)
    nn.init.normal_(a[1].bias)
    b = copy.deepcopy(a)
    x = 3.0 + torch.randn(4, 1, 5, n_times, dtype=torch.float64)
    for training in (True, True, False):
        a.train(training)
        b.train(training)
        out = a._spatial_first(x)
        ref = b[2](b[1](b[0](x)))
        torch.testing.assert_close(out, ref)
        out.square().sum().backward()
        ref.square().sum().backward()
        for i in range(3):
            for pa, pb in zip(a[i].parameters(), b[i].parameters()):
                torch.testing.assert_close(pa.grad, pb.grad)
        for ba, bb in zip(a[1].buffers(), b[1].buffers()):
            torch.testing.assert_close(ba, bb)
        a.zero_grad()
        b.zero_grad()
