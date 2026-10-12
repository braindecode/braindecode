# Authors: Bruno Aristimunha <b.aristimunha@gmail.com>
#
# License: BSD (3-clause)
import torch
from torch import nn

from braindecode.models import MSVTNet


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
