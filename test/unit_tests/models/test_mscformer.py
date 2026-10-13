# Authors: Bruno Aristimunha <b.aristimunha@gmail.com>
#
# License: BSD (3-clause)
import torch

from braindecode.models import MSCFormer


def test_mscformer_patch_embedding_is_the_branches_in_order():
    torch.manual_seed(0)
    model = MSCFormer(n_chans=5, n_outputs=2, n_times=300).double()
    cnn = model.cnn.eval()
    x = torch.randn(3, 1, 5, 300, dtype=torch.float64)
    params = list(cnn.parameters())
    out = cnn(x)
    out.square().sum().backward()
    grads = [p.grad.clone() for p in params]
    cnn.zero_grad()
    ref = cnn.projection(torch.cat([branch(x) for branch in cnn.branches], dim=1))
    ref.square().sum().backward()
    torch.testing.assert_close(out, ref)
    for g, p in zip(grads, params):
        torch.testing.assert_close(g, p.grad)
