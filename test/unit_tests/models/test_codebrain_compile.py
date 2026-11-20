"""CodeBrain's lazily computed kernel norm: it compiles without graph breaks and survives a checkpoint round trip."""
import torch

from braindecode.models import CodeBrain


def _model():
    return CodeBrain(n_chans=4, n_times=400, sfreq=200, n_outputs=2).eval()


def test_codebrain_compiles_without_graph_breaks():
    torch.manual_seed(0)
    model, x = _model(), torch.randn(2, 4, 400)
    compiled = torch.compile(model, backend="eager", fullgraph=True)  # raises on a break
    torch.testing.assert_close(compiled(x), model(x))


def test_codebrain_load_round_trip():
    """A checkpoint saved before or after the first forward reloads to the same output."""
    torch.manual_seed(0)
    x = torch.randn(2, 4, 400)
    for forward_first in (False, True):
        source = _model()
        if forward_first:
            source(x)  # the first forward computes the kernel norm and keeps it
            with torch.no_grad():  # training moves the kernels away from that norm
                for p in source.parameters():
                    p.mul_(1.1)
        loaded = _model()
        loaded.load_state_dict(source.state_dict())
        torch.testing.assert_close(loaded(x), source(x))


def test_first_forward_under_inference_mode_then_train():
    torch.manual_seed(0)
    model = _model()
    x = torch.randn(2, 4, 400)
    with torch.inference_mode():
        model(x)
    model.train()(x).sum().backward()
