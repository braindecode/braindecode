"""CodeBrain's lazily computed kernel norm (compiles without graph breaks, survives a checkpoint round trip) and its reset_head."""
import torch

from braindecode.models import CodeBrain


def _model():
    return CodeBrain(n_chans=4, n_times=400, sfreq=200, n_outputs=2).eval()


def test_codebrain_compiles_without_graph_breaks():
    torch.manual_seed(0)
    model, x = _model(), torch.randn(2, 4, 400)
    compiled = torch.compile(model, backend="eager", fullgraph=True)  # raises on a break
    torch.testing.assert_close(compiled(x), model(x))


def test_codebrain_trains_compiled_without_graph_breaks():
    torch.manual_seed(0)
    model, x = _model().train(), torch.randn(2, 4, 400)
    compiled = torch.compile(model, backend="eager", fullgraph=True)
    for _ in range(2):  # the first step computes the kernel norm, the second reuses it
        compiled(x).sum().backward()


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


def test_codebrain_reset_head_replaces_only_the_output_layer():
    torch.manual_seed(0)
    model = _model()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    model.reset_head(5)
    after = model.state_dict()
    # The keys and shapes are those of a model built with n_outputs=5
    expected = CodeBrain(n_chans=4, n_times=400, sfreq=200, n_outputs=5).state_dict()
    assert {k: v.shape for k, v in after.items()} == {
        k: v.shape for k, v in expected.items()
    }
    # Every weight but the output layer is kept
    changed = {k for k in before if before[k].shape != after[k].shape}
    assert changed == {"final_layer.7.weight", "final_layer.7.bias"}
    for k in before.keys() - changed:
        assert torch.equal(before[k], after[k]), k
    assert model(torch.randn(2, 4, 400)).shape == (2, 5)
