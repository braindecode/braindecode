"""CodeBrain under torch.compile: one graph per forward and no per-block recompiles, on every device."""
import torch
import torch._dynamo

from braindecode.models import CodeBrain


def _model():
    torch.manual_seed(0)
    return CodeBrain(n_chans=4, n_times=400, sfreq=200, n_outputs=2)


def test_codebrain_compiles_to_one_graph():
    model = _model().eval()
    x = torch.randn(2, 4, 400)
    torch._dynamo.reset()
    explanation = torch._dynamo.explain(model)(x)
    assert explanation.graph_break_count == 0, explanation.break_reasons[:3]
    assert explanation.graph_count == 1


def test_codebrain_does_not_recompile_per_block():
    model = _model().train()
    x = torch.randn(2, 4, 400)
    torch._dynamo.reset()
    torch._dynamo.utils.counters.clear()
    compiled = torch.compile(model, backend="eager")
    for _ in range(2):
        compiled(x).sum().backward()
    assert torch._dynamo.utils.counters["stats"]["unique_graphs"] <= 2


def test_uninitialized_checkpoint_initializes_on_first_forward():
    """A model saved before any forward loads with the kernel norm still to compute, as before."""
    x = torch.randn(2, 4, 400)
    source, loaded = _model().eval(), CodeBrain(n_chans=4, n_times=400, sfreq=200, n_outputs=2).eval()
    loaded.load_state_dict(source.state_dict())
    torch.testing.assert_close(loaded(x), source(x))
    for a, b in zip(source.modules(), loaded.modules()):
        if hasattr(a, "kernel_norm"):
            torch.testing.assert_close(b.kernel_norm, a.kernel_norm)
            assert bool(b.kernel_norm_initialized) and b._kernel_norm_ready


def test_initialized_checkpoint_keeps_its_kernel_norm():
    """After loading an initialized checkpoint, the stored norm is used, not recomputed."""
    x = torch.randn(2, 4, 400)
    source = _model().eval()
    source(x)
    loaded = CodeBrain(n_chans=4, n_times=400, sfreq=200, n_outputs=2).eval()
    loaded.load_state_dict(source.state_dict())
    stored = [m.kernel_norm.clone() for m in loaded.modules() if hasattr(m, "kernel_norm")]
    with torch.no_grad():
        for m in loaded.modules():
            if hasattr(m, "kernel_list"):
                for p in m.kernel_list:
                    p.mul_(2.0)
    loaded(x)
    after = [m.kernel_norm for m in loaded.modules() if hasattr(m, "kernel_norm")]
    for a, b in zip(stored, after):
        torch.testing.assert_close(b, a)


def test_first_forward_under_inference_mode_then_train():
    model = _model()
    x = torch.randn(2, 4, 400)
    with torch.inference_mode():
        model.eval()(x)
    model.train()(x).sum().backward()
