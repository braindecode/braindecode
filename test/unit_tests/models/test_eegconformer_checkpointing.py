"""Exact eager-training contract for opt-in activation checkpointing."""

import copy
import io

import pytest
import torch
from torch import nn

from braindecode.models import EEGConformer


@pytest.fixture(autouse=True)
def isolate_acceleration_state(acceleration_caller_state):
    """Restore caller RNG, thread count and grad mode after each case."""
    yield


def _model(**kwargs):
    return EEGConformer(
        n_chans=4,
        chs_info=[{"ch_name": f"EEG{i}"} for i in range(4)],
        sfreq=128,
        n_times=256,
        n_outputs=3,
        n_filters_time=8,
        filter_time_length=5,
        pool_time_length=15,
        pool_time_stride=8,
        num_layers=2,
        num_heads=2,
        drop_prob=0.4,
        att_drop_prob=0.3,
        **kwargs,
    )


def _equal(a, b):
    assert torch.equal(a, b)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("return_features", [False, True])
def test_checkpoint_exact_step(dtype, return_features):
    torch.manual_seed(19)
    off = _model(return_features=return_features).to(dtype)
    on = _model(activation_checkpointing=True, return_features=return_features).to(
        dtype
    )
    on.load_state_dict(off.state_dict(), strict=True)
    assert list(off.state_dict()) == list(on.state_dict())
    x = torch.randn(3, 4, 256, dtype=dtype)
    results = []
    for model in (off, on):
        optimizer = torch.optim.SGD(model.parameters(), lr=0.03, momentum=0.9)
        value = x.clone().requires_grad_()
        torch.manual_seed(29)
        output = model(value)
        forward_rng = torch.get_rng_state().clone()
        loss = output.square().mean()
        loss.backward()
        backward_rng = torch.get_rng_state().clone()
        grads = {
            n: None if p.grad is None else p.grad.clone()
            for n, p in model.named_parameters()
        }
        optimizer.step()
        results.append(
            (
                output.detach(),
                loss.detach(),
                value.grad,
                forward_rng,
                backward_rng,
                grads,
                copy.deepcopy(model.state_dict()),
            )
        )
    for a, b in zip(results[0][:5], results[1][:5], strict=True):
        _equal(a, b)
    for a, b in zip(results[0][5:], results[1][5:], strict=True):
        assert list(a) == list(b)
        for name in a:
            if a[name] is None:
                assert b[name] is None
            else:
                _equal(a[name], b[name])
    off.load_state_dict(on.state_dict(), strict=True)


def test_default_and_explicit_off_are_identical():
    torch.manual_seed(37)
    default = _model()
    rng = torch.get_rng_state().clone()
    torch.manual_seed(37)
    explicit = _model(activation_checkpointing=False)
    _equal(rng, torch.get_rng_state())
    for name, value in default.state_dict().items():
        _equal(value, explicit.state_dict()[name])
    x = torch.randn(2, 4, 256)
    torch.manual_seed(41)
    a = default(x)
    state = torch.get_rng_state().clone()
    torch.manual_seed(41)
    _equal(a, explicit(x))
    _equal(state, torch.get_rng_state())


@pytest.mark.parametrize("kind", ["forward", "pre", "backward", "buffer", "activation"])
def test_unsafe_boundary_rejected(kind):
    model = _model(activation_checkpointing=True)
    block = model.transformer[0]
    if kind == "forward":
        block.register_forward_hook(lambda *args: None)
    elif kind == "pre":
        block.register_forward_pre_hook(lambda *args: None)
    elif kind == "backward":
        block.register_full_backward_hook(lambda *args: None)
    elif kind == "buffer":
        block.register_buffer("counter", torch.tensor(0))
    else:
        block[1].fn[1][1] = nn.BatchNorm1d(8)
    with pytest.raises(RuntimeError, match="activation_checkpointing.*pure"):
        model(torch.randn(2, 4, 256))


def test_outer_hooks_and_buffers_unchanged():
    model = _model(activation_checkpointing=True)
    calls = []
    model.register_forward_hook(lambda *args: calls.append("model"))
    model.patch_embedding.register_forward_hook(lambda *args: calls.append("stem"))
    model(torch.randn(2, 4, 256)).sum().backward()
    assert calls == ["stem", "model"]
    assert model.patch_embedding.shallownet[2].num_batches_tracked == 2


@pytest.mark.parametrize("no_grad", [False, True])
def test_eval_and_no_grad_bypass_checkpoint(no_grad, monkeypatch):
    model = _model(activation_checkpointing=True)
    if not no_grad:
        model.eval()

    def fail(*args, **kwargs):
        raise AssertionError("checkpoint called outside gradient-enabled training")

    monkeypatch.setattr("braindecode.models.eegconformer.checkpoint", fail)
    with torch.set_grad_enabled(not no_grad):
        model(torch.randn(2, 4, 256))


def test_checkpoint_arguments(monkeypatch):
    calls = []

    def spy(function, *args, **kwargs):
        calls.append(kwargs)
        return function(*args)

    monkeypatch.setattr("braindecode.models.eegconformer.checkpoint", spy)
    _model(activation_checkpointing=True)(torch.randn(2, 4, 256))
    assert calls == [{"use_reentrant": False, "preserve_rng_state": True}] * 2


@pytest.mark.parametrize("mode", ["script", "trace", "export"])
def test_opt_in_export_rejected(mode):
    model = _model(activation_checkpointing=True).eval()
    x = torch.randn(2, 4, 256)
    accepted = (RuntimeError, torch.jit.Error, torch._dynamo.exc.Unsupported)
    if mode == "export":
        # Strict export (the default up to torch 2.7) traces with Dynamo, which
        # raises AssertionError naming the forbidden eager-only boundary.
        accepted += (AssertionError,)
    with pytest.raises(accepted, match="activation_checkpointing"):
        if mode == "script":
            torch.jit.script(model)(x)
        elif mode == "trace":
            torch.jit.trace(model, x)
        else:
            torch.export.export(model, (x,))


def test_default_export_executes_after_roundtrip():
    model = _model().eval()
    x = torch.randn(2, 4, 256)
    # Non-strict, the default from torch 2.8: torch 2.6 cannot load a strict
    # export of this model (its serializer splits nn_module_stack on commas).
    exported = torch.export.export(model, (x,), strict=False)
    buffer = io.BytesIO()
    torch.export.save(exported, buffer)
    buffer.seek(0)
    loaded = torch.export.load(buffer).module()
    _equal(model(x), loaded(x))


def test_global_hooks_rejected():
    from torch.nn.modules.module import register_module_forward_hook

    model = _model(activation_checkpointing=True)
    handle = register_module_forward_hook(lambda *args: None)
    try:
        with pytest.raises(
            RuntimeError, match="activation_checkpointing.*global hooks"
        ):
            model(torch.randn(2, 4, 256))
    finally:
        handle.remove()


def test_explicit_off_script_executes():
    model = _model(activation_checkpointing=False).eval()
    x = torch.randn(2, 4, 256)
    _equal(model(x), torch.jit.script(model)(x))


def test_legacy_full_object_pickle_runs_and_resaves():
    import pickle

    model = _model().eval()
    expected_config = model.get_config()
    del model.activation_checkpointing  # Exact pre-feature full-object state.
    restored = pickle.loads(pickle.dumps(model))
    x = torch.randn(2, 4, 256)
    assert restored.activation_checkpointing is False
    assert restored.get_config() == expected_config
    reloaded = pickle.loads(pickle.dumps(restored))
    _equal(restored(x), reloaded(x))
    current = _model().eval()
    current.load_state_dict(restored.state_dict(), strict=True)
    _equal(current(x), restored(x))


@pytest.mark.parametrize("context", ["eval", "no_grad", "inference"])
def test_global_hooks_allowed_when_checkpoint_bypassed(context):
    from torch.nn.modules.module import register_module_forward_hook

    torch.manual_seed(71)
    off = _model()
    on = _model(activation_checkpointing=True)
    on.load_state_dict(off.state_dict(), strict=True)
    if context == "eval":
        off.eval()
        on.eval()
    manager = {
        "eval": torch.enable_grad,
        "no_grad": torch.no_grad,
        "inference": torch.inference_mode,
    }[context]
    x = torch.randn(2, 4, 256)
    results = []
    for model in (off, on):
        calls = []
        handle = register_module_forward_hook(
            lambda module, *args, calls=calls: calls.append(type(module))
        )
        try:
            with manager():
                torch.manual_seed(73)
                output = model(x)
                results.append(
                    (
                        output.detach(),
                        torch.get_rng_state().clone(),
                        copy.deepcopy(model.state_dict()),
                        calls,
                    )
                )
        finally:
            handle.remove()
    _equal(results[0][0], results[1][0])
    _equal(results[0][1], results[1][1])
    assert results[0][3] == results[1][3]
    assert results[0][3]
    for name in results[0][2]:
        _equal(results[0][2][name], results[1][2][name])


@pytest.mark.parametrize("on", [False, True])
def test_checkpoint_local_config_hub_and_pickle(on, tmp_path):
    import json
    import pickle

    model = _model(activation_checkpointing=on).eval()
    config = model.get_config()
    restored = EEGConformer.from_config(json.loads(json.dumps(config))).eval()
    assert restored.activation_checkpointing is on
    restored.load_state_dict(model.state_dict(), strict=True)
    model.save_pretrained(tmp_path)
    loaded = EEGConformer.from_pretrained(tmp_path, local_files_only=True).eval()
    assert loaded.activation_checkpointing is on
    x = torch.randn(2, 4, 256)
    for candidate in (restored, loaded, pickle.loads(pickle.dumps(model))):
        _equal(model(x), candidate(x))


def test_checkpoint_double_backward_exact():
    off = _model().double()
    on = _model(activation_checkpointing=True).double()
    on.load_state_dict(off.state_dict(), strict=True)
    x = torch.randn(2, 4, 256, dtype=torch.float64)
    results = []
    for model in (off, on):
        value = x.clone().requires_grad_()
        torch.manual_seed(79)
        loss = model(value).square().sum()
        first = torch.autograd.grad(loss, value, create_graph=True)[0]
        second = torch.autograd.grad(first.square().sum(), value)[0]
        results.append((first, second, torch.get_rng_state().clone()))
    for a, b in zip(*results, strict=True):
        _equal(a, b)


@pytest.mark.parametrize("backend", ["eager", "inductor"])
@pytest.mark.parametrize("fullgraph", [False, True])
@pytest.mark.parametrize("training", [False, True])
@pytest.mark.parametrize("entry", ["estimator", "native", "wrapper"])
def test_checkpoint_compile_rejected(backend, fullgraph, training, entry):
    from braindecode import EEGClassifier

    torch.compiler.reset()
    model = _model(activation_checkpointing=True).train(training)
    x = torch.randn(2, 4, 256)
    rng = torch.get_rng_state().clone()
    try:
        # Native compilation may reject during setup; wrappers reject on call.
        with pytest.raises(
            (RuntimeError, AssertionError), match="activation_checkpointing.*eager"
        ):
            if entry == "estimator":
                net = EEGClassifier(
                    model,
                    compile=True,
                    compile__backend=backend,
                    compile__fullgraph=fullgraph,
                    train_split=None,
                    verbose=0,
                ).initialize()
                compiled = net.module_
            elif entry == "native":
                model.compile(backend=backend, fullgraph=fullgraph)
                compiled = model
            else:
                compiled = torch.compile(model, backend=backend, fullgraph=fullgraph)
            compiled(x)
        # Rejection must occur before the stem changes BatchNorm/RNG state.
        assert model.patch_embedding.shallownet[2].num_batches_tracked == 1
        _equal(rng, torch.get_rng_state())
        assert model._compiled_call_impl is None
        model(x).sum().backward()
    finally:
        torch.compiler.reset()


@pytest.mark.parametrize("entry", ["estimator", "native", "wrapper"])
def test_checkpoint_compile_disable_runs_eager(entry):
    from braindecode import EEGClassifier

    model = _model(activation_checkpointing=True)
    if entry == "estimator":
        net = EEGClassifier(
            model, compile=True, compile__disable=True, train_split=None, verbose=0
        ).initialize()
        compiled = net.module_
        assert model._compiled_call_impl is None
    elif entry == "native":
        model.compile(disable=True)
        compiled = model
    else:
        compiled = torch.compile(model, disable=True)
    compiled(torch.randn(2, 4, 256)).sum().backward()


@pytest.mark.parametrize("foreign", [False, True])
def test_checkpoint_flag_change_preserves_compile_ownership(foreign):
    from braindecode import EEGClassifier

    torch.compiler.reset()
    model = _model()
    if foreign:
        model.compile(backend="eager")
    net = EEGClassifier(
        model, compile=True, compile__backend="eager", train_split=None, verbose=0
    ).initialize()
    installed = model._compiled_call_impl
    model.activation_checkpointing = True
    try:
        with pytest.raises(
            (RuntimeError, AssertionError), match="activation_checkpointing.*eager"
        ):
            model(torch.randn(2, 4, 256))
        net.set_params(compile=False)
        assert model._compiled_call_impl is (installed if foreign else None)
        if foreign:
            # The caller, not the estimator, owns removal of its compilation.
            model._compiled_call_impl = None
        model(torch.randn(2, 4, 256)).sum().backward()
    finally:
        torch.compiler.reset()


def _run_fresh_process(script, *args):
    """Run ``script`` in a new interpreter that imports this braindecode tree."""
    import os
    import subprocess
    import sys
    from pathlib import Path

    import braindecode

    root = str(Path(braindecode.__file__).resolve().parents[1])
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [root] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
    )
    proc = subprocess.run(
        [sys.executable, "-c", script, *args],
        capture_output=True,
        text=True,
        env=env,
        timeout=600,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return proc.stdout


_MODEL_KWARGS = """dict(
    n_chans=4, chs_info=[{"ch_name": f"EEG{i}"} for i in range(4)], sfreq=128,
    n_times=256, n_outputs=3, n_filters_time=8, filter_time_length=5,
    pool_time_length=15, pool_time_stride=8, num_layers=2, num_heads=2,
    drop_prob=0.4, att_drop_prob=0.3,
)"""


@pytest.mark.skipif(
    torch.__version__ < "2.8", reason="older torch dependencies may import Dynamo"
)
def test_import_and_default_model_do_not_import_dynamo():
    # Importing torch._dynamo costs about a second; only opt-in pays for it.
    script = f"""
import sys
import torch
torch.set_num_threads(1)
import braindecode
assert "torch._dynamo" not in sys.modules, "import braindecode"
import braindecode.models
from braindecode.models import EEGConformer
assert "torch._dynamo" not in sys.modules, "import braindecode.models"
model = EEGConformer(**{_MODEL_KWARGS})
model.activation_checkpointing = False
model(torch.randn(2, 4, 256)).sum().backward()
assert "torch._dynamo" not in sys.modules, "default EEGConformer"
print("NO_DYNAMO")
"""
    assert "NO_DYNAMO" in _run_fresh_process(script)


_LAZY_REJECTION_SCRIPT = f"""
import pickle
import re
import sys
import torch
torch.set_num_threads(1)
from braindecode.models import EEGConformer

route, path = sys.argv[1], sys.argv[2]
if route == "init":
    model = EEGConformer(**{_MODEL_KWARGS}, activation_checkpointing=True)
elif route == "setattr":
    model = EEGConformer(**{_MODEL_KWARGS})
else:
    with open(path, "rb") as fh:
        model = pickle.load(fh)
model.train()
x = torch.randn(2, 4, 256)
compiled = torch.compile(model, backend="eager", fullgraph=False)
if route == "setattr":
    compiled(x)  # Flag off: compiles and runs; the flag change forces a retrace.
    model.activation_checkpointing = True
batches = int(model.patch_embedding.shallownet[2].num_batches_tracked)
rng = torch.get_rng_state().clone()
try:
    compiled(x)
except (RuntimeError, AssertionError) as exc:
    assert re.search("activation_checkpointing.*eager", str(exc)), exc
else:
    raise SystemExit("EXECUTED: checkpointing ran inside torch.compile")
assert int(model.patch_embedding.shallownet[2].num_batches_tracked) == batches
assert torch.equal(torch.get_rng_state(), rng)
print("REJECTED")
"""


@pytest.mark.parametrize("route", ["init", "setattr", "unpickle"])
def test_lazy_dynamo_boundary_rejects_partial_graph_compile(route, tmp_path):
    # Each route must forbid the boundary itself, in a process where no other
    # model has enabled checkpointing; otherwise fullgraph=False falls back to
    # eager checkpointing silently.
    import pickle

    path = tmp_path / "model.pkl"
    with open(path, "wb") as fh:
        pickle.dump(_model(activation_checkpointing=True), fh)
    assert "REJECTED" in _run_fresh_process(_LAZY_REJECTION_SCRIPT, route, str(path))
