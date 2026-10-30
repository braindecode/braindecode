"""Existing estimator compile interface: canonical state and lifecycle contracts."""

from io import BytesIO

import numpy as np
import pytest
import torch
from sklearn.base import clone
from skorch.callbacks import Callback

from braindecode import EEGClassifier


@pytest.fixture(autouse=True)
def isolate_acceleration_state(acceleration_caller_state):
    """Restore caller RNG, thread count and grad mode after each case."""
    yield


@pytest.fixture(autouse=True)
def isolated_compiler_cache():
    # These tests intentionally change backend callables on the same forward
    # code object; avoid a prior case exhausting Dynamo's recompilation budget.
    torch.compiler.reset()
    yield
    torch.compiler.reset()


class TinyClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(24, 2)

    def forward(self, x):
        return self.fc(x.flatten(1))


class Events(Callback):
    def initialize(self):
        self.events_ = []
        return self

    def on_batch_end(self, net, batch, training, **kwargs):
        self.events_.append((training, len(batch[0])))


def make_net(
    compiled=False,
    backend="eager",
    module=TinyClassifier,
    criterion=torch.nn.CrossEntropyLoss,
):
    return EEGClassifier(
        module,
        criterion=criterion,
        compile=compiled,
        compile__backend=backend,
        max_epochs=2,
        batch_size=4,
        train_split=None,
        verbose=0,
        iterator_train__shuffle=False,
        iterator_train__drop_last=False,
        optimizer=torch.optim.SGD,
        lr=0.01,
        callbacks=[("events", Events())],
    )


def data(dtype=torch.float32):
    gen = torch.Generator().manual_seed(81)
    return torch.randn(11, 3, 8, generator=gen, dtype=dtype), torch.arange(11) % 2


def test_compile_checkpoint_keys_strict_roundtrip():
    eager = make_net().initialize()
    compiled = make_net(True).initialize()
    expected = set(eager.module_.state_dict())
    assert set(compiled.module_.state_dict()) == expected
    compiled.module_.load_state_dict(eager.module_.state_dict(), strict=True)
    files = {key: BytesIO() for key in ("f_params", "f_optimizer", "f_criterion")}
    compiled.save_params(**files)
    files["f_params"].seek(0)
    assert set(torch.load(files["f_params"], weights_only=True)) == expected
    for target in (eager, make_net(True).initialize()):
        for file in files.values():
            file.seek(0)
        target.load_params(**files)
        for key, value in compiled.module_.state_dict().items():
            torch.testing.assert_close(
                target.module_.state_dict()[key], value, rtol=0, atol=0
            )


@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.parametrize(
    "backend", ["eager", pytest.param("inductor", marks=pytest.mark.slow)]
)
def test_compile_fit_predictions_history_callbacks_clone_refit(dtype, backend):
    torch.set_num_threads(4)
    X, y = data(dtype)
    nets = []
    for compiled in (False, True):
        torch.manual_seed(33)
        net = make_net(compiled, backend).initialize()
        net.module_.to(dtype=dtype)
        net.partial_fit(X, y)
        nets.append(net)
    eager, compiled = nets
    rtol, atol = (2e-4, 2e-5) if dtype == torch.float32 else (1e-9, 1e-11)
    np.testing.assert_allclose(
        compiled.predict_proba(X), eager.predict_proba(X), rtol=rtol, atol=atol
    )
    np.testing.assert_array_equal(compiled.predict(X), eager.predict(X))
    # New batch sizes exercise recompilation, without dropping/reordering rows.
    for size in (1, 3, 5):
        np.testing.assert_allclose(
            compiled.predict_proba(X[:size]),
            eager.predict_proba(X[:size]),
            rtol=rtol,
            atol=atol,
        )
    for key, value in eager.module_.state_dict().items():
        torch.testing.assert_close(
            compiled.module_.state_dict()[key], value, rtol=rtol, atol=atol
        )
    assert (
        dict(eager.callbacks_)["events"].events_
        == dict(compiled.callbacks_)["events"].events_
        == [(True, 4), (True, 4), (True, 3)] * 2
    )
    assert len(compiled.history) == 2
    for a, b in zip(eager.history, compiled.history):
        assert set(a) == set(b)
        assert [x["train_batch_size"] for x in a["batches"]] == [4, 4, 3]
        np.testing.assert_allclose(
            a["train_loss"], b["train_loss"], rtol=rtol, atol=atol
        )
    fresh = clone(compiled)
    assert fresh.compile and fresh.get_params()["compile__backend"] == backend
    assert not fresh.initialized_
    fresh.fit(X.float(), y)
    old_module, old_optimizer = fresh.module_, fresh.optimizer_
    fresh.fit(X.float(), y)
    assert fresh.module_ is not old_module and fresh.optimizer_ is not old_optimizer
    assert len(fresh.history) == 2


def test_supplied_instances_compile_toggle_ownership():
    calls = []

    def backend(graph, example_inputs):
        def execute(*args):
            calls.append(True)
            return graph.forward(*args)

        return execute

    module, criterion = TinyClassifier(), torch.nn.CrossEntropyLoss()
    net = make_net(True, backend, module, criterion)
    X, y = data()
    net.initialize()
    net.module_(X)
    net.criterion_(net.module_(X), y)
    assert calls
    assert net.module_ is module and net.criterion_ is criterion
    assert module._compiled_call_impl is not None
    assert criterion._compiled_call_impl is not None
    assert set(net.module_.state_dict()) == set(module.state_dict())
    for enabled in (False, True, False):
        net.set_params(compile=enabled)
        before = len(calls)
        net.module_(X)
        assert (len(calls) > before) == enabled
        net.initialize()
        before = len(calls)
        net.module_(X)
        assert (len(calls) > before) == enabled
        assert (module._compiled_call_impl is not None) == enabled
        assert (criterion._compiled_call_impl is not None) == enabled
        assert {
            id(p) for group in net.optimizer_.param_groups for p in group["params"]
        } == {id(p) for p in net.module_.parameters()}
    reused = make_net(module=module).initialize()
    before = len(calls)
    reused.module_(X)
    assert len(calls) == before
    # Independently precompiled user modules are not indiscriminately cleared.
    module.compile(backend=backend)
    user_call = module._compiled_call_impl
    eager = make_net(module=module).initialize()
    eager.module_(X)
    assert len(calls) > before
    assert module._compiled_call_impl is user_call
    # Even a user replacing our callable later must not be cleared.
    net.set_params(compile=True)
    module.compile(backend=backend)
    criterion.compile(backend=backend)
    user_call, user_criterion = (
        module._compiled_call_impl,
        criterion._compiled_call_impl,
    )
    net.set_params(compile=False)
    assert module._compiled_call_impl is user_call
    assert criterion._compiled_call_impl is user_criterion


@pytest.mark.slow
def test_trained_checkpoint_resume_warm_start_and_parameter_identity(tmp_path):
    import copy

    X, y = data()
    net = make_net(True, "inductor").set_params(
        optimizer__momentum=0.9, warm_start=True
    )
    net.fit(X, y)
    module, optimizer = net.module_, net.optimizer_
    net.fit(X, y)
    assert net.module_ is module and net.optimizer_ is optimizer
    assert len(net.history) == 4
    assert net.optimizer_.state
    saved_optimizer = copy.deepcopy(net.optimizer_.state_dict())
    saved_history = copy.deepcopy(net.history.to_list())
    files = {
        key: tmp_path / key
        for key in ("f_params", "f_optimizer", "f_criterion", "f_history")
    }
    net.save_params(**files)
    restored = clone(net).initialize()
    restored.load_params(**files)
    assert restored.history.to_list() == saved_history
    assert (
        restored.optimizer_.state_dict()["param_groups"]
        == saved_optimizer["param_groups"]
    )
    for key, state in saved_optimizer["state"].items():
        torch.testing.assert_close(
            restored.optimizer_.state_dict()["state"][key]["momentum_buffer"],
            state["momentum_buffer"],
            rtol=0,
            atol=0,
        )
    for current in (net, restored):
        ids = {
            id(p) for group in current.optimizer_.param_groups for p in group["params"]
        }
        assert ids == {id(p) for p in current.module_.parameters()}
        current.partial_fit(X, y)
    assert net.module_ is module and net.optimizer_ is optimizer
    assert len(restored.history) == len(net.history) == 6
    for key, value in net.module_.state_dict().items():
        torch.testing.assert_close(
            restored.module_.state_dict()[key], value, rtol=2e-4, atol=2e-5
        )
    np.testing.assert_allclose(
        net.predict_proba(X), restored.predict_proba(X), rtol=2e-4, atol=2e-5
    )


@pytest.mark.slow
def test_direct_compiled_gradient_parity():
    X, y = data()
    eager, compiled = make_net().initialize(), make_net(True, "inductor").initialize()
    compiled.module_.load_state_dict(eager.module_.state_dict(), strict=True)
    for net in (eager, compiled):
        net.criterion_(net.module_(X), y).backward()
    for (name, a), (other, b) in zip(
        eager.module_.named_parameters(),
        compiled.module_.named_parameters(),
        strict=True,
    ):
        assert name == other and a.grad is not None and b.grad is not None
        torch.testing.assert_close(a.grad, b.grad, rtol=2e-4, atol=2e-5)


class CroppedCompileModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv1d(1, 2, 1)

    def forward(self, X):
        return self.conv(X)


@pytest.mark.slow
def test_compiled_estimator_long_trial_chunks():
    from braindecode.training.scoring import iter_trial_prediction_chunks
    from test.unit_tests.training.test_scoring import _InferenceWindows

    class Float32Windows(_InferenceWindows):
        def __getitem__(self, index):
            signal, target, indices = super().__getitem__(index)
            return signal.astype(np.float32), target, indices

    eager = make_net(module=CroppedCompileModule).initialize()
    compiled = (
        make_net(True, "inductor", CroppedCompileModule)
        .set_params(compile__dynamic=False)
        .initialize()
    )
    compiled.module_.load_state_dict(eager.module_.state_dict(), strict=True)

    def chunks(net):
        return list(
            iter_trial_prediction_chunks(
                net.module_, Float32Windows(), windows_per_chunk=1, batch_size=2
            )
        )

    a, b = chunks(eager), chunks(compiled)
    assert len(a) == len(b)
    for left, right in zip(a, b, strict=True):
        assert left.trial_index == right.trial_index
        assert left.prediction_start == right.prediction_start
        np.testing.assert_array_equal(left.target, right.target)
        np.testing.assert_allclose(
            left.prediction, right.prediction, rtol=2e-4, atol=2e-5
        )
    assert eager.module_.training and compiled.module_.training
    iterator = iter_trial_prediction_chunks(
        compiled.module_, Float32Windows(), windows_per_chunk=1, batch_size=2
    )
    next(iterator)
    iterator.close()
    assert compiled.module_.training and torch.is_grad_enabled()


def test_supplied_instance_trained_reference_and_custom_optimizer():
    module = TinyClassifier()
    retained = module.fc.weight
    before = retained.detach().clone()

    def optimizer(params, **kwargs):
        # Supported factory retaining caller Parameter references.
        return torch.optim.SGD(module.parameters(), **kwargs)

    net = make_net(True, "eager", module).set_params(optimizer=optimizer)
    X, y = data()
    net.fit(X, y)
    assert net.module_ is module and net.module_.fc.weight is retained
    assert not torch.equal(retained, before)
    assert module._compiled_call_impl is not None
    net.set_params(compile=False)
    assert net.module_ is module and module._compiled_call_impl is None
    before = retained.detach().clone()
    net.fit(X, y)
    assert module.fc.weight is retained and not torch.equal(retained, before)
    assert module._compiled_call_impl is None


class SharedWeightCriterion(torch.nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.weight = weight
        self.scale = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, prediction, target):
        return (
            self.scale * torch.nn.functional.cross_entropy(prediction, target)
            + 0.01 * self.weight.square().mean()
        )


@pytest.mark.parametrize(
    "backend", ["eager", pytest.param("inductor", marks=pytest.mark.slow)]
)
def test_shared_criterion_parameter_identity_checkpoint_continuation(tmp_path, backend):
    module = TinyClassifier()
    criterion = SharedWeightCriterion(module.fc.weight)
    net = make_net(True, backend, module, criterion).set_params(optimizer__momentum=0.9)
    X, y = data()
    net.fit(X, y)
    assert net.module_ is module and net.criterion_ is criterion
    assert net.criterion_.weight is net.module_.fc.weight
    expected = {id(p) for p in (*module.parameters(), *criterion.parameters())}
    actual = [id(p) for group in net.optimizer_.param_groups for p in group["params"]]
    assert len(actual) == len(expected) and set(actual) == expected
    files = {
        key: tmp_path / key
        for key in ("f_params", "f_optimizer", "f_criterion", "f_history")
    }
    net.save_params(**files)
    other_module = TinyClassifier()
    other_criterion = SharedWeightCriterion(other_module.fc.weight)
    other = (
        make_net(True, backend, other_module, other_criterion)
        .set_params(optimizer__momentum=0.9)
        .initialize()
    )
    other.load_params(**files)
    assert other.criterion_.weight is other.module_.fc.weight
    assert other.history.to_list() == net.history.to_list()
    for current in (net, other):
        current.partial_fit(X, y)
    for key, value in net.criterion_.state_dict().items():
        torch.testing.assert_close(
            other.criterion_.state_dict()[key], value, rtol=2e-4, atol=2e-5
        )
    for key, value in net.module_.state_dict().items():
        torch.testing.assert_close(
            other.module_.state_dict()[key], value, rtol=2e-4, atol=2e-5
        )


def test_compile_options_reinitialize_and_user_replacement():
    calls = []

    def backend(graph, example_inputs):
        def execute(*args):
            calls.append(True)
            return graph.forward(*args)

        return execute

    module = TinyClassifier()
    net = make_net(True, backend, module).initialize()
    X, _ = data()
    net.module_(X)
    first = module._compiled_call_impl
    net.set_params(compile__backend="eager")
    assert net.module_ is module and module._compiled_call_impl is not first
    before = len(calls)
    net.module_(X)
    assert len(calls) == before
    net.set_params(compile__disable=True)
    assert module._compiled_call_impl is None
    net.set_params(compile__disable=False)
    assert module._compiled_call_impl is not None
    # User recompilation after an estimator-owned callable must relinquish it.
    module.compile(backend=backend)
    user_call = module._compiled_call_impl
    net.set_params(compile=False)
    assert module._compiled_call_impl is user_call
    net.module_(X)
    assert len(calls) > before


@pytest.mark.parametrize("supplied", [False, True])
def test_native_global_compile_disable(monkeypatch, supplied):
    # _dynamo.eval_frame._optimize reads this environment variable on each
    # invocation, not only on import; this exercises the native null decorator.
    monkeypatch.setenv("TORCHDYNAMO_DISABLE", "1")
    calls = []

    def backend(graph, example_inputs):
        calls.append(True)
        return graph.forward

    module = TinyClassifier() if supplied else TinyClassifier
    criterion = torch.nn.CrossEntropyLoss()
    net = make_net(True, backend, module, criterion).initialize()
    X, y = data()
    assert net.module_._compiled_call_impl == net.module_._call_impl
    assert net.criterion_._compiled_call_impl == criterion._call_impl
    net.fit(X, y)
    assert not calls
    assert np.isfinite(net.predict_proba(X)).all()
    net.set_params(compile=False)
    assert net.module_._compiled_call_impl is None
    assert criterion._compiled_call_impl is None
    net.partial_fit(X, y)
    assert not calls


@pytest.mark.parametrize("roundtrip", ["pickle", "deepcopy"])
def test_compile_ownership_serialization_and_reinitialize(roundtrip):
    import copy
    import pickle

    net = make_net(True, "eager")
    X, y = data()
    net.fit(X, y)
    original_call = net.module_._compiled_call_impl
    assert "_compile_ownership_" not in net.__getstate__()
    restored = (
        pickle.loads(pickle.dumps(net)) if roundtrip == "pickle" else copy.deepcopy(net)
    )
    assert net.module_._compiled_call_impl is original_call
    assert "_compile_ownership_" not in restored.__dict__
    assert restored.compile and restored.module_._compiled_call_impl is None
    np.testing.assert_allclose(
        restored.predict_proba(X), net.predict_proba(X), rtol=2e-4, atol=2e-5
    )
    fresh = clone(net)
    assert "_compile_ownership_" not in fresh.__dict__ and not fresh.initialized_
    restored.initialize()
    owned_module, owned_call = restored._compile_ownership_["module"]
    assert (
        owned_module is restored.module_
        and owned_call is restored.module_._compiled_call_impl
    )
    assert owned_call is not None
    restored.set_params(compile=False)
    assert restored.module_._compiled_call_impl is None
    assert restored.criterion_._compiled_call_impl is None
