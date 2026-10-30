"""Qualify compiled training and exported execution against the eager oracle.

Eager execution is the oracle throughout. ``compiled`` qualifies a training
step (forward, backward, optimizer update) of the configured model under
``torch.compile``; ``exported`` qualifies forward-only execution of a
``torch.export`` artifact after an actual save/reload round trip.

Every fixture is expected to be supported in every mode. A qualified fixture
that regresses to ``unsupported`` fails here instead of silently skipping;
only (model, mode) pairs listed in ``KNOWN_UNSUPPORTED`` may skip, with reason.
"""

from __future__ import annotations

import functools
import io

import pytest
import torch

from benchmarks.acceleration import bench_execution
from benchmarks.acceleration._common import write_result
from benchmarks.acceleration.bench_execution import MODELS, build_model, run_case

# The EEGNet fixture keeps the default (deprecated-path) classifier on purpose so
# the qualified architecture is braindecode's default; silence only that warning.
pytestmark = pytest.mark.filterwarnings(
    "ignore:Parameter 'final_layer_with_constraint=False' is deprecated"
    ":DeprecationWarning"
)

SEED = 231
BATCH = 4
MODEL_NAMES = tuple(MODELS)
# (model, mode) -> reason. Empty: all fixtures are qualified in all modes.
KNOWN_UNSUPPORTED: dict[tuple[str, str], str] = {}


@pytest.fixture(autouse=True)
def isolate_acceleration_state(acceleration_caller_state):
    """Restore caller RNG, thread count and grad mode after each case."""
    yield


def _small_kwargs(model_name):
    """Deliberately tiny fixtures: fast compile/export, not representative speed."""
    if model_name not in MODEL_NAMES:
        raise ValueError(f"Unknown model: {model_name}")
    return {"n_chans": 8, "n_times": 64, "n_outputs": 2}


@functools.cache
def _cached_case(model_name, mode, batch_size=BATCH, dynamic_batch=False):
    """One run_case per configuration, shared read-only across tests."""
    kwargs = _small_kwargs(model_name)
    return run_case(
        model_name,
        mode=mode,
        device="cpu",
        batch_size=batch_size,
        n_times=kwargs["n_times"],
        n_chans=kwargs["n_chans"],
        n_outputs=kwargs["n_outputs"],
        compile_timeout_s=240,
        dynamic_batch=dynamic_batch,
    )


def _assert_qualified(result, model_name, mode):
    reason = KNOWN_UNSUPPORTED.get((model_name, mode))
    if reason is not None:
        assert result["status"] == "unsupported", (
            f"{model_name}/{mode} is listed as unsupported but returned "
            f"{result['status']}; update KNOWN_UNSUPPORTED"
        )
        pytest.skip(f"known unsupported: {reason}; notes: {result['notes']}")
    assert result["status"] == "ok", result["notes"]
    assert result["correctness"]["status"] == "passed"


@functools.cache
def _post_compile_hooks_observed(model_name):
    """Whether a forward pre-hook added after ``torch.compile`` runs, as in run_case."""
    kwargs = _small_kwargs(model_name)
    model = _seeded(model_name, kwargs).train()
    compiled = torch.compile(model, backend="inductor", fullgraph=False)
    x = torch.randn(BATCH, kwargs["n_chans"], kwargs["n_times"])
    compiled(x).sum().backward()
    seen = []
    with model.register_forward_pre_hook(lambda *args: seen.append(1)):
        compiled(x).sum().backward()
    torch._dynamo.reset()
    return bool(seen)


def _skip_without_post_compile_hooks(model_name):
    # The executed_equal_work check counts the compiled step with a hook added
    # after compilation; torch 2.6 to 2.13 skip it for some models.
    if not _post_compile_hooks_observed(model_name):
        pytest.skip("this torch skips module hooks registered after compile")


def _seeded(model_name, kwargs):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(SEED)
        return build_model(model_name, **kwargs)


def _assert_state_equal(left, right):
    left_state, right_state = left.state_dict(), right.state_dict()
    assert set(left_state) == set(right_state)
    for name in left_state:
        torch.testing.assert_close(
            left_state[name], right_state[name], rtol=1e-4, atol=1e-5, msg=name
        )


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_execution_reference_roundtrip(model_name):
    """Strict state roundtrip; train/eval output, update and buffer parity."""
    kwargs = _small_kwargs(model_name)
    model = _seeded(model_name, kwargs)
    x = torch.randn(
        BATCH,
        kwargs["n_chans"],
        kwargs["n_times"],
        generator=torch.Generator().manual_seed(SEED),
    )
    target = torch.arange(BATCH) % kwargs["n_outputs"]

    def roundtrip(source):
        buffer = io.BytesIO()
        torch.save(source.state_dict(), buffer)
        buffer.seek(0)
        fresh = build_model(model_name, **kwargs)
        fresh.load_state_dict(torch.load(buffer, weights_only=True), strict=True)
        _assert_state_equal(source, fresh)
        return fresh

    reloaded = roundtrip(model)
    outputs = []
    for module in (model, reloaded):
        module.train()
        optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(SEED)
            output = module(x)
        torch.nn.functional.cross_entropy(output, target).backward()
        optimizer.step()
        outputs.append(output.detach())
    assert torch.isfinite(outputs[0]).all()
    torch.testing.assert_close(outputs[0], outputs[1], rtol=1e-4, atol=1e-5)
    with_grad = 0
    for left, right in zip(model.parameters(), reloaded.parameters(), strict=True):
        # Some parameters (e.g. LUNA rotary frequencies) receive no gradient.
        assert (left.grad is None) == (right.grad is None)
        if left.grad is not None:
            assert torch.isfinite(left.grad).all()
            torch.testing.assert_close(left.grad, right.grad, rtol=1e-4, atol=1e-5)
            with_grad += 1
    assert with_grad > 0
    # Parameters and buffers (e.g. BatchNorm running stats) after one update.
    _assert_state_equal(model, reloaded)

    # Trained state survives a second roundtrip; eval-mode outputs agree.
    retrained = roundtrip(model)
    with torch.no_grad():
        expected = model.eval()(x)
        torch.testing.assert_close(retrained.eval()(x), expected, rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(reloaded.eval()(x), expected, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_run_case_eager_reports_training_step(model_name):
    """Eager mode performs one bounded-loss forward/backward/update step."""
    result = _cached_case(model_name, "eager")
    _assert_qualified(result, model_name, "eager")
    parameters = result["config"]["parameters"]
    assert parameters["effective_samples"] == BATCH
    assert parameters["optimizer_updates"] == 1
    assert parameters["loss"] == "cross_entropy"
    assert parameters["timed_configuration"]["dropout_override"] is None
    assert parameters["post_timing_finite"] is True
    assert parameters["compile_setup_s"] is None
    assert parameters["export_setup_s"] is None
    peak = parameters["cpu_peak_rss"]
    assert "process-level" in peak["scope"]
    if bench_execution.resource is None:  # e.g. Windows: no getrusage
        assert peak["ru_maxrss_bytes_after"] is None
    else:
        assert type(peak["ru_maxrss_bytes_after"]) is int
        assert peak["ru_maxrss_bytes_after"] >= peak["ru_maxrss_bytes_before"] > 0


def test_run_case_rejects_unknown_mode():
    with pytest.raises(ValueError, match="Unknown mode"):
        run_case("EEGNet", mode="bogus", device="cpu", batch_size=2, n_times=64)


@pytest.mark.parametrize("model_name", MODEL_NAMES)
@pytest.mark.slow
def test_run_case_compiled_training_matches_eager_oracle(model_name):
    """Compiled training parity, dropout off and dropout on (fallback_random)."""
    _skip_without_post_compile_hooks(model_name)
    result = _cached_case(model_name, "compiled")
    _assert_qualified(result, model_name, "compiled")
    parameters = result["config"]["parameters"]
    assert parameters["effective_samples"] == BATCH
    assert parameters["optimizer_updates"] == 1
    assert parameters["compile_setup_s"] > 0
    assert parameters["post_timing_finite"] is True
    checks = result["correctness"]["metrics"]["checks"]
    dropout_off = checks["dropout_off_parity"]
    assert dropout_off["status"] == "passed"
    assert dropout_off["dropout_override"] == 0.0
    dropout_on = checks["dropout_on_fallback_random_parity"]
    assert dropout_on["status"] == "passed"
    assert dropout_on["dropout_override"] is None
    assert dropout_on["inductor_config_patch"] == {"fallback_random": True}
    timed = checks["timed_configuration_finite"]
    assert timed["status"] == "passed"
    assert timed["dropout_override"] is None
    # Compiler invocation, cache conditions and graph diagnostics are recorded
    # (collected outside headline timing) so a timing claim can be audited.
    diagnostics = parameters["compile_diagnostics"]
    assert diagnostics["backend"] == "inductor"
    assert diagnostics["fullgraph"] is False
    assert diagnostics["dynamo_reset_before_compile"] is True
    for name in ("graph_breaks", "fxgraph_cache_hit", "fxgraph_cache_miss"):
        assert type(diagnostics[name]) is int and diagnostics[name] >= 0
    assert diagnostics["unique_graphs"] >= 1
    assert "inductor_cache_dir" in diagnostics
    assert "fx_graph_cache" in diagnostics
    assert "compiler cache" in result["config"]["cache_policy"]


@pytest.mark.parametrize("model_name", MODEL_NAMES)
@pytest.mark.slow
def test_compiled_and_eager_time_identical_configuration(model_name):
    """Equal work: timed eager and timed compiled run the same model config."""
    _skip_without_post_compile_hooks(model_name)
    eager = _cached_case(model_name, "eager")
    compiled = _cached_case(model_name, "compiled")
    _assert_qualified(eager, model_name, "eager")
    _assert_qualified(compiled, model_name, "compiled")
    eager_parameters = eager["config"]["parameters"]
    compiled_parameters = compiled["config"]["parameters"]
    assert (
        eager_parameters["timed_configuration"]
        == compiled_parameters["timed_configuration"]
    )
    for name in ("effective_samples", "optimizer_updates", "loss", "learning_rate"):
        assert eager_parameters[name] == compiled_parameters[name]
    assert eager["config"]["inputs"] == compiled["config"]["inputs"]


@pytest.mark.parametrize("model_name", MODEL_NAMES)
@pytest.mark.slow
def test_compiled_partial_batch_recompile_is_recorded(model_name):
    """A partial final batch after a full batch keeps parity; recompiles recorded."""
    _skip_without_post_compile_hooks(model_name)
    result = _cached_case(model_name, "compiled")
    _assert_qualified(result, model_name, "compiled")
    shape_change = result["correctness"]["metrics"]["checks"]["partial_batch_parity"]
    assert shape_change["status"] == "passed"
    assert 1 <= shape_change["batch_size"] < BATCH
    before = shape_change["unique_graphs_before"]
    after = shape_change["unique_graphs_after"]
    assert type(before) is int and type(after) is int and after >= before >= 1
    assert shape_change["recompiled"] is (after > before)


@pytest.mark.slow
def test_compile_timeout_is_reported_failure(tmp_path):
    """A compiler stall past the deadline is a reported failure, never a skip."""
    torch._dynamo.reset()
    result = run_case(
        "EEGNet",
        mode="compiled",
        device="cpu",
        batch_size=2,
        n_times=64,
        compile_timeout_s=0.05,
    )
    torch._dynamo.reset()
    assert result["status"] == "failed"
    assert result["correctness"]["status"] != "passed"
    assert result["config"]["parameters"]["compile_timed_out"] is True
    assert any("timed out" in note for note in result["notes"]), result["notes"]
    write_result(result, tmp_path / "timeout.json")  # schema-valid failure record


def test_non_cpu_device_is_unsupported(tmp_path):
    result = run_case("EEGNet", mode="eager", device="cuda", batch_size=2, n_times=64)
    assert result["status"] == "unsupported"
    assert result["notes"]
    assert result["correctness"]["status"] == "not_checked"
    assert result["timing"] is None
    write_result(result, tmp_path / "cuda.json")


def test_invalid_model_configuration_is_unsupported(tmp_path):
    # LUNARotaryBlock needs n_chans divisible by its two heads.
    result = run_case(
        "LUNARotaryBlock",
        mode="eager",
        device="cpu",
        batch_size=2,
        n_times=64,
        n_chans=3,
    )
    assert result["status"] == "unsupported"
    assert any("n_chans" in note for note in result["notes"]), result["notes"]
    write_result(result, tmp_path / "config.json")


@pytest.mark.slow
def test_wrong_exported_artifact_is_reported_failed(monkeypatch, tmp_path):
    """A perturbed exported artifact must yield failed status and correctness."""
    original = bench_execution._export_program

    def perturbed(module, args, dynamic_shapes):
        exported = original(module, args, dynamic_shapes)
        with torch.no_grad():
            for tensor in exported.state_dict.values():
                if tensor.is_floating_point():
                    tensor.add_(1.0)
        return exported

    monkeypatch.setattr(bench_execution, "_export_program", perturbed)
    result = run_case(
        "ShallowFBCSPNet",
        mode="exported",
        device="cpu",
        batch_size=BATCH,
        n_times=64,
        dynamic_batch=True,
    )
    assert result["status"] == "failed"
    assert result["correctness"]["status"] == "failed"
    assert result["correctness"]["metrics"]["failed_check"] == f"batch_{BATCH}"
    assert any("mismatch" in note for note in result["notes"]), result["notes"]
    write_result(result, tmp_path / "wrong.json")


@pytest.mark.parametrize("model_name", MODEL_NAMES)
@pytest.mark.slow
def test_exported_artifact_executes_after_reload(model_name):
    """The exported artifact is actually executed post save/reload, not just typed.

    Build an eval-mode eager oracle, export it, save, reload a fresh
    ``ExportedProgram`` and execute its module directly against the same input.
    """
    result = _cached_case(model_name, "exported")
    _assert_qualified(result, model_name, "exported")
    parameters = result["config"]["parameters"]
    assert parameters["compile_setup_s"] is None
    setup = parameters["export_setup_s"]
    for name in ("export_s", "save_s", "load_s", "unlift_s"):
        assert setup[name] > 0
    assert parameters["optimizer_updates"] == 0

    # Independent, direct proof beyond run_case's own internal check: export,
    # save, reload and execute again here, outside the runner under test.
    kwargs = _small_kwargs(model_name)
    oracle = _seeded(model_name, kwargs).eval()
    x = torch.randn(
        BATCH,
        kwargs["n_chans"],
        kwargs["n_times"],
        generator=torch.Generator().manual_seed(SEED),
    )
    with torch.no_grad():
        expected = oracle(x)
    exported = torch.export.export(oracle, (x,), strict=False)
    buffer = io.BytesIO()
    torch.export.save(exported, buffer)
    buffer.seek(0)
    reloaded = torch.export.load(buffer)
    actual = reloaded.module()(x)
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)


@pytest.mark.slow
def test_exported_timing_binds_module_once_under_no_grad(monkeypatch):
    """Headline exported timing excludes unlifting and runs without autograd."""
    calls = {"module": 0, "timed_requires_grad": []}
    original_module = torch.export.ExportedProgram.module
    original_measure = bench_execution.measure_step

    def counting_module(self, *args, **kwargs):
        calls["module"] += 1
        return original_module(self, *args, **kwargs)

    def spying_measure(run, **kwargs):
        calls["timed_requires_grad"].append(run().requires_grad)
        before = calls["module"]
        measurement = original_measure(run, **kwargs)
        assert calls["module"] == before, "module() rebuilt inside timed calls"
        return measurement

    monkeypatch.setattr(torch.export.ExportedProgram, "module", counting_module)
    monkeypatch.setattr(bench_execution, "measure_step", spying_measure)
    result = run_case(
        "ShallowFBCSPNet",
        mode="exported",
        device="cpu",
        batch_size=BATCH,
        n_times=64,
    )
    assert result["status"] == "ok", result["notes"]
    # >= 1, not == 1: torch.export internals may legitimately call module();
    # the guard is that no timed call rebuilds it (asserted in the spy).
    assert calls["module"] >= 1
    assert calls["timed_requires_grad"] == [False]
    assert result["config"]["autograd"].startswith("no_grad")


@pytest.mark.parametrize("model_name", MODEL_NAMES)
@pytest.mark.parametrize("batch_size", [1, 3])
@pytest.mark.slow
def test_run_case_exported_static_small_batches(model_name, batch_size):
    """Static-shape exports at batch 1 and a non-power-of-two batch execute."""
    result = _cached_case(model_name, "exported", batch_size=batch_size)
    _assert_qualified(result, model_name, "exported")
    assert result["config"]["parameters"]["effective_samples"] == batch_size


@pytest.mark.parametrize("model_name", MODEL_NAMES)
@pytest.mark.slow
def test_run_case_exported_dynamic_batch_partial_shapes(model_name):
    """One dynamic-batch artifact executes at batch 1, a partial batch and B+1."""
    result = _cached_case(model_name, "exported", dynamic_batch=True)
    _assert_qualified(result, model_name, "exported")
    parameters = result["config"]["parameters"]
    assert parameters["dynamic_batch"] is True
    checks = parameters["dynamic_batch_checks"]
    sizes = [check["batch_size"] for check in checks]
    assert sizes == [BATCH, 1, BATCH - 1, BATCH + 1]
    assert all(check["status"] == "passed" for check in checks)
