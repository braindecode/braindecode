"""Cross-model contract gates for every registered Braindecode model.

These tests turn the repository model conventions into machine-enforced
invariants. A newly registered model is automatically included through
models_mandatory_parameters; no per-model test opt-in is required.
"""

from __future__ import annotations

import contextlib
import copy
import functools
import json
import pickle  # nosec B403 - the test reloads a model it just pickled
import re
import sys
from collections import namedtuple
from collections.abc import Mapping, Sequence

import pytest
import torch
from torch import nn
from torch.nn.modules.module import register_module_forward_pre_hook
from torch.nn.utils import parametrize
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten, tree_map_only

import braindecode.functional
from braindecode.functional import _real_dft
from braindecode.models.util import (
    _get_signal_params,
    models_dict,
    models_mandatory_parameters,
)

all_models_dict = dict(models_dict)


def _tensor_leaves(value):
    """Yield tensor leaves from nested model outputs."""
    if torch.is_tensor(value):
        yield value
    elif isinstance(value, Mapping):
        for child in value.values():
            yield from _tensor_leaves(child)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for child in value:
            yield from _tensor_leaves(child)


def _build_case(model_name, required_params, signal_params):
    signal = _get_signal_params(signal_params)
    model_kwargs = _get_signal_params(signal_params, required_params)
    model = all_models_dict[model_name](**model_kwargs).eval()
    x = torch.randn(2, len(signal["chs_info"]), signal["n_times"])
    return model, x


def _materialize(model, x):
    """Run one no-grad forward if lazy parameters still need their shapes."""
    if any(isinstance(p, nn.UninitializedParameter) for p in model.parameters()):
        with torch.no_grad():
            model(x)


def _clone_state(model):
    """Clone persistent tensor state after any lazy first-forward setup."""
    return {name: value.detach().clone() for name, value in model.state_dict().items()}


def _batched_tensor_leaves(value, batch_size):
    """Return tensor leaves whose leading dimension represents the batch."""
    return [
        leaf
        for leaf in _tensor_leaves(value)
        if leaf.ndim > 0 and leaf.shape[0] == batch_size
    ]


@pytest.mark.parametrize(
    "model_name,required_params,signal_params",
    models_mandatory_parameters,
)
def test_registered_model_runtime_contract(
    model_name, required_params, signal_params
):
    """Registered models preserve inputs and emit finite batched tensors."""
    model, x = _build_case(model_name, required_params, signal_params)
    x_before = x.clone()

    with torch.no_grad():
        output = model(x)

    assert torch.equal(x, x_before), (
        f"{model_name} mutated its input tensor during eval-mode forward"
    )

    leaves = list(_tensor_leaves(output))
    assert leaves, f"{model_name} returned no tensor output"

    batched = _batched_tensor_leaves(output, x.shape[0])
    assert batched, (
        f"{model_name} returned no tensor leaf preserving batch dimension "
        f"{x.shape[0]}"
    )

    for leaf in leaves:
        if leaf.is_floating_point() or leaf.is_complex():
            assert torch.isfinite(leaf).all(), (
                f"{model_name} emitted non-finite values in eval-mode forward"
            )

    # A first forward may legitimately materialize lazy parameters. Once warm,
    # however, eval-mode inference must not mutate persistent model state.
    # Reuse the permutation probe below as the second forward so the gate stays
    # cheap even for large foundation models.
    state_before = _clone_state(model)

    # Reordering independent samples must only reorder the corresponding
    # outputs. This catches accidental batch-axis mixing that ordinary shape
    # checks and single-sample tests cannot see.
    permutation = torch.tensor([1, 0], device=x.device)
    with torch.no_grad():
        permuted = model(x.index_select(0, permutation))

    state_after = model.state_dict()
    assert state_before.keys() == state_after.keys()
    for name, expected_state in state_before.items():
        torch.testing.assert_close(
            state_after[name],
            expected_state,
            msg=lambda msg: (
                f"{model_name} mutated persistent state {name!r} during "
                f"eval-mode forward: {msg}"
            ),
        )

    permuted_batched = _batched_tensor_leaves(permuted, x.shape[0])
    assert len(permuted_batched) == len(batched)
    for expected_leaf, actual_leaf in zip(batched, permuted_batched):
        torch.testing.assert_close(
            actual_leaf,
            expected_leaf.index_select(0, permutation),
            msg=lambda msg: (
                f"{model_name} is not batch-permutation equivariant in eval "
                f"mode: {msg}"
            ),
        )
    # Permutation equivariance alone cannot detect all cross-sample mixing: a
    # symmetric batch aggregate can influence every output and still permute
    # correctly. Keep sample 0 fixed, change only sample 1, and require sample
    # 0's outputs to remain invariant.
    composed_x = x.clone()
    composed_x[1].mul_(-3.0).add_(1.0)
    with torch.no_grad():
        recomposed = model(composed_x)

    recomposed_batched = _batched_tensor_leaves(recomposed, x.shape[0])
    assert len(recomposed_batched) == len(batched)
    for expected_leaf, actual_leaf in zip(batched, recomposed_batched):
        torch.testing.assert_close(
            actual_leaf[0],
            expected_leaf[0],
            msg=lambda msg: (
                f"{model_name} leaked information across independent batch "
                f"samples in eval mode: {msg}"
            ),
        )


@pytest.mark.parametrize(
    "model_name,required_params,signal_params",
    models_mandatory_parameters,
)
def test_registered_model_serialization_contract(
    model_name, required_params, signal_params
):
    """Config + state_dict reconstruction, deepcopy and pickle preserve outputs."""
    model, x = _build_case(model_name, required_params, signal_params)

    with torch.no_grad():
        expected = list(_tensor_leaves(model(x)))

    config = model.get_config()
    serialized = json.dumps(config)
    rebuilt = type(model).from_config(json.loads(serialized)).eval()
    _materialize(rebuilt, x)
    rebuilt.load_state_dict(model.state_dict(), strict=True)

    copies = {"config/state round-trip": rebuilt, "deepcopy": copy.deepcopy(model)}
    # torch refuses to pickle modules with parametrizations (weight_norm,
    # max-norm constraints): those models are saved through state_dict only.
    if not any(parametrize.is_parametrized(m) for m in model.modules()):
        copies["pickle"] = pickle.loads(pickle.dumps(model))  # nosec B301
    for how, clone in copies.items():
        with torch.no_grad():
            actual = list(_tensor_leaves(clone(x)))
        assert len(actual) == len(expected), (
            f"{model_name} changed tensor-output structure after {how}"
        )
        for expected_leaf, actual_leaf in zip(expected, actual):
            assert expected_leaf.shape == actual_leaf.shape
            torch.testing.assert_close(actual_leaf, expected_leaf)


# Trainable parameters the default forward does not reach (pretraining heads,
# other read-outs, reference quirks); kept so released checkpoints load.
_UNUSED_IN_FORWARD = {
    "AttnSleep": r"self_attn\.convs\.0\.",  # the reference never convolves the query
    "BrainOmni": r"^blocks\.11\.",  # the reference encode() skips the last block
    "Brant": r"spatial_encoder\.proj_out\.",  # reconstruction head
    "CodeBrain": r"residual_blocks\.7\.(rms_norm|res_conv)|^lm_head_",  # skip-only last block, tokenizer heads
    "DANCE": r"^decoder\.",  # event decoder of detect()
    "MSVTNet": r"^branch_head\.",  # auxiliary branch heads (return_features)
    "PopulationTransformer": r"^spec_prediction_head\.",  # pretraining head
    "SignalJEPA": r"^transformer\.decoder\.",  # pretraining decoder
    "SignalJEPA_Contextual": r"^transformer\.decoder\.",
    "SSTDPN": r"^proto_cpt$",  # prototype loss term
    "STEEGFormer": r"^norm\.",  # only the cls read-out normalises
}


@pytest.mark.parametrize(
    "model_name,required_params,signal_params",
    models_mandatory_parameters,
)
def test_registered_model_training_contract(
    model_name, required_params, signal_params
):
    """A model used under ``torch.inference_mode`` still trains, then evaluates."""
    model, x = _build_case(model_name, required_params, signal_params)
    _materialize(model, x)
    with torch.inference_mode():
        expected = _batched_tensor_leaves(model(x), x.shape[0])

    # One train-mode SGD step; every trainable parameter gets a finite gradient.
    model.train()
    leaves = [t for t in _tensor_leaves(model(x)) if t.requires_grad]
    assert leaves, "no differentiable output"
    assert all(t.shape[0] == x.shape[0] for t in leaves if t.ndim)
    loss = sum(t.float().square().mean() for t in leaves)
    assert torch.isfinite(loss), "non-finite train-mode output"
    loss.backward()
    unused = _UNUSED_IN_FORWARD.get(model_name)
    trainable = {n: p for n, p in model.named_parameters() if p.requires_grad}
    no_grad = [
        n
        for n, p in trainable.items()
        if p.grad is None and not (unused and re.search(unused, n))
    ]
    assert not no_grad, f"no gradient reaches {no_grad}"
    bad = [
        n
        for n, p in trainable.items()
        if p.grad is not None and not torch.isfinite(p.grad).all()
    ]
    assert not bad, f"non-finite gradient in {bad}"
    torch.optim.SGD(trainable.values(), lr=1e-3).step()
    assert all(torch.isfinite(p).all() for p in trainable.values())

    # e.g. keeping the best model: no non-leaf tensor may stay cached.
    model = copy.deepcopy(model)
    with torch.no_grad():
        actual = _batched_tensor_leaves(model.eval()(x), x.shape[0])
    assert [t.shape for t in actual] == [t.shape for t in expected]
    assert all(torch.isfinite(t).all() for t in actual if t.is_floating_point())


# ``.item()`` in forward, which meta tensors cannot answer (see _HOST_SYNC).
_DATA_DEPENDENT_FORWARD = {
    "BaRISTA",
    "BrainOmni",
    "BrainTokenizer",
    "CodeBrain",
    "NeuroRVQTokenizer",
}
# Known misses, not fixed here.
_DEVICE_XFAIL = {
    "LUNA": "default channel locations are cached on the CPU and copied to the "
    "device at every forward",
}


@pytest.mark.parametrize(
    "model_name,required_params,signal_params",
    [
        pytest.param(
            *case,
            marks=[pytest.mark.xfail(reason=_DEVICE_XFAIL[case[0]], strict=True)]
            if case[0] in _DEVICE_XFAIL
            else [],
            id=case[0],
        )
        for case in models_mandatory_parameters
    ],
)
def test_registered_model_follows_device(model_name, required_params, signal_params):
    """``model.to(device)`` moves every tensor forward reads or creates, checked
    on ``meta``: a CPU tensor in forward is a buffer kept as a plain attribute
    or a ``torch.zeros``/``arange``/``tensor`` without ``device=``."""
    model, x = _build_case(model_name, required_params, signal_params)
    _materialize(model, x)
    model.to("meta")
    stray = [
        f"{prefix}.{attr}".lstrip(".")
        for prefix, module in model.named_modules()
        for attr, value in vars(module).items()
        if torch.is_tensor(value) and not value.is_meta
    ]
    assert not stray, f"tensor attributes that .to() does not move: {stray}"
    if model_name in _DATA_DEPENDENT_FORWARD:
        pytest.skip("forward reads tensor values (.item()), which meta tensors lack")
    x = x.to("meta")
    with torch.no_grad(), _record() as log:
        output = model(x)
    assert all(t.is_meta for t in _tensor_leaves(output))
    # 0-dim CPU tensors act as scalars on any device (torch.as_tensor(2.0)).
    on_cpu = sorted(
        {
            op.name
            for op in log.ops
            if any(t[1] != "meta" and t[2] for t in op.ins + op.outs)
        }
    )
    assert not on_cpu, f"forward creates CPU tensors in {on_cpu}"


class _OpLog(TorchDispatchMode):
    """Record the aten ops run under it: name, (dtype, device, ndim, contiguous)
    of the tensor inputs and outputs and the other arguments; outputs of a 3-D
    ``permute`` are remembered by identity. With ``meta_to_cpu``, copies from
    ``meta`` to the CPU return random values (meta tensors hold no data)."""

    def __init__(self, meta_to_cpu=False):
        super().__init__()
        self.ops, self.permuted, self.meta_to_cpu = [], {}, meta_to_cpu

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if (
            self.meta_to_cpu
            and func is torch.ops.aten._to_copy.default
            and args[0].is_meta
            and kwargs.get("device") == torch.device("cpu")
        ):
            dtype = kwargs.get("dtype") or args[0].dtype
            out = torch.randn(args[0].shape).to(dtype)
        else:
            out = func(*args, **kwargs)
        flat = tree_flatten((args, kwargs))[0]
        ins = [t for t in flat if isinstance(t, torch.Tensor)]
        outs = [t for t in tree_flatten(out)[0] if isinstance(t, torch.Tensor)]
        name = func.overloadpacket.__name__
        if name == "permute" and ins[0].dim() == 3:
            self.permuted.update((id(t), t) for t in outs)
        self.ops.append(
            _Op(
                name,
                func._overloadname,
                tuple(_meta(t) for t in ins),
                tuple(_meta(t) for t in outs),
                tree_map_only(torch.Tensor, lambda t: None, args),
                tree_map_only(torch.Tensor, lambda t: None, kwargs),
            )
        )
        return out


# ``args``/``kwargs`` as dispatched, tensors replaced by None.
_Op = namedtuple("_Op", "name overload ins outs args kwargs")


def _meta(t):
    return t.dtype, t.device.type, t.dim(), t.is_contiguous()


@contextlib.contextmanager
def _record(meta_as_hpu=False):
    """Run the block under an ``_OpLog``; RNN modules log whether their input
    comes from a 3-D permute (op ``rnn:<module>``). With ``meta_as_hpu``,
    ``spectral_input`` and ``needs_real_dft`` treat ``meta`` tensors as HPU
    ones, so ``meta`` plays the accelerator without complex dtypes."""
    log = _OpLog(meta_to_cpu=meta_as_hpu)
    original = braindecode.functional.spectral_input
    needs_real_dft, autocast = _real_dft.needs_real_dft, torch.autocast

    def spectral_input(x):
        return original(x.cpu() if x.is_meta else x)

    def meta_autocast(device_type, **kwargs):
        # autocast has no meta device (the real-DFT path disables autocast)
        if device_type == "meta":
            return contextlib.nullcontext()
        return autocast(device_type, **kwargs)

    def rnn_input(module, args):
        if isinstance(module, nn.RNNBase) and isinstance(args[0], torch.Tensor):
            permuted = id(args[0]) in log.permuted
            log.ops.append(
                _Op(f"rnn:{type(module).__name__}", "", (), (), (permuted,), {})
            )

    patched = [
        m
        for name, m in list(sys.modules.items())
        if meta_as_hpu
        and name.startswith("braindecode")
        and getattr(m, "spectral_input", None) is original
    ]
    for m in patched:
        m.spectral_input = spectral_input
    if meta_as_hpu:
        _real_dft.needs_real_dft = lambda x: x.is_meta or needs_real_dft(x)
        torch.autocast = meta_autocast
    hook = register_module_forward_pre_hook(rnn_input)
    try:
        with log:
            yield log
    finally:
        hook.remove()
        _real_dft.needs_real_dft, torch.autocast = needs_real_dft, autocast
        for m in patched:
            m.spectral_input = original


# Gaudi has no complex dtype: an FFT/STFT input goes through
# braindecode.functional.spectral_input, which moves HPU tensors to the CPU.
@pytest.mark.parametrize(
    "model_name,required_params,signal_params", models_mandatory_parameters
)
def test_registered_model_complex_only_after_spectral_input(
    model_name, required_params, signal_params
):
    """With ``meta`` as the device, complex tensors (forward and backward) live
    on the CPU only, i.e. every FFT/STFT input went through ``spectral_input``."""
    if model_name in _DATA_DEPENDENT_FORWARD:
        pytest.skip("forward reads tensor values (.item()), which meta tensors lack")
    model, x = _build_case(model_name, required_params, signal_params)
    _materialize(model, x)
    model.to("meta").train()
    x = x.to("meta")
    with _record(meta_as_hpu=True) as log:
        leaves = [t for t in _tensor_leaves(model(x)) if t.requires_grad]
        sum(t.float().square().mean() for t in leaves).backward()
    found = sorted(
        {
            op.name
            for op in log.ops
            if any(t[0].is_complex and t[1] == "meta" for t in op.outs)
        }
    )
    assert not found, f"complex tensors on the device, not via spectral_input: {found}"


# Host syncs and data-dependent shapes cut the Gaudi lazy graph (and CUDA
# graphs, torch.compile) at every step.
_SYNC_OPS = {
    "_local_scalar_dense",  # .item(), bool(t), float(t), int(t)
    "is_nonzero",
    "item",
    "nonzero",
    "masked_select",
    "_unique",
    "_unique2",
    "unique_dim",
    "unique_consecutive",
}
# Allowed per model: (findings, reason).
_QUANTIZER = (
    ("_local_scalar_dense", "index_put_ with a boolean mask"),
    "codebook k-means init flag, first-batch k-means and dead-code expiry "
    "(the released EuclideanCodebook)",
)
_HOST_SYNC = {
    "BaRISTA": (("_local_scalar_dense",), "eager-only spatial_indices range check"),
    "BENDR": (("_local_scalar_dense",), "LayerDrop coin flip on a CPU torch.rand(1)"),
    "BrainOmni": _QUANTIZER,
    "BrainTokenizer": _QUANTIZER,
    "CodeBrain": (("_local_scalar_dense",), "lazy kernel-norm init flag (buffer)"),
    "EEGSimpleConv": (
        ("_local_scalar_dense",),
        "torchaudio Resample takes the output length from a CPU scalar",
    ),
    "NeuroRVQTokenizer": (
        ("_local_scalar_dense",),
        "codebook k-means init flag (``initted`` buffer), as in the released code",
    ),
    "MetaNeuromotorHand": (
        ("_local_scalar_dense",),
        "training-time masking draws the mask count on the CPU (reference)",
    ),
}


@functools.cache
def _findings(model_name):
    """Host syncs and accelerator/low-precision gaps in one eval-mode and one
    train-mode forward."""
    case = next(c for c in models_mandatory_parameters if c[0] == model_name)
    model, x = _build_case(*case)
    _materialize(model, x)
    with torch.no_grad(), _record() as eval_log:
        model(x)
    model.train()
    with _record() as train_log:
        model(x)
    syncs, gaps = set(), set()
    for op in eval_log.ops + train_log.ops:
        if op.name in _SYNC_OPS:
            syncs.add(op.name)
        elif op.name.startswith("index") and any(
            t[0] == torch.bool for t in op.ins[1:]
        ):
            syncs.add(f"{op.name} with a boolean mask")
        elif (
            op.name == "repeat_interleave"
            and op.overload == "Tensor"
            and op.kwargs.get("output_size") is None
            and len(op.args) < 2
        ):
            syncs.add("repeat_interleave with tensor repeats")
        # Op patterns that broke models on Gaudi or in bfloat16/float16; each
        # one names the fix that removed it.
        if op.name == "avg_pool3d":
            # no CPU bfloat16/float16 kernel (EEGSym, c2aff52b)
            gaps.add("avg_pool3d")
        elif op.name == "elu":
            scale = op.args[2] if len(op.args) > 2 else op.kwargs.get("scale", 1)
            input_scale = (
                op.args[3] if len(op.args) > 3 else op.kwargs.get("input_scale", 1)
            )
            if (scale, input_scale) != (1, 1):
                # nn.SELU: does not train on Gaudi (BrainOmni, ba1264bf)
                gaps.add("elu with scale != 1 (nn.SELU): use scale * F.elu")
        elif op.name == "roll" and not op.ins[0][3]:
            # wrong values in Gaudi eager mode (EMG2QwertyNet, #1249)
            gaps.add("roll of a non-contiguous tensor")
        elif op.name.startswith("rnn:") and op.args[0]:
            # Gaudi lazy mode fails to compile (BrainOmni SEANet LSTM, #1249)
            gaps.add(f"{op.name[4:]} fed by a 3-D permute")
    return {"host_sync": sorted(syncs), "gaps": sorted(gaps)}


@pytest.mark.parametrize(
    "model_name,required_params,signal_params", models_mandatory_parameters
)
def test_registered_model_forward_has_no_host_sync(
    model_name, required_params, signal_params
):
    """Forward reads no tensor value on the host and has no data-dependent shape
    (except the allowed ones in ``_HOST_SYNC``)."""
    allowed = _HOST_SYNC.get(model_name, ((), ""))[0]
    found = [f for f in _findings(model_name)["host_sync"] if f not in allowed]
    assert not found, f"host syncs / data-dependent shapes in forward: {found}"


@pytest.mark.parametrize(
    "model_name,required_params,signal_params", models_mandatory_parameters
)
def test_registered_model_avoids_accelerator_gaps(
    model_name, required_params, signal_params
):
    """Forward avoids ops known to fail on Gaudi or in low precision (cdist in
    bfloat16/float16: test_forward_in_dtype)."""
    found = _findings(model_name)["gaps"]
    assert not found, f"ops with known accelerator/low-precision gaps: {found}"
