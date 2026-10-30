"""EEGClassifier compile lifecycle qualification and serial-pair epoch benchmark.

Use --cell for one fresh process; --suite runs five alternating paired processes.
All measurements include estimator callbacks/history, exclude imports/data creation.
"""

import argparse
import copy
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import skorch
import torch

from braindecode import EEGClassifier
from braindecode.models import EEGConformer, EEGNet, IFNet, ShallowFBCSPNet

MODELS = {c.__name__: c for c in (IFNet, ShallowFBCSPNet, EEGNet, EEGConformer)}


def sync(device):
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def model_config(name, parity=False):
    kwargs = {"n_chans": 8, "n_outputs": 2, "n_times": 512, "sfreq": 128}
    if name == "EEGConformer":
        kwargs.update(num_layers=2, n_filters_time=20, num_heads=4)
    if parity:
        kwargs["drop_prob"] = 0.0
        if name == "EEGConformer":
            kwargs["att_drop_prob"] = 0.0
    return kwargs


def make_net(name, compiled, device, parity=False):
    return EEGClassifier(
        MODELS[name],
        **{"module__" + k: v for k, v in model_config(name, parity).items()},
        compile=compiled,
        compile__backend="inductor",
        compile__dynamic=False,
        max_epochs=1,
        train_split=None,
        batch_size=16,
        iterator_train__shuffle=False,
        iterator_train__drop_last=False,
        iterator_train__num_workers=0,
        optimizer=torch.optim.SGD,
        lr=0.001,
        verbose=0,
        device=device,
    )


def fixtures():
    rng = np.random.default_rng(910)
    return rng.normal(size=(66, 8, 512)).astype("float32"), np.arange(
        66, dtype="int64"
    ) % 2


def qualify(name, device):
    torch.set_num_threads(4)
    X, y = fixtures()
    torch.manual_seed(111)
    eager = make_net(name, False, device, True)
    compiled = make_net(name, True, device, True)
    # Signal inference can call set_params and reinitialize an initialized module;
    # finish the normal inference step BEFORE assigning the shared state oracle.
    for net in (eager, compiled):
        net._set_signal_args(X, y, classes=None)
        net.signal_args_set_ = True
        net.initialize()
        # EEGConformer's head has fixed dropout probabilities not exposed by
        # drop_prob. Disable every Dropout ONLY in this deterministic oracle.
        for layer in net.module_.modules():
            if isinstance(layer, torch.nn.Dropout):
                layer.p = 0.0
    compiled.module_.load_state_dict(
        copy.deepcopy(eager.module_.state_dict()), strict=True
    )
    # Same initial state; one real estimator epoch plus partial-tail prediction.
    eager.partial_fit(X, y)
    compiled.partial_fit(X, y)
    a, b = eager.predict_proba(X), compiled.predict_proba(X)
    np.testing.assert_allclose(a, b, rtol=2e-4, atol=2e-5)
    np.testing.assert_array_equal(eager.predict(X), compiled.predict(X))
    assert set(eager.module_.state_dict()) == set(compiled.module_.state_dict())
    for key, value in eager.module_.state_dict().items():
        torch.testing.assert_close(
            compiled.module_.state_dict()[key], value, rtol=2e-4, atol=2e-5
        )
    assert [b["train_batch_size"] for b in compiled.history[-1]["batches"]] == [
        16,
        16,
        16,
        16,
        2,
    ]
    return {
        "model": name,
        "qualified": True,
        "dropout": 0,
        "rtol": 2e-4,
        "atol": 2e-5,
        "max_probability_error": float(abs(a - b).max()),
    }


def cell(name, compiled, device):
    torch.set_num_threads(4)
    torch.manual_seed(111)
    X, y = fixtures()
    sync(device)
    start = time.perf_counter()
    net = make_net(name, compiled, device)
    net._set_signal_args(X, y, classes=None)
    net.signal_args_set_ = True
    net.initialize()
    sync(device)
    setup = time.perf_counter() - start
    epochs = []
    steady = False
    for i in range(12):
        sync(device)
        start = time.perf_counter()
        net.partial_fit(X, y)
        sync(device)
        epochs.append(time.perf_counter() - start)
        if (
            i >= 3
            and statistics.stdev(epochs[-3:]) / statistics.mean(epochs[-3:]) <= 0.10
        ):
            steady = True
            break
    measured = []
    for _ in range(5):
        sync(device)
        start = time.perf_counter()
        net.partial_fit(X, y)
        sync(device)
        measured.append(time.perf_counter() - start)
    assert [b["train_batch_size"] for b in net.history[-1]["batches"]] == [
        16,
        16,
        16,
        16,
        2,
    ]
    assert np.isfinite(net.predict_proba(X)).all()
    source = Path(__file__).resolve().parents[2]
    return {
        "model": name,
        "compiled": compiled,
        "setup_s": setup,
        "warmup_epochs_s": epochs,
        "steady_epochs_s": measured,
        "steady": steady,
        "median_epoch_s": statistics.median(measured),
        "samples": 66,
        "batch_size": 16,
        "dtype": "float32",
        "device": device,
        "threads": 4,
        "config": model_config(name),
        "torch": torch.__version__,
        "skorch": skorch.__version__,
        "python": sys.version,
        "host": platform.node(),
        "commit": subprocess.check_output(
            ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
        ).strip(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }


def summary(out):
    result = {}
    for name in MODELS:
        rows = []
        for pair in range(5):
            a = json.loads((out / f"{name}-{pair}-eager.json").read_text())
            b = json.loads((out / f"{name}-{pair}-compiled.json").read_text())
            saved = a["median_epoch_s"] - b["median_epoch_s"]
            startup_a = a["setup_s"] + a["warmup_epochs_s"][0]
            startup_b = b["setup_s"] + b["warmup_epochs_s"][0]
            overhead = startup_b - startup_a
            rows.append(
                {
                    "ratio": b["median_epoch_s"] / a["median_epoch_s"],
                    "eager_s": a["median_epoch_s"],
                    "compiled_s": b["median_epoch_s"],
                    "first_epoch_overhead_s": overhead,
                    "break_even_epochs": 1 + math.ceil(max(0, overhead) / saved)
                    if saved > 0
                    else None,
                    "projected_totals": {
                        str(n): {
                            "eager": startup_a + (n - 1) * a["median_epoch_s"],
                            "compiled": startup_b + (n - 1) * b["median_epoch_s"],
                        }
                        for n in (1, 10, 100)
                    },
                    "all_steady": a["steady"] and b["steady"],
                }
            )
        ratios = [x["ratio"] for x in rows]
        q1, _, q3 = statistics.quantiles(ratios, n=4, method="inclusive")
        result[name] = {
            "pairs": rows,
            "median_ratio": statistics.median(ratios),
            "iqr": [q1, q3],
            "recommend": statistics.median(ratios) <= 0.95
            and sum(r < 1 for r in ratios) >= 4
            and all(x["all_steady"] for x in rows),
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", action="store_true")
    parser.add_argument("--qualify", action="store_true")
    parser.add_argument("--cell", choices=MODELS)
    parser.add_argument("--compiled", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.suite:
        args.out.mkdir(parents=True, exist_ok=True)
        for name in MODELS:
            for pair in range(5):
                for compiled in (False, True) if pair % 2 == 0 else (True, False):
                    kind = "compiled" if compiled else "eager"
                    dest = args.out / f"{name}-{pair}-{kind}.json"
                    env = dict(os.environ)
                    cache_root = os.environ.get("COMPILE_ESTIMATOR_CACHE_ROOT", "/work")
                    env["TORCHINDUCTOR_CACHE_DIR"] = f"{cache_root}/cache-{name}-{pair}"
                    cmd = [
                        sys.executable,
                        __file__,
                        "--cell",
                        name,
                        "--out",
                        str(dest),
                        "--device",
                        args.device,
                    ]
                    if compiled:
                        cmd.append("--compiled")
                    with dest.with_suffix(".log").open("w") as log:
                        subprocess.run(
                            cmd,
                            env=env,
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            check=True,
                            timeout=600,
                        )
        (args.out / "summary.json").write_text(json.dumps(summary(args.out), indent=2))
    elif args.qualify:
        args.out.write_text(json.dumps(qualify(args.cell, args.device), indent=2))
    else:
        args.out.write_text(
            json.dumps(cell(args.cell, args.compiled, args.device), indent=2)
        )


if __name__ == "__main__":
    main()
