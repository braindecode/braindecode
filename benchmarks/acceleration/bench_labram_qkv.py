"""Finite synthetic Labram traversals; split cells are expression diagnostics.

No estimator, preprocessing, CUDA or compiled performance claim. Invoke in a fresh
process for each pin/fixture/batch/pass. Raw preconditioning and traversal samples
are retained, with fresh model/optimizer/RNG/iterator per finite traversal.
"""
import argparse
import copy
import hashlib
import json
import os
import resource
import statistics
import subprocess
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from braindecode.models.labram import LABRAM_CHANNEL_ORDER, Labram

FIXTURES = {
    "small": {"n_times": 400, "n_chans": 4, "n_outputs": 2, "embed_dim": 64,
                  "num_layers": 1, "num_heads": 4, "patch_size": 200, "neural_tokenizer": True},
    "representative": {"n_times": 1000, "n_chans": 128, "n_outputs": 4, "embed_dim": 200,
                           "num_layers": 2, "num_heads": 4, "patch_size": 200, "neural_tokenizer": True},
}


def _hash(tensor):
    return hashlib.sha256(tensor.detach().contiguous().numpy().tobytes()).hexdigest()


def _record_preconditioning(run_probe, *, synchronize, sidecar_path):
    samples, start = [], time.perf_counter()
    steady = False
    while time.perf_counter() - start < 120:
        synchronize()
        t = time.perf_counter()
        run_probe()
        synchronize()
        samples.append(time.perf_counter() - t)
        if time.perf_counter() - start >= 20 and len(samples) >= 5:
            tail = samples[-5:]
            if max(tail) / min(tail) <= 1.03:
                steady = True
                break
    result = {"samples_s": samples, "steady": steady, "floor_s": 20, "cap_s": 120,
                  "last_five_max_min": 1.03, "reset": "fresh model/optimizer/RNG/iterator"}
    Path(sidecar_path).write_text(json.dumps(result, indent=2) + "\n")
    return result


def _summary(samples):
    ordered = sorted(samples)
    q = statistics.quantiles(ordered, n=4, method="inclusive")
    return {"raw_s": samples, "median_s": statistics.median(samples), "iqr_s": q[2]-q[0],
                "max_min_ratio": max(samples)/min(samples)}


def run_case(*, cell_id, fixture, batch_size, pass_kind, mode, seed,
             device, device_name, output=None, samples=7):
    if mode != "eager" or device != "cpu":
        raise ValueError("This bounded benchmark qualifies eager CPU only")
    if cell_id not in ("split_diagnostic", "model_forward", "synthetic_train_32"):
        raise ValueError("Unsupported bounded cell")
    if cell_id == "model_forward" and pass_kind != "forward":
        raise ValueError("model_forward is forward only")
    if cell_id == "synthetic_train_32" and pass_kind != "forward_backward":
        raise ValueError("synthetic_train_32 includes backward and SGD")
    torch.set_num_threads(1)
    affinity = sorted(os.sched_getaffinity(0))
    cores = {(Path(f"/sys/devices/system/cpu/cpu{c}/topology/physical_package_id").read_text().strip(),
              Path(f"/sys/devices/system/cpu/cpu{c}/topology/core_id").read_text().strip()) for c in affinity}
    assert len(cores) == int(os.environ.get("SLURM_CPUS_PER_TASK", len(cores)))
    assert len(affinity) == len(cores), "SMT siblings are not whole-core placement"
    root = Path(__file__).resolve().parents[2]
    import braindecode.models.labram as lab
    assert Path(lab.__file__).resolve() == root / "braindecode/models/labram.py"
    status = subprocess.check_output(["git", "-C", str(root), "status", "--porcelain"], text=True)
    assert not status, status
    record = {"cell_id": cell_id, "fixture": fixture, "batch_size": batch_size,
                  "pass_kind": pass_kind, "mode": mode, "seed": seed, "device": device,
                  "device_name": device_name, "torch": torch.__version__,
                  "commit": subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip(),
                  "source_sha256": hashlib.sha256(Path(lab.__file__).read_bytes()).hexdigest(),
                  "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  "hostname": os.uname().nodename, "affinity": affinity, "physical_cores": len(cores),
                  "threads": torch.get_num_threads(), "job": os.environ.get("SLURM_JOB_ID")}
    if cell_id == "split_diagnostic":
        # Deliberately independent of the imported production source pin.
        generator = torch.Generator().manual_seed(seed)
        source = torch.randn(3, 2, 4, 65, 16, generator=generator)
        use_unbind = fixture == "unbind"
        def traverse():
            start = time.perf_counter()
            for _ in range(100):
                x = source.detach().clone().requires_grad_(pass_kind == "forward_backward")
                if use_unbind:
                    q, k, v = x.unbind(0)
                else:
                    q, k, v = x[0], x[1], x[2]
                y = (q * k + v).sum()
                if pass_kind == "forward_backward":
                    y.backward()
            return {"traversal_s": time.perf_counter()-start, "examples": 100, "updates": 0}
        record.update(diagnostic_not_imported_production=True, expression=fixture,
                      shape=list(source.shape), clone_reset_included=True)
    else:
        config = FIXTURES[fixture]
        torch.manual_seed(seed)
        template = Labram(**config)
        state = copy.deepcopy(template.state_dict())
        g = torch.Generator().manual_seed(320026)
        data = torch.randn(32, config["n_chans"], config["n_times"], generator=g)
        labels = torch.arange(32) % config["n_outputs"]
        names = list(LABRAM_CHANNEL_ORDER[:config["n_chans"]])
        train = pass_kind == "forward_backward"
        record.update(config=config, actual_embed_dim=template.embed_dim,
                      parameters=sum(p.numel() for p in template.parameters()),
                      dropouts={n:m.p for n,m in template.named_modules() if isinstance(m, nn.Dropout)},
                      data_hash=_hash(data), labels_hash=_hash(labels), channel_names=names,
                      initial_state_hashes={n:_hash(p) for n,p in state.items()},
                      examples=32, expected_updates=(32+batch_size-1)//batch_size if train else 0,
                      workers=0, shuffle=False, synthetic_not_estimator=True)
        def traverse():
            start = time.perf_counter()
            model = Labram(**config)
            model.load_state_dict(state, strict=True)
            model.train(train)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0)
            loader = DataLoader(TensorDataset(data, labels), batch_size=batch_size,
                                shuffle=False, num_workers=0)
            torch.manual_seed(seed + 1)
            setup = time.perf_counter()-start
            begin = time.perf_counter()
            count = updates = 0
            with torch.set_grad_enabled(train):
                for x, target in loader:
                    optimizer.zero_grad(set_to_none=True)
                    y = model(x, ch_names=names)
                    if train:
                        nn.functional.cross_entropy(y, target).backward()
                        optimizer.step()
                        updates += 1
                    count += len(x)
            elapsed = time.perf_counter()-begin
            assert count == 32
            assert updates == record["expected_updates"]
            return {"setup_s": setup, "traversal_s": elapsed,
                        "setup_plus_traversal_s": time.perf_counter()-start,
                        "examples": count, "updates": updates,
                        "next_rng_hash": _hash(torch.get_rng_state())}
    record["cold"] = traverse()
    sidecar = str(output) + ".preconditioning.json"
    record["preconditioning"] = _record_preconditioning(traverse, synchronize=lambda: None,
                                                        sidecar_path=sidecar)
    rows = [traverse() for _ in range(samples)]
    record["samples"] = rows
    record["summary"] = {key:_summary([r[key] for r in rows])
                         for key in rows[0] if key.endswith("_s")}
    record["maxrss_kib_process_high_water"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    record["status"] = "ok" if record["preconditioning"]["steady"] else "unqualified_nonsteady"
    assert subprocess.check_output(["git", "-C", str(root), "status", "--porcelain"], text=True) == ""
    return record


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name, default in (("cell-id", "model_forward"), ("fixture", "small"),
                          ("pass-kind", "forward"), ("mode", "eager"),
                          ("device", "cpu"), ("device-name", "cpu")):
        p.add_argument("--"+name, default=default)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--seed", type=int, default=20261026)
    p.add_argument("--samples", type=int, default=7)
    p.add_argument("--require-clean", action="store_true")
    p.add_argument("--output", required=True)
    args = vars(p.parse_args())
    args.pop("require_clean")  # Clean is always mandatory, never optional.
    output = args["output"]
    result = run_case(**args)
    Path(output).write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
