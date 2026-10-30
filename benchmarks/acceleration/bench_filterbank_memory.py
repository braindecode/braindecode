"""Continuous filterbank memory and timing-breakdown accounting.

Characterizes the default `order_by_frequency_band=False`,
`drop_original_signals=True` branch of
`braindecode.preprocessing.preprocess.filterbank` only: no production filtering algorithm is introduced, wrapped or
changed by this module. `run_case` reports the produced array's own
unavoidable byte size (`get_data().nbytes` of the real `filterbank()` output)
distinctly from whole-process high-water RSS (never a fabricated
transient-isolated peak computed from a high-water mark), and reports
filter-vs-assembly wall time as a separate, non-headline instrumentation
built from the identical public MNE calls `filterbank` already issues, which
is checked equal to the real call's output before being trusted. This task
delivers evidence only; it does not claim a speedup or change production
arrays. Every case here runs single-process, single-worker, single BLAS/OMP
thread, serial.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import platform
import resource
import subprocess
import tempfile
from pathlib import Path
from time import perf_counter
from warnings import warn

import mne
import numpy as np
import torch

import braindecode
from benchmarks.acceleration import _common
from benchmarks.acceleration._common import measure_step, write_result
from braindecode.preprocessing.preprocess import filterbank

ROOT = Path(__file__).resolve().parents[2]


def _git(*args):
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()


def _dirty_source_hashes():
    status = _git("status", "--porcelain")
    hashes = {}
    for line in status.splitlines():
        path = line[3:].strip()
        if path.startswith('"') and path.endswith('"'):
            path = path[1:-1]
        # A rename line is "old -> new"; hash the new (current) path only.
        path = path.split(" -> ")[-1]
        full = ROOT / path
        if full.is_file():
            hashes[path] = hashlib.sha256(full.read_bytes()).hexdigest()
    return hashes


def _synthetic_raw(n_chans: int, n_times: int, sfreq: float = 256.0, seed: int = 0):
    """Synthetic continuous Raw owned by this benchmark: fixed sinusoid plus
    an impulse per channel, explicit double precision. Not a repository
    fixture; never written into Git."""
    rng = np.random.default_rng(seed)
    ch_names = [f"ch{i}" for i in range(n_chans)]
    times = np.arange(n_times, dtype=np.float64) / sfreq
    data = np.zeros((n_chans, n_times), dtype=np.float64)
    for i in range(n_chans):
        data[i] = np.sin(2 * np.pi * (4.0 + i) * times)
        data[i, n_times // 2] += 1.0
    data += 1e-6 * rng.standard_normal(data.shape)
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    return mne.io.RawArray(data, info, verbose="ERROR")


def _load_raw(n_chans, n_times, mode, tmp_dir, sfreq=256.0, seed=0):
    raw = _synthetic_raw(n_chans, n_times, sfreq=sfreq, seed=seed)
    if mode == "preloaded":
        return raw
    if mode == "saved_reloaded":
        path = Path(tmp_dir) / f"synthetic-{n_chans}-{n_times}-{seed}_raw.fif"
        # Explicit double precision on both sides of the round trip, so the
        # saved/reloaded case is a fair comparison against the preloaded
        # case rather than an artifact of FIF's default single precision.
        raw.save(path, fmt="double", overwrite=True, verbose="ERROR")
        reloaded = mne.io.read_raw_fif(path, preload=True, verbose="ERROR")
        assert reloaded.get_data().dtype == np.float64
        return reloaded
    raise ValueError(f"Unknown mode: {mode!r}")


# Mirrors the default order_by_frequency_band=False branch only: the info
# fields it copies back onto each per-band filtered Raw, so merging that
# Raw's info with every other one in add_channels() stays consistent.
_INFO_FIELDS_TO_PRESERVE = (
    "description",
    "line_freq",
    "device_info",
    "helium_info",
    "experimenter",
    "proj_name",
)


def _instrumented_filterbank(raw, frequency_bands, drop_original_signals=True):
    """Separate, non-headline timing breakdown built from the identical
    public MNE calls `filterbank` issues (short-channel-name warn guard,
    then copy+filter+info-field-preserve per band, then
    add_channels/reorder_channels/drop_channels). This is not an alternative
    algorithm: it mirrors the default order_by_frequency_band=False branch only,
    including the
    warn guard and the `_INFO_FIELDS_TO_PRESERVE` copy-back, so that the
    waveform, warnings and info fields it produces all stay identical to a
    real `filterbank` call. `run_case` below only re-checks the produced
    waveform data per call (`np.testing.assert_array_equal` on
    `get_data()`); the dedicated regression tests
    `test_filterbank_instrumented_replica_preserves_info_fields` and
    `test_filterbank_instrumented_replica_warns_on_long_channel_names`
    (test_preprocess.py) are what actually exercise and lock in the
    warning/info-field parity claimed above — not every call site re-checks
    every one of those every time."""
    if not all(len(ch_name) < 8 for ch_name in raw.ch_names):
        warn(
            "Try to use shorter channel names, since frequency band "
            "annotation requires an estimated 4-8 chars depending on the "
            "frequency ranges. Will truncate to 15 chars (mne max)."
        )
    original_ch_names = raw.ch_names
    all_filtered = []
    filter_time = 0.0
    for l_freq, h_freq in frequency_bands:
        start = perf_counter()
        filtered = raw.copy()
        filtered.filter(l_freq=l_freq, h_freq=h_freq, verbose="ERROR")
        ch_names = filtered.info.ch_names
        ch_types = filtered.info.get_channel_types()
        sampling_freq = filtered.info["sfreq"]
        info = mne.create_info(
            ch_names=ch_names, ch_types=ch_types, sfreq=sampling_freq
        )
        for field in _INFO_FIELDS_TO_PRESERVE:
            value = filtered.info.get(field)
            if value is not None:
                info[field] = value
        filtered.info = info
        filtered.rename_channels(
            {old: (old + f"_{l_freq}-{h_freq}")[-15:] for old in filtered.ch_names}
        )
        filter_time += perf_counter() - start
        all_filtered.append(filtered)
    start = perf_counter()
    raw.add_channels(all_filtered)
    chs_by_freq_band = []
    for i in range(len(original_ch_names)):
        chs_by_freq_band.extend(raw.ch_names[i :: len(original_ch_names)])
    raw.reorder_channels(chs_by_freq_band)
    if drop_original_signals:
        raw.drop_channels(original_ch_names)
    assembly_time = perf_counter() - start
    return raw, filter_time, assembly_time


def _process_high_water_rss_bytes():
    """Whole-process high-water RSS, not an isolated/transient allocation
    peak. Linux `ru_maxrss` is kilobytes, macOS is already bytes; any other
    platform is documented unsupported rather than guessed."""
    system = platform.system()
    maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if system == "Linux":
        return int(maxrss) * 1024, True
    if system == "Darwin":
        return int(maxrss), True
    return None, False


def run_case(
    n_chans: int,
    n_times: int,
    bands: list[tuple[float, float]],
    *,
    mode: str = "preloaded",
    workers: int = 1,
    sfreq: float = 256.0,
    seed: int = 0,
    min_run_time: float = 0.3,
    device_name: str = "cpu",
) -> dict:
    """Characterize one filterbank case's memory/time footprint.

    Does not change, wrap or replace the production filter algorithm.
    Reports the produced array's unavoidable `nbytes` separately from the
    whole-process high-water RSS, and reports filter-vs-assembly wall time
    as a separate, non-headline instrumentation. `workers` is accepted for
    forward schema compatibility; only the single-worker case
    (`workers == 1`) is implemented.
    """
    if workers != 1:
        raise NotImplementedError("Only the single-worker case is implemented")
    if mode not in ("preloaded", "saved_reloaded"):
        raise ValueError(f"Unknown mode: {mode!r}")

    torch.set_num_threads(1)

    with tempfile.TemporaryDirectory() as tmp_dir:
        base_raw = _load_raw(n_chans, n_times, mode, tmp_dir, sfreq=sfreq, seed=seed)

        reference_copy = base_raw.copy()
        filterbank(reference_copy, bands, drop_original_signals=True, verbose="ERROR")
        reference_data = reference_copy.get_data()

        replica_source = base_raw.copy()
        replica_result, filter_time_s, assembly_time_s = _instrumented_filterbank(
            replica_source, bands, drop_original_signals=True
        )
        np.testing.assert_array_equal(replica_result.get_data(), reference_data)

        def timed_call():
            copy = base_raw.copy()
            filterbank(copy, bands, drop_original_signals=True, verbose="ERROR")
            return copy

        measurement = measure_step(
            timed_call,
            device=torch.device("cpu"),
            warmup=2,
            min_run_time=min_run_time,
        )

    gc.collect()
    rss_bytes, rss_supported = _process_high_water_rss_bytes()
    final_output_bytes = int(reference_data.nbytes)

    dirty_hashes = _dirty_source_hashes()
    record = {
        "schema_version": 1,
        "case_id": f"filterbank/bands{len(bands)}/{mode}",
        "status": "ok",
        "implementation": "reference",
        "config": {
            "inputs": [
                {
                    "name": "raw_data",
                    "shape": [n_chans, n_times],
                    "dtype": "float64",
                    "strides": [n_times, 1],
                    "stride_unit": "elements",
                }
            ],
            "seed": seed,
            "threads": 1,
            "workers": workers,
            "cache_policy": (
                "fresh synthetic Raw per case; preloaded or save/reload "
                "fmt=double; warmup before measurement; OS page cache "
                "uncontrolled"
            ),
            "autograd": "not applicable (no torch tensors involved)",
            "precision": "float64 throughout (MNE Raw storage; save fmt=double)",
            "parameters": {
                "n_chans": n_chans,
                "n_times": n_times,
                "bands": [list(band) for band in bands],
                "mode": mode,
                "sfreq": sfreq,
            },
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "dependencies": {
                "mne": str(mne.__version__),
                "numpy": str(np.__version__),
                "braindecode": str(braindecode.__version__),
                "torch": str(torch.__version__),
            },
            "device": "cpu",
            "device_name": device_name,
        },
        "source": {
            "commit": _git("rev-parse", "HEAD"),
            "dirty": bool(dirty_hashes),
            "dirty_source_hashes": dirty_hashes,
        },
        "correctness": {
            "status": "passed",
            "method": (
                "instrumented replica (identical public MNE calls) output "
                "equals real filterbank() output via "
                "np.testing.assert_array_equal"
            ),
            "metrics": {
                "final_output_bytes": final_output_bytes,
                "process_high_water_rss_bytes": rss_bytes,
                "process_high_water_rss_supported": rss_supported,
                "filter_time_s": filter_time_s,
                "assembly_time_s": assembly_time_s,
            },
        },
        **measurement,
        "notes": [
            (
                "Characterizes existing filterbank(); no production filtering "
                "algorithm change. final_output_bytes is the produced Raw's own "
                "get_data().nbytes, not a formula or an estimate."
            ),
            (
                "process_high_water_rss_bytes is the whole-process cumulative "
                "high-water mark (ru_maxrss), not an isolated/transient peak: it "
                "includes imports, setup and every call made in this process, "
                "by design; never used to fabricate an incremental delta."
            ),
            (
                "filter_time_s/assembly_time_s come from a separate, "
                "non-headline instrumented replica of filterbank's own MNE call "
                "sequence, checked equal to the real call before being "
                "reported; the timing/memory fields above are the headline "
                "measurement of the real, unmodified filterbank() call."
            ),
            (
                "This task delivers memory/time evidence only; it does not "
                "claim a speedup or change production arrays."
            ),
        ],
    }
    _common._validate(record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n-chans", type=int, default=32)
    parser.add_argument("--n-times", type=int, default=2560)
    parser.add_argument("--min-run-time", type=float, default=0.2)
    parser.add_argument("--device-name", required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    band_sets = {
        1: [(4.0, 8.0)],
        3: [(4.0, 8.0), (8.0, 13.0), (13.0, 30.0)],
        9: [(float(lo), float(lo + 4.0)) for lo in range(2, 38, 4)],
    }
    for n_bands, bands in band_sets.items():
        for mode in ("preloaded", "saved_reloaded"):
            record = run_case(
                args.n_chans,
                args.n_times,
                bands,
                mode=mode,
                min_run_time=args.min_run_time,
                device_name=args.device_name,
            )
            write_result(
                record, args.output / f"filterbank-bands{n_bands}-{mode}.json"
            )


if __name__ == "__main__":
    main()
