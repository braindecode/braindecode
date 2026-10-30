"""Regression test for the ``_zarr_to_memmap`` concurrency race.

Historically ``_zarr_to_memmap`` used a double primitive::

    if not npy_path.exists():                 # TOCTOU check
        ...write tmp_path...
        tmp_path.rename(npy_path)             # replaces destination

The gap between the ``exists()`` check and the ``rename`` let multiple
processes materialise the same zarr group concurrently and repeatedly
*replace* the published ``.npy`` inode.  On local POSIX filesystems
this produced wasted I/O; on NFSv3 it left ``.nfsXXXX`` silly-rename
files behind and triggered ``SIGBUS`` when worker processes
page-faulted on mmap'd inodes that a concurrent rename had unlinked.

The fixed implementation must guarantee that the published ``.npy``
inode is created **exactly once** and is **never replaced**, even
under arbitrary concurrency.  We verify this by spawning several
worker processes that all race on the same ``(zarr_path, group)`` and
requiring every worker to observe the *same* inode number for the
final file.

Authors: Pierre Guetschel <pierre.guetschel@gmail.com>

License: BSD (3-clause)
"""

from __future__ import annotations

import multiprocessing as mp
import os
from pathlib import Path

import numpy as np
import pytest

zarr = pytest.importorskip("zarr")

from braindecode.datasets.base import _zarr_to_memmap

# ~16 MB at float64: wide enough that 8 writers racing on the same
# group overlap for long enough to exercise the TOCTOU window on a
# local filesystem.
ARRAY_SHAPE = (256, 8 * 1024)
CHUNKS = (64, 8 * 1024)

# Hard caps so a hang in one child process cannot wedge the whole
# test session.  CI runners vary wildly in speed (macOS in particular
# can be 3–4x slower than a typical laptop), so these are generous.
_BARRIER_TIMEOUT_S = 120.0
_POOL_TIMEOUT_S = 600.0


def _make_zarr(tmp_path: Path) -> Path:
    """Create a tiny on-disk zarr store with one group and one array."""
    zarr_path = tmp_path / "dataset.zarr"
    root = zarr.open(str(zarr_path), mode="w")
    grp = root.create_group("recording_0")
    try:
        arr = grp.create_array(
            "data", shape=ARRAY_SHAPE, dtype="float32", chunks=CHUNKS
        )
    except AttributeError:
        # zarr<3 fallback
        arr = grp.zeros("data", shape=ARRAY_SHAPE, dtype="float32", chunks=CHUNKS)
    arr[:] = np.arange(int(np.prod(ARRAY_SHAPE)), dtype="float32").reshape(ARRAY_SHAPE)
    return zarr_path


# Global barrier handle, populated in each pool worker by the pool
# initializer.  ``multiprocessing`` synchronisation primitives cannot
# be passed as ``starmap`` args under the ``spawn`` start method, so
# the initializer is the canonical way to share them.
_BARRIER: mp.synchronize.Barrier | None = None


def _init_worker(barrier) -> None:
    global _BARRIER
    _BARRIER = barrier


def _worker(zarr_path_str: str, group: str):
    """Wait at the rendezvous barrier, then race on the same group.

    Returns a tuple of observable signals that the test asserts on:
    ``(pid, npy_path, inode, checksum, shape, dtype)``.  The inode is
    the critical one: if the buggy rename-replace path is taken,
    different workers will observe different inodes at the same path.
    """
    assert _BARRIER is not None, "pool initializer was not invoked"
    # Cap the barrier wait so one crashed sibling cannot wedge the
    # whole pool forever — without this the test can hang for the
    # full CI job timeout on slow runners where a worker dies during
    # spawn/import.
    _BARRIER.wait(timeout=_BARRIER_TIMEOUT_S)

    npy_path = _zarr_to_memmap(Path(zarr_path_str), group)

    # Stat right after the call so we capture whichever inode is
    # currently published at this path for this worker.
    inode = os.stat(npy_path).st_ino

    # Force real page faults on the memmap — this mirrors the
    # production access pattern that originally raised SIGBUS.
    arr = np.load(npy_path, mmap_mode="c")
    checksum = float(arr[0, 0]) + float(arr[-1, -1]) + float(arr.sum())
    return (
        os.getpid(),
        str(npy_path),
        inode,
        checksum,
        tuple(arr.shape),
        str(arr.dtype),
    )


def _run_race(tmp_path: Path, n_workers: int):
    zarr_path = _make_zarr(tmp_path)
    cache_dir = zarr_path.parent / f".{zarr_path.name}_memmap"

    ctx = mp.get_context("spawn")
    barrier = ctx.Barrier(n_workers)
    with ctx.Pool(n_workers, initializer=_init_worker, initargs=(barrier,)) as pool:
        # starmap_async(...).get(timeout=...) gives us a hard wall-
        # clock cap on the whole pool: if any worker deadlocks or
        # stalls, the test fails with TimeoutError instead of hanging
        # until the CI job runner kills it.
        async_result = pool.starmap_async(
            _worker,
            [(str(zarr_path), "recording_0")] * n_workers,
        )
        results = async_result.get(timeout=_POOL_TIMEOUT_S)
    return cache_dir, results


@pytest.mark.parametrize("n_workers", [2, 4, 8])
def test_zarr_to_memmap_is_concurrency_safe(tmp_path, n_workers):
    """N processes racing on the same group must publish one inode.

    Asserts the full concurrency contract of ``_zarr_to_memmap``:

    1. every worker returned (no ``SIGBUS``);
    2. every worker agrees on the published path;
    3. every worker observed the *same* inode at that path (this is
       the single assertion that fails deterministically on the
       buggy ``tmp_path.rename(npy_path)`` implementation);
    4. every worker read the same bytes;
    5. the cache directory is clean — exactly one file, no leftover
       ``*.tmp.npy`` or ``*.lock`` or ``.nfsXXXX`` debris;
    6. the published file has the expected shape, dtype and content.
    """
    cache_dir, results = _run_race(tmp_path, n_workers)

    # (1) No worker was killed.
    assert len(results) == n_workers

    # (2) Path agreement.
    paths = {r[1] for r in results}
    assert paths == {str(cache_dir / "recording_0.npy")}

    # (3) Inode uniqueness — the race-detector.
    inodes = {r[2] for r in results}
    assert len(inodes) == 1, (
        "workers observed distinct inodes at "
        f"{cache_dir / 'recording_0.npy'}: {inodes}. "
        "This proves the published file was replaced at least once, "
        "which is the TOCTOU rename-replace race."
    )

    # (4) Content agreement across readers.
    checksums = {r[3] for r in results}
    assert len(checksums) == 1, f"workers disagreed on file content: {checksums}"

    # (5) Cache dir is clean.
    leftover = sorted(p.name for p in cache_dir.iterdir())
    assert leftover == ["recording_0.npy"], (
        f"unexpected leftover state in cache dir: {leftover}"
    )

    # (6) Published file is well-formed.
    arr = np.load(cache_dir / "recording_0.npy", mmap_mode="r")
    assert arr.shape == ARRAY_SHAPE
    assert arr.dtype == np.float64
    expected = (
        np.arange(int(np.prod(ARRAY_SHAPE)), dtype="float32")
        .reshape(ARRAY_SHAPE)
        .astype(np.float64)
    )
    np.testing.assert_array_equal(np.asarray(arr), expected)


def test_zarr_to_memmap_reuses_existing_cache(tmp_path):
    """A pre-existing ``.npy`` cache must be returned unmodified.

    Guarantees backwards compatibility with caches written by older
    braindecode versions: the fast-path is a single ``stat`` and no
    re-materialisation happens.
    """
    zarr_path = _make_zarr(tmp_path)
    cache_dir = zarr_path.parent / f".{zarr_path.name}_memmap"

    first = _zarr_to_memmap(zarr_path, "recording_0")
    inode_before = os.stat(first).st_ino
    mtime_before = os.stat(first).st_mtime_ns

    second = _zarr_to_memmap(zarr_path, "recording_0")
    assert Path(second) == Path(first)
    assert os.stat(second).st_ino == inode_before
    assert os.stat(second).st_mtime_ns == mtime_before

    # Still no leftover debris.
    leftover = sorted(p.name for p in cache_dir.iterdir())
    assert leftover == ["recording_0.npy"]


@pytest.mark.parametrize(
    "shape,chunks,budget",
    [
        ((4, 103), (4, 16), 64),
        ((3, 5, 19), (2, 3, 7), 80),
        ((2, 3, 5), (9, 8, 7), 8),
        ((0, 5), (2, 3), 64),
    ],
)
def test_iter_zarr_blocks_bounds_and_coverage(shape, chunks, budget):
    """Catch missing subdivision, chunk crossings, overlaps and tail omissions."""
    from braindecode.datasets.base import _iter_zarr_blocks

    coverage = np.zeros(shape, dtype=np.uint8)
    block_bytes = []
    for selection in _iter_zarr_blocks(shape, chunks, max_bytes=budget):
        assert len(selection) == len(shape)
        lengths = []
        for part, size, chunk in zip(selection, shape, chunks):
            assert part.step in (None, 1)
            assert 0 <= part.start < part.stop <= size
            assert part.start // chunk == (part.stop - 1) // chunk
            lengths.append(part.stop - part.start)
        block_bytes.append(int(np.prod(lengths)) * 8)
        coverage[selection] += 1
    assert np.all(coverage == 1)
    if block_bytes:
        assert max(block_bytes) <= budget


@pytest.mark.parametrize("budget", [0, 7])
def test_iter_zarr_blocks_rejects_impossible_budget(budget):
    from braindecode.datasets.base import _iter_zarr_blocks

    with pytest.raises(ValueError, match="max_bytes"):
        list(_iter_zarr_blocks((2, 3), (1, 2), max_bytes=budget))


def _small_zarr(tmp_path, shape=(4, 103), chunks=(4, 16)):
    path = tmp_path / "small.zarr"
    source = np.arange(int(np.prod(shape)), dtype=np.float32).reshape(shape) / 13
    group = zarr.open(str(path), mode="w").create_group("recording_0")
    try:
        array = group.create_array("data", shape=shape, chunks=chunks, dtype="f4")
    except AttributeError:
        array = group.zeros("data", shape=shape, chunks=chunks, dtype="f4")
    array[:] = source
    return path, source


@pytest.mark.parametrize("shape,chunks", [((4, 103), (4, 16)), ((3, 5, 19), (2, 3, 7))])
def test_zarr_temporal_chunks_bound_conversion(tmp_path, monkeypatch, shape, chunks):
    """Catch full-time-axis reads while checking real Zarr conversion exactly."""
    path, source = _small_zarr(tmp_path, shape, chunks)
    selections = []
    original = zarr.Array.__getitem__

    def read_spy(self, selection):
        selections.append(selection)
        return original(self, selection)

    monkeypatch.setattr(zarr.Array, "__getitem__", read_spy)
    actual = np.load(
        _zarr_to_memmap(path, "recording_0", bounded_conversion=True), mmap_mode="r"
    )
    assert actual.dtype == np.float64
    assert np.array_equal(actual, source.astype(np.float64))
    assert selections
    for selection in selections:
        parts = selection if isinstance(selection, tuple) else (selection,)
        parts += (slice(None),) * (len(shape) - len(parts))
        for part, size, chunk in zip(parts, shape, chunks):
            start, stop, step = part.indices(size)
            assert step == 1
            assert stop - start <= chunk
            assert start // chunk == (stop - 1) // chunk


def test_zarr_to_memmap_same_pid_writers_preserve_live_reader(tmp_path, monkeypatch):
    """Catch shared temp names even when both writers reach publication together."""
    import threading

    from braindecode.datasets import base

    path, source = _small_zarr(tmp_path)
    barrier = threading.Barrier(2)
    published = threading.Event()
    reader_ready = threading.Event()
    link = os.link
    results, errors, temp_paths, published_inodes = [], [], [], []
    live_readers = []

    def coordinated_link(src, dst):
        temp_paths.append(Path(src))
        barrier.wait(timeout=10)
        if threading.current_thread().name == "second-writer":
            assert reader_ready.wait(timeout=10)
        try:
            link(src, dst)
        finally:
            if threading.current_thread().name == "first-writer":
                published.set()

    def worker():
        try:
            result = _zarr_to_memmap(path, "recording_0")
            results.append(result)
            published_inodes.append(result.stat().st_ino)
            if threading.current_thread().name == "first-writer":
                live_readers.append(np.load(result, mmap_mode="r"))
                reader_ready.set()
        except Exception as error:  # noqa: BLE001 -- relay thread failures to pytest
            errors.append(error)
            reader_ready.set()

    monkeypatch.setattr(base.os, "link", coordinated_link)
    threads = [
        threading.Thread(target=worker, name=name, daemon=True)
        for name in ("first-writer", "second-writer")
    ]
    try:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=20)
        assert all(not thread.is_alive() for thread in threads)
        assert not errors, errors
        assert published.is_set()
        assert len(results) == 2
        assert len(set(temp_paths)) == 2
        assert len(set(published_inodes)) == 1
        assert results[0].stat().st_ino == published_inodes[0]
        assert np.array_equal(live_readers[0], source.astype(np.float64))
        assert np.array_equal(np.load(results[1]), source.astype(np.float64))
        assert sorted(p.name for p in results[0].parent.iterdir()) == [
            "recording_0.npy"
        ]
    finally:
        barrier.abort()
        reader_ready.set()
        for thread in threads:
            thread.join(timeout=1)


def test_zarr_to_memmap_failed_writer_cleans_only_owned_temp(tmp_path, monkeypatch):
    """A mid-conversion failure must not remove another writer's temporary."""
    path, source = _small_zarr(tmp_path, shape=(8, 103), chunks=(4, 16))
    cache = path.parent / f".{path.name}_memmap"
    cache.mkdir()
    other = cache / "recording_0.other-writer.tmp.npy"
    other.write_bytes(b"owned by another writer")
    original = zarr.Array.__getitem__
    calls = 0

    def failing_read(self, selection):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected read failure")
        return original(self, selection)

    with monkeypatch.context() as patch:
        patch.setattr(zarr.Array, "__getitem__", failing_read)
        with pytest.raises(RuntimeError, match="injected read failure"):
            _zarr_to_memmap(path, "recording_0")
    assert list(cache.iterdir()) == [other]
    assert other.read_bytes() == b"owned by another writer"
    result = _zarr_to_memmap(path, "recording_0")
    assert np.array_equal(np.load(result), source.astype(np.float64))
    assert set(cache.iterdir()) == {other, result}


def test_zarr_to_memmap_reads_legacy_cache_without_source(tmp_path):
    """A legacy float64 cache is reused even when the source is unavailable."""
    path = tmp_path / "legacy.zarr"
    cache = tmp_path / ".legacy.zarr_memmap"
    cache.mkdir()
    expected = np.array([[1.25, -3.5]], dtype=np.float64)
    np.save(cache / "recording_0.npy", expected)
    before = (cache / "recording_0.npy").stat()
    result = _zarr_to_memmap(path, "recording_0")
    assert result.stat().st_ino == before.st_ino
    assert result.stat().st_mtime_ns == before.st_mtime_ns
    assert np.array_equal(np.load(result), expected)


def test_iter_zarr_blocks_default_budget_without_large_allocation():
    """Large source chunks must split even without an explicit byte budget."""
    from braindecode.datasets.base import _iter_zarr_blocks

    shape = (3, 4097, 4099)
    blocks = list(_iter_zarr_blocks(shape, shape))
    sizes = [int(np.prod([s.stop - s.start for s in block])) for block in blocks]
    assert max(sizes) * 8 <= 67108864
    assert sum(sizes) == int(np.prod(shape))


@pytest.mark.parametrize(
    "shape,chunks", [((2, 3), (1,)), ((2, 3), (1, 0)), ((-1, 3), (1, 2))]
)
def test_iter_zarr_blocks_rejects_invalid_geometry(shape, chunks):
    from braindecode.datasets.base import _iter_zarr_blocks

    with pytest.raises(ValueError):
        list(_iter_zarr_blocks(shape, chunks))


@pytest.mark.parametrize("epochs", [False, True])
def test_zarr_cache_preserves_dataset_samples_targets_and_metadata(tmp_path, epochs):
    """Conversion must not rescale EEG units or change window ordering/labels."""
    import mne
    import pandas as pd

    from braindecode.datasets import EEGWindowsDataset, WindowsDataset

    shape, chunks = ((3, 4, 19), (2, 3, 7)) if epochs else ((4, 103), (4, 16))
    path, source = _small_zarr(tmp_path, shape, chunks)
    source *= np.float32(1e-6)
    zarr.open(str(path), mode="a")["recording_0"]["data"][:] = source
    metadata = pd.DataFrame(
        {
            "i_window_in_trial": [0, 1, 2],
            "i_start_in_trial": [0, 31, 77],
            "i_stop_in_trial": [19, 50, 96],
            "target": [2, 0, 1],
            "trial_note": ["first", "second", "third"],
        }
    )
    before = metadata.copy(deep=True)
    info = mne.create_info(["C3", "C4", "P3", "P4"], 128, "eeg")
    cls = WindowsDataset if epochs else EEGWindowsDataset
    dataset = cls._from_zarr(
        path, "recording_0", metadata, {"subject": "synthetic"}, info.to_json_dict()
    )
    assert len(dataset) == 3
    for index, (start, stop, target) in enumerate(
        [(0, 19, 2), (31, 50, 0), (77, 96, 1)]
    ):
        x, y, crop = dataset[index]
        expected = source[index] if epochs else source[:, start:stop]
        assert np.array_equal(x, expected)
        assert y == target
        assert crop == [index, start, stop]
    pd.testing.assert_frame_equal(dataset.metadata, before)
    assert dataset.description["subject"] == "synthetic"
    mne_object = dataset.windows if epochs else dataset.raw
    assert mne_object.info["sfreq"] == 128
    assert mne_object.ch_names == ["C3", "C4", "P3", "P4"]
    assert np.array_equal(mne_object.get_data(), source.astype(np.float64))


@pytest.mark.parametrize("chunks", [(1, 1), (1, 1000000000)])
def test_iter_zarr_blocks_first_yield_does_not_pool_origins(monkeypatch, chunks):
    """Guard both chunk and subdivision scheduling without allocating huge pools."""
    import builtins

    from braindecode.datasets import base

    consumed = 0

    def guarded_range(*args):
        nonlocal consumed
        values = builtins.range(*args)
        if len(values) <= 100:
            return values

        def guarded_origins():
            nonlocal consumed
            for value in values:
                consumed += 1
                if consumed > 8:
                    raise AssertionError("eager origin consumption before first blocks")
                yield value

        return guarded_origins()

    monkeypatch.setattr(base, "range", guarded_range, raising=False)
    blocks = base._iter_zarr_blocks((1, 1000000000), chunks, max_bytes=8)
    try:
        assert next(blocks) == (slice(0, 1), slice(0, 1))
        assert next(blocks) == (slice(0, 1), slice(1, 2))
        assert next(blocks) == (slice(0, 1), slice(2, 3))
        assert consumed <= 8
    finally:
        blocks.close()


def test_iter_zarr_blocks_preserves_chunk_then_subdivision_order():
    from braindecode.datasets.base import _iter_zarr_blocks

    blocks = list(_iter_zarr_blocks((3, 5), (2, 3), max_bytes=16))
    assert blocks == [
        (slice(0, 1), slice(0, 2)),
        (slice(0, 1), slice(2, 3)),
        (slice(1, 2), slice(0, 2)),
        (slice(1, 2), slice(2, 3)),
        (slice(0, 1), slice(3, 5)),
        (slice(1, 2), slice(3, 5)),
        (slice(2, 3), slice(0, 2)),
        (slice(2, 3), slice(2, 3)),
        (slice(2, 3), slice(3, 5)),
    ]
