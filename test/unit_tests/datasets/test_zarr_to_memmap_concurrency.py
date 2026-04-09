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

from braindecode.datasets.base import _zarr_to_memmap  # noqa: E402

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
    arr[:] = np.arange(
        int(np.prod(ARRAY_SHAPE)), dtype="float32"
    ).reshape(ARRAY_SHAPE)
    return zarr_path


# Global barrier handle, populated in each pool worker by the pool
# initializer.  ``multiprocessing`` synchronisation primitives cannot
# be passed as ``starmap`` args under the ``spawn`` start method, so
# the initializer is the canonical way to share them.
_BARRIER: "mp.synchronize.Barrier | None" = None


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
    checksum = (
        float(arr[0, 0]) + float(arr[-1, -1]) + float(arr.sum())
    )
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
    with ctx.Pool(
        n_workers, initializer=_init_worker, initargs=(barrier,)
    ) as pool:
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
    assert len(checksums) == 1, (
        f"workers disagreed on file content: {checksums}"
    )

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
