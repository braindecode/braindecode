"""Independent historical Zarr oracle (Braindecode BSD-3-Clause)."""

# Verbatim function from df66b2ed5d04b303041017d36c577a99935fc4a7
# Source: braindecode/datasets/base.py::_zarr_to_memmap
# SHA256 of function text (def through final return, including final newline):
# 31e2f601b4d64f012411ebcd7fd21b8e5bfb379688518849e45ddeb1815b8a45
# Keep the body frozen; changes to the candidate must not update this oracle.
import os
from pathlib import Path

import numpy as np
import zarr


def _zarr_to_memmap(zarr_path, group_name):
    """Materialise a zarr array as a float64 ``.npy`` memmap, race-free.

    Decompress the ``group_name`` array inside ``zarr_path`` into a
    float64 ``.npy`` file inside a sibling ``.<zarr>_memmap/`` cache
    directory and return the path to that file.  The dtype is float64
    because MNE objects wrap the memmap zero-copy and MNE requires
    float64 internally.  Callers open the result with
    ``np.load(path, mmap_mode='c')`` so preprocessing writes land in
    anonymous copy-on-write pages and never mutate the on-disk file.

    Concurrency contract
    --------------------
    Safe to call from any number of threads or processes concurrently
    on the same ``(zarr_path, group_name)``, on local POSIX, NFSv3,
    Lustre and SMB.  The published ``.npy`` file is **created exactly
    once and never replaced**: subsequent callers either hit the
    already-materialised file on the fast path or lose the
    publication race and discard their own copy.

    The atomic publication step is ``os.link``, not ``rename``.
    ``os.link(tmp, dst)`` either creates ``dst`` pointing at ``tmp``'s
    inode or fails with :class:`FileExistsError` when ``dst`` already
    exists — the "create only if absent" primitive we need, available
    on every POSIX filesystem.  ``rename`` was the wrong primitive:
    it atomically *replaces* its destination, which unlinks the inode
    that concurrent readers may already have ``mmap``'d.  On NFSv3
    that produced ``.nfsXXXX`` silly-rename files and intermittent
    ``SIGBUS`` when workers page-faulted on the unlinked inode.

    Under ``N`` concurrent writers, every racer materialises the data
    into its own per-pid temp file and then tries to publish it; at
    most one ``os.link`` wins, and the losers silently ``unlink``
    their temp file in a ``finally`` block.  The wasted I/O scales
    with the number of racers; for the workloads braindecode targets
    (a handful of concurrent workers per group) it is negligible.

    Crash recovery: if a writer dies mid-run the only debris is a
    ``<group>.<pid>.tmp.npy`` file.  Such files are namespaced by pid
    so they can never collide with a live writer, they are never
    read by the fast path, and they do not prevent subsequent
    callers from producing a correct result.  Caches written by
    earlier braindecode versions remain valid: the fast path only
    requires ``<group>.npy`` to exist.
    """
    zarr_p = Path(zarr_path)
    cache_dir = zarr_p.parent / f".{zarr_p.name}_memmap"
    npy_path = cache_dir / f"{group_name}.npy"

    # Fast path: the file is already fully materialised.  A single
    # stat() is all the steady-state workload pays.
    if npy_path.exists():
        return npy_path

    cache_dir.mkdir(parents=True, exist_ok=True)
    arr = zarr.open(zarr_path, mode="r")[group_name]["data"]
    tmp_path = cache_dir / f"{group_name}.{os.getpid()}.tmp.npy"
    try:
        # Write chunk-by-chunk to avoid materialising the whole zarr
        # array in RAM.
        mm = np.lib.format.open_memmap(
            tmp_path, mode="w+", dtype=np.float64, shape=arr.shape
        )
        try:
            chunk0 = arr.chunks[0] if hasattr(arr, "chunks") else arr.shape[0]
            for start in range(0, arr.shape[0], chunk0):
                end = min(start + chunk0, arr.shape[0])
                mm[start:end] = np.asarray(arr[start:end], dtype=np.float64)
            mm.flush()
        finally:
            del mm

        # Atomic publication.  ``os.link`` never replaces an existing
        # destination, so the published inode is immutable for the
        # lifetime of the cache directory.  Concurrent readers never
        # see their mmap'd inode vanish from under them.
        try:
            os.link(tmp_path, npy_path)
        except FileExistsError:
            # Another writer beat us to it.  Their file is the
            # canonical one; ours is discarded in the finally block.
            pass
    finally:
        # Always remove our private per-pid temp file.  It is
        # namespaced by pid so this cannot clobber a concurrent
        # writer's in-progress tmp file.
        try:
            os.unlink(tmp_path)
        except FileNotFoundError:
            pass

    return npy_path
