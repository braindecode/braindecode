"""Historical conversion/permission compatibility, independent of block scheduler."""

import os
import shutil
import stat
import subprocess

import numpy as np
import pytest

from braindecode.datasets import base

zarr = pytest.importorskip("zarr")


@pytest.fixture
def historical_converter():
    """Use a frozen independent oracle, including in history-free checkouts."""
    from test.unit_tests.datasets._historical_zarr import _zarr_to_memmap

    return _zarr_to_memmap


def _store(directory, shape, chunks):
    directory.mkdir()
    path = directory / "source.zarr"
    source = np.arange(np.prod(shape), dtype=np.float32).reshape(shape) / 13
    source.flat[:4] = [np.nan, -0.0, np.inf, -np.inf]
    group = zarr.open(str(path), mode="w").create_group("recording_0")
    try:
        array = group.create_array("data", shape=shape, chunks=chunks, dtype="f4")
    except AttributeError:
        array = group.zeros("data", shape=shape, chunks=chunks, dtype="f4")
    array[:] = source
    return path


@pytest.mark.parametrize("shape,chunks", [((7, 103), (3, 16)), ((5, 3, 19), (2, 2, 7))])
@pytest.mark.parametrize("bounded", [False, True])
def test_conversion_matches_historical_bytes_and_default_allocations(
    tmp_path, monkeypatch, historical_converter, shape, chunks, bounded
):
    """Check actual conversion calls/array bytes, not merely final values."""
    reference_path = _store(tmp_path / "reference", shape, chunks)
    candidate_path = _store(tmp_path / "candidate", shape, chunks)
    reads, conversions = [], []
    original_read, original_asarray = zarr.Array.__getitem__, np.asarray

    def read(self, selection):
        result = original_read(self, selection)
        reads.append(selection)
        return result

    def asarray(value, *args, **kwargs):
        result = original_asarray(value, *args, **kwargs)
        if isinstance(value, np.ndarray) and kwargs.get("dtype") == np.float64:
            conversions.append((result.shape, result.dtype.str, result.nbytes))
        return result

    with monkeypatch.context() as patch:
        patch.setattr(zarr.Array, "__getitem__", read)
        patch.setattr(np, "asarray", asarray)
        reference = historical_converter(reference_path, "recording_0")
        historical_reads, historical_conversions = reads[:], conversions[:]
        reads.clear()
        conversions.clear()
        kwargs = {"bounded_conversion": True} if bounded else {}
        candidate = base._zarr_to_memmap(candidate_path, "recording_0", **kwargs)
    assert candidate.read_bytes() == reference.read_bytes()
    assert np.load(candidate).tobytes() == np.load(reference).tobytes()
    if bounded:
        assert len(reads) > len(historical_reads)
        assert max(nbytes for _, _, nbytes in conversions) <= 67108864
    else:
        assert reads == historical_reads
        assert conversions == historical_conversions
        assert reads == [
            slice(start, min(start + chunks[0], shape[0]))
            for start in range(0, shape[0], chunks[0])
        ]
        assert historical_conversions


@pytest.mark.parametrize("mask,expected", [(0o002, 0o664), (0o022, 0o644)])
@pytest.mark.parametrize("bounded", [False, True])
def test_new_cache_honors_normal_creation_umask(tmp_path, mask, expected, bounded):
    path = _store(tmp_path / "input", (4, 17), (2, 5))
    old_mask = os.umask(mask)
    try:
        ordinary = tmp_path / "ordinary.npy"
        mm = np.lib.format.open_memmap(
            ordinary, mode="w+", dtype=np.float64, shape=(4, 17)
        )
        mm.flush()
        del mm
        kwargs = {"bounded_conversion": True} if bounded else {}
        result = base._zarr_to_memmap(path, "recording_0", **kwargs)
    finally:
        os.umask(old_mask)
    assert stat.S_IMODE(result.stat().st_mode) == stat.S_IMODE(ordinary.stat().st_mode)
    assert stat.S_IMODE(result.stat().st_mode) == expected


@pytest.mark.parametrize("bounded", [False, True])
def test_existing_cache_permissions_and_inode_are_never_changed(tmp_path, bounded):
    path = tmp_path / "absent.zarr"
    cache = tmp_path / ".absent.zarr_memmap"
    cache.mkdir()
    existing = cache / "recording_0.npy"
    np.save(existing, np.array([[1.0]]))
    existing.chmod(0o600)
    before = existing.stat()
    result = base._zarr_to_memmap(path, "recording_0", bounded_conversion=bounded)
    after = result.stat()
    assert (after.st_ino, after.st_mode, after.st_mtime_ns) == (
        before.st_ino,
        before.st_mode,
        before.st_mtime_ns,
    )


def test_cache_inherits_default_acl_like_ordinary_creation(tmp_path):
    if not shutil.which("setfacl") or not shutil.which("getfacl"):
        pytest.skip("POSIX ACL tools unavailable")
    path = _store(tmp_path / "input", (4, 17), (2, 5))
    cache = path.parent / f".{path.name}_memmap"
    cache.mkdir()
    setup = subprocess.run(
        ["setfacl", "-m", "d:u::rw-,d:g::rw-,d:o::r--", str(cache)],
        capture_output=True,
        text=True,
        check=False,
    )
    if setup.returncode:
        pytest.skip(f"filesystem default ACL unsupported: {setup.stderr}")
    old_mask = os.umask(0o077)
    try:
        ordinary = cache / "ordinary.npy"
        with ordinary.open("xb"):
            pass
        result = base._zarr_to_memmap(path, "recording_0")
    finally:
        os.umask(old_mask)

    def acl(file):
        return subprocess.check_output(["getfacl", "-cp", str(file)], text=True)

    assert acl(result) == acl(ordinary)
    assert stat.S_IMODE(result.stat().st_mode) == 0o664
