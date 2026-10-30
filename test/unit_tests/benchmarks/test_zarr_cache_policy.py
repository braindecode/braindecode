"""The private benchmark must explicitly select the bounded experiment."""

from benchmarks.acceleration import bench_zarr_cache


def test_candidate_converter_explicitly_opts_into_bounded_policy(monkeypatch):
    calls = []

    def convert(path, group, *, bounded_conversion=False):
        calls.append((path, group, bounded_conversion))
        return "cache.npy"

    monkeypatch.setattr(bench_zarr_cache.base, "_zarr_to_memmap", convert)
    converter = bench_zarr_cache._converter("candidate", bench_zarr_cache.BASELINE)
    assert converter("source.zarr", "recording_0") == "cache.npy"
    assert calls == [("source.zarr", "recording_0", True)]
