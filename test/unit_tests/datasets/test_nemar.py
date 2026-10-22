# Authors: Adam Mounir <am91ris@gmail.com>
#
# License: BSD (3-clause)

import functools
import http.server
import json
import os
import pickle  # nosec B403: the test pickles its own windows
import threading

import numpy as np
import pandas as pd
import pytest

zarr = pytest.importorskip("zarr")
pytest.importorskip("fsspec")
pytest.importorskip("pyarrow")

from braindecode.datasets import nemar  # noqa: E402
from braindecode.datasets.nemar import NEMARDataset  # noqa: E402
from braindecode.preprocessing import create_fixed_length_windows  # noqa: E402

SFREQ = 250.0
SHARD = 300  # samples per shard, NEMAR uses 300 s; small here to test boundaries


def _write_store(path, digital, scale, offset, channels, group="eeg_250hz"):
    root = zarr.open_group(str(path), mode="w", zarr_format=3)
    grp = root.create_group(group)
    grp.attrs.update(
        {
            "modality": "EEG",
            "rate": SFREQ,
            "n_channels": len(channels),
            "n_samples": digital.shape[1],
            "channels": channels,
        }
    )
    arr = grp.create_array(
        "0",
        shape=digital.shape,
        dtype="int16",
        chunks=(digital.shape[0], 100),
        shards=(digital.shape[0], SHARD),
    )
    arr[:] = digital
    arr.attrs.update({"scale": list(scale), "offset": list(offset)})


@pytest.fixture
def nemar_dataset(tmp_path, monkeypatch):
    """A fake NEMAR dataset with two recordings, laid out as on zarr.nemar.org."""
    rng = np.random.default_rng(0)
    dataset_id = "nm000000"
    data_base = tmp_path / dataset_id / "zarr"
    channels = [
        {"label": "Cz", "channel_type": "EEG", "unit": "µV", "row_index": 0},
        {"label": "Pz", "channel_type": "EEG", "unit": "mV", "row_index": 1},
        {"label": "HEOG", "channel_type": "HEOG", "unit": "µV", "row_index": 2},
    ]
    scale = np.array([0.5, 0.01, 2.0])
    offset = np.array([10.0, -1.0, 0.0])
    stores, physical = [], {}
    for subject, n_times in [("01", 1000), ("02", 700)]:
        path = f"sub-{subject}/eeg/sub-{subject}_task-rest_eeg.vhdr"
        zarr_path = path.replace(".vhdr", ".zarr")
        digital = rng.integers(-1000, 1000, size=(3, n_times), dtype=np.int16)
        _write_store(data_base / zarr_path, digital, scale, offset, channels)
        units = np.array([1e-6, 1e-3, 1e-6])[:, None]
        physical[subject] = (digital * scale[:, None] + offset[:, None]) * units
        stores.append(
            {
                "path": path,
                "zarr": zarr_path,
                "groups": [{"name": "eeg_250hz", "modality": "EEG", "rate": SFREQ}],
                "power_line_frequency": 50.0,
            }
        )
    index = {"data_base": f"{data_base}/", "stores": stores}
    (data_base / "index.json").write_text(json.dumps(index))
    pd.DataFrame(
        {
            "store_path": [stores[0]["zarr"]] * 5,
            "onset_s": [0.5, 1.0, 1.5, 2.0, np.nan],
            "duration_s": [0.1, 0.1, np.nan, 0.1, 0.1],
            "trial_type": ["stimulus", "stimulus", "response", None, "stimulus"],
            "value": [1, 2, 3, 4, 5],
        }
    ).to_parquet(data_base / "events.parquet")
    monkeypatch.setattr(nemar, "NEMAR_ZARR_URL", str(tmp_path))
    return dataset_id, physical


def test_nemar_dataset_reads_physical_values(nemar_dataset):
    dataset_id, physical = nemar_dataset
    ds = NEMARDataset(dataset_id)
    assert len(ds.datasets) == 2
    assert list(ds.description["subject"]) == ["01", "02"]
    assert list(ds.description["task"]) == ["rest", "rest"]
    raw = ds.datasets[0].raw
    assert raw.info["sfreq"] == SFREQ
    assert raw.info["line_freq"] == 50.0
    assert raw.get_channel_types() == ["eeg", "eeg", "eog"]
    np.testing.assert_allclose(raw.get_data(), physical["01"])
    # A read across a shard boundary, and one inside a single shard.
    np.testing.assert_allclose(
        raw.get_data(start=250, stop=650), physical["01"][:, 250:650]
    )
    np.testing.assert_allclose(
        raw.get_data(picks=["Pz"], start=10, stop=20), physical["01"][1:2, 10:20]
    )


def test_nemar_dataset_filters(nemar_dataset):
    dataset_id, _ = nemar_dataset
    ds = NEMARDataset(dataset_id, subjects="02")
    assert list(ds.description["subject"]) == ["02"]
    assert len(NEMARDataset(dataset_id, modalities="eeg").datasets) == 2
    with pytest.raises(ValueError, match="No recording"):
        NEMARDataset(dataset_id, modalities="IEEG")


def test_nemar_dataset_annotations_follow_mne_bids(nemar_dataset):
    dataset_id, _ = nemar_dataset
    ds = NEMARDataset(dataset_id)
    annotations = ds.datasets[0].raw.annotations
    # "stimulus" has two values, so it is split like mne_bids does; a missing
    # trial_type falls back to the value; an event without onset is dropped.
    assert list(annotations.description) == [
        "stimulus/1",
        "stimulus/2",
        "response",
        "4",
    ]
    np.testing.assert_allclose(annotations.onset, [0.5, 1.0, 1.5, 2.0])
    np.testing.assert_allclose(annotations.duration, [0.1, 0.1, 0.0, 0.1])
    assert len(ds.datasets[1].raw.annotations) == 0


def test_nemar_dataset_windows_and_pickle(nemar_dataset):
    dataset_id, physical = nemar_dataset
    ds = NEMARDataset(dataset_id)
    windows = create_fixed_length_windows(
        ds,
        window_size_samples=200,
        window_stride_samples=200,
        on_last_window="drop",
        preload=False,
    )
    # A pickled copy reads the data itself, as a spawned data loader worker does.
    windows = pickle.loads(pickle.dumps(windows))  # nosec B301
    X, _, (_, start, stop) = windows[2]
    np.testing.assert_allclose(X, physical["01"][:, start:stop], rtol=1e-6)
    X, _, (_, start, stop) = windows[len(windows) - 1]
    np.testing.assert_allclose(X, physical["02"][:, start:stop], rtol=1e-6)


class _RangeHandler(http.server.SimpleHTTPRequestHandler):
    """Serves files with byte ranges, as S3 does; counts the bytes it sends,
    and answers 503 to the first `failures` requests."""

    failures = 0
    sent = 0

    def log_message(self, *args):
        pass

    def do_GET(self):
        if type(self).failures > 0:
            type(self).failures -= 1
            self.send_error(503)
            return
        path = self.translate_path(self.path)
        if not os.path.isfile(path):
            # S3 answers 403 rather than 404 for a missing object.
            self.send_error(403)
            return
        with open(path, "rb") as f:
            body = f.read()
        status = 200
        if "Range" in self.headers:
            first, last = self.headers["Range"][len("bytes=") :].split("-")
            if not first:  # suffix range: the last `last` bytes
                body = body[-int(last) :]
            else:
                body = body[int(first) : int(last) + 1 if last else None]
            status = 206
        type(self).sent += len(body)
        self.send_response(status)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture
def nemar_http(nemar_dataset, tmp_path, monkeypatch):
    """The fake dataset served over HTTP, as zarr.nemar.org serves it."""
    dataset_id, physical = nemar_dataset
    handler = functools.partial(_RangeHandler, directory=str(tmp_path))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_port}"
    index_path = tmp_path / dataset_id / "zarr" / "index.json"
    index = json.loads(index_path.read_text())
    index["data_base"] = f"{url}/{dataset_id}/zarr/"
    index_path.write_text(json.dumps(index))
    monkeypatch.setattr(nemar, "NEMAR_ZARR_URL", url)
    yield dataset_id, physical
    server.shutdown()


def test_nemar_dataset_reads_over_http(nemar_http):
    dataset_id, physical = nemar_http
    ds = NEMARDataset(dataset_id)
    # sub-02 has 700 samples: its last shard is partial.
    for raw, subject in zip([d.raw for d in ds.datasets], ["01", "02"]):
        np.testing.assert_allclose(raw.get_data(), physical[subject])


def test_nemar_dataset_reads_only_the_chunks_it_needs(nemar_http):
    dataset_id, physical = nemar_http
    raw = NEMARDataset(dataset_id, subjects="01").datasets[0].raw
    raw.get_data(start=0, stop=1)  # opens the array: metadata, first chunk
    _RangeHandler.sent = 0
    # 50 samples inside one 100-sample chunk of a 300-sample shard: the
    # shard index and that chunk, not the shard.
    np.testing.assert_allclose(
        raw.get_data(start=410, stop=460), physical["01"][:, 410:460]
    )
    chunk_bytes = 3 * 100 * 2
    assert 0 < _RangeHandler.sent < 3 * chunk_bytes


def test_nemar_dataset_retries_failed_reads(nemar_http, monkeypatch):
    dataset_id, physical = nemar_http
    raw = NEMARDataset(dataset_id, subjects="01").datasets[0].raw
    monkeypatch.setattr(nemar.time, "sleep", lambda s: None)
    monkeypatch.setattr(_RangeHandler, "failures", 2)
    np.testing.assert_allclose(
        raw.get_data(start=250, stop=650), physical["01"][:, 250:650]
    )
    assert _RangeHandler.failures == 0


def test_nemar_dataset_does_not_retry_client_errors(nemar_http, tmp_path, monkeypatch):
    aiohttp = pytest.importorskip("aiohttp")
    dataset_id, _ = nemar_http
    raw = NEMARDataset(dataset_id, subjects="01").datasets[0].raw
    store = tmp_path / dataset_id / "zarr" / "sub-01/eeg/sub-01_task-rest_eeg.zarr"
    (store / "eeg_250hz/0/c/0/1").unlink()

    def sleep(s):
        raise AssertionError("a 4xx was retried")

    monkeypatch.setattr(nemar.time, "sleep", sleep)
    with pytest.raises(aiohttp.ClientResponseError) as e:
        raw.get_data(start=300, stop=600)
    assert e.value.status == 403


def test_nemar_dataset_data_loader_workers(nemar_http):
    """Forked workers open their own array: an HTTP one opened in the parent
    is bound to the parent's event loop."""
    torch = pytest.importorskip("torch")
    dataset_id, physical = nemar_http
    ds = NEMARDataset(dataset_id, subjects="01")
    ds.datasets[0].raw.get_data(start=0, stop=10)  # opened in the parent
    windows = create_fixed_length_windows(
        ds,
        window_size_samples=200,
        window_stride_samples=200,
        on_last_window="drop",
        preload=False,
    )
    loader = torch.utils.data.DataLoader(
        windows, batch_size=2, num_workers=2, multiprocessing_context="fork"
    )
    X = torch.cat([batch[0] for batch in loader]).numpy()
    expected = physical["01"][:, : 5 * 200].reshape(3, 5, 200).transpose(1, 0, 2)
    np.testing.assert_allclose(X, expected, rtol=1e-6)


def test_nemar_dataset_incomplete_conversion(nemar_dataset, tmp_path):
    dataset_id, _ = nemar_dataset
    data_base = tmp_path / dataset_id / "zarr"
    (data_base / "events.parquet").unlink()
    index = json.loads((data_base / "index.json").read_text())
    index["pending_count"] = 1
    (data_base / "index.json").write_text(json.dumps(index))
    with pytest.warns(UserWarning):
        ds = NEMARDataset(dataset_id)
    assert len(ds.datasets[0].raw.annotations) == 0


def test_nemar_dataset_unknown_id(nemar_dataset):
    with pytest.raises(ValueError, match="serves no Zarr copy"):
        NEMARDataset("nm999999")


def test_make_info_meg_types_and_units():
    channels = [
        dict(label="MEG 001", channel_type="MEGGRADAXIAL", unit="T"),
        dict(label="MEG 002", channel_type="MEGGRADPLANAR", unit="fT/cm"),
        dict(label="REF 001", channel_type="MEGREFMAG", unit="fT"),
        dict(label="TRIG", channel_type="TRIG", unit="V"),
    ]
    info, _ = nemar._make_info(
        dict(channels=channels, rate=250.0),
        dict(scale=[2.0] * 4, offset=[0.0] * 4),
    )
    assert info.get_channel_types() == ["mag", "grad", "ref_meg", "stim"]
    np.testing.assert_allclose(
        [ch["cal"] for ch in info["chs"]], [2.0, 2e-13, 2e-15, 2.0]
    )
