"""Stream recordings from the NEMAR Zarr serving copy.

NEMAR (https://nemar.org) publishes, next to each BIDS dataset, a derived
Zarr v3 copy of its recordings that can be read slice by slice over HTTP:
https://docs.nemar.org/platform/zarr/. This module wraps each recording in
a lazy :class:`mne.io.BaseRaw`, so the rest of braindecode (windowing,
preprocessing, data loaders) works unchanged while only the samples that
are actually read get downloaded.
"""

# Authors: Adam Mounir <am91ris@gmail.com>
#
# License: BSD (3-clause)

from __future__ import annotations

import http.client
import json
import logging
import os
import threading
import time
import urllib.error
import urllib.request
import warnings
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

import mne
import numpy as np
import pandas as pd
from mne._fiff.utils import _mult_cal_one
from mne.io import BaseRaw
from mne.utils import _soft_import
from mne_bids import get_entities_from_fname
from mne_bids.utils import _get_ch_type_mapping

from .base import BaseConcatDataset, RawDataset

zarr = _soft_import("zarr", purpose="streaming NEMAR datasets", strict=False)
fsspec = _soft_import("fsspec", purpose="streaming NEMAR datasets", strict=False)

log = logging.getLogger(__name__)

NEMAR_ZARR_URL = "https://zarr.nemar.org"

# Shards are fetched as parallel byte ranges, each with a deadline: some S3
# connections stall at a few tens of kB/s while others run at MB/s, so a
# whole-shard request can outlast any timeout, again and again when retried.
# A 4 MiB range takes 1-2 s on a healthy connection; one still running after
# PART_DEADLINE s is dropped and asked anew, on a new connection.
PART_SIZE = 4 * 1024**2
PART_DEADLINE = 20.0
PART_RETRIES = 10
N_CONNECTIONS = 8

# mne_bids leaves MEG out of this mapping (it reads MEG types from the raw
# file), so the BIDS MEG types are added here. Axial gradiometers are "mag"
# in MNE, as in its KIT and CTF readers.
_BIDS_TO_MNE_TYPE = {
    **_get_ch_type_mapping("bids", "mne"),
    "MEGMAG": "mag",
    "MEGGRADAXIAL": "mag",
    "MEGGRADPLANAR": "grad",
    "MEGREFMAG": "ref_meg",
    "MEGREFGRADAXIAL": "ref_meg",
    "MEGREFGRADPLANAR": "ref_meg",
}
_UNIT_TO_SI = {
    "V": 1.0,
    "mV": 1e-3,
    "µV": 1e-6,
    "μV": 1e-6,
    "uV": 1e-6,
    "nV": 1e-9,
    "T": 1.0,
    "pT": 1e-12,
    "fT": 1e-15,
    "T/m": 1.0,
    "T/cm": 1e2,
    "fT/cm": 1e-13,
}

# Decoded int16 shards, per process: (url, group, shard index) -> array.
_SHARDS: OrderedDict = OrderedDict()
_ARRAYS: dict = {}
_LOCK = threading.Lock()
_PID = os.getpid()


def _get_shard_cache():
    """Per-process caches; a forked DataLoader worker starts with empty ones."""
    global _LOCK, _PID
    if _PID != os.getpid():
        _LOCK = threading.Lock()
        _SHARDS.clear()
        _ARRAYS.clear()
        _PID = os.getpid()
    return _SHARDS, _ARRAYS


def _open_level0(url, group):
    _, arrays = _get_shard_cache()
    key = (url, group)
    if key not in arrays:
        # zarr_format=3 skips the v2 metadata probes, which NEMAR's S3
        # answers with 403 rather than 404.
        arrays[key] = zarr.open_array(url, path=f"{group}/0", mode="r", zarr_format=3)
    return arrays[key]


class _NEMARRaw(BaseRaw):
    """Raw backed by one channel group of a NEMAR Zarr store, read lazily.

    Samples are fetched one shard (NEMAR stores 300 s per shard) at a time,
    as parallel byte ranges over HTTP, and the decoded shards are kept in a
    per-process LRU cache of at most ``cache_size`` bytes.
    """

    def __init__(
        self,
        url,
        group,
        info,
        n_times,
        offset_over_scale,
        shard_len,
        cache_size,
        annotations=None,
    ):
        super().__init__(
            info,
            preload=False,
            last_samps=[n_times - 1],
            filenames=[None],
            raw_extras=[
                dict(
                    url=url,
                    group=group,
                    offset_over_scale=offset_over_scale,
                    shard_len=shard_len,
                    cache_size=cache_size,
                )
            ],
            orig_format="short",
        )
        if annotations is not None:
            self.set_annotations(annotations)

    def _read_segment_file(self, data, idx, fi, start, stop, cals, mult):
        extra = self._raw_extras[fi]
        shard_len = extra["shard_len"]
        first, last = start // shard_len, (stop - 1) // shard_len
        block = np.concatenate(
            [_get_shard(extra, i) for i in range(first, last + 1)], axis=1
        )
        block = block[:, start - first * shard_len : stop - first * shard_len]
        # physical = (digital + offset / scale) * scale, the scale (and the
        # unit conversion to SI) being the channel calibration in info.
        block = block + extra["offset_over_scale"][:, np.newaxis]
        _mult_cal_one(data, block, idx, cals, mult)


def _get_shard(extra, i):
    shards, _ = _get_shard_cache()
    key = (extra["url"], extra["group"], i)
    with _LOCK:
        if key in shards:
            shards.move_to_end(key)
            return shards[key]
    shard_len = extra["shard_len"]
    if extra["url"].startswith(("http://", "https://")):
        shard = _fetch_shard_http(extra["url"], extra["group"], i, shard_len)
    else:
        # Reading a whole shard costs one read; zarr fetches the shard object
        # at once instead of one per inner chunk.
        arr = _open_level0(extra["url"], extra["group"])
        shard = arr[:, i * shard_len : (i + 1) * shard_len]
    if extra["cache_size"]:
        with _LOCK:
            shards[key] = shard
            while sum(s.nbytes for s in shards.values()) > extra["cache_size"]:
                shards.popitem(last=False)
    return shard


def _level0_meta(url, group):
    _, arrays = _get_shard_cache()
    key = (url, group, "zarr.json")
    if key not in arrays:
        with urllib.request.urlopen(
            f"{url}/{group}/0/zarr.json", timeout=PART_DEADLINE
        ) as f:
            arrays[key] = json.load(f)
    return arrays[key]


def _fetch_range(url, start, stop):
    """Bytes [start, stop) of url, asked anew when they miss PART_DEADLINE."""
    for attempt in range(PART_RETRIES):
        t0 = time.monotonic()
        try:
            request = urllib.request.Request(
                url, headers={"Range": f"bytes={start}-{stop - 1}"}
            )
            out = bytearray()
            with urllib.request.urlopen(request, timeout=PART_DEADLINE) as f:
                while chunk := f.read(256 * 1024):
                    out += chunk
                    if time.monotonic() - t0 > PART_DEADLINE:
                        raise TimeoutError(f"{len(out)} B in {PART_DEADLINE} s")
            if len(out) != stop - start:
                raise OSError(f"{len(out)} B for a range of {stop - start}")
            return bytes(out)
        except (OSError, http.client.HTTPException) as e:  # stalls, resets, 5xx
            client_error = isinstance(e, urllib.error.HTTPError) and e.code < 500
            if client_error or attempt == PART_RETRIES - 1:
                raise
            log.warning(
                "%s bytes %d-%d: %r, retry %d", url, start, stop, e, attempt + 1
            )
            time.sleep(min(2**attempt, 30))


def _fetch_shard_http(url, group, i, shard_len):
    """Shard i of the level-0 array: its bytes in parallel ranges, decoded by zarr."""
    meta = _level0_meta(url, group)
    n_channels, n_times = meta["shape"]
    width = min(shard_len, n_times - i * shard_len)
    sep = meta.get("chunk_key_encoding", {}).get("configuration", {})
    sep = sep.get("separator", "/")
    shard_url = f"{url}/{group}/0/c{sep}0{sep}{i}"
    try:
        head = urllib.request.Request(shard_url, method="HEAD")
        with urllib.request.urlopen(head, timeout=PART_DEADLINE) as f:
            size = int(f.headers["Content-Length"])
    except urllib.error.HTTPError as e:
        # A shard never written holds the fill value; NEMAR's S3 answers 403
        # rather than 404 for a missing object.
        if e.code in (403, 404):
            return np.full(
                (n_channels, width), meta.get("fill_value", 0), meta["data_type"]
            )
        raise
    bounds = [(a, min(a + PART_SIZE, size)) for a in range(0, size, PART_SIZE)]
    with ThreadPoolExecutor(N_CONNECTIONS) as pool:
        data = b"".join(pool.map(lambda b: _fetch_range(shard_url, *b), bounds))
    # The shard as the only chunk of a one-shard array: same codecs, so zarr
    # decodes it, and trims the last, partial shard, as it would remotely.
    one = dict(meta, shape=[n_channels, width])
    buffer = zarr.core.buffer.default_buffer_prototype().buffer
    store = zarr.storage.MemoryStore(
        {
            "zarr.json": buffer.from_bytes(json.dumps(one).encode()),
            f"c{sep}0{sep}0": buffer.from_bytes(data),
        }
    )
    return zarr.open_array(store, mode="r", zarr_format=3)[:]


def _make_info(group_attrs, array_attrs):
    chans = group_attrs["channels"]
    ch_names = [c["label"] for c in chans]
    ch_types = [
        _BIDS_TO_MNE_TYPE.get(str(c.get("channel_type", "MISC")).upper(), "misc")
        for c in chans
    ]
    info = mne.create_info(ch_names, group_attrs["rate"], ch_types)
    scale = np.asarray(array_attrs["scale"], dtype=np.float64)
    offset = np.asarray(array_attrs["offset"], dtype=np.float64)
    for ch, c, s in zip(info["chs"], chans, scale):
        ch["cal"] = s * _UNIT_TO_SI.get(str(c.get("unit", "")).strip(), 1.0)
    if group_attrs.get("power_line_frequency"):
        info["line_freq"] = group_attrs["power_line_frequency"]
    return info, offset / scale


def _fetch_json(fs, urls):
    return {u: json.loads(v) for u, v in fs.cat(urls).items()}


def _load_events(fs, data_base, store_paths):
    # Only the selected recordings' rows: the parquet covers the whole dataset
    # and can weigh tens of MB, while its row groups are read on demand.
    try:
        with fs.open(data_base + "events.parquet", "rb") as f:
            return pd.read_parquet(
                f,
                columns=["store_path", "onset_s", "duration_s", "trial_type", "value"],
                filters=[("store_path", "in", list(store_paths))],
            )
    except FileNotFoundError:
        return None


def _annotations(events, store_path):
    if events is None:
        return None
    ev = events[events["store_path"] == store_path]
    ev = ev[ev["onset_s"].notna() & (ev["trial_type"].notna() | ev["value"].notna())]
    if not len(ev):
        return None
    # Same labels as mne_bids.read_raw_bids: trial_type, falling back to
    # value, and "trial_type/value" when a trial_type has several values.
    trial_type = ev["trial_type"].astype("string")
    value = ev["value"].map(_format_value).astype("string")
    labels = trial_type.fillna(value)
    n_values = value.groupby(trial_type).transform("nunique")
    multi = trial_type.notna() & value.notna() & (n_values > 1)
    labels[multi] = trial_type[multi] + "/" + value[multi]
    return mne.Annotations(
        onset=ev["onset_s"].to_numpy(dtype=float),
        duration=np.clip(ev["duration_s"].fillna(0).to_numpy(dtype=float), 0, None),
        description=labels.to_numpy(dtype=str),
    )


def _format_value(value):
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return value if pd.isna(value) else str(value)


class NEMARDataset(BaseConcatDataset):
    """Stream a dataset from the NEMAR Zarr serving copy.

    NEMAR serves every converted recording of its datasets as a Zarr v3
    store that can be read in slices over HTTP, without downloading the
    BIDS files [NEMARZarr]_. Each channel group of each recording (one
    group per modality, e.g. ``eeg_250hz``) becomes a
    :class:`~braindecode.datasets.RawDataset` whose raw is lazy: building
    the dataset only downloads metadata, and samples are fetched the first
    time they are read, e.g. by
    :func:`~braindecode.preprocessing.create_windows_from_events` with
    ``preload=False`` and a data loader. Preprocessing that needs a whole
    recording (``raw.load_data()``) downloads that recording only.

    The serving copy is derived from the BIDS source: it is resampled to a
    rate set per modality (e.g. 250 Hz for EEG and MEG, the group name
    gives it) and stored as int16. Use :class:`~braindecode.datasets.BIDSDataset`
    on the downloaded BIDS files when the original samples matter.

    .. versionadded:: 1.9

    Parameters
    ----------
    dataset_id : str
        NEMAR dataset identifier, e.g. ``"nm000132"``. Converted datasets are
        listed at https://zarr.nemar.org/catalog.json.
    subjects, sessions, tasks, runs : str | list of str | None
        Keep only recordings with these BIDS entities. ``None`` keeps all.
    modalities : str | list of str | None
        Keep only channel groups of these modalities (e.g. ``"EEG"``,
        ``"IEEG"``, ``"EMG"``). ``None`` keeps all.
    target_name : str | tuple | None
        Name(s) of the description field(s) used as target.
    cache_size : int
        Bytes of decoded shards to keep in memory per process (and per
        data loader worker). ``0`` disables the cache.

    References
    ----------
    .. [NEMARZarr] NEMAR Zarr serving copy,
       https://docs.nemar.org/platform/zarr/
    """

    def __init__(
        self,
        dataset_id: str,
        subjects: str | list[str] | None = None,
        sessions: str | list[str] | None = None,
        tasks: str | list[str] | None = None,
        runs: str | list[str] | None = None,
        modalities: str | list[str] | None = None,
        target_name: str | tuple[str, ...] | None = None,
        cache_size: int = 512 * 1024**2,
    ):
        if zarr is False or fsspec is False:
            raise ImportError(
                "zarr and fsspec are required: pip install braindecode[nemar]"
            )
        index_url = f"{NEMAR_ZARR_URL}/{dataset_id}/zarr/index.json"
        try:
            with fsspec.open(index_url, "rb") as f:
                index = json.load(f)
        except FileNotFoundError as e:
            raise ValueError(
                f"NEMAR serves no Zarr copy of {dataset_id!r}; converted datasets "
                f"are listed at {NEMAR_ZARR_URL}/catalog.json."
            ) from e
        data_base = index["data_base"]
        fs, _ = fsspec.core.url_to_fs(data_base)

        wanted = {
            entity: {str(v)} if isinstance(v, (str, int)) else {str(x) for x in v}
            for entity, v in dict(
                subject=subjects, session=sessions, task=tasks, run=runs
            ).items()
            if v is not None
        }
        if isinstance(modalities, str):
            modalities = [modalities]
        wanted_modalities = (
            None if modalities is None else {m.upper() for m in modalities}
        )

        selected = []
        for store in index["stores"]:
            entities = get_entities_from_fname(store["path"], on_error="ignore")
            if any(str(entities.get(k)) not in v for k, v in wanted.items()):
                continue
            for group in store["groups"]:
                if (
                    wanted_modalities is None
                    or group["modality"].upper() in wanted_modalities
                ):
                    selected.append((store, group, entities))
        if not selected:
            raise ValueError(f"No recording of {dataset_id} matches the filters.")
        if index.get("pending_count") or index.get("failure_count"):
            warnings.warn(
                f"{dataset_id}: {index.get('pending_count', 0)} recordings are not "
                f"converted yet and {index.get('failure_count', 0)} failed; they "
                "are not served.",
                UserWarning,
            )

        # Two small metadata documents per group, fetched concurrently.
        urls = []
        for store, group, _ in selected:
            base = f"{data_base}{store['zarr']}/{group['name']}"
            urls += [f"{base}/zarr.json", f"{base}/0/zarr.json"]
        meta = _fetch_json(fs, urls)
        events = _load_events(
            fs, data_base, {store["zarr"] for store, _, _ in selected}
        )

        datasets = []
        for store, group, entities in selected:
            url = data_base + store["zarr"]
            base = f"{url}/{group['name']}"
            group_attrs = meta[f"{base}/zarr.json"]["attributes"]
            array_meta = meta[f"{base}/0/zarr.json"]
            group_attrs.setdefault(
                "power_line_frequency", store.get("power_line_frequency")
            )
            info, offset_over_scale = _make_info(group_attrs, array_meta["attributes"])
            raw = _NEMARRaw(
                url,
                group["name"],
                info,
                n_times=array_meta["shape"][1],
                offset_over_scale=offset_over_scale,
                shard_len=array_meta["chunk_grid"]["configuration"]["chunk_shape"][1],
                cache_size=cache_size,
                annotations=_annotations(events, store["zarr"]),
            )
            description = {
                "dataset": dataset_id,
                **{k: v for k, v in entities.items() if v is not None},
                "group": group["name"],
                "modality": group["modality"],
                "path": store["path"],
            }
            datasets.append(RawDataset(raw, description, target_name=target_name))
        super().__init__(datasets)
