"""Exercise the installed wheel, its package data and a spawned loader offline."""

from importlib.metadata import distribution
from pathlib import Path
from tempfile import TemporaryDirectory


def main():
    import mne
    import numpy as np
    import torch
    from packaging.version import Version
    from torch.utils.data import DataLoader

    import braindecode
    from braindecode.datasets import BaseConcatDataset, RawDataset
    from braindecode.models import EEGNet
    from braindecode.models.util import get_summary_table
    from braindecode.preprocessing import create_fixed_length_windows

    wheel = distribution("braindecode")
    assert (
        Path(braindecode.__file__).resolve()
        == wheel.locate_file("braindecode/__init__.py").resolve()
    )
    assert Version(wheel.version) == Version(braindecode.__version__)
    assert "EEGNet" in get_summary_table().index

    samples = np.arange(512, dtype=float).reshape(2, 256) / 512
    info = mne.create_info(["C3", "C4"], 128, "eeg")
    raw = mne.io.RawArray(samples, info, verbose=False)
    dataset = BaseConcatDataset([RawDataset(raw, {"label": 1}, target_name="label")])
    windows = create_fixed_length_windows(dataset, window_size_samples=128)
    loader = DataLoader(
        windows, batch_size=2, num_workers=1, multiprocessing_context="spawn"
    )
    x, y, (_, starts, stops) = next(iter(loader))
    expected = torch.tensor(
        np.stack((samples[:, :128], samples[:, 128:])), dtype=torch.float32
    )
    torch.testing.assert_close(x, expected)
    assert (
        y.tolist() == [1, 1]
        and starts.tolist() == [0, 128]
        and stops.tolist() == [128, 256]
    )

    model = EEGNet(n_chans=2, n_outputs=2, n_times=128).eval()
    assert "EEGNet" in model._repr_html_()
    with torch.inference_mode(), TemporaryDirectory() as folder:
        logits = model(x)
        assert logits.shape == (2, 2) and torch.isfinite(logits).all()
        model.save_pretrained(folder)
        loaded = EEGNet.from_pretrained(folder, strict=True).eval()
        torch.testing.assert_close(loaded(x), logits)
    print(f"Verified braindecode {wheel.version} in {wheel.locate_file('braindecode')}")


if __name__ == "__main__":
    main()
