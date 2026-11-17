"""Optional exact-source Braindecode API integration tests (no network).

Requires Braindecode and huggingface_hub in the local environment. A local
roundtrip is not published-checkpoint parity or scientific performance evidence.
"""
import pytest

from braindecode.models._pretrained_audit import audit_model

pytest.importorskip("braindecode")
pytest.importorskip("huggingface_hub")


def test_eegnet_local_roundtrip_marks_all_parameters_loaded(tmp_path):
    from braindecode.models import EEGNet

    EEGNet(n_chans=4, n_times=128, n_outputs=2).save_pretrained(tmp_path)
    model = EEGNet.from_pretrained(tmp_path, local_files_only=True)
    report = audit_model(model)
    assert report["loaded_fraction"] == 1.0
    assert report["unexpected_new_or_adapted"] == []


def test_eegnet_changed_head_is_expected_but_new(tmp_path):
    from braindecode.models import EEGNet

    EEGNet(n_chans=4, n_times=128, n_outputs=2).save_pretrained(tmp_path)
    model = EEGNet.from_pretrained(tmp_path, n_outputs=3, local_files_only=True)
    report = audit_model(model, expected_new=("final_layer.",))
    assert report["unexpected_new_or_adapted"] == []
    assert report["loaded_fraction"] < 1.0
    assert any(p["status"] == "new" and p["expected_new"] for p in report["parameters"])
