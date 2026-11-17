"""Local CPU tests: no pretrained checkpoint downloads required."""
import pytest
import torch
from torch import nn

from braindecode.models._pretrained_audit import audit_model


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(4, 3)
        self.fc_norm = nn.LayerNorm(3)
        self.head = nn.Linear(3, 2)


def model():
    m = Tiny()
    m._pretrained_provenance = {
        "repo_id": "example/revision", "revision": "deadbeef",
        "loaded": {"backbone.weight", "backbone.bias"}, "mixed": set()
    }
    return m


def test_loaded_and_expected_head():
    m = model()
    r = audit_model(m, expected_new=("head.",))
    assert r["counts"]["loaded"] == m.backbone.weight.numel() + m.backbone.bias.numel()
    assert r["counts"]["total_materialized"] == sum(p.numel() for p in m.parameters())
    assert r["unexpected_new_or_adapted"] == ["fc_norm.bias", "fc_norm.weight"]


def test_frozen_new_projection():
    m = model()
    for p in m.parameters():
        p.requires_grad_(False)
    assert audit_model(m, expected_new=("head.",))["frozen_unexpected"] == [
        "fc_norm.bias", "fc_norm.weight"
    ]


def test_mixed_parameters_not_fully_loaded():
    m = model()
    m._pretrained_provenance["mixed"] = {"backbone.weight"}
    r = audit_model(m, expected_new=("head.", "fc_norm."))
    assert r["counts"]["adapted"] == m.backbone.weight.numel()
    assert r["counts"]["loaded"] == m.backbone.bias.numel()
    assert r["unexpected_new_or_adapted"] == ["backbone.weight"]


def test_alias_counted_once():
    m = nn.Module()
    m.a = nn.Linear(3, 3, bias=False)
    m.b = m.a
    m._pretrained_provenance = {"loaded": {"b.weight"}, "mixed": set()}
    r = audit_model(m)
    assert r["counts"]["total_materialized"] == 9
    assert r["counts"]["loaded"] == 9
    assert r["parameters"][0]["aliases"] == ["a.weight", "b.weight"]


def test_lazy_parameter():
    m = nn.Module()
    m.layer = nn.LazyLinear(3)
    m._pretrained_provenance = {"loaded": set(), "mixed": set()}
    r = audit_model(m)
    assert r["uninitialized"] == ["layer.bias", "layer.weight"]
    assert r["counts"]["total_materialized"] == 0


def test_missing_provenance():
    with pytest.raises(ValueError, match="no Braindecode"):
        audit_model(nn.Linear(3, 2))


def test_exact_allowed_name_is_not_prefix():
    r = audit_model(model(), expected_new=("head.weight", "fc_norm"))
    assert "head.bias" in r["unexpected_new_or_adapted"]
    assert "head.weight" not in r["unexpected_new_or_adapted"]
    assert "fc_norm.bias" in r["unexpected_new_or_adapted"]


def test_read_only_deterministic():
    m = model()
    before = {name: p.detach().clone() for name, p in m.named_parameters()}
    a, b = audit_model(m), audit_model(m)
    assert a == b
    for name, p in m.named_parameters():
        assert torch.equal(before[name], p)


def test_no_feature_scope_not_assessed():
    r = audit_model(model(), expected_new=("head.",))
    assert r["feature_scope_status"] == "not_assessed"
    assert r["feature_unloaded"] is None


def test_feature_scope_reports_expected_random_projection():
    m = model()
    r = audit_model(m, expected_new=("head.", "fc_norm."), feature_scope=("backbone.", "fc_norm."))
    assert r["unexpected_new_or_adapted"] == []
    assert r["feature_unloaded"] == ["fc_norm.bias", "fc_norm.weight"]


def test_expected_frozen_feature_is_still_not_pretrained():
    m = model()
    for p in m.parameters():
        p.requires_grad_(False)
    r = audit_model(m, expected_new=("head.", "fc_norm."), feature_scope=("backbone.", "fc_norm."))
    assert r["frozen_unexpected"] == []
    assert r["feature_unloaded_frozen"] == ["fc_norm.bias", "fc_norm.weight"]


def test_buffers_are_optional_and_not_frozen_parameters():
    m = model()
    m.register_buffer("running_estimate", torch.ones(3))
    r0 = audit_model(m, feature_scope=("running_estimate",))
    r1 = audit_model(m, include_buffers=True, feature_scope=("running_estimate",))
    assert r0["feature_unloaded"] == []
    assert r1["feature_unloaded"] == ["running_estimate"]
    assert r1["feature_unloaded_frozen"] == []
    assert r1["counts"]["new"] == r0["counts"]["new"] + 3


def test_stale_provenance_keys_not_silently_counted():
    m = model()
    m._pretrained_provenance["loaded"].add("removed_module.weight")
    m._pretrained_provenance["mixed"].add("removed_module.bias")
    r = audit_model(m)
    assert r["stale_loaded_keys"] == ["removed_module.weight"]
    assert r["stale_adapted_keys"] == ["removed_module.bias"]


def test_empty_feature_scope_declared_explicitly():
    r = audit_model(model(), feature_scope=())
    assert r["feature_scope_status"] == "declared"
    assert r["feature_unloaded"] == []


def test_uninitialized_buffer_not_counted():
    m = model()
    m.register_buffer("lazy", nn.parameter.UninitializedBuffer())
    r = audit_model(m, include_buffers=True, feature_scope=("lazy",))
    assert "lazy" in r["uninitialized"]
    assert "lazy" in r["feature_unloaded"]
