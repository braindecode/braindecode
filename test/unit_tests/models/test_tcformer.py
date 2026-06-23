# License: MIT
import os

import pytest
import torch

from braindecode.models import TCFormer

# ponytail: only TCFormer-specific checks live here. forward / final_layer /
# activation / drop_prob / export / compile / config / HuggingFace / batch-1 /
# categorization / summary are already exercised for every registered model by
# the shared model test suite (test_integration, test_config, test_huggingface,
# test_models, test_model_categorization).

_REF = os.environ.get(
    "TCFORMER_REF_DIR", "/Users/bruaristimunha/Projects/braindecode-v2/TCFormer"
)


def test_param_count_and_forward():
    # 77,820 params == paper Table 1 / Tables 2 & 4 headline (N=2) config. This
    # exact integer is the structural fidelity anchor (runs in CI, no clone).
    model = TCFormer(n_chans=22, n_outputs=4, n_times=1000)
    assert sum(p.numel() for p in model.parameters()) == 77_820
    assert model.eval()(torch.randn(2, 22, 1000)).shape == (2, 4)


@pytest.mark.skipif(not os.path.isdir(_REF), reason="upstream clone absent")
def test_matches_reference():
    """Port is numerically identical to the upstream model given equal weights."""
    ref = _load_reference()(
        n_channels=22,
        n_classes=4,
        F1=32,
        temp_kernel_lengths=(20, 32, 64),
        d_group=16,
        D=2,
        pool_length_1=8,
        pool_length_2=7,
        dropout_conv=0.4,
        use_group_attn=True,
        q_heads=4,
        kv_heads=2,
        trans_depth=2,
        trans_dropout=0.4,
        tcn_depth=2,
        kernel_length_tcn=4,
        dropout_tcn=0.3,
    ).eval()
    bd = TCFormer(n_chans=22, n_outputs=4, n_times=1000).eval()

    # Trunk aligns 1:1 by state_dict position; the constrained classifier
    # (parametrized weight) is copied by name.
    ref_trunk = [
        (k, v)
        for k, v in ref.state_dict().items()
        if not k.startswith("tcn_head.classifier")
    ]
    bd_trunk = [
        (k, v) for k, v in bd.state_dict().items() if not k.startswith("final_layer")
    ]
    bd.load_state_dict(
        {bk: rv for (_, rv), (bk, _) in zip(ref_trunk, bd_trunk)}, strict=False
    )
    cls = ref.tcn_head.classifier.linear
    bd.final_layer.conv.parametrizations.weight.original.data.copy_(cls.weight.data)
    bd.final_layer.conv.bias.data.copy_(cls.bias.data)

    x = torch.randn(4, 22, 1000)
    with torch.no_grad():
        assert torch.allclose(ref(x), bd(x), atol=1e-5)


def _load_reference():
    """Import the upstream TCFormerModule with its Lightning/util deps stubbed."""
    src = open(os.path.join(_REF, "models", "tcformer.py")).read()
    src = src.replace(
        "from .classification_module import ClassificationModule",
        "class ClassificationModule:\n    def __init__(self, *a, **k): pass",
    )
    for line in (
        "from .channel_group_attention import ChannelGroupAttention",
        "from utils.weight_initialization import glorot_weight_zero_bias",
        "from utils.latency  import measure_latency",
        "from .modules import CausalConv1d, Conv1dWithConstraint",
    ):
        src = src.replace(line, "")
    prelude = (
        "import sys\nimport torch\nfrom torch import nn, Tensor\n"
        "from torch.nn import functional as F\n"
        "from einops import rearrange\nfrom einops.layers.torch import Rearrange\n"
        f"sys.path.insert(0, {os.path.join(_REF, 'models')!r})\n"
        "from modules import CausalConv1d, Conv1dWithConstraint\n"
        "from channel_group_attention import ChannelGroupAttention\n"
        "measure_latency = glorot_weight_zero_bias = lambda *a, **k: None\n"
    )
    ns: dict = {}
    exec(prelude + src, ns)  # noqa: S102  (test-only, controlled source)
    return ns["TCFormerModule"]
