# Authors: Bruno Aristimunha <b.aristimunha@gmail.com>
#
# License: BSD (3-clause)
"""Keep the number of private copies of standard blocks in ``braindecode.models``
from growing.

Every model file that re-implements attention, a transformer block, an MLP, a
positional encoding, a rotary embedding, a norm, a patch embedding or a residual
wrapper is listed below. A new private copy fails this test: use the shared
implementation in :mod:`braindecode.modules` / :mod:`braindecode.functional`,
or add the class here with a one-line reason in its docstring (for example a
parameter layout that released weights depend on).

Removing a copy requires removing its line here, so the list stays exact.
"""

import ast
import re
from pathlib import Path

import braindecode.models

_BLOCK_NAME = re.compile(
    r"Attention|Attn|MHA|MHSA|TransformerEncoder|TransformerBlock|EncoderLayer|EncoderBlock|Block|MLP|Mlp|FeedForward|FFN|PositionalEncoding|PositionalEmbedding|PosEncoder|Rotary|RoPE|RMSNorm|LayerNorm|GroupNorm|PatchEmbed|Residual"
)

# (module, class): private copies of standard blocks present on 2026-10-17.
_KNOWN_PRIVATE_BLOCKS = {
    ("atcnet", "_ConvBlock"),
    ("atcnet", "_AttentionBlock"),
    ("atcnet", "_TCNResidualBlock"),
    ("atcnet", "_MHA"),
    ("attentionbasenet", "_ChannelAttentionBlock"),
    ("attn_sleep", "_SEBasicBlock"),
    ("attn_sleep", "_MultiHeadedAttention"),
    ("attn_sleep", "_ResidualLayerNormAttn"),
    ("attn_sleep", "_ResidualLayerNormFF"),
    ("attn_sleep", "_EncoderLayer"),
    ("attn_sleep", "_PositionwiseFeedForward"),
    ("axon", "_GatedFeedForward"),
    ("axon", "_AxisAttention"),
    ("axon", "_AXONBlock"),
    ("barista", "_DilatedConvBlock"),
    ("barista", "_TransformerEncoderLayer"),
    ("barista", "_RotarySelfAttention"),
    ("bendr", "_TFixupEncoderLayer"),
    ("biot", "_PositionalEncoding"),
    ("brainomni", "_SpatialTemporalBlock"),
    ("brainomni", "_RotaryPositionalEmbedding"),
    ("brainomni", "_MultiHeadAttentionRoPE"),
    ("brainomni", "_SEANetResBlock"),
    ("cbramod", "_PatchEmbedding"),
    ("cbramod", "_TransformerEncoder"),
    ("codebrain", "_RMSNorm"),
    ("codebrain", "_ResidualBlock"),
    ("codebrain", "_ResidualGroup"),
    ("codebrain", "_PatchEmbedding"),
    ("contrawr", "_ResBlock"),
    ("csbrain", "_CSBrainEncoderLayer"),
    ("ctnet", "_PatchEmbeddingEEGNet"),
    ("ctnet", "_TransformerEncoderBlock"),
    ("ctnet", "_TransformerEncoder"),
    ("ctnet", "_PositionalEncoding"),
    ("diver1", "_AnyVariateEncoderLayer"),
    ("diver1", "_AnyVariateAttention"),
    ("diver1", "_RotaryEmbedding"),
    ("eegconformer", "_PatchEmbedding"),
    ("eegconformer", "_TransformerEncoderBlock"),
    ("eegconformer", "_TransformerEncoder"),
    ("eegdino", "_PatchEmbedding"),
    ("eegdino", "_MLP"),
    ("eeginception_mi", "_ResidualModuleMI"),
    ("eegitnet", "_TCBlock"),
    ("eegpt", "_PatchEmbed"),
    ("eegsym", "_InceptionBlock"),
    ("eegsym", "_ResidualBlock"),
    ("eegsym", "_TemporalBlock"),
    ("eegsym", "_ChannelMergingBlock"),
    ("eegsym", "_TemporalMergingBlock"),
    ("eegsym", "_OutputBlock"),
    ("eegtcnet", "_TCNBlock"),
    ("emg2qwerty", "_RotationInvariantMLP"),
    ("emg2qwerty", "_MultiBandRotationInvariantMLP"),
    ("epint", "_RoPEAttention"),
    ("guetschel2026", "_LinearPatchEmbedding"),
    ("guetschel2026", "_MultiHeadAttention"),
    ("guetschel2026", "_TransformerEncoderLayer"),
    ("guetschel2026", "_TransformerEncoder"),
    ("ifnet", "_SpatioTemporalFeatureBlock"),
    ("labram", "_PatchEmbed"),
    ("labram", "_Attention"),
    ("labram", "_WindowsAttentionBlock"),
    ("luna", "_RotarySelfAttentionBlock"),
    ("luna", "_RotaryEmbedding"),
    ("luna", "_FeedForwardBlock"),
    ("luna", "_CrossAttentionBlock"),
    ("luna", "_PatchEmbedNetwork"),
    ("mapa", "_WithinArrayBlock"),
    ("medformer", "_PositionalEmbedding"),
    ("medformer", "_ListPatchEmbedding"),
    ("medformer", "_AttentionLayer"),
    ("medformer", "_FullAttention"),
    ("medformer", "_EncoderLayer"),
    ("meta_neuromotor", "_RotationInvariantMPFMLP"),
    ("meta_neuromotor", "_Residual"),
    ("meta_neuromotor", "_MultiHeadAttention"),
    ("mirepnet", "_PatchEmbedding"),
    ("mirepnet", "_TransformerEncoderBlock"),
    ("mirepnet", "_TransformerEncoder"),
    ("mscformer", "_MultiScalePatchEmbeddingCNN"),
    ("mscformer", "_TransformerEncoderBlock"),
    ("mscformer", "_TransformerEncoder"),
    ("mscformer", "_PositionalEncoding"),
    ("msvtnet", "_PositionalEncoding"),
    ("mvpformer", "_WaveletPatchEmbed"),
    ("mvpformer", "_MVPAttention"),
    ("mvpformer", "_SwiGLUMLP"),
    ("mvpformer", "_MVPABlock"),
    ("neuropose", "_EncoderBlock"),
    ("neuropose", "_ResBlock"),
    ("neuropose", "_DecoderBlock"),
    ("neurorvq_tokenizer", "_ResidualVectorQuantizer"),
    ("patchedtransformer", "_LayerNorm"),
    ("patchedtransformer", "_MHSA"),
    ("patchedtransformer", "_FeedForward"),
    ("patchedtransformer", "_TransformerEncoderLayer"),
    ("patchedtransformer", "_TransformerEncoder"),
    ("popt", "_PopTSpatialPositionalEncoding"),
    ("seizure_transformer", "_ConvBlock"),
    ("seizure_transformer", "_ResidualConvBlock"),
    ("sensingdynamics", "_ConvBlock"),
    ("signal_jepa", "_GroupNorm"),
    ("signal_jepa", "_PosEncoder"),
    ("sleepfm", "_SleepFMAttentionPooling"),
    ("sparcnet", "_DenseBlock"),
    ("sstdpn", "_SpatSpectralAttn"),
    ("steegformer", "_TemporalPositionalEncoding"),
    ("steegformer", "_TransformerEncoderBlock"),
    ("tcformer", "_GroupedQueryAttention"),
    ("tcformer", "_TransformerBlock"),
    ("tcformer", "_ChannelGroupAttention"),
    ("tcformer", "_MultiKernelConvBlock"),
    ("tcformer", "_TCNResidualBlock"),
    ("tcn", "_TemporalBlock"),
    ("tfm_tokenizer", "_GroupNorm"),
    ("tidnet", "_ConvBlock2D"),
    ("tmsanet", "_TMSAAttention"),
    ("tmsanet", "_TMSABlock"),
    ("usleep", "_EncoderBlock"),
    ("usleep", "_DecoderBlock"),
    ("vemg2pose", "_Conv1dBlock"),
    ("vemg2pose", "_MlpPoseDecoder"),
    ("zuna", "_RotaryPositionEmbedding"),
    ("zuna", "_Attention"),
    ("zuna", "_FeedForward"),
    ("zuna", "_TransformerBlock"),
}


def _private_block_classes():
    found = set()
    for path in sorted(Path(braindecode.models.__file__).parent.glob("*.py")):
        for node in ast.parse(path.read_text()).body:
            if (
                isinstance(node, ast.ClassDef)
                and node.name.startswith("_")
                and _BLOCK_NAME.search(node.name)
            ):
                found.add((path.stem, node.name))
    return found


def test_no_new_private_blocks():
    found = _private_block_classes()
    new = sorted(found - _KNOWN_PRIVATE_BLOCKS)
    assert not new, (
        "New private copies of standard blocks in braindecode.models: "
        f"{new}. Reuse braindecode.modules / braindecode.functional, or add the "
        "class to _KNOWN_PRIVATE_BLOCKS with a reason in its docstring."
    )
    gone = sorted(_KNOWN_PRIVATE_BLOCKS - found)
    assert not gone, f"Removed from braindecode.models; drop from _KNOWN_PRIVATE_BLOCKS: {gone}"
