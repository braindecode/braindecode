# Authors: Bruno Aristimunha <b.aristimunha@gmail.com>
#
# License: BSD (3-clause)
"""Enforce mechanical model-review rules in :mod:`braindecode.models`.

Each rule uses pure AST inspection with an explicit ``(module_stem, symbol)``
allowlist of existing baseline entries. Adding a new violation fails CI;
removing a baseline entry requires dropping its line here so allowlists
stay exact and shrink over time.
"""

import ast
import re
from collections.abc import Callable
from pathlib import Path

import pytest

import braindecode.models

_INFRA_STEMS = frozenset({"__init__", "base", "config", "util"})
_SHORT_TUPLE = 8
_MATH_ATOMS = frozenset({"pi", "e", "tau", "inf", "nan"})
_MATH_MODULES = frozenset({"math", "np", "numpy", "torch"})
_NUMERIC_CALLS = frozenset(
    {"sqrt", "log", "log2", "log10", "exp", "pow", "radians", "float", "int"}
)

_VENDOR_RE = re.compile(
    r"(?<![a-z])(?:hpus?|habana|gaudi|mps|cuda|xla|tpu|xpu|rocm)(?![a-z])",
    re.I,
)
_DOWNLOAD_PRIMITIVES = frozenset(
    {
        "hf_hub_download",
        "snapshot_download",
        "pooch",
        "load_state_dict_from_url",
        "download_url_to_file",
        "urlretrieve",
        "urlopen",
    }
)
_URL_RE = re.compile(r"https?://|huggingface\.co/", re.I)
_LOAD_OVERRIDES = frozenset({"load_state_dict", "_load_from_state_dict"})
_LOAD_HOOK_RE = re.compile(r"_?register_(?:module_)?load_state_dict\w*")

# C4: load_state_dict / _load_from_state_dict overrides and load hooks (excluding base.py).
_KNOWN_LOAD_STATE_DICT_OVERRIDES: set[tuple[str, str]] = {
    # Existing v2.0.0 compatibility / intentional hooks
    ("attn_sleep", "_SEBasicBlock._load_from_state_dict"),
    ("brant", "Brant.load_state_dict"),
    ("gcnsnet", "GCNsNet.__init__"),
    ("labram", "Labram._load_from_state_dict"),
    ("luna", "LUNA._register_channel_locations"),
    # Weight re-hosting is in the backlog for these models
    ("codebrain", "CodeBrain.load_state_dict"),
    ("codebrain", "_GConv._load_from_state_dict"),
    ("duin", "DuIN.load_state_dict"),
    ("eegpt", "EEGPT._load_from_state_dict"),
    ("guetschel2026", "Guetschel2026.load_state_dict"),
    ("popt", "PopulationTransformer.load_state_dict"),
}

# C5: download primitives and non-docstring URLs (excluding base.py).
_KNOWN_DOWNLOADS: set[tuple[str, str]] = set()

# C6: vendor identifiers, imports, and non-docstring backend strings in executable code.
_KNOWN_VENDOR_REFERENCES: set[tuple[str, str]] = {
    ("eegnex", "_TemporalConv.forward"),
    ("eegsym", "_BatchNorm3d.forward"),
    ("eegsym", "_TemporalConv3d.forward"),
    ("luna", "_FrequencyFeatureEmbedder.forward"),  # "mps" complex-angle guard
    ("msvtnet", "_TSConv.forward"),
    ("zuna", "<module>"),
    ("zuna", "_RotaryPositionEmbedding.forward"),
    ("zuna", "_define_hpu_pairwise_rotary"),
    ("zuna", "_define_hpu_pairwise_rotary.rotary"),
    ("zuna", "_hpu_fused_rotary"),
}

# C7: private cross-model imports from non-infra model modules.
_KNOWN_PRIVATE_IMPORTS: set[tuple[str, str]] = {
    ("csbrain", "cbramod._PatchEmbedding"),
    ("fbmsnet", "fbcnet._valid_layers"),
    ("guetschel2026", "signal_jepa._pos_encode_time"),
    ("neurorvq", "labram._Attention"),
    ("neurorvq_tokenizer", "neurorvq._MultiScaleTemporalConv"),
    ("neurorvq_tokenizer", "neurorvq._block"),
    ("neurorvq_tokenizer", "neurorvq._channel_slots"),
    ("neurorvq_tokenizer", "neurorvq._check_channel_strategy"),
    ("neurorvq_tokenizer", "neurorvq._modality"),
    ("neurorvq_tokenizer", "neurorvq._stack_scales"),
    ("util", "bendr._BENDR_TARGET_CHS_INFO"),
    ("util", "labram._LABRAM_TARGET_CHS_INFO"),
}

# C11: module-level numeric/short-tuple constants and private-block __init__ numeric defaults.
_KNOWN_CONFIGURING_CONSTANTS: set[tuple[str, str]] = {
    ("atcnet", "_AttentionBlock.dropout"),
    ("atcnet", "_AttentionBlock.head_dim"),
    ("atcnet", "_AttentionBlock.in_shape"),
    ("atcnet", "_AttentionBlock.num_heads"),
    ("atcnet", "_ConvBlock.depth_mult"),
    ("atcnet", "_ConvBlock.dropout"),
    ("atcnet", "_ConvBlock.kernel_length_1"),
    ("atcnet", "_ConvBlock.kernel_length_2"),
    ("atcnet", "_ConvBlock.n_filters"),
    ("atcnet", "_ConvBlock.pool_size_1"),
    ("atcnet", "_ConvBlock.pool_size_2"),
    ("atcnet", "_MHA.dropout"),
    ("atcnet", "_TCNResidualBlock.dilation"),
    ("atcnet", "_TCNResidualBlock.dropout"),
    ("atcnet", "_TCNResidualBlock.kernel_size"),
    ("atcnet", "_TCNResidualBlock.n_filters"),
    ("attentionbasenet", "_ChannelAttentionBlock.drop_prob"),
    ("attentionbasenet", "_ChannelAttentionBlock.freq_idx"),
    ("attentionbasenet", "_ChannelAttentionBlock.in_channels"),
    ("attentionbasenet", "_ChannelAttentionBlock.kernel_size"),
    ("attentionbasenet", "_ChannelAttentionBlock.n_codewords"),
    ("attentionbasenet", "_ChannelAttentionBlock.pool_length"),
    ("attentionbasenet", "_ChannelAttentionBlock.pool_stride"),
    ("attentionbasenet", "_ChannelAttentionBlock.reduction_rate"),
    ("attentionbasenet", "_ChannelAttentionBlock.seq_len"),
    ("attentionbasenet", "_ChannelAttentionBlock.temp_filter_length"),
    ("attentionbasenet", "_FeatureExtractor.drop_prob"),
    ("attentionbasenet", "_FeatureExtractor.n_temporal_filters"),
    ("attentionbasenet", "_FeatureExtractor.pool_length"),
    ("attentionbasenet", "_FeatureExtractor.pool_stride"),
    ("attentionbasenet", "_FeatureExtractor.spatial_expansion"),
    ("attentionbasenet", "_FeatureExtractor.temporal_filter_length"),
    ("attn_sleep", "_MRCNN.kernel_size"),
    ("attn_sleep", "_MRCNN_COARSE_DEEP_PADDING"),
    ("attn_sleep", "_MRCNN_COARSE_TAIL_POOL"),
    ("attn_sleep", "_MultiHeadedAttention.dropout"),
    ("attn_sleep", "_PositionwiseFeedForward.dropout"),
    ("attn_sleep", "_SEBasicBlock.reduction"),
    ("attn_sleep", "_SEBasicBlock.stride"),
    ("attn_sleep", "_SELayer.reduction"),
    ("bendr", "_BENDRContextualizer.dropout"),
    ("bendr", "_BENDRContextualizer.heads"),
    ("bendr", "_BENDRContextualizer.hidden_feedforward"),
    ("bendr", "_BENDRContextualizer.layer_drop"),
    ("bendr", "_BENDRContextualizer.layers"),
    ("bendr", "_BENDRContextualizer.position_encoder"),
    ("bendr", "_BENDRContextualizer.start_token"),
    ("bendr", "_ConvEncoderBENDR.dropout"),
    ("bendr", "_ConvEncoderBENDR.enc_downsample"),
    ("bendr", "_ConvEncoderBENDR.enc_width"),
    ("bendr", "_ConvEncoderBENDR.encoder_h"),
    ("biot", "_BIOTEncoder.attn_dropout"),
    ("biot", "_BIOTEncoder.attn_layer_dropout"),
    ("biot", "_BIOTEncoder.drop_prob"),
    ("biot", "_BIOTEncoder.emb_size"),
    ("biot", "_BIOTEncoder.hop_length"),
    ("biot", "_BIOTEncoder.max_seq_len"),
    ("biot", "_BIOTEncoder.n_chans"),
    ("biot", "_BIOTEncoder.n_fft"),
    ("biot", "_BIOTEncoder.n_layers"),
    ("biot", "_BIOTEncoder.num_heads"),
    ("biot", "_PatchFrequencyEmbedding.emb_size"),
    ("biot", "_PatchFrequencyEmbedding.n_freq"),
    ("brainbert", "_STFTSpectrogram.clip"),
    ("brainbert", "_STFTSpectrogram.idx_freq_cutoff"),
    ("brainbert", "_STFTSpectrogram.min_frames"),
    ("brainbert", "_STFTSpectrogram.noverlap"),
    ("brainbert", "_STFTSpectrogram.nperseg"),
    ("brainmodule", "_ChannelDropout.dropout_prob"),
    ("brainmodule", "_ConvSequence.dilation_growth"),
    ("brainmodule", "_ConvSequence.dilation_period"),
    ("brainmodule", "_ConvSequence.dropout"),
    ("brainmodule", "_ConvSequence.dropout_input"),
    ("brainmodule", "_ConvSequence.glu"),
    ("brainmodule", "_ConvSequence.glu_context"),
    ("brainmodule", "_ConvSequence.kernel_size"),
    ("brainmodule", "_ScaledEmbedding.scale"),
    ("cbramod", "_PatchEmbedding.drop_prob"),
    ("codebrain", "_Conv.dilation"),
    ("codebrain", "_Conv.kernel_size"),
    ("codebrain", "_GConv.channels"),
    ("codebrain", "_GConv.d_state"),
    ("codebrain", "_GConv.decay_max"),
    ("codebrain", "_GConv.decay_min"),
    ("codebrain", "_GConv.dropout"),
    ("codebrain", "_GConv.init_scale"),
    ("codebrain", "_GConv.kernel_dim"),
    ("codebrain", "_GConv.l_max"),
    ("codebrain", "_PatchEmbedding.pos_kernel"),
    ("codebrain", "_PatchEmbedding.proj_kernel_size"),
    ("codebrain", "_PatchEmbedding.proj_padding"),
    ("codebrain", "_PatchEmbedding.proj_refine_kernel"),
    ("codebrain", "_PatchEmbedding.spectral_dropout"),
    ("codebrain", "_RMSNorm.eps"),
    ("codebrain", "_ResidualBlock.swa_window_size"),
    ("codebrain", "_ResidualGroup.swa_window_size"),
    ("contrawr", "_ResBlock.drop_prob"),
    ("contrawr", "_ResBlock.kernel_size"),
    ("contrawr", "_ResBlock.padding"),
    ("contrawr", "_ResBlock.stride"),
    ("csbrain", "REGION_CENTRAL"),
    ("csbrain", "REGION_FRONTAL"),
    ("csbrain", "REGION_OCCIPITAL"),
    ("csbrain", "REGION_PARIETAL"),
    ("csbrain", "REGION_TEMPORAL"),
    ("csbrain", "_CSBrainEncoderLayer.dim_feedforward"),
    ("csbrain", "_CSBrainEncoderLayer.dropout"),
    ("csbrain", "_CrossScaleTemporalEmbedding.kernel_sizes"),
    ("csbrain", "_RegionEmbedding.kernel_sizes"),
    ("ctnet", "_PatchEmbeddingEEGNet.depth_multiplier"),
    ("ctnet", "_PatchEmbeddingEEGNet.drop_prob"),
    ("ctnet", "_PatchEmbeddingEEGNet.kernel_size"),
    ("ctnet", "_PatchEmbeddingEEGNet.n_chans"),
    ("ctnet", "_PatchEmbeddingEEGNet.n_filters_time"),
    ("ctnet", "_PatchEmbeddingEEGNet.pool_size_1"),
    ("ctnet", "_PatchEmbeddingEEGNet.pool_size_2"),
    ("deepsleepnet", "_CNNPath.deep_kernel_size"),
    ("deepsleepnet", "_CNNPath.drop_prob"),
    ("deepsleepnet", "_CNNPath.first_kernel_size"),
    ("deepsleepnet", "_CNNPath.first_padding"),
    ("deepsleepnet", "_CNNPath.first_stride"),
    ("deepsleepnet", "_CNNPath.n_filters_1"),
    ("deepsleepnet", "_CNNPath.n_filters_2"),
    ("deepsleepnet", "_CNNPath.pool1_kernel_size"),
    ("deepsleepnet", "_CNNPath.pool1_padding"),
    ("deepsleepnet", "_CNNPath.pool1_stride"),
    ("deepsleepnet", "_CNNPath.pool2_kernel_size"),
    ("deepsleepnet", "_CNNPath.pool2_padding"),
    ("deepsleepnet", "_CNNPath.pool2_stride"),
    ("dgcnn", "_LearnableAdjacency.n_neighbors"),
    ("diver1", "_PatchCNN.depth"),
    ("diver1", "_PatchCNN.kernel_size"),
    ("diver1", "_RotaryEmbedding.base"),
    ("diver1", "_SinusoidalCoordEmbedding.scale"),
    ("diver1", "_SinusoidalCoordEmbedding.temperature"),
    ("eegconformer", "_FullyConnected.drop_prob_1"),
    ("eegconformer", "_FullyConnected.drop_prob_2"),
    ("eegconformer", "_FullyConnected.hidden_channels"),
    ("eegconformer", "_FullyConnected.out_channels"),
    ("eegdino", "_ClassificationHead.drop_prob_1"),
    ("eegdino", "_ClassificationHead.drop_prob_2"),
    ("eegdino", "_MLP.drop"),
    ("eegdino", "_PatchEmbedding.drop_prob"),
    ("eeginception_mi", "_InceptionModuleMI.kernel_unit_s"),
    ("eeginception_mi", "_InceptionModuleMI.sfreq"),
    ("eegitnet", "_TCBlock.drop_prob"),
    ("eegpt", "_ChannelProjection.max_norm"),
    ("eegpt", "_EEGTransformer.attn_drop_rate"),
    ("eegpt", "_EEGTransformer.depth"),
    ("eegpt", "_EEGTransformer.drop_path_rate"),
    ("eegpt", "_EEGTransformer.drop_rate"),
    ("eegpt", "_EEGTransformer.embed_dim"),
    ("eegpt", "_EEGTransformer.embed_num"),
    ("eegpt", "_EEGTransformer.init_std"),
    ("eegpt", "_EEGTransformer.mlp_ratio"),
    ("eegpt", "_EEGTransformer.n_chans"),
    ("eegpt", "_EEGTransformer.n_times"),
    ("eegpt", "_EEGTransformer.num_heads"),
    ("eegpt", "_EEGTransformer.patch_size"),
    ("eegpt", "_EEGTransformer.return_attention_layer"),
    ("eegpt", "_LinearConstraintProbe.dropout_p"),
    ("eegpt", "_LinearConstraintProbe.probe1_max_norm"),
    ("eegpt", "_LinearConstraintProbe.probe2_max_norm"),
    ("eegpt", "_LinearConstraintProbe.probe_hidden_dim"),
    ("eegpt", "_PatchEmbed.embed_dim"),
    ("eegpt", "_PatchEmbed.n_chans"),
    ("eegpt", "_PatchEmbed.n_times"),
    ("eegpt", "_PatchEmbed.patch_size"),
    ("eegsym", "_OutputBlock.n_residual"),
    ("eegsym", "_ResidualBlock.spatial_resnet_repetitions"),
    ("eegtcnet", "_EEGNetTC.depth_multiplier"),
    ("eegtcnet", "_EEGNetTC.drop_prob"),
    ("eegtcnet", "_EEGNetTC.filter_1"),
    ("eegtcnet", "_EEGNetTC.kern_length"),
    ("emg2qwerty", "_LogSpectrogram.log_eps"),
    ("emg2qwerty", "_MultiBandRotationInvariantMLP.num_bands"),
    ("emg2qwerty", "_MultiBandRotationInvariantMLP.offsets"),
    ("emg2qwerty", "_MultiBandRotationInvariantMLP.stack_dim"),
    ("emg2qwerty", "_RotationInvariantMLP.offsets"),
    ("emg2qwerty", "_SpecAugment.freq_mask_param"),
    ("emg2qwerty", "_SpecAugment.n_freq_masks"),
    ("emg2qwerty", "_SpecAugment.n_time_masks"),
    ("emg2qwerty", "_SpecAugment.prob"),
    ("emg2qwerty", "_SpecAugment.time_mask_param"),
    ("fblightconvnet", "_LightweightConv1d.heads"),
    ("fblightconvnet", "_LightweightConv1d.kernel_size"),
    ("fblightconvnet", "_LightweightConv1d.padding"),
    ("fbmsnet", "_MixedConv2d.dilation"),
    ("fbmsnet", "_MixedConv2d.kernels_weights"),
    ("fbmsnet", "_MixedConv2d.stride"),
    ("geonirs", "_GeometryEncoder.geometry_dim"),  # structural 6D coordinate vector
    ("guetschel2026", "_GaussianRandomProjection.seed"),
    ("guetschel2026", "_MASK_LENGTHS"),
    ("ifnet", "_SpatioTemporalFeatureBlock.dim"),
    ("ifnet", "_SpatioTemporalFeatureBlock.drop_prob"),
    ("ifnet", "_SpatioTemporalFeatureBlock.kernel_sizes"),
    ("ifnet", "_SpatioTemporalFeatureBlock.n_bands"),
    ("ifnet", "_SpatioTemporalFeatureBlock.out_channels"),
    ("ifnet", "_SpatioTemporalFeatureBlock.stride_factor"),
    ("labram", "_Attention.attn_drop"),
    ("labram", "_Attention.num_heads"),
    ("labram", "_Attention.proj_drop"),
    ("labram", "_LABRAM_N_TIME_SLOTS"),
    ("labram", "_PatchEmbed.emb_dim"),
    ("labram", "_PatchEmbed.in_channels"),
    ("labram", "_PatchEmbed.n_codebooks"),
    ("labram", "_PatchEmbed.n_times"),
    ("labram", "_PatchEmbed.patch_size"),
    ("labram", "_SegmentPatch.emb_dim"),
    ("labram", "_SegmentPatch.n_chans"),
    ("labram", "_SegmentPatch.n_times"),
    ("labram", "_SegmentPatch.patch_size"),
    ("labram", "_TemporalConv.in_channels"),
    ("labram", "_TemporalConv.kernel_size_1"),
    ("labram", "_TemporalConv.kernel_size_2"),
    ("labram", "_TemporalConv.num_groups"),
    ("labram", "_TemporalConv.out_channels"),
    ("labram", "_TemporalConv.padding_1"),
    ("labram", "_TemporalConv.padding_2"),
    ("labram", "_TemporalConv.stride_1"),
    ("labram", "_WindowsAttentionBlock.attn_drop"),
    ("labram", "_WindowsAttentionBlock.drop"),
    ("labram", "_WindowsAttentionBlock.drop_path"),
    ("labram", "_WindowsAttentionBlock.mlp_ratio"),
    ("luna", "_ChannelEmbeddings.number_channels"),
    ("luna", "_ClassificationHeadWithQueries.drop_decoder"),
    ("luna", "_ClassificationHeadWithQueries.drop_ffn"),
    ("luna", "_ClassificationHeadWithQueries.embed_dim"),
    ("luna", "_ClassificationHeadWithQueries.input_dim"),
    ("luna", "_ClassificationHeadWithQueries.num_classes"),
    ("luna", "_ClassificationHeadWithQueries.num_heads"),
    ("luna", "_ClassificationHeadWithQueries.num_queries"),
    ("luna", "_CrossAttentionBlock.dropout_p"),
    ("luna", "_CrossAttentionBlock.ff_dim"),
    ("luna", "_FeedForwardBlock.dropout"),
    ("luna", "_PatchEmbedNetwork.embed_dim"),
    ("luna", "_PatchEmbedNetwork.patch_size"),
    ("luna", "_PatchReconstructionHeadWithQueries.embed_dim"),
    ("luna", "_PatchReconstructionHeadWithQueries.input_dim"),
    ("luna", "_PatchReconstructionHeadWithQueries.num_heads"),
    ("luna", "_PatchReconstructionHeadWithQueries.num_queries"),
    ("luna", "_RotaryEmbedding.theta"),
    ("luna", "_RotarySelfAttentionBlock.attn_drop"),
    ("luna", "_RotarySelfAttentionBlock.num_heads"),
    ("luna", "_RotarySelfAttentionBlock.proj_drop"),
    ("mapa", "_INPUT_CLIP_Z"),
    ("mapa", "_SUP_TAPS"),
    ("medformer", "_FullAttention.attention_dropout"),
    ("medformer", "_MedformerLayer.dropout"),
    ("meta_neuromotor", "_Conv1d.dilation"),
    ("meta_neuromotor", "_Conv1d.groups"),
    ("meta_neuromotor", "_Conv1d.stride"),
    ("meta_neuromotor", "_MaskAug.mask_value"),
    ("meta_neuromotor", "_MultiHeadAttention.dropout"),
    ("meta_neuromotor", "_MultiHeadAttention.lpad"),
    ("meta_neuromotor", "_MultiHeadAttention.stride"),
    ("meta_neuromotor", "_MultivariatePowerFrequencyFeatures.fs"),
    ("meta_neuromotor", "_Residual.dropout"),
    ("meta_neuromotor", "_Residual.weight"),
    ("meta_neuromotor", "_RotationInvariantMPFMLP.num_adjacent_cov"),
    ("meta_neuromotor", "_RotationInvariantMPFMLP.offsets"),
    ("meta_neuromotor", "_Window.dilation"),
    ("meta_neuromotor", "_Window.lpad"),
    ("meta_neuromotor", "_Window.stride"),
    ("msvtnet", "_Transformer.drop_prob"),
    ("msvtnet", "_Transformer.num_layers"),
    ("mvpformer", "_MVPABlock.attn_drop"),
    ("mvpformer", "_MVPABlock.eps"),
    ("mvpformer", "_MVPABlock.local_window"),
    ("mvpformer", "_MVPABlock.resid_drop"),
    ("mvpformer", "_MVPAttention.attn_drop"),
    ("mvpformer", "_MVPAttention.local_window"),
    ("mvpformer", "_MVPAttention.resid_drop"),
    ("patchedtransformer", "_FeedForward.drop_prob"),
    ("patchedtransformer", "_MHSA.drop_prob"),
    ("patchedtransformer", "_Patcher.d_input"),
    ("patchedtransformer", "_Patcher.n_chans"),
    ("patchedtransformer", "_Patcher.n_times"),
    ("patchedtransformer", "_TransformerEncoderLayer.drop_prob"),
    ("popt", "_PopTInputEmbedding.drop_prob"),
    ("popt", "_PopTInputEmbedding.max_len"),
    ("popt", "_PopTSpatialPositionalEncoding.max_len"),
    ("sensingdynamics", "_ConvBlock.dilation"),
    ("sensingdynamics", "_ConvBlock.stride"),
    ("signal_jepa", "_BaseSignalJEPA.drop_prob"),
    ("signal_jepa", "_BaseSignalJEPA.pos_encoder__sfreq_features"),
    ("signal_jepa", "_BaseSignalJEPA.pos_encoder__spat_dim"),
    ("signal_jepa", "_BaseSignalJEPA.pos_encoder__time_dim"),
    ("signal_jepa", "_BaseSignalJEPA.transformer__d_model"),
    ("signal_jepa", "_BaseSignalJEPA.transformer__nhead"),
    ("signal_jepa", "_BaseSignalJEPA.transformer__num_decoder_layers"),
    ("signal_jepa", "_BaseSignalJEPA.transformer__num_encoder_layers"),
    ("signal_jepa", "_ConvFeatureEncoder.drop_prob"),
    ("signal_jepa", "_PosEncoder.max_seconds"),
    ("sinc_shallow", "_SincFilter.bandwidth"),
    ("sinc_shallow", "_SincFilter.min_freq"),
    ("sparcnet", "_DenseBlock.drop_prob"),
    ("sparcnet", "_DenseBlock.kernel_size_conv1"),
    ("sparcnet", "_DenseBlock.kernel_size_conv2"),
    ("sparcnet", "_DenseBlock.padding_conv2"),
    ("sparcnet", "_DenseBlock.stride_conv1"),
    ("sparcnet", "_DenseBlock.stride_conv2"),
    ("sparcnet", "_DenseLayer.drop_prob"),
    ("sparcnet", "_DenseLayer.kernel_size_conv1"),
    ("sparcnet", "_DenseLayer.kernel_size_conv2"),
    ("sparcnet", "_DenseLayer.padding_conv2"),
    ("sparcnet", "_DenseLayer.stride_conv1"),
    ("sparcnet", "_DenseLayer.stride_conv2"),
    ("sparcnet", "_TransitionLayer.kernel_size_trans"),
    ("sparcnet", "_TransitionLayer.stride_trans"),
    ("sstdpn", "_DepthwiseTemporalConv1d.kernel_size"),
    ("sstdpn", "_DepthwiseTemporalConv1d.n_spectral_filters_temporal"),
    ("sstdpn", "_DepthwiseTemporalConv1d.num_heads"),
    ("sstdpn", "_DepthwiseTemporalConv1d.padding"),
    ("sstdpn", "_DepthwiseTemporalConv1d.stride"),
    ("sstdpn", "_GlobalContextVarPool1D.padding"),
    ("sstdpn", "_SSTEncoder.n_fused_filters"),
    ("sstdpn", "_SSTEncoder.n_spectral_filters_temporal"),
    ("sstdpn", "_SSTEncoder.spt_attn_epsilon"),
    ("sstdpn", "_SSTEncoder.spt_attn_global_context_kernel"),
    ("sstdpn", "_SSTEncoder.temporal_conv_kernel_size"),
    ("sstdpn", "_SpatSpectralAttn.epsilon"),
    ("sstdpn", "_SpatSpectralAttn.global_context_kernel"),
    ("sstdpn", "_VariancePool1D.padding"),
    ("steegformer", "_CHANNELS_VOCAB_SIZE"),
    ("steegformer", "_TemporalPositionalEncoding.max_len"),
    ("tcformer", "_ChannelGroupAttention.reduction"),
    ("tcformer", "_ClassificationHead.max_norm"),
    ("tcformer", "_GroupedQueryAttention.drop_prob"),
    ("tcformer", "_TransformerBlock.drop_path_rate"),
    ("tcformer", "_TransformerBlock.drop_prob"),
    ("tcformer", "_TransformerBlock.mlp_ratio"),
    ("tidnet", "_ConvBlock2D.dilation"),
    ("tidnet", "_ConvBlock2D.drop_prob"),
    ("tidnet", "_ConvBlock2D.groups"),
    ("tidnet", "_ConvBlock2D.padding"),
    ("tidnet", "_ConvBlock2D.stride"),
    ("tidnet", "_DenseFilter.bottleneck"),
    ("tidnet", "_DenseFilter.dim"),
    ("tidnet", "_DenseFilter.drop_prob"),
    ("tidnet", "_DenseFilter.filter_len"),
    ("tidnet", "_DenseSpatialFilter.bottleneck"),
    ("tidnet", "_DenseSpatialFilter.drop_prob"),
    ("tidnet", "_DenseSpatialFilter.in_ch"),
    ("tidnet", "_TemporalFilter.drop_prob"),
    ("usleep", "_DecoderBlock.in_channels"),
    ("usleep", "_DecoderBlock.kernel_size"),
    ("usleep", "_DecoderBlock.out_channels"),
    ("usleep", "_DecoderBlock.upsample"),
    ("usleep", "_EncoderBlock.downsample"),
    ("usleep", "_EncoderBlock.in_channels"),
    ("usleep", "_EncoderBlock.kernel_size"),
    ("usleep", "_EncoderBlock.out_channels"),
    ("util", "_ASEG_SLOT_PERMUTATION"),  # FreeSurfer slot permutation table
    ("zuna", "_Attention.norm_eps"),
    ("zuna", "_FeedForward.multiple_of"),
    ("zuna", "_TransformerBlock.multiple_of"),
}


def _model_sources() -> dict[str, str]:
    models_dir = Path(braindecode.models.__file__).parent
    return {path.stem: path.read_text() for path in sorted(models_dir.glob("*.py"))}


def _scan_all(
    sources: dict[str, str],
    finder: Callable[[str, str], set[tuple[str, str]]],
) -> set[tuple[str, str]]:
    found: set[tuple[str, str]] = set()
    for stem, source in sources.items():
        found.update(finder(source, stem))
    return found


def _check_allowlist(
    found: set[tuple[str, str]],
    allowlist: set[tuple[str, str]],
    *,
    rule: str,
    remediation: str,
) -> None:
    new = sorted(found - allowlist)
    assert not new, f"New {rule} violations in braindecode.models: {new}. {remediation}"
    gone = sorted(allowlist - found)
    assert not gone, (
        f"Removed from braindecode.models; drop from {rule} allowlist: {gone}"
    )


def _scope_table(tree: ast.AST) -> list[tuple[int, int, str]]:
    out: list[tuple[int, int, str]] = []

    def visit(node: ast.AST, prefix: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                name = f"{prefix}.{child.name}" if prefix else child.name
                out.append((child.lineno, child.end_lineno or child.lineno, name))
                visit(child, name)
            else:
                visit(child, prefix)

    visit(tree, "")
    return out


def _symbol_at(scopes: list[tuple[int, int, str]], line: int) -> str:
    inside = [(end - start, name) for start, end, name in scopes if start <= line <= end]
    return min(inside)[1] if inside else "<module>"


def _docstring_node_ids(tree: ast.AST) -> set[int]:
    docs: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(
            node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        ) and node.body:
            first = node.body[0]
            if (
                isinstance(first, ast.Expr)
                and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)
            ):
                docs.add(id(first.value))
    return docs


def _iter_identifiers(tree: ast.AST) -> list[tuple[int, str]]:
    out: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            out.append((node.lineno, node.id))
        elif isinstance(node, ast.Attribute):
            out.append((node.end_lineno or node.lineno, node.attr))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.append((node.lineno, node.name))
        elif isinstance(node, ast.arg):
            out.append((node.lineno, node.arg))
        elif isinstance(node, ast.keyword) and node.arg:
            out.append((node.lineno, node.arg))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                out.append((node.lineno, f"{alias.name}.{alias.asname or ''}"))
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            for alias in node.names:
                out.append((node.lineno, f"{mod}.{alias.name}.{alias.asname or ''}"))
    return out


def _is_device_type_expr(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "type"
        and (
            (isinstance(node.value, ast.Attribute) and node.value.attr == "device")
            or (isinstance(node.value, ast.Name) and node.value.id == "device")
        )
    )


def _extract_non_cpu_strings(node: ast.AST) -> list[str]:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return [node.value] if node.value.lower() != "cpu" else []
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        vals: list[str] = []
        for elt in node.elts:
            if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                if elt.value.lower() != "cpu":
                    vals.append(elt.value)
        return vals
    return []


def _is_number(node: ast.AST) -> bool:
    if isinstance(node, ast.Constant):
        return isinstance(node.value, (int, float, complex)) and not isinstance(
            node.value, bool
        )
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        return _is_number(node.operand)
    if isinstance(node, ast.BinOp):
        return _is_number(node.left) and _is_number(node.right)
    if isinstance(node, ast.Attribute):
        return (
            isinstance(node.value, ast.Name)
            and node.value.id in _MATH_MODULES
            and node.attr in _MATH_ATOMS
        )
    if isinstance(node, ast.Call) and node.args and not node.keywords:
        if isinstance(node.func, ast.Attribute):
            name = node.func.attr
        elif isinstance(node.func, ast.Name):
            name = node.func.id
        else:
            return False
        first = node.args[0]
        if name == "float" and isinstance(first, ast.Constant) and isinstance(first.value, str):
            return first.value.strip().lower().lstrip("+-") in {
                "inf",
                "infinity",
                "nan",
            }
        return name in _NUMERIC_CALLS and all(_is_number(a) for a in node.args)
    return False


def _is_numeric_value(node: ast.AST) -> bool:
    if isinstance(node, (ast.Tuple, ast.List)):
        return 0 < len(node.elts) <= _SHORT_TUPLE and all(
            _is_number(elt) for elt in node.elts
        )
    return _is_number(node)


def _is_type_checking_guard(test: ast.expr) -> bool:
    return (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (
        isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"
    )


def _module_assignments(body: list[ast.stmt]) -> list[ast.Assign | ast.AnnAssign]:
    out: list[ast.Assign | ast.AnnAssign] = []
    for node in body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            out.append(node)
        elif isinstance(node, ast.If):
            if _is_type_checking_guard(node.test):
                out.extend(_module_assignments(node.orelse))
            else:
                out.extend(_module_assignments(node.body + node.orelse))
        elif isinstance(node, ast.Try):
            handlers = [s for h in node.handlers for s in h.body]
            out.extend(
                _module_assignments(node.body + node.orelse + node.finalbody + handlers)
            )
    return out


def _find_c4_state_dict_overrides(source: str, stem: str) -> set[tuple[str, str]]:
    if stem == "base":
        return set()
    tree = ast.parse(source)
    scopes = _scope_table(tree)
    found: set[tuple[str, str]] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for fn in node.body:
                if (
                    isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and fn.name in _LOAD_OVERRIDES
                ):
                    found.add((stem, _symbol_at(scopes, fn.lineno)))
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                func_name = node.func.attr
            elif isinstance(node.func, ast.Name):
                func_name = node.func.id
            else:
                func_name = ""
            if _LOAD_HOOK_RE.fullmatch(func_name):
                found.add((stem, _symbol_at(scopes, node.lineno)))
    return found


def _find_c5_downloads(source: str, stem: str) -> set[tuple[str, str]]:
    if stem == "base":
        return set()
    tree = ast.parse(source)
    scopes = _scope_table(tree)
    doc_ids = _docstring_node_ids(tree)
    found: set[tuple[str, str]] = set()
    for line, name in _iter_identifiers(tree):
        if set(re.split(r"\W+", name)) & _DOWNLOAD_PRIMITIVES:
            found.add((stem, _symbol_at(scopes, line)))
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and id(node) not in doc_ids
            and _URL_RE.search(node.value)
        ):
            found.add((stem, _symbol_at(scopes, node.lineno)))
    return found


def _find_c6_vendor_references(source: str, stem: str) -> set[tuple[str, str]]:
    tree = ast.parse(source)
    scopes = _scope_table(tree)
    doc_ids = _docstring_node_ids(tree)
    found: set[tuple[str, str]] = set()
    for line, name in _iter_identifiers(tree):
        if _VENDOR_RE.search(name):
            found.add((stem, _symbol_at(scopes, line)))
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and id(node) not in doc_ids
            and _VENDOR_RE.search(node.value)
        ):
            found.add((stem, _symbol_at(scopes, node.lineno)))
        elif isinstance(node, ast.Compare):
            exprs = [node.left, *node.comparators]
            if any(_is_device_type_expr(e) for e in exprs) and any(
                _extract_non_cpu_strings(e) for e in exprs
            ):
                found.add((stem, _symbol_at(scopes, node.lineno)))
    return found


def _attr_chain(node: ast.AST) -> list[str] | None:
    parts: list[str] = []
    cur: ast.AST = node
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        parts.append(cur.id)
        parts.reverse()
        return parts
    return None


def _find_c7_private_imports(source: str, stem: str) -> set[tuple[str, str]]:
    tree = ast.parse(source)
    found: set[tuple[str, str]] = set()
    imported_aliases: dict[str, str] = {}
    imported_full_paths: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.level == 0 and node.module.startswith("braindecode.models."):
                other = node.module.split(".")[2]
            elif node.level == 1:
                other = node.module.split(".")[0]
            else:
                continue
            if other == stem or other in _INFRA_STEMS:
                continue
            for alias in node.names:
                if alias.name.startswith("_"):
                    found.add((stem, f"{other}.{alias.name}"))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("braindecode.models."):
                    parts = alias.name.split(".")
                    other = parts[2]
                    if other == stem or other in _INFRA_STEMS:
                        continue
                    if alias.asname:
                        imported_aliases[alias.asname] = other
                    else:
                        imported_full_paths.add(other)

    if imported_aliases or imported_full_paths:
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr.startswith("_"):
                chain = _attr_chain(node.value)
                if not chain:
                    continue
                if len(chain) == 1 and chain[0] in imported_aliases:
                    found.add((stem, f"{imported_aliases[chain[0]]}.{node.attr}"))
                elif (
                    len(chain) == 3
                    and chain[0] == "braindecode"
                    and chain[1] == "models"
                    and chain[2] in imported_full_paths
                ):
                    found.add((stem, f"{chain[2]}.{node.attr}"))
    return found


def _find_c11_configuring_constants(source: str, stem: str) -> set[tuple[str, str]]:
    tree = ast.parse(source)
    found: set[tuple[str, str]] = set()
    for node in _module_assignments(tree.body):
        if node.value is None or not _is_numeric_value(node.value):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        for target in targets:
            if isinstance(target, (ast.Attribute, ast.Subscript)):
                continue
            for sub in ast.walk(target):
                if isinstance(sub, ast.Name) and not sub.id.startswith("__"):
                    found.add((stem, sub.id))
    for cls in ast.walk(tree):
        if not (isinstance(cls, ast.ClassDef) and cls.name.startswith("_")):
            continue
        for fn in cls.body:
            if not (isinstance(fn, ast.FunctionDef) and fn.name == "__init__"):
                continue
            args = fn.args
            positional = args.posonlyargs + args.args
            pairs = list(
                zip(positional[len(positional) - len(args.defaults) :], args.defaults)
            )
            pairs.extend(
                (arg, default)
                for arg, default in zip(args.kwonlyargs, args.kw_defaults)
                if default is not None
            )
            for arg, default in pairs:
                if _is_numeric_value(default):
                    found.add((stem, f"{cls.name}.{arg.arg}"))
    return found


def test_no_new_load_state_dict_overrides() -> None:
    found = _scan_all(_model_sources(), _find_c4_state_dict_overrides)
    _check_allowlist(
        found,
        _KNOWN_LOAD_STATE_DICT_OVERRIDES,
        rule="C4 (load_state_dict override/hook)",
        remediation=(
            "Re-host weights cleanly instead of adding checkpoint conversion overrides/hooks."
        ),
    )


def test_no_new_model_downloads() -> None:
    found = _scan_all(_model_sources(), _find_c5_downloads)
    _check_allowlist(
        found,
        _KNOWN_DOWNLOADS,
        rule="C5 (download primitive / URL)",
        remediation=(
            "Use PyTorchModelHubMixin in base.py instead of model-specific downloads or URLs."
        ),
    )


def test_no_new_vendor_references() -> None:
    found = _scan_all(_model_sources(), _find_c6_vendor_references)
    _check_allowlist(
        found,
        _KNOWN_VENDOR_REFERENCES,
        rule="C6 (vendor branch / identifier / string)",
        remediation=(
            "Keep model code device-agnostic; hardware-specific gaps belong in plugins."
        ),
    )


def test_no_new_private_cross_model_imports() -> None:
    found = _scan_all(_model_sources(), _find_c7_private_imports)
    _check_allowlist(
        found,
        _KNOWN_PRIVATE_IMPORTS,
        rule="C7 (private cross-model import)",
        remediation=(
            "Move shared private helpers to braindecode.modules or braindecode.models.util."
        ),
    )


def test_no_new_configuring_constants() -> None:
    found = _scan_all(_model_sources(), _find_c11_configuring_constants)
    _check_allowlist(
        found,
        _KNOWN_CONFIGURING_CONSTANTS,
        rule="C11 (configuring constant)",
        remediation=(
            "Expose configuring numbers as keyword-only __init__ parameters on the model."
        ),
    )


@pytest.mark.parametrize(
    ("snippet", "expected_symbol"),
    [
        (
            "class Dummy:\n    def load_state_dict(self, state_dict, strict=True):\n        pass\n",
            "Dummy.load_state_dict",
        ),
        (
            "class Dummy:\n    def _load_from_state_dict(self, *args):\n        pass\n",
            "Dummy._load_from_state_dict",
        ),
        (
            "class Dummy:\n    def __init__(self):\n        self.register_load_state_dict_pre_hook(lambda *a: None)\n",
            "Dummy.__init__",
        ),
        (
            "class Dummy:\n    def __init__(self):\n        self.register_load_state_dict_post_hook(lambda *a: None)\n",
            "Dummy.__init__",
        ),
        (
            "class Dummy:\n    def __init__(self):\n        self.register_module_load_state_dict_pre_hook(lambda *a: None)\n",
            "Dummy.__init__",
        ),
        (
            "class Dummy:\n    def __init__(self):\n        self.register_module_load_state_dict_post_hook(lambda *a: None)\n",
            "Dummy.__init__",
        ),
    ],
)
def test_unallowlisted_load_state_dict_override_fails(
    snippet: str, expected_symbol: str
) -> None:
    found = _find_c4_state_dict_overrides(snippet, "dummy_model")
    assert found == {("dummy_model", expected_symbol)}
    with pytest.raises(AssertionError, match="New C4"):
        _check_allowlist(found, set(), rule="C4", remediation="re-host")


def test_load_state_dict_base_excluded_and_stale_allowlist_fails() -> None:
    src = "class EEGModuleMixin:\n    def load_state_dict(self):\n        pass\n"
    assert _find_c4_state_dict_overrides(src, "base") == set()
    with pytest.raises(AssertionError, match="drop from C4 allowlist"):
        _check_allowlist(
            set(),
            {("dummy_model", "Dummy.load_state_dict")},
            rule="C4",
            remediation="re-host",
        )


def test_unallowlisted_download_fails_and_docstrings_ignored() -> None:
    bad = (
        "class Dummy:\n"
        "    def load(self):\n"
        "        return hf_hub_download('org/repo', 'w.pt')\n"
        "    def url(self):\n"
        "        return 'https://example.com/weights.pt'\n"
    )
    found = _find_c5_downloads(bad, "dummy_model")
    assert found == {
        ("dummy_model", "Dummy.load"),
        ("dummy_model", "Dummy.url"),
    }
    with pytest.raises(AssertionError, match="New C5"):
        _check_allowlist(found, set(), rule="C5", remediation="use mixin")

    benign = (
        '"""See https://example.com/paper."""\n'
        "# Adapted from https://github.com/example/repo\n"
        "class Dummy:\n"
        '    """Example: hf_hub_download(\'org/repo\', \'w.pt\')."""\n'
        "    pass\n"
    )
    assert _find_c5_downloads(benign, "dummy_model") == set()
    assert _find_c5_downloads(bad, "base") == set()


def test_unallowlisted_vendor_reference_fails_and_benign_strings_ignored() -> None:
    bad = (
        "from habana_frameworks.torch import hpex\n"
        "class _Block:\n"
        "    def forward(self, x):\n"
        "        if x.device.type == 'hpu' or x.device.type == 'mps' or x.device.type == 'cuda':\n"
        "            return x\n"
        "        return x\n"
    )
    found = _find_c6_vendor_references(bad, "dummy_model")
    assert found == {
        ("dummy_model", "<module>"),
        ("dummy_model", "_Block.forward"),
    }
    with pytest.raises(AssertionError, match="New C6"):
        _check_allowlist(found, set(), rule="C6", remediation="keep agnostic")

    benign = (
        "# Workaround for Gaudi / HPU real DFT\n"
        "class Dummy:\n"
        '    """Not implemented with CUDA or Triton kernels."""\n'
        "    def forward(self, x):\n"
        "        if x.device.type == 'cpu':\n"
        "            return 'throughput n_outputs'\n"
        "        return x\n"
    )
    assert _find_c6_vendor_references(benign, "dummy_model") == set()


def test_unallowlisted_private_cross_model_import_fails() -> None:
    bad = (
        "from braindecode.models.other import _priv_a\n"
        "from .another import _priv_b\n"
        "import braindecode.models.third as third\n"
        "import braindecode.models.fourth\n"
        "X = third._priv_c\n"
        "Y = braindecode.models.fourth._priv_d\n"
    )
    found = _find_c7_private_imports(bad, "dummy_model")
    assert found == {
        ("dummy_model", "other._priv_a"),
        ("dummy_model", "another._priv_b"),
        ("dummy_model", "third._priv_c"),
        ("dummy_model", "fourth._priv_d"),
    }
    with pytest.raises(AssertionError, match="New C7"):
        _check_allowlist(found, set(), rule="C7", remediation="move to modules")

    benign = (
        "from braindecode.models.util import _get_pad\n"
        "from braindecode.models.other import PublicBlock\n"
        "from .dummy_model import _self_helper\n"
    )
    assert _find_c7_private_imports(benign, "dummy_model") == set()


def test_unallowlisted_configuring_constant_fails_and_tables_ignored() -> None:
    bad = (
        "_CONST = 4\n"
        "class _Block:\n"
        "    def __init__(self, k=3, *, scale=1.5):\n"
        "        pass\n"
    )
    found = _find_c11_configuring_constants(bad, "dummy_model")
    assert found == {
        ("dummy_model", "_CONST"),
        ("dummy_model", "_Block.k"),
        ("dummy_model", "_Block.scale"),
    }
    with pytest.raises(AssertionError, match="New C11"):
        _check_allowlist(found, set(), rule="C11", remediation="expose kwarg")

    benign = (
        "from typing import TYPE_CHECKING\n"
        "__all__ = (1,)\n"
        "if TYPE_CHECKING:\n"
        "    _TYPE_ONLY = 1\n"
        "_TABLE = ((1, 2), (3, 4))\n"
        "class _Block:\n"
        "    def __init__(self, flag: bool = True):\n"
        "        pass\n"
    )
    assert _find_c11_configuring_constants(benign, "dummy_model") == set()
