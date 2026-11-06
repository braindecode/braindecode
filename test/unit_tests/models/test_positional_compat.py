"""Positional calls written for braindecode 1.8.1 bind the same way.

New ``__init__`` parameters must be added after the 1.8.1 ones: inserting one in
the middle shifts every later positional argument by one (for example
``on_non_divisible`` in Labram, CBraMod and LUNA took the value meant for
``embed_dim``, ``dim_feedforward`` or ``num_queries``).
"""

import inspect

import braindecode.models as models

# Positional-or-keyword parameters of each model's ``__init__`` in 1.8.1, in order.
_POSITIONAL_1_8_1 = """
ATCNet: n_chans n_outputs input_window_seconds sfreq conv_block_n_filters conv_block_kernel_length_1 conv_block_kernel_length_2 conv_block_pool_size_1 conv_block_pool_size_2 conv_block_depth_mult conv_block_dropout n_windows head_dim num_heads att_drop_prob tcn_depth tcn_kernel_size tcn_drop_prob tcn_activation concat max_norm_const conv_max_norm_const chs_info n_times
AttentionBaseNet: n_times n_chans n_outputs chs_info sfreq input_window_seconds n_temporal_filters temp_filter_length_inp spatial_expansion pool_length_inp pool_stride_inp drop_prob_inp ch_dim temp_filter_length pool_length pool_stride drop_prob_attn attention_mode reduction_rate use_mlp freq_idx n_codewords kernel_size activation extra_params
AttnSleep: sfreq n_tce d_model d_ff n_attn_heads drop_prob activation_mrcnn activation input_window_seconds n_outputs after_reduced_cnn_size return_feats chs_info n_chans n_times
BDTCN: n_chans n_outputs chs_info n_times sfreq input_window_seconds n_blocks n_filters kernel_size drop_prob activation
BENDR: n_chans n_outputs n_times chs_info input_window_seconds sfreq encoder_h contextualizer_hidden projection_head drop_prob layer_drop activation transformer_layers transformer_heads position_encoder_length enc_width enc_downsample start_token final_layer encoder_only
BIOT: embed_dim num_heads num_layers sfreq hop_length return_feature n_outputs n_chans chs_info n_times input_window_seconds activation drop_prob max_seq_len att_drop_prob att_layer_drop_prob
BrainModule: n_chans n_outputs n_times sfreq chs_info input_window_seconds hidden_dim depth kernel_size growth dilation_growth dilation_period use_merger n_virtual_channels merger_drop_prob conv_drop_prob dropout_input batch_norm activation n_subjects subject_dim subject_layers subject_layers_dim subject_layers_id embedding_scale n_fft fft_complex channel_dropout_prob channel_dropout_type glu glu_context
CBraMod: n_outputs n_chans chs_info n_times input_window_seconds sfreq patch_size dim_feedforward n_layer nhead activation emb_dim channels_kernel_stride_padding_norm drop_prob return_encoder_output
CTNet: n_outputs n_chans sfreq chs_info n_times input_window_seconds activation_patch activation_transformer cnn_drop_prob att_positional_drop_prob final_drop_prob num_heads embed_dim num_layers n_filters_time kernel_size depth_multiplier pool_size_1 pool_size_2
CodeBrain: n_outputs n_chans chs_info n_times input_window_seconds sfreq patch_size res_channels skip_channels out_channels num_res_layers drop_prob s4_bidirectional s4_layernorm s4_lmax s4_d_state conv_out_chans conv_groups proj_kernel_size proj_padding proj_refine_kernel pos_kernel spectral_dropout mlp_hidden_multiplier swa_window_size codebook_size_t codebook_size_f pretrain_mode activation
ContraWR: n_chans n_outputs sfreq emb_size res_channels steps activation drop_prob stride_res kernel_size_res padding_res chs_info n_times input_window_seconds
DANCE: n_outputs n_chans chs_info n_times input_window_seconds sfreq
DGCNN: n_outputs n_chans chs_info n_times input_window_seconds sfreq n_filters cheb_order n_neighbors mlp_dims activation drop_prob
Deep4Net: n_chans n_outputs n_times final_conv_length n_filters_time n_filters_spat filter_time_length pool_time_length pool_time_stride n_filters_2 filter_length_2 n_filters_3 filter_length_3 n_filters_4 filter_length_4 activation_first_conv_nonlin first_pool_mode first_pool_nonlin activation_later_conv_nonlin later_pool_mode later_pool_nonlin drop_prob split_first_layer batch_norm batch_norm_alpha stride_before_pool chs_info input_window_seconds sfreq
DeepSleepNet: n_outputs return_feats n_chans chs_info n_times input_window_seconds sfreq activation_large activation_small drop_prob bilstm_hidden_size bilstm_num_layers small_n_filters_1 small_n_filters_2 small_first_kernel_size small_first_stride small_first_padding small_pool1_kernel_size small_pool1_stride small_pool1_padding small_deep_kernel_size small_pool2_kernel_size small_pool2_stride small_pool2_padding large_n_filters_1 large_n_filters_2 large_first_kernel_size large_first_stride large_first_padding large_pool1_kernel_size large_pool1_stride large_pool1_padding large_deep_kernel_size large_pool2_kernel_size large_pool2_stride large_pool2_padding
EEGConformer: n_outputs n_chans n_filters_time filter_time_length pool_time_length pool_time_stride drop_prob num_layers num_heads att_drop_prob final_fc_length return_features activation activation_transfor n_times chs_info input_window_seconds sfreq
EEGDINO: n_outputs n_chans chs_info n_times input_window_seconds sfreq patch_size n_layer nhead dim_feedforward channels_kernel_stride_padding_norm n_channel_embeddings n_global_tokens global_token_layer activation patch_activation drop_prob return_features return_encoder_output
EEGITNet: n_outputs n_chans n_times chs_info input_window_seconds sfreq n_filters_time kernel_length pool_kernel tcn_in_channel tcn_kernel_size tcn_padding drop_prob tcn_dilatation activation
EEGInceptionERP: n_chans n_outputs n_times sfreq drop_prob scales_samples_s n_filters activation batch_norm_alpha depth_multiplier pooling_sizes chs_info input_window_seconds
EEGInceptionMI: n_chans n_outputs input_window_seconds sfreq n_convs n_filters kernel_unit_s activation chs_info n_times
EEGMiner: method n_chans n_outputs n_times chs_info input_window_seconds sfreq filter_f_mean filter_bandwidth filter_shape group_delay clamp_f_mean
EEGNeX: n_chans n_outputs n_times chs_info input_window_seconds sfreq activation depth_multiplier filter_1 filter_2 drop_prob kernel_block_1_2 kernel_block_4 dilation_block_4 avg_pool_block4 kernel_block_5 dilation_block_5 avg_pool_block5 max_norm_conv max_norm_linear
EEGNet: n_chans n_outputs n_times final_conv_length pool_mode F1 D F2 kernel_length
EEGPT: n_outputs n_chans chs_info n_times input_window_seconds sfreq patch_size patch_stride embed_num embed_dim depth num_heads mlp_ratio drop_prob attn_drop_rate drop_path_rate init_std qkv_bias patch_module norm_layer layer_norm_eps return_encoder_output chan_proj_type n_chans_target chan_conv_max_norm final_layer
EEGSimpleConv: n_outputs n_chans sfreq feature_maps n_convs resampling_freq kernel_size return_feature activation chs_info n_times input_window_seconds
EEGSym: n_chans n_outputs n_times chs_info input_window_seconds sfreq filters_per_branch scales_time drop_prob activation spatial_resnet_repetitions left_right_chs middle_chs
EEGTCNet: n_chans n_outputs n_times chs_info input_window_seconds sfreq activation depth_multiplier filter_1 kern_length drop_prob depth kernel_size filters max_norm_const drop_prob_eeg drop_prob_tcn tcn_batch_norm
EMG2QwertyNet: n_outputs n_chans sfreq num_bands electrodes_per_band n_fft hop_length log_eps mlp_features rotation_offsets pooling block_channels kernel_width log_softmax activation drop_prob spec_augment n_time_masks time_mask_param n_freq_masks freq_mask_param spec_augment_prob return_feature n_times input_window_seconds chs_info
FBCNet: n_chans n_outputs chs_info n_times input_window_seconds sfreq n_bands n_filters_spat temporal_layer n_dim stride_factor activation linear_max_norm cnn_max_norm filter_parameters
FBLightConvNet: n_chans n_outputs chs_info n_times input_window_seconds sfreq n_bands n_filters_spat n_dim stride_factor win_len heads weight_softmax bias activation verbose filter_parameters
FBMSNet: n_chans n_outputs chs_info n_times input_window_seconds sfreq n_bands n_filters_spat temporal_layer n_dim stride_factor dilatability activation kernels_weights cnn_max_norm linear_max_norm verbose filter_parameters return_features
IFNet: n_chans n_outputs n_times chs_info input_window_seconds sfreq bands n_filters_spat kernel_sizes stride_factor drop_prob linear_max_norm activation verbose filter_parameters
LUNA: n_outputs n_chans n_times sfreq chs_info input_window_seconds patch_size num_queries embed_dim depth num_heads mlp_ratio norm_layer drop_path drop_prob_chan attn_drop activation
Labram: n_times n_outputs chs_info n_chans sfreq input_window_seconds patch_size learned_patcher embed_dim conv_in_channels conv_out_channels num_layers num_heads mlp_ratio qkv_bias qk_norm qk_scale drop_prob attn_drop_prob drop_path_prob norm_layer init_values use_abs_pos_emb use_mean_pooling init_scale neural_tokenizer attn_head_dim activation
MEDFormer: n_chans n_outputs n_times chs_info input_window_seconds sfreq patch_len_list embed_dim num_heads drop_prob no_inter_attn num_layers dim_feedforward activation_trans single_channel output_attention activation_class
MSVTNet: n_chans n_outputs n_times input_window_seconds sfreq chs_info n_filters_list conv1_kernels_size conv2_kernel_size depth_multiplier pool1_size pool2_size drop_prob num_heads ffn_expansion_factor att_drop_prob num_layers activation return_features
MVPFormer: n_outputs n_chans chs_info n_times input_window_seconds sfreq
MetaNeuromotorHand: n_outputs n_chans sfreq mpf_window_length mpf_stride mpf_n_fft mpf_fft_stride mpf_frequency_bins mask_max_num_masks mask_max_lengths mask_dims mask_value invariance_hidden_dims invariance_offsets num_adjacent_cov conformer_input_dim conformer_ffn_dim conformer_kernel_size conformer_stride conformer_num_heads conformer_attn_window_size conformer_num_layers drop_prob time_reduction_stride log_softmax activation invariance_activation n_times input_window_seconds chs_info
NeuroPose: n_outputs n_chans chs_info n_times input_window_seconds sfreq
PBT: n_chans n_outputs n_times chs_info input_window_seconds sfreq d_input embed_dim num_layers num_heads drop_prob learnable_cls bias_transformer activation
REVE: n_outputs n_chans chs_info n_times input_window_seconds sfreq embed_dim depth heads head_dim mlp_dim_ratio use_geglu freqs patch_size patch_overlap attention_pooling
SCCNet: n_chans n_outputs n_times chs_info input_window_seconds sfreq n_spatial_filters n_spatial_filters_smooth drop_prob activation batch_norm_momentum
SPARCNet: n_chans n_times n_outputs block_layers growth_rate bottleneck_size drop_prob conv_bias batch_norm activation kernel_size_conv0 kernel_size_conv1 kernel_size_conv2 kernel_size_pool stride_pool stride_conv0 stride_conv1 stride_conv2 padding_pool padding_conv0 padding_conv2 kernel_size_trans stride_trans chs_info input_window_seconds sfreq
SSTDPN: n_chans n_times n_outputs input_window_seconds sfreq chs_info n_spectral_filters_temporal n_fused_filters temporal_conv_kernel_size mvp_kernel_sizes return_features proto_sep_maxnorm proto_cpt_std spt_attn_global_context_kernel spt_attn_epsilon spt_attn_mode activation
STEEGFormer: n_outputs n_chans chs_info n_times input_window_seconds sfreq patch_size embed_dim depth num_heads mlp_ratio drop_prob drop_path activation global_pool n_chans_pos chan_pos_idx
SensingDynamics: n_outputs n_chans chs_info n_times input_window_seconds sfreq
ShallowFBCSPNet: n_chans n_outputs n_times n_filters_time filter_time_length n_filters_spat pool_time_length pool_time_stride final_conv_length conv_nonlin pool_mode activation_pool_nonlin split_first_layer batch_norm batch_norm_alpha drop_prob chs_info input_window_seconds sfreq
SignalJEPA: n_outputs n_chans chs_info n_times input_window_seconds sfreq
SignalJEPA_Contextual: n_outputs n_chans chs_info n_times input_window_seconds sfreq
SignalJEPA_PostLocal: n_outputs n_chans chs_info n_times input_window_seconds sfreq
SignalJEPA_PreLocal: n_outputs n_chans chs_info n_times input_window_seconds sfreq
SincShallowNet: num_time_filters time_filter_len depth_multiplier activation drop_prob first_freq min_freq freq_stride padding bandwidth pool_size pool_stride n_chans n_outputs n_times input_window_seconds sfreq chs_info
SleepStagerBlanco2020: n_chans sfreq n_conv_chans input_window_seconds n_outputs n_groups max_pool_size drop_prob apply_batch_norm return_feats activation chs_info n_times
SleepStagerChambon2018: n_chans sfreq n_conv_chs time_conv_size_s max_pool_size_s pad_size_s activation input_window_seconds n_outputs drop_prob apply_batch_norm return_feats chs_info n_times
SyncNet: n_chans n_times n_outputs chs_info input_window_seconds sfreq num_filters filter_width pool_size activation ampli_init_values omega_init_values beta_init_values phase_init_values
TCFormer: n_outputs n_chans chs_info n_times input_window_seconds sfreq n_filters_time temp_kernel_lengths depth_multiplier pool_length_1 pool_length_2 temp_kernel_length_2 group_dim se_reduction n_transformer_layers q_heads kv_heads mlp_ratio drop_path_max tcn_depth tcn_kernel_length classifier_max_norm drop_prob_conv drop_prob_trans drop_prob_tcn activation activation_ffn
TIDNet: n_chans n_outputs n_times input_window_seconds sfreq chs_info s_growth t_filters drop_prob pooling temp_layers spat_layers temp_span bottleneck summary activation
TSception: n_chans n_outputs input_window_seconds chs_info n_times sfreq number_filter_temp number_filter_spat hidden_size drop_prob activation pool_size inception_windows
USleep: n_chans sfreq depth n_time_filters complexity_factor with_skip_connection n_outputs input_window_seconds time_conv_size_s ensure_odd_conv_size activation chs_info n_times
VEMG2Pose: n_outputs n_chans chs_info n_times input_window_seconds sfreq
ZUNA: n_outputs n_chans chs_info n_times input_window_seconds sfreq
"""
POSITIONAL_1_8_1 = {
    name.strip(): params.split()
    for name, params in (
        line.split(":") for line in _POSITIONAL_1_8_1.strip().splitlines()
    )
}


def test_1_8_1_positional_parameters_keep_their_position():
    # One test over all models: a parametrized REVE case would be skipped as a
    # network test by the conftest.
    moved = {}
    for name, old in POSITIONAL_1_8_1.items():
        params = [
            p.name
            for p in inspect.signature(getattr(models, name)).parameters.values()
            if p.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
        ]
        if params[: len(old)] != old:
            moved[name] = params[: len(old)]
    assert len(POSITIONAL_1_8_1) == 61
    assert not moved, moved
