"""
Labram module.
Authors: Wei-Bang Jiang
         Bruno Aristimunha <b.aristimunha@gmail.com>
License: BSD 3 clause
"""

from collections import OrderedDict
from warnings import warn

import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange
from torch.nn.init import trunc_normal_

from braindecode.functional import rescale_parameter
from braindecode.models.base import EEGModuleMixin
from braindecode.modules import MLP, DropPath


class Labram(EEGModuleMixin, nn.Module):
    """Labram from Jiang, W B et al (2024) [Jiang2024]_.

    .. figure:: https://arxiv.org/html/2405.18765v1/x1.png
        :align: center
        :alt: Labram Architecture.

    Large Brain Model for Learning Generic Representations with Tremendous
    EEG Data in BCI from [Jiang2024]_

    This is an **adaptation** of the code [Code2024]_ from the Labram model.

    The model is transformer architecture with **strong** inspiration from
    BEiTv2 [BeiTv2]_.

    The models can be used in two modes:
    - Neural Tokenizor: Design to get an embedding layers (e.g. classification).
    - Neural Decoder: To extract the ampliture and phase outputs with a VQSNP.

    The braindecode's modification is to allow the model to be used in
    with an input shape of (batch, n_chans, n_times), if neural tokenizer
    equals True. The original implementation uses (batch, n_chans, n_patches,
    patch_size) as input with static segmentation of the input data.

    The models have the following sequence of steps:
    if neural tokenizer:
        - SegmentPatch: Segment the input data in patches;
        - TemporalConv: Apply a temporal convolution to the segmented data;
        - Residual adding cls, temporal and position embeddings (optional);
        - WindowsAttentionBlock: Apply a windows attention block to the data;
        - LayerNorm: Apply layer normalization to the data;
        - Linear: An head linear layer to transformer the data into classes.

    else:
        - PatchEmbed: Apply a patch embedding to the input data;
        - Residual adding cls, temporal and position embeddings (optional);
        - WindowsAttentionBlock: Apply a windows attention block to the data;
        - LayerNorm: Apply layer normalization to the data;
        - Linear: An head linear layer to transformer the data into classes.

    .. versionadded:: 0.9

    Parameters
    ----------
    patch_size : int
        The size of the patch to be used in the patch embedding.
    emb_size : int
        The dimension of the embedding.
    in_channels : int
        The number of convolutional input channels.
    out_channels : int
        The number of convolutional output channels.
    n_layers :  int (default=12)
        The number of attention layers of the model.
    att_num_heads : int (default=10)
        The number of attention heads.
    mlp_ratio : float (default=4.0)
        The expansion ratio of the mlp layer
    qkv_bias :  bool (default=False)
        If True, add a learnable bias to the query, key, and value tensors.
    qk_norm : Pytorch Normalize layer (default=None)
        If not None, apply LayerNorm to the query and key tensors
    qk_scale : float (default=None)
        If not None, use this value as the scale factor. If None,
        use head_dim**-0.5, where head_dim = dim // num_heads.
    drop_prob : float (default=0.0)
        Dropout rate for the attention weights.
    attn_drop_prob : float (default=0.0)
        Dropout rate for the attention weights.
    drop_path_prob : float (default=0.0)
        Dropout rate for the attention weights used on DropPath.
    norm_layer : Pytorch Normalize layer (default=nn.LayerNorm)
        The normalization layer to be used.
    init_values : float (default=None)
        If not None, use this value to initialize the gamma_1 and gamma_2
        parameters.
    use_abs_pos_emb : bool (default=True)
        If True, use absolute position embedding.
    use_mean_pooling : bool (default=True)
        If True, use mean pooling.
    init_scale : float (default=0.001)
        The initial scale to be used in the parameters of the model.
    neural_tokenizer : bool (default=True)
        The model can be used in two modes: Neural Tokenizor or Neural Decoder.
    attn_head_dim : bool (default=None)
        The head dimension to be used in the attention layer, to be used only
        during pre-training.
    activation: nn.Module, default=nn.GELU
        Activation function class to apply. Should be a PyTorch activation
        module class like ``nn.ReLU`` or ``nn.ELU``. Default is ``nn.GELU``.

    References
    ----------
    .. [Jiang2024] Wei-Bang Jiang, Li-Ming Zhao, Bao-Liang Lu. 2024, May.
       Large Brain Model for Learning Generic Representations with Tremendous
       EEG Data in BCI. The Twelfth International Conference on Learning
       Representations, ICLR.
    .. [Code2024] Wei-Bang Jiang, Li-Ming Zhao, Bao-Liang Lu. 2024. Labram
       Large Brain Model for Learning Generic Representations with Tremendous
       EEG Data in BCI. GitHub https://github.com/935963004/LaBraM
       (accessed 2024-03-02)
    .. [BeiTv2] Zhiliang Peng, Li Dong, Hangbo Bao, Qixiang Ye, Furu Wei. 2024.
       BEiT v2: Masked Image Modeling with Vector-Quantized Visual Tokenizers.
       arXiv:2208.06366 [cs.CV]
    """

    def __init__(
        self,
        n_times=None,
        n_outputs=None,
        chs_info=None,
        n_chans=None,
        sfreq=None,
        input_window_seconds=None,
        patch_size=200,
        emb_size=200,
        in_channels=1,
        out_channels=8,
        n_layers=12,
        att_num_heads=10,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_norm=None,
        qk_scale=None,
        drop_prob=0.0,
        attn_drop_prob=0.0,
        drop_path_prob=0.0,
        norm_layer=nn.LayerNorm,
        init_values=None,
        use_abs_pos_emb=True,
        use_mean_pooling=True,
        init_scale=0.001,
        neural_tokenizer=True,
        attn_head_dim=None,
        activation: nn.Module = nn.GELU,
    ):
        super().__init__(
            n_outputs=n_outputs,
            n_chans=n_chans,
            chs_info=chs_info,
            n_times=n_times,
            input_window_seconds=input_window_seconds,
            sfreq=sfreq,
        )
        del n_outputs, n_chans, chs_info, n_times, input_window_seconds, sfreq

        self.patch_size = patch_size
        self.num_features = self.emb_size = emb_size
        self.neural_tokenizer = neural_tokenizer
        self.init_scale = init_scale

        if patch_size > self.n_times:
            warn(
                f"patch_size ({patch_size}) > n_times ({self.n_times}); "
                f"setting patch_size = {self.n_times}.",
                UserWarning,
            )
            self.patch_size = self.n_times
            self.num_features = None
            self.emb_size = None
        else:
            self.patch_size = patch_size
        self.n_path = self.n_times // self.patch_size

        if neural_tokenizer and in_channels != 1:
            warn(
                "The model is in Neural Tokenizer mode, but the variable "
                + "`in_channels` is different from the default values."
                + "`in_channels` is only needed for the Neural Decoder mode."
                + "in_channels is not used in the Neural Tokenizer mode.",
                UserWarning,
            )
            in_channels = 1
            # If you can use the model in Neural Tokenizer mode,
        # temporal conv layer will be use over the patched dataset
        if neural_tokenizer:
            self.patch_embed = nn.Sequential(
                OrderedDict(
                    [
                        (
                            "segment_patch",
                            _SegmentPatch(
                                n_times=self.n_times,
                                patch_size=self.patch_size,
                                n_chans=self.n_chans,
                                emb_dim=self.patch_size,
                            ),
                        ),
                        (
                            "temporal_conv",
                            _TemporalConv(
                                out_channels=out_channels, activation=activation
                            ),
                        ),
                    ]
                )
            )
        else:
            # If not, the model will be used as Neural Decoder mode
            # So the input here will be after the VQVAE encoder
            # To be used to extract the ampliture and phase outputs.
            # Adding inside a Sequential to use the same convention as the
            # Neural Tokenizer mode.
            self.patch_embed = nn.Sequential()
            self.patch_embed.add_module(
                "segment_patch",
                _PatchEmbed(
                    n_times=self.n_times,
                    patch_size=patch_size,
                    in_channels=in_channels,
                    emb_dim=self.emb_size,
                ),
            )

        with torch.no_grad():
            dummy = torch.zeros(1, self.n_chans, self.n_times)
            out = self.patch_embed(dummy)
        # out.shape for tokenizer: (1, n_chans, emb_dim)
        # for decoder:        (1, n_patch, patch_size, emb_dim), but we want last dim
        self.emb_size = out.shape[-1]
        self.num_features = self.emb_size

        # Defining the parameters
        # Creating a parameter list with cls token]
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.emb_size))
        # Positional embedding and time embedding are complementary
        # one is for the spatial information and the other is for the temporal
        # information.
        # The time embedding is used to encode something in the number of
        # patches, and the position embedding is used to encode the channels'
        # information.
        if use_abs_pos_emb:
            self.position_embedding = nn.Parameter(
                torch.zeros(1, self.n_chans + 1, self.emb_size),
                requires_grad=True,
            )
        else:
            self.position_embedding = None

        self.temporal_embedding = nn.Parameter(
            torch.zeros(1, self.patch_embed[0].n_patchs + 1, self.emb_size),
            requires_grad=True,
        )
        self.pos_drop = nn.Dropout(p=drop_prob)

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_prob, n_layers)
        ]  # stochastic depth decay rule
        self.blocks = nn.ModuleList(
            [
                _WindowsAttentionBlock(
                    dim=self.emb_size,
                    num_heads=att_num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_norm=qk_norm,
                    qk_scale=qk_scale,
                    drop=drop_prob,
                    attn_drop=attn_drop_prob,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    init_values=init_values,
                    window_size=(
                        self.patch_embed[0].patch_shape
                        if not neural_tokenizer
                        else None
                    ),
                    attn_head_dim=attn_head_dim,
                    activation=activation,
                )
                for i in range(n_layers)
            ]
        )
        self.norm = nn.Identity() if use_mean_pooling else norm_layer(self.emb_size)
        self.fc_norm = norm_layer(self.emb_size) if use_mean_pooling else None

        if self.n_outputs > 0:
            self.final_layer = nn.Linear(self.emb_size, self.n_outputs)
        else:
            self.final_layer = nn.Identity()

        self.apply(self._init_weights)
        self.fix_init_weight_and_init_embedding()

    def fix_init_weight_and_init_embedding(self):
        """
        Fix the initial weight and the initial embedding.
        Initializing with truncated normal distribution.
        """
        trunc_normal_(self.cls_token, std=0.02)
        trunc_normal_(self.temporal_embedding, std=0.02)

        if self.position_embedding is not None:
            trunc_normal_(self.position_embedding, std=0.02)

        if isinstance(self.final_layer, nn.Linear):
            trunc_normal_(self.final_layer.weight, std=0.02)

        for layer_id, layer in enumerate(self.blocks):
            rescale_parameter(layer.attn.proj.weight.data, layer_id + 1)
            rescale_parameter(layer.mlp[-2].weight.data, layer_id + 1)

        if isinstance(self.final_layer, nn.Linear):
            self.final_layer.weight.data.mul_(self.init_scale)
            self.final_layer.bias.data.mul_(self.init_scale)

    @staticmethod
    def _init_weights(layer):
        """
        Initialize the weights of the model for each layer layer.

        If the layer is a linear layer, the weight will be initialized
        with a truncated normal distribution with std=0.02.

        If m.bias is not None, the bias will be initialized with a constant
        value of 0.

        If the layer is a layer normalization layer, the bias will be
        initialized with a constant value of 0, and the weight will be
        initialized with a constant value of 1.

        Parameters
        ----------
        m : torch.nn.Module
            The layer of the pytorch model
        """

        if isinstance(layer, nn.Linear):
            trunc_normal_(layer.weight, std=0.02)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)
        elif isinstance(layer, nn.LayerNorm):
            nn.init.constant_(layer.bias, 0)
            nn.init.constant_(layer.weight, 1.0)

    def get_num_layers(self):
        """
        Convenience method to get the number of layers in the model.
        """
        return len(self.blocks)

    def forward_features(
        self,
        x,
        input_chans=None,
        return_patch_tokens=False,
        return_all_tokens=False,
    ):
        """
        Forward the features of the model.

        Parameters
        ----------
        x : torch.Tensor
            The input data with shape (batch, n_chans, n_patches, patch size),
            if neural decoder or (batch, n_chans, n_times), if neural tokenizer.
        input_chans : int
            The number of input channels.
        return_patch_tokens : bool
            Whether to return the patch tokens.
        return_all_tokens : bool
            Whether to return all the tokens.

        Returns
        -------
        x : torch.Tensor
            The output of the model.
        """
        if self.neural_tokenizer:
            batch_size, nch, n_patch, temporal = self.patch_embed.segment_patch(x).shape
        else:
            batch_size, nch, n_patch = self.patch_embed(x).shape
        x = self.patch_embed(x)
        # add the [CLS] token to the embedded patch tokens
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)

        x = torch.cat((cls_tokens, x), dim=1)

        # Positional Embedding
        if input_chans is not None:
            pos_embed_used = self.position_embedding[:, input_chans]
        else:
            pos_embed_used = self.position_embedding

        if self.position_embedding is not None:
            pos_embed = self._adj_position_embedding(
                pos_embed_used=pos_embed_used, batch_size=batch_size
            )
            x += pos_embed

        # The time embedding is added across the channels after the [CLS] token
        if self.neural_tokenizer:
            num_ch = self.n_chans
        else:
            num_ch = n_patch
        time_embed = self._adj_temporal_embedding(
            num_ch=num_ch, batch_size=batch_size, dim_embed=temporal
        )
        x[:, 1:, :] += time_embed

        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        if self.fc_norm is not None:
            if return_all_tokens:
                return self.fc_norm(x)
            temporal = x[:, 1:, :]
            if return_patch_tokens:
                return self.fc_norm(temporal)
            return self.fc_norm(temporal.mean(1))
        else:
            if return_all_tokens:
                return x
            elif return_patch_tokens:
                return x[:, 1:]
            return x[:, 0]

    def forward(
        self,
        x,
        input_chans=None,
        return_patch_tokens=False,
        return_all_tokens=False,
    ):
        """
        Forward the input EEG data through the model.

        Parameters
        ----------
        x: torch.Tensor
            The input data with shape (batch, n_chans, n_times)
            or (batch, n_chans, n_patches, patch size).
        input_chans: int
            An input channel to select some dimensions
        return_patch_tokens: bool
            Return the patch tokens
        return_all_tokens: bool
            Return all the tokens

        Returns
        -------
        torch.Tensor
            The output of the model with dimensions (batch, n_outputs)
        """
        x = self.forward_features(
            x,
            input_chans=input_chans,
            return_patch_tokens=return_patch_tokens,
            return_all_tokens=return_all_tokens,
        )
        x = self.final_layer(x)
        return x

    def get_classifier(self):
        """
        Get the classifier of the model.

        Returns
        -------
        torch.nn.Module
            The classifier of the head model.
        """
        return self.final_layer

    def reset_classifier(self, n_outputs):
        """
        Reset the classifier with the new number of classes.

        Parameters
        ----------
        n_outputs : int
            The new number of classes.
        """
        self.n_outputs = n_outputs
        self.final_layer = (
            nn.Linear(self.emb_dim, self.n_outputs)
            if self.n_outputs > 0
            else nn.Identity()
        )

    def _adj_temporal_embedding(self, num_ch, batch_size, dim_embed=None):
        """
        Adjust the dimensions of the time embedding to match the
        number of channels.

        Parameters
        ----------
        num_ch : int
            The number of channels or number of code books vectors.
        batch_size : int
            Batch size of the input data.

        Returns
        -------
        temporal_embedding : torch.Tensor
            The adjusted time embedding to be added across the channels
            after the [CLS] token. (x[:, 1:, :] += time_embed)
        """
        if dim_embed is None:
            cut_dimension = self.patch_size
        else:
            cut_dimension = dim_embed
        # first step will be match the time_embed to the number of channels
        temporal_embedding = self.temporal_embedding[:, 1:cut_dimension, :]
        # Add a new dimension to the time embedding
        # e.g. (batch, 62, 200) -> (batch, 1, 62, 200)
        temporal_embedding = temporal_embedding.unsqueeze(1)
        # Expand the time embedding to match the number of channels
        # or number of patches from
        temporal_embedding = temporal_embedding.expand(batch_size, num_ch, -1, -1)
        # Flatten the intermediate dimensions
        temporal_embedding = temporal_embedding.flatten(1, 2)
        return temporal_embedding

    def _adj_position_embedding(self, pos_embed_used, batch_size):
        """
        Adjust the dimensions of position embedding to match the
        number of patches.

        Parameters
        ----------
        pos_embed_used : torch.Tensor
            The position embedding to be adjusted.
        batch_size : int
            The number of batches.

        Returns
        -------
        pos_embed : torch.Tensor
            The adjusted position embedding
        """
        # [CLS] token has no position embedding
        pos_embed = pos_embed_used[:, 1:, :]
        # Adding a new dimension to the position embedding
        pos_embed = pos_embed.unsqueeze(2)
        # Need to expand the position embedding to match the number of
        # n_patches
        pos_embed = pos_embed.expand(batch_size, -1, self.patch_embed[0].n_patchs, -1)
        # Flatten the intermediate dimensions,
        # such as the number of patches and the "channels" dim
        pos_embed = pos_embed.flatten(1, 2)
        # Get the base position embedding
        # This is the position embedding for the [CLS] token
        base_pos = pos_embed[:, 0:1, :].expand(batch_size, -1, -1)
        # Concatenate the base position embedding with the
        # position embedding
        pos_embed = torch.cat((base_pos, pos_embed), dim=1)
        return pos_embed


class _SegmentPatch(nn.Module):
    """Segment and Patch for EEG data.

    Adapted Patch Embedding inspired in the Visual Transform approach
    to extract the learned segmentor, we expect get the input shape as:
    (Batch, Number of Channels, number of times points).

    We apply a 2D convolution with kernel size of (1, patch_size)
    and a stride of (1, patch_size).

    The results output shape will be:
    (Batch, Number of Channels, Number of patches, patch size).

    This way, we learned a convolution to segment the input shape.

    The number of patches is calculated as the number of samples divided
    by the patch size.

    Parameters:
    -----------
    n_times: int (default=2000)
        Number of temporal components of the input tensor.
    in_chans: int (default=1)
        number of electrods from the EEG signal
    emb_dim: int (default=200)
        Number of n_output to be used in the convolution, here,
        we used the same as patch_size.
    patch_size: int (default=200)
        Size of the patch, default is 1-seconds with 200Hz.
    Returns:
    --------
    x_patched: torch.Tensor
        Output tensor of shape (batch, n_chans, num_patches, emb_dim).
    """

    def __init__(
        self, n_times=2000, patch_size=200, n_chans=1, emb_dim=200, learned_patcher=True
    ):
        super().__init__()

        self.n_times = n_times
        self.patch_size = patch_size
        self.n_patchs = n_times // patch_size
        self.emb_dim = emb_dim
        self.n_chans = n_chans
        self.learned_patcher = learned_patcher

        self.patcher = nn.Conv1d(
            in_channels=1,
            out_channels=self.emb_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

        self.adding_extra_dim = Rearrange(
            pattern="batch nchans temporal -> (batch nchans) 1 temporal"
        )

    def forward(self, x):
        """
        Using an 1D convolution to generate segments of EEG signal.

        Parameters:
        -----------
        X: Tensor
            [batch, n_chans, n_times]

        Returns:
        --------
        X_patch: Tensor
            [batch, n_chans, n_times//patch_size, patch_size]
        """
        batch_size, _, _ = x.shape
        # Input shape: [batch, n_chs, n_times]

        # First, rearrange input to treat the channel dimension 'n_chs' as
        # separate 'dimension' in batch for Conv1d
        # This requires reshaping x to have a height of 1 for each EEG sample.
        if self.learned_patcher:
            x = self.adding_extra_dim(x)

            # Apply the convolution along the temporal dimension
            # Conv2d output shape: [(batch*n_chs), emb_dim, n_patches]
            x = self.patcher(x)

            # Now, rearrange output to get back to a batch-first format,
            # combining embedded patches with channel information
            # Assuming you want [batch, n_chs, n_patches, emb_dim]
            # as output, which keeps channel information
            # This treats each patch embedding as a feature alongside channels
            x = rearrange(
                x,
                pattern="(batch nchans) embed npatchs -> batch nchans npatchs embed",
                batch=batch_size,
                nchans=self.n_chans,
            )
        else:
            x = x.view(
                batch_size,
                self.n_chans,
                self.n_times // self.patch_size,
                self.patch_size,
            )
        return x


class _PatchEmbed(nn.Module):
    """EEG to Patch Embedding.

    This code is used when we want to apply the patch embedding
    after the codebook layer.

    Parameters:
    -----------
    n_times: int (default=2000)
        Number of temporal components of the input tensor.
    patch_size: int (default=200)
        Size of the patch, default is 1-seconds with 200Hz.
    in_channels: int (default=1)
        Number of input channels for to be used in the convolution.
    emb_dim: int (default=200)
        Number of out_channes to be used in the convolution, here,
        we used the same as patch_size.
    n_codebooks: int (default=62)
        Number of patches to be used in the convolution, here,
        we used the same as n_times // patch_size.
    """

    def __init__(
        self, n_times=2000, patch_size=200, in_channels=1, emb_dim=200, n_codebooks=62
    ):
        super().__init__()
        self.n_times = n_times
        self.patch_size = patch_size
        self.patch_shape = (1, self.n_times // self.patch_size)
        n_patchs = n_codebooks * (self.n_times // self.patch_size)

        self.n_patchs = n_patchs

        self.proj = nn.Conv2d(
            in_channels=in_channels,
            out_channels=emb_dim,
            kernel_size=(1, self.patch_size),
            stride=(1, self.patch_size),
        )

        self.merge_transpose = Rearrange(
            "Batch ch patch spatch -> Batch patch spatch ch",
        )

    def forward(self, x):
        """
        Apply the convolution to the input tensor.
        then merge the output tensor to the desired shape.

        Parameters:
        -----------
        x: torch.Tensor
            Input tensor of shape (Batch, Channels, n_patchs, patch_size).

        Return:
        -------
        x: torch.Tensor
            Output tensor of shape (Batch, n_patchs, patch_size, channels).
        """
        x = self.proj(x)
        x = self.merge_transpose(x)
        return x


class _Attention(nn.Module):
    """
    Attention with the options of Window-based multi-head self attention (W-MSA).

    This code is strong inspired by:
    https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer.py#L77

    Basically, the attention module is a linear layer that takes the input
    tensor and returns the output tensor. The input tensor is first passed
    through a linear layer to get the query, key, and value tensors. Then,
    the query tensor is multiplied by the scale factor and the result is
    multiplied by the transpose of the key tensor.

    The flag window_size is used to determine if the attention is
    window-based or not.

    Parameters:
    -----------
    dim: int
        Number of input features.
    num_heads: int (default=8)
        Number of attention heads.
    qkv_bias: bool (default=False)
        If True, add a learnable bias to the query, key, and value tensors.
    qk_norm: nn.LayerNorm (default=None)
        If not None, apply LayerNorm to the query and key tensors.
    qk_scale: float (default=None)
        If not None, use this value as the scale factor. If None,
        use head_dim**-0.5, where head_dim = dim // num_heads.
    attn_drop: float (default=0.0)
        Dropout rate for the attention weights.
    proj_drop: float (default=0.0)
        Dropout rate for the output tensor.
    window_size: bool (default=None)
        If not None, use window-based multi-head self attention based on Swin Transformer.
    attn_head_dim: int (default=None)
        If not None, use this value as the head_dim. If None, use dim // num_heads.
    """

    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_norm=None,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        window_size=None,
        attn_head_dim=None,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        if qk_norm is not None:
            self.q_norm = qk_norm(head_dim)
            self.k_norm = qk_norm(head_dim)
        else:
            self.q_norm = None
            self.k_norm = None

        if window_size:
            self.window_size = window_size
            self.num_relative_distance = (2 * window_size[0] - 1) * (
                2 * window_size[1] - 1
            ) + 3
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros(self.num_relative_distance, num_heads)
            )  # 2*Wh-1 * 2*Ww-1, nH
            # cls to token & token 2 cls & cls to cls

            # get pair-wise relative position index for each token inside the window
            coords_h = torch.arange(window_size[0])
            coords_w = torch.arange(window_size[1])
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
            coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            # 2, Wh*Ww, Wh*Ww
            relative_coords = relative_coords.permute(
                1, 2, 0
            ).contiguous()  # Wh*Ww, Wh*Ww, 2
            relative_coords[:, :, 0] += window_size[0] - 1
            # shift to start from 0
            relative_coords[:, :, 1] += window_size[1] - 1
            relative_coords[:, :, 0] *= 2 * window_size[1] - 1
            relative_position_index = torch.zeros(
                size=(window_size[0] * window_size[1] + 1,) * 2,
                dtype=relative_coords.dtype,
            )
            relative_position_index[1:, 1:] = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
            relative_position_index[0, 0:] = self.num_relative_distance - 3
            relative_position_index[0:, 0] = self.num_relative_distance - 2
            relative_position_index[0, 0] = self.num_relative_distance - 1

            self.register_buffer("relative_position_index", relative_position_index)
        else:
            self.window_size = None
            self.relative_position_bias_table = None
            self.relative_position_index = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
        self,
        x: torch.Tensor,
        return_attention=False,
        return_qkv=False,
    ):
        """
        Apply the attention mechanism to the input tensor.

        Parameters:
        -----------
        x: torch.Tensor
            Input tensor of shape (Batch, N, C).
        return_attention: bool (default=False)
            If True, return the attention weights.
        return_qkv: bool (default=False)
            If True, return the query, key, and value tensors together with
            the output tensor.
        Returns:
        --------
        x: torch.Tensor
            Output tensor of shape (Batch, N, C).
        qkv: torch.Tensor (optional)
            Query, key, and value tensors of shape
            (Batch, N, 3, num_heads, C // num_heads).
        """
        B, N, _ = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat(
                (
                    self.q_bias,
                    torch.zeros_like(self.v_bias, requires_grad=False),
                    self.v_bias,
                )
            )
        qkv = nn.functional.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple) (B, H, N, C)
        if self.q_norm is not None:
            q = self.q_norm(q).type_as(v)
        if self.k_norm is not None:
            k = self.k_norm(k).type_as(v)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        if self.relative_position_bias_table is not None:
            relative_position_bias = self.relative_position_bias_table[
                self.relative_position_index.view(-1)
            ].view(
                self.window_size[0] * self.window_size[1] + 1,
                self.window_size[0] * self.window_size[1] + 1,
                -1,
            )  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.permute(
                2, 0, 1
            ).contiguous()  # nH, Wh*Ww, Wh*Ww
            attn = attn + relative_position_bias.unsqueeze(0)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        if return_attention:
            return attn

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)

        x = self.proj(x)
        x = self.proj_drop(x)

        if return_qkv:
            return x, qkv

        return x


class _WindowsAttentionBlock(nn.Module):
    """Blocks of Windows Attention with Layer norm and MLP.

    Notes: This code is strong inspired by:
    BeiTv2 from Microsoft.

    Parameters:
    -----------
    dim: int
        Number of input features.
    num_heads: int (default=8)
        Number of attention heads.
    mlp_ratio: float (default=4.0)
        Ratio to increase the hidden features from input features in the MLP layer
    qkv_bias: bool (default=False)
        If True, add a learnable bias to the query, key, and value tensors.
    qk_norm: nn.LayerNorm (default=None)
        If not None, apply LayerNorm to the query and key tensors.
    qk_scale: float (default=None)
        If not None, use this value as the scale factor. If None,
        use head_dim**-0.5, where head_dim = dim // num_heads.
    drop: float (default=0.0)
        Dropout rate for the output tensor.
    attn_drop: float (default=0.0)
        Dropout rate for the attention weights.
    drop_path: float (default=0.0)
        Dropout rate for the output tensor.
    init_values: float (default=None)
        If not None, use this value to initialize the gamma_1 and gamma_2
        parameters.
    activation: nn.GELU (default)
        Activation function.
    norm_layer: nn.LayerNorm (default)
        Normalization layer.
    window_size: bool (default=None)
        If not None, use window-based multi-head self attention based on
        Swin Transformer.
    attn_head_dim: int (default=None)
        If not None, use this value as the head_dim. If None,
        the classes use dim // num_heads

    Returns:
    --------
    x: torch.Tensor
        Output tensor of shape (Batch, N, C). [I think]

    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_norm=None,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        init_values=None,
        activation: nn.Module = nn.GELU,
        norm_layer=nn.LayerNorm,
        window_size=None,
        attn_head_dim=None,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = _Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            window_size=window_size,
            attn_head_dim=attn_head_dim,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(
            in_features=dim,
            hidden_features=[mlp_hidden_dim],
            activation=activation,
            drop=drop,
        )

        if init_values is not None and init_values > 0:
            self.gamma_1 = nn.Parameter(
                init_values * torch.ones((dim)), requires_grad=True
            )
            self.gamma_2 = nn.Parameter(
                init_values * torch.ones((dim)), requires_grad=True
            )
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x, return_attention=False, return_qkv=False):
        """
        Apply the attention mechanism to the input tensor.
        Parameters
        ----------
        x: torch.Tensor
            Input tensor of shape (Batch, chs, npatchs, patch).
        return_attention: bool (default=False)
            If True, return the attention weights.
        return_qkv: bool (default=False)
            If True, return the query, key, and value tensors together with
            the output tensor.

        Returns
        -------
        torch.Tensor
            Output tensor of shape (Batch, chs, npatchs, patch).
        """

        if return_attention:
            return self.attn(self.norm1(x), return_attention=True)
        if return_qkv:
            y, qkv = self.attn(self.norm1(x), return_qkv=return_qkv)
            x = x + self.drop_path(self.gamma_1 * y)
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
            return x, qkv

        if self.gamma_1 is None:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class _TemporalConv(nn.Module):
    """
    Temporal Convolutional Module inspired by Visual Transformer.

    In this module we apply the follow steps three times repeatedly
    to the input tensor, reducing the temporal dimension only in the first.
    - Apply a 2D convolution.
    - Apply a GELU activation function.
    - Apply a GroupNorm with 4 groups.

    Parameters:
    -----------
    in_chans: int (default=1)
        Number of input channels.
    out_chans: int (default=8)
        Number of output channels.
    num_groups: int (default=4)
        Number of groups for GroupNorm.
    kernel_size_1: tuple (default=(1, 15))
        Kernel size for the first convolution.
    kernel_size_2: tuple (default=(1, 3))
        Kernel size for the second and third convolutions.
    stride_1: tuple (default=(1, 8))
        Stride for the first convolution.
    padding_1: tuple (default=(0, 7))
        Padding for the first convolution.
    padding_2: tuple (default=(0, 1))
        Padding for the second and third convolutions.
    activation: nn.Module, default=nn.GELU
        Activation function class to apply. Should be a PyTorch activation
        module class like ``nn.ReLU`` or ``nn.ELU``. Default is ``nn.GELU``.

    Returns:
    --------
    x: torch.Tensor
        Output tensor of shape (Batch, NA, Temporal Channel).
    """

    def __init__(
        self,
        in_channels=1,
        out_channels=8,
        num_groups=4,
        kernel_size_1=(1, 15),
        stride_1=(1, 8),
        padding_1=(0, 7),
        kernel_size_2=(1, 3),
        padding_2=(0, 1),
        activation: nn.Module = nn.GELU,
    ):
        super().__init__()

        # Here, we use the Rearrange layer from einops to flatten the input
        # tensor to a 2D tensor, so we can apply 2D convolutions.
        self.channel_patch_flatten = Rearrange(
            "Batch chs npat spatch -> Batch () (chs npat) spatch"
        )

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size_1,
            stride=stride_1,
            padding=padding_1,
        )
        self.act_layer_1 = activation()
        self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)

        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size_2,
            padding=padding_2,
        )
        self.act_layer_2 = activation()
        self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)

        self.conv3 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size_2,
            padding=padding_2,
        )
        self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)
        self.act_layer_3 = activation()

        self.transpose_temporal_channel = Rearrange("Batch C NA T -> Batch NA (T C)")

    def forward(self, x):
        """
        Apply 3 steps of 2D convolution, GELU activation function,
        and GroupNorm.

        Parameters:
        -----------
        x: torch.Tensor
            Input tensor of shape (Batch, Channels, n_patchs, size_patch).

        Returns:
        --------
        x: torch.Tensor
            Output tensor of shape (Batch, NA, Temporal Channel).
        """
        x = self.channel_patch_flatten(x)
        x = self.act_layer_1(self.norm1(self.conv1(x)))
        x = self.act_layer_2(self.norm2(self.conv2(x)))
        x = self.act_layer_3(self.norm3(self.conv3(x)))
        x = self.transpose_temporal_channel(x)
        return x
