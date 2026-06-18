"""CPU reference transcription of the upstream MVPFormer numerics.

This module is the *oracle* used to test braindecode's :class:`MVPFormer`
reimplementation for numerical equivalence. It is a faithful, CPU-only
transcription of the pure-PyTorch path of the original implementation
(https://github.com/IBM/multi-variate-parallel-transformer, Apache-2.0,
Copyright IBM Corp. 2024-2025), with all GPU-only machinery removed:

- ``flash_attn`` RMSNorm -> :class:`torch.nn.RMSNorm`
- the Triton ``flashmvpa`` kernel / ``MVPFormerGQAFlashAttention`` -> dropped
  (we transcribe only the documented CPU path ``MVPFormerGQAAttention._rel_attn``)
- ``deepspeed`` / ``loralib`` / ``torchtune`` / HF ``GPT2`` plumbing -> dropped

It lives under ``test/`` on purpose: it is not shipped, it only proves that the
braindecode modules compute the same thing as the reference. Components are
added just-in-time as the corresponding braindecode module is implemented
(TDD), so this file grows alongside ``braindecode/models/mvpformer.py``.
"""

from types import SimpleNamespace

import torch
from torch import nn

from braindecode.models.mvpformer import _WaveletPatchEmbed


def make_ref_config(**kwargs):
    """Minimal config namespace exposing the attributes the upstream attention
    reads off its ``GPT2Config``-derived config object."""
    defaults = dict(
        hidden_size=16,
        num_attention_heads=4,
        n_head_kv=2,
        n_inner=32,
        intermediate_size=32,
        global_att=True,
        max_position_embeddings=16,
        n_channels=16,
        n_layer=2,
        scale_attn_weights=True,
        scale_attn_by_inverse_layer_idx=False,
        attn_pdrop=0.0,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        layer_norm_epsilon=1e-5,
        hidden_act="silu",
        mlp_bias=False,
        pretraining_tp=1,
    )
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


class RefMVPAttention(nn.Module):
    """Near-verbatim transcription of upstream
    ``layers.mvpa.MVPFormerGQAAttention`` (the documented CPU path), with the
    LoRA / Flash / KV-cache / cross-attention branches removed."""

    def __init__(self, config, layer_idx=None):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.n_head_kv
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.embed_kv_dim = self.head_dim * self.num_kv_heads
        self.global_att = config.global_att
        self.scale_attn_weights = config.scale_attn_weights
        self.is_cross_attention = False
        self.scale_attn_by_inverse_layer_idx = config.scale_attn_by_inverse_layer_idx
        self.layer_idx = layer_idx
        self.q_attn = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.c_attn = nn.Linear(self.embed_dim, 2 * self.embed_kv_dim, bias=False)
        self.position_net = nn.Linear(self.embed_dim, self.embed_kv_dim, bias=False)
        self.channel_net = nn.Linear(self.embed_dim, self.embed_kv_dim, bias=False)
        self.c_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.attn_bias = nn.Parameter(torch.Tensor(3 * self.embed_kv_dim))
        nn.init.normal_(self.attn_bias, std=0.02)
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

    @staticmethod
    def repeat_kv(hidden_states, n_rep):
        batch, num_key_value_heads, clen, tlen, head_dim = hidden_states.shape
        if n_rep == 1:
            return hidden_states
        hidden_states = hidden_states[:, :, None, :, :, :].expand(
            batch, num_key_value_heads, n_rep, clen, tlen, head_dim
        )
        return hidden_states.reshape(
            batch, num_key_value_heads * n_rep, clen, tlen, head_dim
        )

    @staticmethod
    def repeat_channel(hidden_states, n_rep):
        batch, num_key_value_heads, head_dim, clen = hidden_states.shape
        if n_rep == 1:
            return hidden_states
        hidden_states = hidden_states[:, :, :, :, None].expand(
            batch, num_key_value_heads, head_dim, clen, n_rep
        )
        return hidden_states.reshape(batch, num_key_value_heads, head_dim, clen * n_rep)

    @staticmethod
    def repeat_time(hidden_states, n_rep):
        batch, num_key_value_heads, head_dim, tlen = hidden_states.shape
        if n_rep == 1:
            return hidden_states
        hidden_states = hidden_states[:, :, :, None, :].expand(
            batch, num_key_value_heads, head_dim, n_rep, tlen
        )
        return hidden_states.reshape(batch, num_key_value_heads, head_dim, n_rep * tlen)

    @staticmethod
    def _rel_shift(x):
        zero_pad_shape = x.size()[:2] + (x.size(3), 1)
        x_review_shape = x.size()[:2] + (x.size(3), x.size(2))
        zero_pad = torch.zeros(zero_pad_shape, device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x.view(x_review_shape)], dim=-1)
        x_padded_shape = x.size()[:2] + (x.size(2) + 1, x.size(3))
        x_padded = x_padded.view(
            x_padded_shape[0], x_padded_shape[1], x_padded_shape[2], x_padded_shape[3]
        )
        x = x_padded[..., 1:, :].view_as(x)
        return x

    @staticmethod
    def _rel_shift_chan(x):
        chan_size = x.shape[-1]
        if chan_size > 1:
            upper_val = torch.cat(
                [
                    torch.arange(1, chan_size - i, dtype=torch.int32)
                    for i in range(chan_size - 1)
                ]
            )
        else:
            upper_val = torch.tensor([], dtype=torch.int32)
        idxes = torch.triu_indices(chan_size, chan_size, offset=1)
        shifting_idxes = torch.zeros(chan_size, chan_size, dtype=torch.int32)
        shifting_idxes[..., idxes[0], idxes[1]] = upper_val
        shifting_idxes.transpose(-2, -1)[..., idxes[0], idxes[1]] = upper_val
        shifting_idxes = (chan_size - 1 - shifting_idxes).repeat(
            x.shape[-2] // chan_size, 1
        )
        return x[..., torch.arange(x.size(-2)).unsqueeze(1), shifting_idxes]

    def _split_heads(self, tensor, num_heads, attn_head_size):
        new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
        tensor = tensor.view(new_shape)
        return tensor.permute(0, 3, 1, 2, 4)

    def _merge_heads(self, tensor, num_heads, attn_head_size):
        tensor = tensor.permute(0, 2, 3, 1, 4).contiguous()
        new_shape = tensor.size()[:-2] + (num_heads * attn_head_size,)
        return tensor.view(new_shape)

    def _rel_attn(
        self,
        query,
        global_key,
        time_key,
        channel_key,
        value,
        attention_mask=None,
        head_mask=None,
    ):
        bsz, _, pos_len, chan_len, _ = query.size()
        tot_len = pos_len * chan_len
        global_key_bias, time_key_bias, channel_key_bias = self.attn_bias.split(
            self.embed_kv_dim, dim=0
        )
        if self.global_att:
            global_key_bias = self._split_heads(
                global_key_bias[None, None, None, :], self.num_kv_heads, self.head_dim
            )
        time_key_bias = self._split_heads(
            time_key_bias[None, None, None, :], self.num_kv_heads, self.head_dim
        )
        channel_key_bias = self._split_heads(
            channel_key_bias[None, None, None, :], self.num_kv_heads, self.head_dim
        )
        if self.global_att:
            global_key_bias = self.repeat_kv(global_key_bias, self.num_kv_groups)
        channel_key_bias = self.repeat_kv(channel_key_bias, self.num_kv_groups)
        time_key_bias = self.repeat_kv(time_key_bias, self.num_kv_groups)
        if self.global_att:
            global_key = self.repeat_kv(global_key, self.num_kv_groups)
            global_key = global_key.reshape(
                bsz, self.num_heads, tot_len, self.head_dim
            ).transpose(-1, -2)
        time_key = self.repeat_kv(time_key, self.num_kv_groups)
        channel_key = self.repeat_kv(channel_key, self.num_kv_groups)
        time_key = time_key.squeeze(-2).transpose(-1, -2)
        channel_key = channel_key.squeeze(-3).transpose(-1, -2)
        if self.global_att:
            global_query_head = (query + global_key_bias).reshape(
                bsz, self.num_heads, -1, self.head_dim
            )
        time_query_head = (query + time_key_bias).reshape(
            bsz, self.num_heads, -1, self.head_dim
        )
        channel_query_head = (query + channel_key_bias).reshape(
            bsz, self.num_heads, -1, self.head_dim
        )
        if self.global_att:
            global_att = torch.matmul(global_query_head, global_key)
        time_att = torch.matmul(time_query_head, time_key)
        channel_att = torch.matmul(channel_query_head, channel_key)
        time_att = self._rel_shift(time_att)
        channel_att = self._rel_shift_chan(channel_att)
        attn_weights = self.repeat_channel(time_att, chan_len) + self.repeat_time(
            channel_att, pos_len
        )
        if self.global_att:
            window_mask = torch.logical_and(
                torch.tril(
                    torch.ones((pos_len, pos_len), device=query.device, dtype=bool),
                    diagonal=10,
                ),
                torch.triu(
                    torch.ones((pos_len, pos_len), device=query.device, dtype=bool),
                    diagonal=-10,
                ),
            )
            window_mask = window_mask.repeat_interleave(chan_len, 0).repeat_interleave(
                chan_len, 1
            )
            window_mask[-chan_len:] = 1
            attn_weights += global_att.masked_fill(~window_mask, 0.0)
        if self.scale_attn_weights:
            attn_weights = attn_weights / torch.full(
                [],
                value.size(-1) ** 0.5,
                dtype=attn_weights.dtype,
                device=attn_weights.device,
            )
        if self.scale_attn_by_inverse_layer_idx:
            attn_weights = attn_weights / float(self.layer_idx + 1)
        if not self.is_cross_attention:
            causal_mask = (
                torch.tril(
                    torch.ones((pos_len, pos_len), device=query.device, dtype=bool)
                )
                .repeat_interleave(chan_len, 0)
                .repeat_interleave(chan_len, 1)
            )
            mask_value = torch.finfo(attn_weights.dtype).min
            attn_weights = attn_weights.masked_fill(~causal_mask, mask_value)
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        if value.dtype == torch.float16:
            attn_weights = torch.nn.functional.softmax(
                attn_weights, dim=-1, dtype=torch.bfloat16
            ).to(value.dtype)
        else:
            attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        if head_mask is not None:
            attn_weights = attn_weights * head_mask
        value = self.repeat_kv(value, self.num_kv_groups)
        attn_output = torch.matmul(attn_weights, value.flatten(2, 3))
        return attn_output.view_as(value), attn_weights

    def forward(
        self,
        hidden_states,
        positional_embedding,
        channel_embedding,
        attention_mask=None,
        head_mask=None,
    ):
        query = self.q_attn(hidden_states)
        key, value = self.c_attn(hidden_states).split(self.embed_kv_dim, dim=-1)
        time_key = self.position_net(positional_embedding).unsqueeze(2)
        channel_key = self.channel_net(channel_embedding).unsqueeze(1)
        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_kv_heads, self.head_dim)
        value = self._split_heads(value, self.num_kv_heads, self.head_dim)
        time_key = self._split_heads(time_key, self.num_kv_heads, self.head_dim)
        channel_key = self._split_heads(channel_key, self.num_kv_heads, self.head_dim)
        attn_output, _ = self._rel_attn(
            query, key, time_key, channel_key, value, attention_mask, head_mask
        )
        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)
        return attn_output


class RefBlock(nn.Module):
    """Transcription of upstream ``models.mvpformer.MVPFormerBlock``.

    Parallel attention + MLP (Megatron/Wang2021 style):
    ``x + attn(ln_1(x)) + mlp(ln_1(x))``. Note upstream instantiates ``ln_2``
    but never uses it in forward (it is dead); kept here so the reference can
    load a real checkpoint that still carries ``ln_2.weight``."""

    def __init__(self, config, layer_idx=None):
        super().__init__()
        from transformers.models.llama.modeling_llama import LlamaMLP

        self.ln_1 = nn.RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.attn = RefMVPAttention(config, layer_idx=layer_idx)
        self.ln_2 = nn.RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.mlp = LlamaMLP(config)

    def forward(
        self,
        hidden_states,
        positional_embedding,
        channel_embedding,
        attention_mask=None,
    ):
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_output = self.attn(
            hidden_states, positional_embedding, channel_embedding, attention_mask
        )
        mlp_output = self.mlp(hidden_states)
        return residual + mlp_output + attn_output


class RefMVPFormerModel(nn.Module):
    """Full upstream classification path: per-segment wavelet encoder ->
    learnable segment/channel embeddings -> stack of MVPA blocks -> RMSNorm ->
    last-segment channel-mean pooling -> linear head. Mirrors upstream
    ``MVPFormerModel.forward`` + ``ClassificationHMVPFormer`` pooling.

    Input is the pre-segmented signal ``(B, n_segments, n_chans, segment_len)``.
    """

    def __init__(self, config, segment_len, n_outputs):
        super().__init__()
        self.encoder = _WaveletPatchEmbed(segment_len, config.hidden_size)
        self.positional_embedding = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )
        self.channel_embedding = nn.Embedding(config.n_channels, config.hidden_size)
        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList(
            [RefBlock(config, layer_idx=i) for i in range(config.n_layer)]
        )
        self.ln_f = nn.RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.head = nn.Linear(config.hidden_size, n_outputs, bias=False)

    def forward(self, x):
        bsz, seg, ch, length = x.shape
        x_flat = x.reshape(bsz * seg * ch, length)
        embeds = self.encoder(x_flat).reshape(bsz, seg, ch, -1)
        pos = self.positional_embedding(torch.arange(seg)).unsqueeze(0)
        chan = self.channel_embedding(torch.arange(ch)).unsqueeze(0)
        hidden = self.drop(embeds)
        for block in self.h:
            hidden = block(hidden, pos, chan)
        hidden = self.ln_f(hidden)
        pooled = hidden[:, -1].mean(dim=1)
        return self.head(pooled)
