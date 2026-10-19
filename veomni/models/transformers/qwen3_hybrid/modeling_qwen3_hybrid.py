# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Qwen3-Hybrid: Qwen3 with Kimi Delta Attention (KDA) linear-attention layers and NoPE full-attention layers.

The full-attention layers, MLPs, norms, embeddings and LM head are the VeOmni Qwen3 modules, so a Qwen3-Hybrid and
a Qwen3 of the same shape differ only in the token mixer of the ``linear_attention`` layers (and RoPE on the
others). There is no Hugging Face modeling to patch, so this file binds ``VeomniOp`` handles directly.

KDA (Kimi Linear, arXiv:2510.26692) layer, per head with ``K = linear_head_dim``, ``V = linear_value_head_dim``:
``q, k, v`` = SiLU(short causal conv(linear(x))); ``q, k`` are L2-normalized in the kernel; per-channel forget gate
``g = -exp(A_log) * softplus(f_proj(x) + dt_bias)`` (low-rank ``f_proj``); ``beta = sigmoid(b_proj(x))``; the
gated delta rule runs over ``[K, V]`` states; the output is ``o_proj(RMSNorm(o) * sigmoid(g_proj(x)))``.
Parameter names follow ``fla.layers.KimiDeltaAttention``.

Generation uses the transformers ``DynamicCache`` built from ``layer_types``: a KV cache on the full-attention layers
and, on each KDA layer, the ``[B, H, K, V]`` recurrent state plus the last ``linear_conv_kernel_dim`` inputs of the
q / k / v short convolutions (three conv states). Cached prefill expects left padding, as ``generate`` does.
"""

from functools import partial

import torch
import torch.nn.functional as F
from torch import nn
from transformers import initialization as init
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.masking_utils import create_causal_mask
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs
from transformers.utils.generic import merge_with_config_defaults
from transformers.utils.output_capturing import capture_outputs

from veomni.models.loss_utils import ForCausalLMLoss
from veomni.models.utils.attention_utils import prepare_dense_attention_inputs
from veomni.ops import VeomniOp
from veomni.ops.config import resolve_op_impl
from veomni.utils.device import IS_NPU_AVAILABLE

from .configuration_qwen3_hybrid import LINEAR_ATTENTION, Qwen3HybridConfig


if IS_NPU_AVAILABLE:
    from ..qwen3.generated.patched_modeling_qwen3_npu import (
        Qwen3Attention,
        Qwen3ForCausalLM,
        Qwen3MLP,
        Qwen3RMSNorm,
        Qwen3RotaryEmbedding,
    )
else:
    from ..qwen3.generated.patched_modeling_qwen3_gpu import (
        Qwen3Attention,
        Qwen3ForCausalLM,
        Qwen3MLP,
        Qwen3RMSNorm,
        Qwen3RotaryEmbedding,
    )


# Inference inputs at most this long use the token-recurrent KDA kernel instead of the chunked one (FLA's choice).
FUSED_RECURRENT_MAX_LEN = 64


def _no_rope(query: torch.Tensor, key: torch.Tensor, cos: object, sin: object) -> tuple[torch.Tensor, torch.Tensor]:
    """Identity stand-in for the ``rope`` op: NoPE attention."""
    return query, key


class KDAGatedRMSNorm(nn.Module):
    """Per-head ``weight * RMSNorm(o) * sigmoid(gate)`` on the KDA output (FLA ``FusedRMSNormGated``)."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        self.veomni_rms_norm_gated = VeomniOp(
            "rms_norm_gated", "standard", resolve_op_impl("rms_norm_gated_implementation")
        )

    def forward(self, hidden_states: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        shape = hidden_states.shape
        out = self.veomni_rms_norm_gated(
            hidden_states.reshape(-1, shape[-1]),
            gate.reshape(-1, shape[-1]),
            self.weight,
            eps=self.variance_epsilon,
            activation="sigmoid",
        )
        return out.view(shape)


class KimiDeltaAttention(nn.Module):
    """KDA token mixer: varlen (``cu_seqlens``) training through the FLA chunk kernel; padded batches and cached
    generation (prefill, then decode from the cached conv windows and recurrent state) at inference."""

    # Cache conv-state slots of the q, k and v short convolutions.
    CONV_STATE_IDX = {"q": 0, "k": 1, "v": 2}

    def __init__(self, config: Qwen3HybridConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.linear_num_heads
        self.head_k_dim = config.linear_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.num_heads * self.head_k_dim
        self.value_dim = self.num_heads * self.head_v_dim
        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.allow_neg_eigval = config.linear_allow_neg_eigval
        gate_rank = config.linear_gate_low_rank_dim or self.head_v_dim

        self.q_proj = nn.Linear(self.hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.value_dim, bias=False)
        # ``*conv1d*`` names keep these depthwise kernels on AdamW under Muon (``_DEFAULT_ADAMW_NAME_PATTERNS``).
        self.q_conv1d = self._short_conv(self.key_dim)
        self.k_conv1d = self._short_conv(self.key_dim)
        self.v_conv1d = self._short_conv(self.value_dim)
        self.f_proj = nn.Sequential(
            nn.Linear(self.hidden_size, gate_rank, bias=False),
            nn.Linear(gate_rank, self.key_dim, bias=False),
        )
        self.b_proj = nn.Linear(self.hidden_size, self.num_heads, bias=False)
        self.A_log = nn.Parameter(torch.empty(self.num_heads, dtype=torch.float32))
        self.dt_bias = nn.Parameter(torch.empty(self.key_dim, dtype=torch.float32))
        self.g_proj = nn.Sequential(
            nn.Linear(self.hidden_size, gate_rank, bias=False),
            nn.Linear(gate_rank, self.value_dim, bias=True),
        )
        self.o_norm = KDAGatedRMSNorm(self.head_v_dim, eps=config.rms_norm_eps)
        self.o_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)

        self.veomni_causal_conv1d = VeomniOp(
            "causal_conv1d", "standard", resolve_op_impl("causal_conv1d_implementation")
        )
        kda_impl = resolve_op_impl("kda_implementation")
        self.veomni_chunk_kda = VeomniOp("chunk_kda", "standard", kda_impl)
        self.veomni_fused_recurrent_kda = VeomniOp("fused_recurrent_kda", "standard", kda_impl)

    def _short_conv(self, channels: int) -> nn.Conv1d:
        return nn.Conv1d(
            channels,
            channels,
            kernel_size=self.conv_kernel_size,
            groups=channels,
            padding=self.conv_kernel_size - 1,
            bias=False,
        )

    def _conv(
        self,
        x: torch.Tensor,
        name: str,
        cu_seqlens: torch.Tensor | None,
        cache: Cache | None,
        continue_from_cache: bool,
    ) -> torch.Tensor:
        """SiLU(causal depthwise conv) of ``x`` ``[B, T, D]``; with a cache, store its last inputs and, when continuing,
        convolve over the cached window followed by ``x``."""
        conv = getattr(self, f"{name}_conv1d")
        weight = conv.weight.squeeze(1)
        if cache is not None:
            window = cache.update_conv_state(
                x.transpose(1, 2),
                self.layer_idx,
                state_idx=self.CONV_STATE_IDX[name],
                conv_kernel_size=self.conv_kernel_size,
            )
            if continue_from_cache:
                # ``window`` is the cached last ``W`` inputs followed by ``x``: the last ``T`` conv outputs are new.
                out = F.conv1d(window.to(x.dtype), conv.weight, groups=weight.shape[0])[..., -x.shape[1] :]
                return F.silu(out).transpose(1, 2)
        return self.veomni_causal_conv1d(x, weight, None, cu_seqlens, activation="silu")

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        cache: Cache | None = None,
    ) -> torch.Tensor:
        """``hidden_states`` ``[B, T, D]``. Packed training passes ``cu_seqlens`` (``B == 1``). A padded batch passes a
        2-D 0/1 ``attention_mask``: padded tokens are zeroed, so they add nothing to the state (zero q / k / v) and a
        left-padded row keeps its zero initial state until its first real token. ``cache`` (generation) stores the
        conv windows and the final recurrent state, and continues from them when it already holds a state."""
        seq_len = hidden_states.shape[1]
        if cache is not None and cu_seqlens is not None:
            raise ValueError("KDA does not support a generation cache on packed (cu_seqlens) inputs.")
        if attention_mask is not None and attention_mask.ndim == 2 and cu_seqlens is None:
            hidden_states = hidden_states * attention_mask[:, -seq_len:, None].to(hidden_states.dtype)
        continue_from_cache = cache is not None and cache.has_previous_state(self.layer_idx)

        q = self._conv(self.q_proj(hidden_states), "q", cu_seqlens, cache, continue_from_cache)
        k = self._conv(self.k_proj(hidden_states), "k", cu_seqlens, cache, continue_from_cache)
        v = self._conv(self.v_proj(hidden_states), "v", cu_seqlens, cache, continue_from_cache)
        g = self.f_proj(hidden_states)
        beta = self.b_proj(hidden_states).sigmoid()
        if self.allow_neg_eigval:
            beta = beta * 2.0

        head_k = (*q.shape[:-1], self.num_heads, self.head_k_dim)
        q, k, g = q.reshape(head_k), k.reshape(head_k), g.view(head_k)
        v = v.reshape(*v.shape[:-1], self.num_heads, self.head_v_dim)

        initial_state = cache.layers[self.layer_idx].recurrent_states[0] if continue_from_cache else None
        use_recurrent = not self.training and seq_len <= FUSED_RECURRENT_MAX_LEN
        kda = self.veomni_fused_recurrent_kda if use_recurrent else self.veomni_chunk_kda
        o, final_state = kda(
            q,
            k,
            v,
            g,
            beta,
            self.A_log,
            self.dt_bias,
            initial_state,
            cu_seqlens,
            output_final_state=cache is not None,
            use_qk_l2norm_in_kernel=True,
        )
        if cache is not None:
            cache.update_recurrent_state(final_state, self.layer_idx)

        gate = self.g_proj(hidden_states).view(*o.shape)
        return self.o_proj(self.o_norm(o, gate).flatten(-2))


class Qwen3HybridAttention(Qwen3Attention):
    """Qwen3 full attention; with ``full_attention_use_rope=False`` the rope op is the identity (NoPE)."""

    def __init__(self, config: Qwen3HybridConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        if not config.full_attention_use_rope:
            self.veomni_rope = _no_rope


class Qwen3HybridDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Qwen3HybridConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_type = config.layer_types[layer_idx]
        if self.layer_type == LINEAR_ATTENTION:
            self.linear_attn = KimiDeltaAttention(config, layer_idx)
        else:
            self.self_attn = Qwen3HybridAttention(config, layer_idx)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        linear_attn_cu_seqlens: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        """``attention_mask`` is the full-attention mask; KDA layers read the 2-D ``padding_mask`` and the
        packed ``linear_attn_cu_seqlens`` instead."""
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        if self.layer_type == LINEAR_ATTENTION:
            hidden_states = self.linear_attn(
                hidden_states, attention_mask=padding_mask, cu_seqlens=linear_attn_cu_seqlens
            )
        else:
            # SDPA / eager cannot read packed cu_seqlens: they get an isolating dense mask instead.
            kwargs, attention_mask = prepare_dense_attention_inputs(
                kwargs,
                impl=self.self_attn.veomni_attn.impl,
                attention_mask=attention_mask,
                hidden_states=hidden_states,
            )
            hidden_states, _ = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                position_embeddings=position_embeddings,
                **kwargs,
            )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


class Qwen3HybridPreTrainedModel(PreTrainedModel):
    config: Qwen3HybridConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen3HybridDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _supports_attention_backend = True
    # Recurrent KDA state: the cache cannot be cropped / rolled back token by token (no assisted decoding).
    _is_stateful = True
    _can_record_outputs = {
        "hidden_states": Qwen3HybridDecoderLayer,
        "attentions": Qwen3HybridAttention,
    }

    @torch.no_grad()
    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, KimiDeltaAttention):
            # FLA / Kimi-Linear init: per-head decay rate exp(A_log) ~ U(1, 16), zero gate bias, and PyTorch's
            # default conv init (U(+-1/sqrt(width))) rather than the std-0.02 normal the base applied to Conv1d.
            init.copy_(module.A_log, torch.empty_like(module.A_log).uniform_(1, 16).log_())
            init.zeros_(module.dt_bias)
            for conv in (module.q_conv1d, module.k_conv1d, module.v_conv1d):
                bound = 1.0 / conv.kernel_size[0] ** 0.5
                init.uniform_(conv.weight, -bound, bound)


class Qwen3HybridModel(Qwen3HybridPreTrainedModel):
    def __init__(self, config: Qwen3HybridConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Qwen3HybridDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config=config) if config.full_attention_use_rope else None
        self.gradient_checkpointing = False
        self.post_init()

    @merge_with_config_defaults
    @capture_outputs
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        if use_cache and past_key_values is None:
            # One cache layer per ``layer_types`` entry: DynamicLayer (KV) for full attention, LinearAttentionLayer
            # (``config.number_of_conv_states`` conv windows + recurrent state) for KDA.
            past_key_values = DynamicCache(config=self.config)
        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            position_ids = position_ids.unsqueeze(0)

        # Packed (dyn_bsz) batches carry varlen metadata from the collator; KDA must restart its state at each
        # sample boundary exactly as flash attention does.
        linear_attn_cu_seqlens = kwargs.pop("linear_attn_cu_seq_lens_q", None)
        if linear_attn_cu_seqlens is None:
            linear_attn_cu_seqlens = kwargs.get("cu_seq_lens_q")

        causal_mask = create_causal_mask(
            config=self.config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )
        hidden_states = inputs_embeds
        position_embeddings = (
            self.rotary_emb(hidden_states, position_ids) if self.rotary_emb is not None else (None, None)
        )

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                padding_mask=attention_mask,
                linear_attn_cu_seqlens=linear_attn_cu_seqlens,
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states, past_key_values=past_key_values if use_cache else None
        )


class Qwen3HybridForCausalLM(Qwen3HybridPreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}
    _fsdp_plan = {"lm_head": "keep_full_weight"}

    def __init__(self, config: Qwen3HybridConfig):
        super().__init__(config)
        self.model = Qwen3HybridModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.veomni_ce = VeomniOp(
            "cross_entropy_loss", "standard", resolve_op_impl("cross_entropy_loss_implementation")
        )
        self.loss_function = partial(ForCausalLMLoss, op=self.veomni_ce)
        self.post_init()

    # Same head and loss path as Qwen3 (fused linear cross-entropy on ``labels``).
    forward = Qwen3ForCausalLM.forward


__all__ = [
    "KimiDeltaAttention",
    "Qwen3HybridForCausalLM",
    "Qwen3HybridModel",
    "Qwen3HybridPreTrainedModel",
]
