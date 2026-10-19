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

"""Qwen3-Hybrid configuration: Qwen3 with most attention layers replaced by Kimi Delta Attention (KDA)."""

from huggingface_hub.dataclasses import strict
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config


LINEAR_ATTENTION = "linear_attention"
FULL_ATTENTION = "full_attention"


@strict
class Qwen3HybridConfig(Qwen3Config):
    """Qwen3 decoder whose layers alternate KDA linear attention and full softmax attention.

    Every Qwen3 field keeps its meaning; the full-attention layers and every MLP / norm are the Qwen3 ones.

    Args:
        full_attention_interval: With ``layer_types`` unset, layer ``i`` is full attention iff
            ``(i + 1) % full_attention_interval == 0`` (default 4: three KDA layers, then one attention layer).
        full_attention_use_rope: ``False`` (default) runs the full-attention layers without positional
            encoding (NoPE); the causal KDA layers carry the position information.
        linear_num_heads: KDA heads (query/key heads == value heads).
        linear_head_dim: KDA query/key head dimension (FLA kernels support <= 256).
        linear_value_head_dim: KDA value head dimension.
        linear_conv_kernel_dim: Depthwise causal short-convolution width on q, k and v.
        linear_gate_low_rank_dim: Rank of the low-rank forget-gate and output-gate projections;
            ``None`` uses ``linear_value_head_dim`` (the FLA / Kimi-Linear choice).
        linear_allow_neg_eigval: Scale ``beta`` to ``(0, 2)`` so the transition can have negative eigenvalues.
    """

    model_type = "qwen3_hybrid"

    full_attention_interval: int = 4
    full_attention_use_rope: bool = False
    linear_num_heads: int = 7
    linear_head_dim: int = 64
    linear_value_head_dim: int = 64
    linear_conv_kernel_dim: int = 4
    linear_gate_low_rank_dim: int | None = None
    linear_allow_neg_eigval: bool = False
    use_cache: bool = True

    @property
    def number_of_conv_states(self) -> int:
        """Short-conv states per KDA layer in the generation cache (q, k and v), read by ``DynamicCache``."""
        return 3

    def __post_init__(self, **kwargs):
        if self.layer_types is None:
            self.layer_types = [
                FULL_ATTENTION if (i + 1) % self.full_attention_interval == 0 else LINEAR_ATTENTION
                for i in range(self.num_hidden_layers)
            ]
        super().__post_init__(**kwargs)
        unknown = set(self.layer_types) - {LINEAR_ATTENTION, FULL_ATTENTION}
        if unknown:
            raise ValueError(f"Qwen3-Hybrid layer_types must be {LINEAR_ATTENTION!r} or {FULL_ATTENTION!r}: {unknown}")
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError(f"len(layer_types)={len(self.layer_types)} != num_hidden_layers={self.num_hidden_layers}")
        if self.linear_head_dim > 256:
            raise ValueError(f"linear_head_dim={self.linear_head_dim}: the FLA KDA kernels support at most 256.")


__all__ = ["Qwen3HybridConfig"]
