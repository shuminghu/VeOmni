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

"""Kimi Delta Attention (KDA) kernels.

Two families share this package and one calling convention: FLA layouts ``q/k/g [B, T, H, K]``,
``v [B, T, H, V]``, ``beta [B, T, H]`` (already sigmoid-ed), the *raw* gate ``g`` plus ``A_log [H]`` and
``dt_bias [H * K]`` (the log-space decay ``-exp(A_log) * softplus(g + dt_bias)`` is computed inside the op),
optional ``initial_state [N, H, K, V]`` and varlen ``cu_seqlens [N + 1]`` (batch 1). Both return
``(o, final_state)``.

``chunk_kda`` is the training kernel (chunked WY form, forward and backward). ``fused_recurrent_kda`` is the
token-recurrent kernel for short inference inputs; its FLA row is forward only. ``eager`` rows are the FP32
token recurrence (``reference.py``), the numerical reference for both.
"""

from ...platform import GpuKernelRequirement
from ...registry import register_op
from .chunk_kda.standard import eager as chunk_eager
from .chunk_kda.standard import fla as chunk_fla
from .fused_recurrent_kda.standard import eager as recurrent_eager
from .fused_recurrent_kda.standard import fla as recurrent_fla


_GPU = GpuKernelRequirement()

register_op(
    "chunk_kda",
    "standard",
    "eager",
    description="PyTorch FP32 token recurrence of Kimi Delta Attention",
    wrapper=chunk_eager.wrapper,
)

register_op(
    "chunk_kda",
    "standard",
    "fla",
    description="flash-linear-attention chunked Kimi Delta Attention with variable-length support",
    wrapper=chunk_fla.wrapper,
    requirement=_GPU,
    requires=("fla",),
)

register_op(
    "fused_recurrent_kda",
    "standard",
    "eager",
    description="PyTorch FP32 token recurrence of Kimi Delta Attention",
    wrapper=recurrent_eager.wrapper,
)

register_op(
    "fused_recurrent_kda",
    "standard",
    "fla",
    description="flash-linear-attention fused token-recurrent Kimi Delta Attention (forward only)",
    wrapper=recurrent_fla.wrapper,
    requirement=_GPU,
    requires=("fla",),
)
